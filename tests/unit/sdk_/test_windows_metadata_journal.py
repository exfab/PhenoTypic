"""Handle-bound Windows metadata-journal adapter contracts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


class _FakeCFunction:
    """ctypes-like callable whose declared ABI can be inspected on Linux."""

    def __call__(self, *_args: object) -> int:
        return 1


def test_ctypes_binding_declares_pointer_width_safe_signatures() -> None:
    """Missing argtypes can truncate HANDLE and pointer arguments on 64-bit Windows."""
    from phenotypic.sdk_._windows_metadata_journal import _CtypesWindowsApi

    api = _CtypesWindowsApi.__new__(_CtypesWindowsApi)
    kernel_names = (
        "CreateFileW",
        "GetFileInformationByHandleEx",
        "SetFileInformationByHandle",
        "FlushFileBuffers",
        "ReadFile",
        "WriteFile",
        "SetFilePointerEx",
        "GetFileSizeEx",
        "CloseHandle",
        "LockFileEx",
        "UnlockFileEx",
    )
    api.kernel32 = SimpleNamespace(
        **{name: _FakeCFunction() for name in kernel_names}
    )
    api.ntdll = SimpleNamespace(
        NtCreateFile=_FakeCFunction(),
        RtlNtStatusToDosError=_FakeCFunction(),
    )

    api._bind()

    for name in (*kernel_names, "NtCreateFile", "RtlNtStatusToDosError"):
        assert getattr(getattr(api, name), "argtypes", None), name


class _MemoryWindowsApi:
    """In-memory NT-handle model that rejects path-based child operations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.handles: dict[int, tuple[str, ...]] = {}
        self.directories: set[tuple[str, ...]] = set()
        self.files: dict[tuple[str, ...], bytes] = {}
        self.reparse: set[tuple[str, ...]] = set()
        self.flushed: list[tuple[str, ...]] = []
        self._next_handle = 10
        anchor = (root.anchor,)
        self.directories.add(anchor)
        cursor = anchor
        for part in root.parts[1:]:
            cursor += (part,)
            self.directories.add(cursor)

    def _handle(self, path: tuple[str, ...]) -> int:
        self._next_handle += 1
        self.handles[self._next_handle] = path
        return self._next_handle

    @staticmethod
    def _relative_name(name: str) -> None:
        assert name not in {"", ".", ".."}
        assert "/" not in name and "\\" not in name

    def open_anchor(self, anchor: str, *, share_delete: bool) -> int:
        assert share_delete is False
        return self._handle((anchor,))

    def open_directory(
        self,
        parent: int,
        name: str,
        *,
        create: bool,
        share_delete: bool,
    ) -> int:
        assert share_delete is False
        self._relative_name(name)
        path = self.handles[parent] + (name,)
        if path not in self.directories:
            if not create:
                raise FileNotFoundError(name)
            self.directories.add(path)
        return self._handle(path)

    def open_file(
        self,
        parent: int,
        name: str,
        *,
        create_new: bool,
        share_delete: bool,
    ) -> int:
        assert share_delete is False
        self._relative_name(name)
        path = self.handles[parent] + (name,)
        if create_new:
            if path in self.files:
                raise FileExistsError(name)
            self.files[path] = b""
        elif path not in self.files:
            raise FileNotFoundError(name)
        return self._handle(path)

    def handle_info(self, handle: int):
        from phenotypic.sdk_._windows_metadata_journal import WindowsHandleInfo

        path = self.handles[handle]
        return WindowsHandleInfo(
            volume_serial=1,
            file_id="/".join(path).encode().ljust(16, b"\0")[:16],
            attributes=0x400 if path in self.reparse else 0,
            reparse_tag=0xA000000C if path in self.reparse else 0,
        )

    def write_all(self, handle: int, payload: bytes) -> None:
        self.files[self.handles[handle]] = payload

    def read_all(self, handle: int) -> bytes:
        return self.files[self.handles[handle]]

    def flush(self, handle: int) -> None:
        self.flushed.append(self.handles[handle])

    def rename(
        self,
        handle: int,
        parent: int,
        name: str,
        *,
        replace: bool,
    ) -> None:
        self._relative_name(name)
        source = self.handles[handle]
        destination = self.handles[parent] + (name,)
        if destination in self.files and not replace:
            raise FileExistsError(name)
        self.files[destination] = self.files.pop(source)
        for held, path in tuple(self.handles.items()):
            if path == source:
                self.handles[held] = destination

    def delete(self, handle: int) -> None:
        self.files.pop(self.handles[handle], None)

    def close(self, handle: int) -> None:
        self.handles.pop(handle, None)

    @contextmanager
    def lock(self, _handle: int):
        yield


def test_session_publishes_and_reads_only_through_relative_held_handles(
    tmp_path: Path,
) -> None:
    """A full-path child open or delete-sharing mutation breaks this test."""
    from phenotypic.sdk_._windows_metadata_journal import WindowsJournalSession

    root = tmp_path / "output"
    root.mkdir()
    api = _MemoryWindowsApi(root)
    status = root / ".phenotypic" / "metadata_migration" / "status.json"

    with WindowsJournalSession(root, api=api) as session:
        session.publish_absent_bytes(
            status, b'{"state":"complete"}\n', role="status"
        )
        assert session.read_bytes(status, role="status") == (
            b'{"state":"complete"}\n'
        )

    assert b'{"state":"complete"}\n' in api.files.values()
    assert len(api.flushed) >= 2


def test_session_no_clobber_preserves_competing_authority(tmp_path: Path) -> None:
    """Changing immutable publication to replace=True corrupts the competitor."""
    from phenotypic.sdk_._windows_metadata_journal import WindowsJournalSession

    root = tmp_path / "output"
    root.mkdir()
    api = _MemoryWindowsApi(root)
    receipt = root / ".phenotypic" / "metadata_migration" / "receipt.json"

    with WindowsJournalSession(root, api=api) as session:
        session.publish_absent_bytes(receipt, b"competitor", role="receipt")
        with pytest.raises(ValueError, match="Competing"):
            session.publish_absent_bytes(receipt, b"replacement", role="receipt")
        assert session.read_bytes(receipt, role="receipt") == b"competitor"


def test_session_rejects_reparse_component_before_publication(
    tmp_path: Path,
) -> None:
    """Omitting handle-level reparse inspection redirects authority traversal."""
    from phenotypic.sdk_._windows_metadata_journal import WindowsJournalSession

    root = tmp_path / "output"
    root.mkdir()
    api = _MemoryWindowsApi(root)
    root_tuple = (root.anchor, *root.parts[1:])
    api.reparse.add(root_tuple + (".phenotypic",))
    status = root / ".phenotypic" / "metadata_migration" / "status.json"

    with WindowsJournalSession(root, api=api) as session:
        with pytest.raises(ValueError, match="reparse"):
            session.publish_absent_bytes(status, b"unsafe", role="status")

    assert b"unsafe" not in api.files.values()


def test_public_migration_routes_receipt_replay_through_windows_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leaving the public fallback disconnected makes fresh Windows migration fail."""
    import phenotypic.sdk_._metadata_migration as migration
    from phenotypic.sdk_ import (
        BundleLayout,
        migrate_preflighted_metadata_bundle,
        preflight_metadata_schema,
    )
    from phenotypic.sdk_._metadata_migration import NON_IMAGE_KINDS
    from phenotypic.sdk_._windows_metadata_journal import WindowsJournalSession

    output = tmp_path / "output"
    deliverables = output / "deliverables"
    measurements = output / "results" / "dataset" / "measurements"
    deliverables.mkdir(parents=True)
    measurements.mkdir(parents=True)
    target = measurements / "_dataset_aggregated.parquet"
    pd.DataFrame({"MetadataGenetic_Strain": ["WT"]}).to_parquet(
        target, index=False
    )
    layout = BundleLayout(deliverables_base=deliverables, output_root=output)
    report = preflight_metadata_schema(layout, kinds=NON_IMAGE_KINDS)
    api = _MemoryWindowsApi(output)

    @contextmanager
    def open_fake_session(_root: Path):
        with WindowsJournalSession(output, api=api) as session:
            yield session

    monkeypatch.setattr(migration, "_JOURNAL_DIR_FD_SUPPORTED", False)
    monkeypatch.setattr(
        migration,
        "_windows_journal_capabilities_available",
        lambda: True,
        raising=False,
    )
    monkeypatch.setattr(
        migration,
        "open_windows_journal_session",
        open_fake_session,
        raising=False,
    )

    result = migrate_preflighted_metadata_bundle(
        layout, report=report, kinds=NON_IMAGE_KINDS
    )

    assert result.status == "applied", result.conflicts
    migrated = pd.read_parquet(target)
    assert list(migrated.columns) == ["Metadata_Strain"]
    assert any(b'"state": "complete"' in payload for payload in api.files.values())
