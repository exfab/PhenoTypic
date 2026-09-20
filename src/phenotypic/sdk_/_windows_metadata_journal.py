"""Handle-bound Windows authority I/O for metadata migration journals.

The public migration module owns receipt semantics. This module owns only the
Windows namespace-safety boundary: every child is opened relative to a held
directory handle, reparse points are rejected from handle metadata, directory
handles deny delete sharing, and publication renames an already-open temporary
handle relative to its held parent without replacement.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._identity_io_windows import (
    WindowsHandleInfo,
    WindowsJournalUnavailable,
    _CtypesWindowsApi,
    _FILE_ATTRIBUTE_REPARSE_POINT,
    _FileDispositionInfo,  # noqa: F401  (re-exported for callers)
    _FileRenameInfoHeader,  # noqa: F401  (re-exported for callers)
    _IoStatusBlock,  # noqa: F401  (re-exported for callers)
    _Overlapped,  # noqa: F401  (re-exported for callers)
    _WindowsApi,
)


@dataclass(frozen=True)
class _HeldDirectory:
    handle: int
    identity: tuple[int, bytes]


class WindowsJournalSession:
    """Hold and validate the authority directory chain for one transaction."""

    def __init__(self, root: Path, *, api: _WindowsApi | None = None) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        self._api: _WindowsApi = api or _CtypesWindowsApi()
        self._directories: dict[Path, _HeldDirectory] = {}
        self._entered = False

    def __enter__(self) -> WindowsJournalSession:
        anchor = self.root.anchor
        if not anchor:
            raise WindowsJournalUnavailable(
                "Windows metadata journal root has no filesystem anchor"
            )
        anchor_path = Path(anchor)
        handle = self._api.open_anchor(anchor, share_delete=False)
        try:
            held = self._validated_directory(handle, role="filesystem anchor")
        except BaseException:
            self._api.close(handle)
            raise
        self._directories[anchor_path] = held
        self._entered = True
        try:
            self._directory(self.root, create=False)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close held directory handles from leaf to anchor."""
        if not self._entered:
            return
        for held in reversed(tuple(self._directories.values())):
            self._api.close(held.handle)
        self._directories.clear()
        self._entered = False

    def _validated_directory(self, handle: int, *, role: str) -> _HeldDirectory:
        info = self._api.handle_info(handle)
        if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"Windows metadata {role} is a reparse point")
        if len(info.file_id) != 16 or not any(info.file_id):
            raise WindowsJournalUnavailable(
                f"Windows metadata {role} lacks a stable FILE_ID_INFO identity"
            )
        return _HeldDirectory(handle=handle, identity=info.identity)

    def _verify_directories(self) -> None:
        for path, held in self._directories.items():
            info = self._api.handle_info(held.handle)
            if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(
                    f"Windows metadata directory became a reparse point: {path}"
                )
            if info.identity != held.identity:
                raise ValueError(
                    f"Windows metadata directory identity changed: {path}"
                )

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        candidate = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"Windows metadata authority escapes its root: {candidate}"
            ) from exc
        for part in relative.parts:
            if part in {"", ".", ".."} or "/" in part or "\\" in part:
                raise ValueError(
                    f"Unsafe Windows metadata authority component: {part!r}"
                )
        return relative.parts

    def _directory(self, path: Path, *, create: bool) -> _HeldDirectory:
        if not self._entered:
            raise RuntimeError("Windows journal session is not open")
        candidate = Path(os.path.abspath(os.fspath(path)))
        cached = self._directories.get(candidate)
        if cached is not None:
            self._verify_directories()
            return cached
        parts = self._relative_parts(candidate)
        current_path = self.root
        root_held = self._directories.get(current_path)
        if root_held is None:
            anchor_path = Path(self.root.anchor)
            root_held = self._directories[anchor_path]
            current_path = anchor_path
            root_parts = self.root.parts[1:]
            walk_parts = root_parts
        else:
            walk_parts = parts
        held = root_held
        for part in walk_parts:
            next_path = current_path / part
            cached = self._directories.get(next_path)
            if cached is None:
                handle = self._api.open_directory(
                    held.handle,
                    part,
                    create=create and next_path.is_relative_to(self.root),
                    share_delete=False,
                )
                try:
                    cached = self._validated_directory(
                        handle, role=f"directory component {part!r}"
                    )
                except BaseException:
                    self._api.close(handle)
                    raise
                self._directories[next_path] = cached
            held = cached
            current_path = next_path
        if current_path != candidate:
            # The root was already cached; walk only the requested relative tail.
            held = self._directories[self.root]
            current_path = self.root
            for part in parts:
                next_path = current_path / part
                cached = self._directories.get(next_path)
                if cached is None:
                    handle = self._api.open_directory(
                        held.handle,
                        part,
                        create=create,
                        share_delete=False,
                    )
                    try:
                        cached = self._validated_directory(
                            handle, role=f"directory component {part!r}"
                        )
                    except BaseException:
                        self._api.close(handle)
                        raise
                    self._directories[next_path] = cached
                held = cached
                current_path = next_path
        self._verify_directories()
        return held

    def _open_file(self, path: Path, *, create_new: bool) -> tuple[int, _HeldDirectory]:
        parts = self._relative_parts(path)
        if not parts:
            raise ValueError("Windows metadata authority path names a directory")
        parent = self._directory(Path(path).parent, create=create_new)
        handle = self._api.open_file(
            parent.handle,
            parts[-1],
            create_new=create_new,
            share_delete=False,
        )
        try:
            info = self._api.handle_info(handle)
            if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError("Windows metadata authority child is a reparse point")
            if len(info.file_id) != 16 or not any(info.file_id):
                raise WindowsJournalUnavailable(
                    "Windows metadata authority child lacks FILE_ID_INFO"
                )
            self._verify_directories()
        except BaseException:
            self._api.close(handle)
            raise
        return handle, parent

    def read_bytes(self, path: Path, *, role: str) -> bytes:
        """Read one regular authority while its directory chain is pinned."""
        handle, _parent = self._open_file(path, create_new=False)
        try:
            before = self._api.handle_info(handle)
            payload = self._api.read_all(handle)
            after = self._api.handle_info(handle)
            if before.identity != after.identity:
                raise ValueError(f"Windows metadata {role} identity changed")
            self._verify_directories()
            return payload
        finally:
            self._api.close(handle)

    def exists(self, path: Path) -> bool:
        """Return whether one handle-bound regular authority exists."""
        try:
            handle, _parent = self._open_file(path, create_new=False)
        except FileNotFoundError:
            return False
        self._api.close(handle)
        return True

    def hold_directory(self, path: Path) -> bool:
        """Pin an existing authority directory, returning false when absent."""
        try:
            self._directory(path, create=False)
        except FileNotFoundError:
            return False
        return True

    def _publish_bytes(
        self,
        path: Path,
        payload: bytes,
        *,
        role: str,
        replace: bool,
    ) -> None:
        target = Path(os.path.abspath(os.fspath(path)))
        parent = self._directory(target.parent, create=True)
        temp_name = f".{target.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
        temp_path = target.with_name(temp_name)
        handle, opened_parent = self._open_file(temp_path, create_new=True)
        if opened_parent != parent:
            self._api.close(handle)
            raise ValueError(f"Windows metadata {role} parent identity changed")
        published = False
        try:
            initial = self._api.handle_info(handle)
            self._api.write_all(handle, payload)
            self._api.flush(handle)
            self._verify_directories()
            try:
                self._api.rename(
                    handle,
                    parent.handle,
                    target.name,
                    replace=replace,
                )
            except FileExistsError as exc:
                raise ValueError(f"Competing Windows metadata {role} exists") from exc
            self._api.flush(handle)
            final = self._api.handle_info(handle)
            if final.identity != initial.identity:
                raise ValueError(
                    f"Windows metadata {role} identity changed during publication"
                )
            self._verify_directories()
            published = True
        finally:
            if not published:
                try:
                    self._api.delete(handle)
                except OSError:
                    pass
            self._api.close(handle)

    def publish_absent_bytes(self, path: Path, payload: bytes, *, role: str) -> None:
        """Publish immutable authority with handle-relative no-replace rename."""
        self._publish_bytes(path, payload, role=role, replace=False)

    def replace_bytes(self, path: Path, payload: bytes, *, role: str) -> None:
        """Replace mutable receipt state under the held writer lock."""
        self._publish_bytes(path, payload, role=role, replace=True)

    @contextmanager
    def writer_lock(self, path: Path) -> Iterator[None]:
        """Lock one handle-bound writer file for the session lifetime."""
        try:
            handle, _parent = self._open_file(path, create_new=False)
        except FileNotFoundError:
            try:
                handle, _parent = self._open_file(path, create_new=True)
            except FileExistsError:
                handle, _parent = self._open_file(path, create_new=False)
        try:
            with self._api.lock(handle):
                self._verify_directories()
                yield
                self._verify_directories()
        finally:
            self._api.close(handle)


def windows_journal_supported() -> bool:
    """Return whether all required native Windows primitives can be bound."""
    try:
        _CtypesWindowsApi()
    except WindowsJournalUnavailable:
        return False
    return True


@contextmanager
def open_windows_journal_session(root: Path) -> Iterator[WindowsJournalSession]:
    """Open one native handle-bound Windows journal transaction."""
    with WindowsJournalSession(root) as session:
        yield session


__all__ = [
    "WindowsHandleInfo",
    "WindowsJournalSession",
    "WindowsJournalUnavailable",
    "open_windows_journal_session",
    "windows_journal_supported",
]
