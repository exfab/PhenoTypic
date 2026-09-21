"""The refusal contract every identity-IO backend must satisfy.

Written once and parameterized over backends. The ``native`` lane is whichever
backend this platform actually selects -- POSIX on Linux and macOS, Windows on
Windows. The ``fake-windows`` lane runs the Windows backend against the
in-memory handle model in :mod:`tests.unit.sdk_._identity_io_fake`, so the
Windows walk, read, listing and stream logic is exercised on every platform.
Passing both lanes is the definition of a correct backend.
"""

from __future__ import annotations

import os
from unittest import mock
from pathlib import Path
from typing import Callable, ContextManager

import pytest

from phenotypic.sdk_ import _identity_io, _identity_io_windows
from tests.unit.sdk_._identity_io_fake import memory_api_factory

pytestmark = pytest.mark.platform_io

BackendOpener = Callable[[Path], ContextManager[_identity_io.HeldDirectory]]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "store" / "nested").mkdir(parents=True)
    (tmp_path / "store" / "root.json").write_bytes(b'{"ok": true}')
    (tmp_path / "store" / "nested" / "chunk").write_bytes(b"0123456789")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_bytes(b"secret")
    return tmp_path


@pytest.fixture
def native_backend() -> BackendOpener:
    """The backend this platform selects, with no model standing in for it."""
    if not _identity_io.identity_io_available():
        pytest.fail("no identity-IO backend on this platform")
    return _identity_io.open_identity_directory


@pytest.fixture(params=["native", "fake-windows"])
def backend(request: pytest.FixtureRequest) -> BackendOpener:
    """Run the contract against the live backend and the in-memory model."""
    if request.param == "native":
        if not _identity_io.identity_io_available():
            pytest.fail("no identity-IO backend on this platform")
        return _identity_io.open_identity_directory

    def _open(path: Path) -> ContextManager[_identity_io.HeldDirectory]:
        return _identity_io_windows.open_identity_directory(
            path, api=memory_api_factory(path)
        )

    return _open


def test_a_held_directory_reads_its_own_regular_files(
    backend: BackendOpener, tree: Path
) -> None:
    with backend(tree / "store") as held:
        assert held.read_regular_bytes("root.json") == b'{"ok": true}'


def test_a_child_directory_stays_held(
    backend: BackendOpener, tree: Path
) -> None:
    with backend(tree / "store") as held:
        nested = held.child_directory("nested")
        assert nested.read_regular_bytes("chunk") == b"0123456789"


def test_reading_yields_the_stat_of_the_same_open_file(
    backend: BackendOpener, tree: Path
) -> None:
    """Both ``store_publication_token`` branches fold this ``stat_result``.

    The probe confirmed ``os.fstat`` on an adopted descriptor agrees exactly
    with ``os.stat`` on the path, which is what makes the handle branch and
    the path branch agree by construction instead of 409-ing every tile.
    """
    expected = os.stat(tree / "store" / "root.json")
    with backend(tree / "store") as held:
        payload, actual = held.read_regular_with_stat("root.json")
    assert payload == b'{"ok": true}'
    assert (actual.st_ino, actual.st_size, actual.st_mtime_ns) == (
        expected.st_ino,
        expected.st_size,
        expected.st_mtime_ns,
    )


def test_listing_excludes_dot_entries(
    backend: BackendOpener, tree: Path
) -> None:
    """Windows reports ``.`` and ``..``; POSIX's ``os.listdir`` reports neither.

    The 2026-09-20 probe observed ``['.', '..', 'alpha.json', 'beta.json']``
    from a real directory handle, so the model reports them too and the filter
    is load-bearing on both lanes.
    """
    with backend(tree / "store") as held:
        assert sorted(held.list_names()) == ["nested", "root.json"]


def test_listing_spans_multiple_buffer_fills(
    backend: BackendOpener, tmp_path: Path
) -> None:
    """A 64 KiB fill holds a few hundred entries; a transition dir holds more.

    Without this the continuation branch of the listing loop never runs on a
    small fixture, and a truncated listing reads as "no evidence" rather than
    as a failure.
    """
    directory = tmp_path / "many"
    directory.mkdir()
    expected = {f"entry_{index:05d}.json" for index in range(2000)}
    for name in expected:
        (directory / name).write_bytes(b"x")
    with backend(directory) as held:
        assert set(held.list_names()) == expected


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_a_non_canonical_component_is_refused(
    backend: BackendOpener, tree: Path, name: str
) -> None:
    """Refusal 1. Traversal must die at the component, not the resolved path."""
    with backend(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes(name)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_a_symlinked_entry_is_refused(
    backend: BackendOpener, tree: Path
) -> None:
    """Refusal 2. The link points outside; following it would leak `secret`."""
    link = tree / "store" / "escape.json"
    try:
        link.symlink_to(tree / "outside" / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not create a symlink here")
    with backend(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("escape.json")


def test_a_directory_is_refused_where_a_file_is_required(
    backend: BackendOpener, tree: Path
) -> None:
    """Refusal 3, and it must be refusal 3 that fires.

    Matching the message is not pedantry. A directory has ``st_nlink >= 2``,
    so the multi-link guard refuses it first and this test passed with the
    type check deleted -- found by mutation, not by reading. A directory on a
    filesystem that reports ``st_nlink == 1`` (some network and FUSE mounts)
    would then reach the read with nothing left to stop it.
    """
    with backend(tree / "store") as held:
        with pytest.raises(
            _identity_io.IdentityRefused, match="not a regular file"
        ):
            held.read_regular_bytes("nested")


def test_a_read_torn_by_a_concurrent_write_is_refused(tree: Path) -> None:
    """The torn-read guard, which no lane reached until this test.

    ``read_regular_with_stat`` stats the same open description before and
    after the read so the publication token cannot be computed over two
    different generations of a file. Nothing in the suite provoked that, so
    the branch was dead weight a reader would assume was covered -- it is
    simulated here by making the second stat disagree with the first.
    """
    from phenotypic.sdk_ import _identity_io_posix

    real_fstat = os.fstat
    calls = {"n": 0}

    class _Grown:
        """The same stat, reporting a larger file on the second look."""

        def __init__(self, base: os.stat_result) -> None:
            self._base = base

        def __getattr__(self, name: str) -> object:
            return getattr(self._base, name)

        @property
        def st_size(self) -> int:
            return self._base.st_size + 1

    def _fstat(fd: int) -> object:
        calls["n"] += 1
        result = real_fstat(fd)
        return _Grown(result) if calls["n"] > 1 else result

    with _identity_io.open_identity_directory(tree / "store") as held:
        with mock.patch.object(_identity_io_posix.os, "fstat", _fstat):
            with pytest.raises(
                _identity_io.IdentityRefused, match="changed during the read"
            ):
                held.read_regular_with_stat("root.json")


def test_a_file_is_refused_where_a_directory_is_required(
    backend: BackendOpener, tree: Path
) -> None:
    """Refusal 3, the other direction."""
    with backend(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.child_directory("root.json")


def test_a_multi_link_file_is_refused(
    backend: BackendOpener, tree: Path
) -> None:
    """Refusal 4. A hard link is a second name for authority bytes."""
    try:
        os.link(tree / "store" / "root.json", tree / "store" / "alias.json")
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem will not create a hard link")
    with backend(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("alias.json")


def test_a_missing_entry_raises_file_not_found(
    backend: BackendOpener, tree: Path
) -> None:
    """Refusal 6: absence is evidence, not failure; callers branch on it."""
    with backend(tree / "store") as held:
        with pytest.raises(FileNotFoundError):
            held.read_regular_bytes("absent.json")


def test_max_bytes_bounds_a_read(
    backend: BackendOpener, tree: Path
) -> None:
    with backend(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("root.json", max_bytes=4)


def test_a_missing_root_raises_file_not_found(
    backend: BackendOpener, tree: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        with backend(tree / "absent"):
            pass


def test_a_stream_outlives_the_hold(
    backend: BackendOpener, tree: Path
) -> None:
    """Browse hands this stream to send_file and returns; the hold is gone."""
    with backend(tree / "store") as held:
        stream = held.child_directory("nested").open_regular_stream("chunk")
    try:
        assert stream.read() == b"0123456789"
    finally:
        stream.close()


def test_a_stream_is_seekable_for_range_requests(
    backend: BackendOpener, tree: Path
) -> None:
    with backend(tree / "store") as held:
        stream = held.child_directory("nested").open_regular_stream("chunk")
    try:
        stream.seek(4)
        assert stream.read(3) == b"456"
    finally:
        stream.close()


def test_a_swapped_directory_fails_reverification(
    native_backend: BackendOpener, tree: Path
) -> None:
    """Refusal 5. The held identity, not the path, is the authority.

    Native lane only: the in-memory model is a snapshot taken when the hold
    opens, so it cannot observe a rename made afterwards. Running it there
    would assert nothing while reporting a pass.
    """
    with native_backend(tree / "store") as held:
        nested = held.child_directory("nested")
        os.rename(tree / "store" / "nested", tree / "store" / "gone")
        (tree / "store" / "nested").mkdir()
        with pytest.raises(_identity_io.IdentityRefused):
            nested.reverify()


def test_a_streamed_member_does_not_lock_the_file(
    native_backend: BackendOpener, tree: Path
) -> None:
    """A read-only viewer must never block a concurrent CLI run.

    Native lane only: what is under test is the operating system's sharing
    mode, and the model has none. On the fake lane the write would succeed
    whatever mask the backend had requested.
    """
    with native_backend(tree / "store") as held:
        stream = held.open_regular_stream("root.json")
    try:
        (tree / "store" / "root.json").write_bytes(b'{"ok": false}')
    finally:
        stream.close()


def test_the_directory_entry_name_offset_matches_the_probe() -> None:
    """A wrong offset mis-slices every name, silently, only on Windows.

    ``list_names`` reads each entry's name at
    ``offset + _FileFullDirInfo.FileName.offset``. The 2026-09-20 probe
    measured that offset as 68 against a real Win32 buffer. Nothing else here
    parses the struct -- the fake lane overrides ``list_names`` outright -- so
    without this the declaration's agreement with the probe rests on someone
    having counted the fields correctly by eye. A field inserted, reordered or
    mis-typed would leave every Linux lane green and hand the real runner
    garbage names of the shape the probe saw when it ran the wrong layout:
    ``['\\x00', '\\x14BETA~1.J', ...]``.
    """
    assert _identity_io_windows._FileFullDirInfo.FileName.offset == 68


def test_the_windows_backend_binds_supported_before_the_facade_import() -> None:
    """A journal-first import on Windows re-enters this module mid-execution.

    ``_identity_io`` calls ``_select_backend()`` at its own bottom and reads
    ``SUPPORTED`` off ``_identity_io_windows``. ``_windows_metadata_journal``
    imports ``_identity_io_windows`` at *its* top, so on Windows an
    ``import phenotypic.sdk_._windows_metadata_journal`` reaches
    ``_select_backend()`` while ``_identity_io_windows`` is still running its
    own module body. That resolves only because ``SUPPORTED`` is bound before
    the facade import at the bottom of the file; move the import back to the
    top and the whole package stops importing on Windows with an
    ``AttributeError`` nothing here would catch.

    Linux never takes the ``os.name == "nt"`` branch, so there is no way to
    execute the failure. Reading the source is what is left.
    """
    source = Path(_identity_io_windows.__file__).read_text(encoding="utf-8")
    assert source.index("\nSUPPORTED = ") < source.index(
        "\nfrom ._identity_io import ("
    )


def test_a_genuine_permission_failure_is_not_reported_as_a_refusal(
    tree: Path,
) -> None:
    """``ERROR_ACCESS_DENIED`` alone never proves a type mismatch.

    ``RtlNtStatusToDosError`` folds ``STATUS_FILE_IS_A_DIRECTORY`` onto the
    same code a denied ACL produces, so the backend disambiguates by re-asking
    the held parent for the same name as a directory. This pins the other half
    of that branch: when the entry is *not* a directory, the original
    ``OSError`` must survive rather than being relabelled a refusal that a
    caller would read as "this store is unsafe".

    Fake lane only -- there is no way to provoke a Win32 error code from the
    POSIX backend.
    """
    api = memory_api_factory(tree / "store")
    real_open_regular_read = api.open_regular_read

    def _denied(parent: int, name: str) -> int:
        if name == "root.json":
            raise PermissionError(5, "access denied")
        return real_open_regular_read(parent, name)

    api.open_regular_read = _denied  # type: ignore[method-assign]
    with _identity_io_windows.open_identity_directory(
        tree / "store", api=api
    ) as held:
        with pytest.raises(PermissionError):
            held.read_regular_bytes("root.json")
