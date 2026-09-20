# Windows Identity-Bound Directory I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--mode recompile` and the Browse OME-Zarr store route work on Windows by routing both through one platform-neutral, identity-bound directory facade.

**Architecture:** A new private `sdk_/_identity_io.py` exposes `open_identity_directory()` and a `HeldDirectory` protocol. Two backends implement it: the existing POSIX dir-fd code, moved; and a Windows backend built on the `NtCreateFile` relative-open plumbing that already ships in `sdk_/_windows_metadata_journal.py`. The three consumers (recompile recovery, the Browse tile route, `store_publication_token`) stop branching on the platform.

**Tech Stack:** Python 3.11+, ctypes against `ntdll`/`kernel32`, pytest, `uv` as the only runner.

**Spec:** `docs/superpowers/specs/2026-09-19-windows-identity-io/design.md` — read it before Task 1; the refusal contract in it is the acceptance criterion for Tasks 1–5.

## Global Constraints

- **`uv` is the only runner.** Never bare `python`/`pip`. Tests: `uv run pytest ...`.
- **Never weaken a refusal.** Both backends must refuse all six cases in the spec's *Refusal contract*. Widening where the guarantees hold is the goal; changing what they are is out of scope.
- **Opens are always relative to a held handle/descriptor.** No backend may re-open by absolute path after the initial root open. That is the TOCTOU window these modules exist to close.
- **Read-only.** Nothing in this plan writes, creates, renames or deletes a filesystem entry. `open_identity_directory` takes no `create=` parameter.
- **Windows-only code cannot be verified on the HPCC.** Any claim about Win32 behaviour comes from a runner, never from reasoning. Tasks 0 and 9 are where that happens.
- **A skipped test is not a passing test.** No task may make its coverage conditional on the platform without the Task 8 assertion that the expected backend is active.
- **Lint and type-check every changed file:** `uv run ruff check <paths>` and `uv run mypy src/phenotypic` before each commit.

---

### Task 0: Probe the two unverified Win32 mechanisms

The spec names two mechanisms taken from documentation rather than observation. The journal's rename shipped on exactly that kind of assumption and was wrong for weeks because its tests mocked the API. This task buys certainty for about five minutes of CI.

**Files:**
- Create (throwaway, never merged): `win_identity_probe.py`, `.github/workflows/diag-windows-identity.yml` on a branch `diag/windows-identity-probe`

**Interfaces:**
- Consumes: nothing.
- Produces: two answers that Tasks 4 and 5 depend on — the working `FILE_INFO_BY_HANDLE_CLASS` value for directory listing, and whether an `open_osfhandle` stream can be range-served.

- [ ] **Step 1: Write the probe script**

```python
"""Throwaway: confirm directory listing and stream hand-off on Windows."""
import ctypes, importlib.util, io, msvcrt, os, sys, tempfile
from ctypes import wintypes
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "wj", "src/phenotypic/sdk_/_windows_metadata_journal.py"
)
wj = importlib.util.module_from_spec(spec)
sys.modules["wj"] = wj
spec.loader.exec_module(wj)

api = wj._CtypesWindowsApi()
root = Path(tempfile.mkdtemp(prefix="identity-probe-"))
(root / "d").mkdir()
for name in ("alpha.json", "beta.json"):
    (root / "d" / name).write_bytes(b"payload-" + name.encode())

anchor = api.open_anchor(str(root), share_delete=True)
directory = api.open_directory(anchor, "d", create=False, share_delete=True)


class _FullDirInfo(ctypes.Structure):
    _fields_ = [
        ("NextEntryOffset", ctypes.c_uint32),
        ("FileIndex", ctypes.c_uint32),
        ("CreationTime", ctypes.c_int64),
        ("LastAccessTime", ctypes.c_int64),
        ("LastWriteTime", ctypes.c_int64),
        ("ChangeTime", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("AllocationSize", ctypes.c_int64),
        ("FileAttributes", ctypes.c_uint32),
        ("FileNameLength", ctypes.c_uint32),
        ("EaSize", ctypes.c_uint32),
        ("FileName", ctypes.c_uint16 * 1),
    ]


class _StandardInfo(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", ctypes.c_uint32),
        ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


def listing(restart_class: int, continue_class: int) -> list[str]:
    names: list[str] = []
    buffer = ctypes.create_string_buffer(64 * 1024)
    info_class = restart_class
    while api.GetFileInformationByHandleEx(
        directory, info_class, buffer, ctypes.sizeof(buffer)
    ):
        info_class = continue_class
        offset = 0
        while True:
            entry = ctypes.cast(
                ctypes.byref(buffer, offset), ctypes.POINTER(_FullDirInfo)
            ).contents
            name_offset = offset + _FullDirInfo.FileName.offset
            name = ctypes.wstring_at(
                ctypes.byref(buffer, name_offset), entry.FileNameLength // 2
            )
            names.append(name)
            if entry.NextEntryOffset == 0:
                break
            offset += entry.NextEntryOffset
    error = ctypes.get_last_error()
    print(f"  listing ended with error {error} (18 == ERROR_NO_MORE_FILES)")
    return names


print("FileFullDirectoryRestartInfo(15)/FileFullDirectoryInfo(14):")
print("  names:", sorted(listing(15, 14)))

handle = api.open_file(directory, "alpha.json", create_new=False, share_delete=True)
fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
stream = os.fdopen(fd, "rb")
print("stream: size", os.fstat(stream.fileno()).st_size, "seekable", stream.seekable())
stream.seek(4)
print("stream: seek+read ->", stream.read(4))
api.close(directory)
api.close(anchor)
print("stream survives directory close ->", stream.read(3))

buffer = ctypes.create_string_buffer(ctypes.sizeof(_StandardInfo))
handle2 = api.open_file(
    api.open_anchor(str(root / "d"), share_delete=True),
    "beta.json",
    create_new=False,
    share_delete=True,
)
ok = api.GetFileInformationByHandleEx(handle2, 1, buffer, ctypes.sizeof(buffer))
standard = ctypes.cast(buffer, ctypes.POINTER(_StandardInfo)).contents
print("FileStandardInfo(1):", ok, "links", standard.NumberOfLinks,
      "size", standard.EndOfFile, "directory", standard.Directory)
```

- [ ] **Step 2: Add the throwaway workflow**

```yaml
name: diag-windows-identity
on:
  push:
    branches: [diag/windows-identity-probe]
permissions:
  contents: read
jobs:
  probe:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: python win_identity_probe.py
```

- [ ] **Step 3: Push the branch and read the output**

```bash
git worktree add -b diag/windows-identity-probe ../diag-identity HEAD
# copy both files in, commit, then:
GIT_SSH_COMMAND="ssh -i /rhome/anguy344/.ssh/github_agent -o IdentitiesOnly=yes -o BatchMode=yes" \
  git push -u origin diag/windows-identity-probe
gh run list --branch diag/windows-identity-probe -L 1
```

Expected: both `alpha.json` and `beta.json` listed (plus possibly `.`/`..`, which the backend must filter); the stream reports size 15, is seekable, and still reads after the directory handles close; `FileStandardInfo` reports `links 1`.

Three more checks belong in the same probe, each load-bearing for a decision above:

1. **Does `os.fstat` on an `open_osfhandle` fd return the same `st_ino` and `st_ctime_ns` as `os.stat(path)`?** B2 turns entirely on this — print both and compare. If they differ, the two `store_publication_token` branches disagree and every Browse tile request 409s.
2. **Does the read-only mask succeed where the journal's write mask fails?** Create a file, deny write on its ACL with `icacls`, then try `open_regular_read` and `open_file`. This is B5's only real test.
3. **Keep the listing probe going through `api.open_directory`**, i.e. a `FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT` NT handle — not a plain `CreateFileW` handle. The probe already does; say so, so nobody "simplifies" it into probing the wrong thing.

- [ ] **Step 4: Record the answers in the spec, delete the branch**

Append a short "Probe results (YYYY-MM-DD)" section to the spec with the observed values. If listing failed with class 15/14, retry with `FileIdBothDirectoryRestartInfo(11)`/`FileIdBothDirectoryInfo(10)` and record which worked — Task 4 uses whichever the runner accepted.

```bash
GIT_SSH_COMMAND="ssh -i /rhome/anguy344/.ssh/github_agent -o IdentitiesOnly=yes -o BatchMode=yes" \
  git push origin --delete diag/windows-identity-probe
git worktree remove --force ../diag-identity && git branch -D diag/windows-identity-probe
gh api repos/exfab/PhenoTypic/branches/diag/windows-identity-probe   # expect 404
```

- [ ] **Step 5: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-09-19-windows-identity-io/design.md
git commit -m "docs(spec): record Windows identity-IO probe results"
```

---

### Task 1: The facade and the POSIX backend

**Files:**
- Create: `src/phenotypic/sdk_/_identity_io.py`
- Create: `src/phenotypic/sdk_/_identity_io_posix.py`
- Create: `tests/unit/sdk_/test_identity_io_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces, and every later task depends on these exact names:
  - `open_identity_directory(path: Path) -> AbstractContextManager[HeldDirectory]`
  - `identity_io_available() -> bool`
  - `active_backend_name() -> str` — `"posix"`, `"windows"`, or `"unavailable"`
  - `class IdentityIoUnavailable(RuntimeError)`
  - `HeldDirectory` protocol: `path: Path`, `child_directory(name)`, `list_names()`, `read_regular_bytes(name, *, max_bytes=None)`, `open_regular_stream(name)`, `reverify()`
  - `class IdentityRefused(ValueError)` — raised for every refusal except a missing entry, which raises `FileNotFoundError`

- [ ] **Step 1: Write the failing contract test**

The suite is written once and parameterized; Task 3 adds the second backend to the fixture. Note what each test is really pinning — a test named after a mechanism instead of a guarantee is how a guard rots.

```python
"""The refusal contract every identity-IO backend must satisfy."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import _identity_io


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "store" / "nested").mkdir(parents=True)
    (tmp_path / "store" / "root.json").write_bytes(b'{"ok": true}')
    (tmp_path / "store" / "nested" / "chunk").write_bytes(b"0123456789")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_bytes(b"secret")
    return tmp_path


def test_a_held_directory_reads_its_own_regular_files(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        assert held.read_regular_bytes("root.json") == b'{"ok": true}'


def test_a_child_directory_stays_held(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        nested = held.child_directory("nested")
        assert nested.read_regular_bytes("chunk") == b"0123456789"


def test_listing_excludes_dot_entries(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        assert sorted(held.list_names()) == ["nested", "root.json"]


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_a_non_canonical_component_is_refused(tree: Path, name: str) -> None:
    """Refusal 1. Traversal must die at the component, not at the resolved path."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes(name)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_a_symlinked_entry_is_refused(tree: Path) -> None:
    """Refusal 2. The link points outside; following it would leak `secret`."""
    link = tree / "store" / "escape.json"
    try:
        link.symlink_to(tree / "outside" / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not create a symlink here")
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("escape.json")


def test_a_directory_is_refused_where_a_file_is_required(tree: Path) -> None:
    """Refusal 3."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("nested")


def test_a_file_is_refused_where_a_directory_is_required(tree: Path) -> None:
    """Refusal 3, the other direction."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.child_directory("root.json")


def test_a_multi_link_file_is_refused(tree: Path) -> None:
    """Refusal 4. A hard link is a second name for authority bytes."""
    try:
        os.link(tree / "store" / "root.json", tree / "store" / "alias.json")
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem will not create a hard link")
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("alias.json")


def test_a_missing_entry_raises_file_not_found(tree: Path) -> None:
    """Refusal 6: absence is evidence, not failure; callers branch on it."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(FileNotFoundError):
            held.read_regular_bytes("absent.json")


def test_max_bytes_bounds_a_read(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("root.json", max_bytes=4)


def test_a_swapped_directory_fails_reverification(tree: Path) -> None:
    """Refusal 5. The held identity, not the path, is the authority."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        nested = held.child_directory("nested")
        os.rename(tree / "store" / "nested", tree / "store" / "gone")
        (tree / "store" / "nested").mkdir()
        with pytest.raises(_identity_io.IdentityRefused):
            nested.reverify()


def test_a_missing_root_raises_file_not_found(tree: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with _identity_io.open_identity_directory(tree / "absent"):
            pass
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -x -q`
Expected: collection error — `cannot import name '_identity_io'`.

- [ ] **Step 3: Write the facade**

```python
"""Reach a file through directories whose identity this process holds.

Both consumers of this module -- recompile transition recovery and the Browse
store route -- must prove that the bytes they read came from the directory they
validated, not from one swapped underneath them between the check and the open.
A path-based open cannot prove that; an open relative to a held descriptor or
handle can. This module is the one place that knows how.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


class IdentityIoUnavailable(RuntimeError):
    """This platform cannot bind directory access to held identities."""


class IdentityRefused(ValueError):
    """A component, type, link count or identity failed the contract."""


@runtime_checkable
class HeldDirectory(Protocol):
    """One directory held open by identity for the life of the hold."""

    path: Path

    def child_directory(self, name: str) -> "HeldDirectory": ...

    def list_names(self) -> tuple[str, ...]: ...

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes: ...

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]: ...

    def open_regular_stream(self, name: str) -> BinaryIO: ...

    def reverify(self) -> None: ...


def validate_component(name: str) -> str:
    """Return *name* if it is one canonical path component, else refuse."""
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise IdentityRefused(f"not a canonical path component: {name!r}")
    return name


def _select_backend():
    if os.name == "posix":
        from . import _identity_io_posix

        if _identity_io_posix.SUPPORTED:
            return _identity_io_posix
    elif os.name == "nt":
        from . import _identity_io_windows

        if _identity_io_windows.SUPPORTED:
            return _identity_io_windows
    return None


# Keep this call at the BOTTOM of the module. Each backend imports
# ``HeldDirectory``/``IdentityRefused``/``validate_component`` from here at its
# own module scope, so it may only use names defined ABOVE this line: when
# ``_select_backend()`` runs, this module is still partially initialized.
_BACKEND = _select_backend()


def identity_io_available() -> bool:
    """Whether this platform can hold directories by identity."""
    return _BACKEND is not None


def active_backend_name() -> str:
    """Return ``posix``, ``windows`` or ``unavailable``."""
    return "unavailable" if _BACKEND is None else _BACKEND.BACKEND_NAME


def open_identity_directory(path: Path) -> AbstractContextManager[HeldDirectory]:
    """Hold *path* by identity, refusing to follow any link on the way in.

    Args:
        path: Absolute directory path to hold.

    Returns:
        A context manager yielding the held directory. Every descendant reached
        through it is opened relative to a handle this process holds.

    Raises:
        IdentityIoUnavailable: If no backend supports this platform.
        FileNotFoundError: If *path* does not exist.
        IdentityRefused: If *path* is not a canonical, non-link directory.
    """
    if _BACKEND is None:
        raise IdentityIoUnavailable(
            "this platform cannot bind directory access to held identities"
        )
    return _BACKEND.open_identity_directory(Path(path))
```

- [ ] **Step 4: Write the POSIX backend**

Lift the mechanics from `_cli_recompile_recovery.py:241-305` and `_gui/browse/_tile_routes.py:91-145`; do not invent new ones.

```python
"""POSIX backend: directory file descriptors with ``O_NOFOLLOW``."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from ._identity_io import HeldDirectory, IdentityRefused, validate_component

BACKEND_NAME = "posix"

# ``os.stat`` is required by ``reverify``, which re-resolves a child's name
# inside its held parent. The constant this replaces
# (``_cli_recompile_recovery.py:219-222``) also required mkdir/unlink/rename
# for a transition *writer* that no longer exists (`:246-250`); this facade is
# read-only, so those are deliberately dropped and `os.stat` deliberately kept.
SUPPORTED = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_NONBLOCK")
    and os.listdir in os.supports_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
)

_DIR_FLAGS = (
    (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW) if SUPPORTED else os.O_RDONLY
)
_FILE_FLAGS = (
    (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW) if SUPPORTED else os.O_RDONLY
)


def _refuse_os_error(exc: OSError, subject: Path) -> "IdentityRefused":
    """Translate a no-follow open's refusal errno into the contract's error.

    ``O_NOFOLLOW`` on a symlink raises ``ELOOP``; ``O_DIRECTORY`` on a file
    raises ``ENOTDIR``. Both are refusals 2 and 3, not transport failures --
    the code this backend replaces mapped them the same way
    (``_cli_recompile_recovery.py:273-274``).
    """
    return IdentityRefused(f"{errno.errorcode.get(exc.errno, exc.errno)}: {subject}")


_REFUSED_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR, errno.EISDIR, errno.EMLINK})


class _PosixHeldDirectory:
    """A directory pinned by an open descriptor and its `st_dev`/`st_ino`.

    ``parent``/``name`` are how ``reverify`` re-resolves this directory: an
    identity read back from ``self._fd`` is the identity that fd was opened
    with and can never differ, so a check written that way cannot fail.
    """

    def __init__(
        self,
        path: Path,
        fd: int,
        owned: bool,  # retained for symmetry; _close always owns its own fd
        *,
        parent: "_PosixHeldDirectory | None" = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self._fd = fd
        self._owned = owned
        self._parent = parent
        self._name = name
        self._closed = False
        identity = os.fstat(fd)
        if not stat.S_ISDIR(identity.st_mode):
            raise IdentityRefused(f"not a directory: {path}")
        self._identity = (identity.st_dev, identity.st_ino)
        self._children: list[_PosixHeldDirectory] = []

    def child_directory(self, name: str) -> HeldDirectory:
        validate_component(name)
        try:
            child_fd = os.open(name, _DIR_FLAGS, dir_fd=self._fd)
        except OSError as exc:
            if exc.errno in _REFUSED_ERRNOS:
                raise _refuse_os_error(exc, self.path / name) from exc
            raise
        try:
            child = _PosixHeldDirectory(
                self.path / name, child_fd, owned=True, parent=self, name=name
            )
        except BaseException:
            os.close(child_fd)
            raise
        self._children.append(child)
        return child

    def list_names(self) -> tuple[str, ...]:
        self.reverify()
        return tuple(sorted(os.listdir(self._fd)))

    def _open_regular_fd(self, name: str) -> int:
        validate_component(name)
        try:
            file_fd = os.open(name, _FILE_FLAGS, dir_fd=self._fd)
        except OSError as exc:
            if exc.errno in _REFUSED_ERRNOS:
                raise _refuse_os_error(exc, self.path / name) from exc
            raise
        try:
            identity = os.fstat(file_fd)
            if not stat.S_ISREG(identity.st_mode):
                raise IdentityRefused(f"not a regular file: {self.path / name}")
            if identity.st_nlink != 1:
                raise IdentityRefused(
                    f"not a single-link file: {self.path / name}"
                )
        except BaseException:
            os.close(file_fd)
            raise
        return file_fd

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes:
        file_fd = self._open_regular_fd(name)
        try:
            size = os.fstat(file_fd).st_size
            if max_bytes is not None and size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(file_fd)
        self.reverify()
        return payload

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]:
        """Return the bytes and the stat of the same open file description.

        The torn-read comparison lives here rather than in the caller: the
        publication token folds ``st_mtime_ns``/``st_ctime_ns``/``st_ino`` and
        must be computed over one coherent read
        (``_io_constants.py:1946-1970``).
        """
        file_fd = self._open_regular_fd(name)
        try:
            before = os.fstat(file_fd)
            if max_bytes is not None and before.st_size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                payload = stream.read()
            after = os.fstat(file_fd)
        finally:
            os.close(file_fd)
        changed = (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_ino,
        )
        if changed or len(payload) != after.st_size:
            raise IdentityRefused(f"{self.path / name} changed during the read")
        return payload, after

    def open_regular_stream(self, name: str) -> BinaryIO:
        file_fd = self._open_regular_fd(name)
        try:
            stream = os.fdopen(file_fd, "rb")
        except BaseException:
            os.close(file_fd)
            raise
        return stream

    def reverify(self) -> None:
        """Re-resolve this directory's NAME and compare it to what is held.

        The stat result is only ever compared, never opened through, so this
        re-introduces no TOCTOU window: a hostile swap makes the check fail and
        can never make the facade open the wrong object, because every open
        still goes through the held descriptor.

        A child re-resolves one component inside its held parent, so no
        ancestor is re-walked. The root has no held parent and re-walks its
        whole prefix -- it catches its own rename, deletion or replacement, but
        not the swap of an ancestor. That asymmetry is deliberate and is why
        the spec's refusal 5 binds children.
        """
        if self._parent is None or self._name is None:
            current = os.lstat(self.path)
        else:
            current = os.stat(
                self._name, dir_fd=self._parent._fd, follow_symlinks=False
            )
        if (current.st_dev, current.st_ino) != self._identity:
            raise IdentityRefused(f"identity changed: {self.path}")

    def _close(self) -> None:
        """Close children leaf-first, then self. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for child in self._children:
            child._close()
        os.close(self._fd)


@contextmanager
def open_identity_directory(path: Path) -> Iterator[HeldDirectory]:
    """Hold *path* by identity through a no-follow directory descriptor."""
    root = Path(os.path.abspath(os.fspath(path)))
    fd = os.open(root, _DIR_FLAGS)
    held: _PosixHeldDirectory | None = None
    try:
        held = _PosixHeldDirectory(root, fd, owned=True)
        yield held
    finally:
        # In the ``finally``, not after the ``yield``: when the ``with`` body
        # raises, the generator resumes *at* the yield and anything below it is
        # skipped, leaking every child descriptor in ``_children``. In the
        # Browse route that is one leak per failing request -- the path most
        # exercised under load.
        if held is not None:
            held._close()
        else:
            os.close(fd)
```

- [ ] **Step 5: Run the contract suite**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q`
Expected: PASS (the hard-link and symlink tests run for real on Linux).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src/phenotypic/sdk_/_identity_io.py src/phenotypic/sdk_/_identity_io_posix.py tests/unit/sdk_/test_identity_io_contract.py
uv run mypy src/phenotypic/sdk_/_identity_io.py src/phenotypic/sdk_/_identity_io_posix.py
git add src/phenotypic/sdk_/_identity_io.py src/phenotypic/sdk_/_identity_io_posix.py tests/unit/sdk_/test_identity_io_contract.py
git commit -m "feat(sdk): add identity-bound directory facade with a POSIX backend"
```

---

### Task 2: Relocate the Windows ctypes API

Mechanical move, done alone so a reviewer can see it is mechanical. The journal's behaviour must not change.

**Files:**
- Create: `src/phenotypic/sdk_/_identity_io_windows.py` (receives the class)
- Modify: `src/phenotypic/sdk_/_windows_metadata_journal.py` (import it instead of defining it)

**Interfaces:**
- Consumes: Task 1's `IdentityRefused`, `validate_component`.
- Produces: `_CtypesWindowsApi`, `WindowsHandleInfo`, `_WindowsApi` protocol and the `_FILE_*` constants importable from `_identity_io_windows`; `SUPPORTED: bool`; `BACKEND_NAME = "windows"`.

- [ ] **Step 1: Move the structures, constants and class**

Move from `_windows_metadata_journal.py` into `_identity_io_windows.py`, unchanged: `WindowsHandleInfo` (`:26`), the `_WindowsApi` protocol (`:42`), every `_FILE_*`/`_OBJ_*`/`_ERROR_*` constant (`:371-406`), the ctypes structures (`:405-460`), `_CtypesWindowsApi` (`:466`), **and `WindowsJournalUnavailable` (`:23`)**. Leave `WindowsJournalSession` and everything journal-specific where it is.

**The exception must travel with the class or the import is circular.**
`_CtypesWindowsApi` raises `WindowsJournalUnavailable` at `:471`, `:479`,
`:606` and `:685` — all inside the moved code. Leaving it behind gives
`_windows_metadata_journal → _identity_io_windows → _windows_metadata_journal`.
It is also in the journal's `__all__` (`:922`), so the re-export below is what
keeps that surface intact.

- [ ] **Step 2: Re-export from the journal so its imports keep working**

```python
from ._identity_io_windows import (
    WindowsHandleInfo,
    WindowsJournalUnavailable,
    _CtypesWindowsApi,
    _FileDispositionInfo,  # noqa: F401  (re-exported for callers)
    _FileRenameInfoHeader,  # noqa: F401  (re-exported for callers)
    _IoStatusBlock,  # noqa: F401  (re-exported for callers)
    _Overlapped,  # noqa: F401  (re-exported for callers)
    _WindowsApi,
)
```

**Eight names, not six.** `tests/unit/sdk_/test_windows_metadata_journal.py`
imports `_FileDispositionInfo` and `_Overlapped` from the journal at `:23-27`
and asserts on them at `:71-76`, so a shorter list fails at **collection** —
the same symptom this task attributes to a circular import, which makes it
easy to misdiagnose.

**This step is also a subtraction.** After the move nothing left in the
journal uses `ctypes`, `Any` or `Protocol`, so `import ctypes` must go and the
typing import narrows to `Iterator`, or ruff's F401 fails Step 4.

- [ ] **Step 3: Prove the move changed nothing**

Run: `uv run pytest tests/unit/sdk_/test_windows_metadata_journal.py -q`
Expected: PASS, same count as before the move (the in-memory model exercises this on Linux).

Read the result carefully: a circular import surfaces as a **collection
error**, not as a count mismatch, so "the numbers match" is not the check —
"it collected at all, and the numbers match" is.

- [ ] **Step 4: Commit**

```bash
uv run ruff check src/phenotypic/sdk_/_identity_io_windows.py src/phenotypic/sdk_/_windows_metadata_journal.py
git add src/phenotypic/sdk_/_identity_io_windows.py src/phenotypic/sdk_/_windows_metadata_journal.py
git commit -m "refactor(sdk): move the Windows ctypes handle API into _identity_io_windows"
```

---

### Task 3: The Windows backend — walk, read, reverify

**Files:**
- Modify: `src/phenotypic/sdk_/_identity_io_windows.py`
- Create: `tests/unit/sdk_/_identity_io_fake.py` (in-memory handle model, shared by tests)
- Modify: `tests/unit/sdk_/test_identity_io_contract.py` (parameterize over backends)

**Interfaces:**
- Consumes: `_CtypesWindowsApi` (Task 2), `HeldDirectory`/`IdentityRefused`/`validate_component` (Task 1).
- Produces: `open_identity_directory(path, *, api=None)` in the Windows backend — the `api` keyword exists so tests inject the fake; production passes nothing.

- [ ] **Step 1: Extend the in-memory model**

Start from `_MemoryWindowsApi` in `tests/unit/sdk_/test_windows_metadata_journal.py:79` (which already refuses path-based child operations) and add what this backend needs: `list_names(handle)`, `link_count(handle)`, `is_directory(handle)`, `adopt_descriptor(handle)`, and a `stream(handle)` returning `io.BytesIO`. Keep the refusal of path-based children — that is what makes the fake catch a backend that forgets to route through a held handle.

Three details, each of which otherwise makes the fake lane lie:

1. **`share_delete` is asserted `False` today** in the fake's `open_anchor` (`:106`), `open_directory` (`:117`) and `open_file` (`:139`), because the journal opens that way. This backend always passes `share_delete=True`, so every fake-lane test errors on that assert. Make the expected value a constructor argument rather than deleting the assert — it pins the journal's stricter sharing, which this design deliberately does not inherit.
2. **`memory_api_factory(path)` scans the real tree at construction** and models it in memory. That is what makes `test_listing_spans_multiple_buffer_fills` meaningful, and it only works because those 2,000 files are written *before* the hold opens.
3. **Three tests cannot run in the fake lane at all**, because they mutate the real tree after the hold is taken and an in-memory model cannot observe it: `test_a_swapped_directory_fails_reverification` (`os.rename`), `test_a_streamed_member_does_not_lock_the_file` (`write_bytes`), and `test_a_multi_link_file_is_refused` (`os.link` — the fake has no hard-link concept, so a link count would be invented rather than modelled). Restrict those three to the `native` lane explicitly, and say so in each test's docstring: a lane that silently does not exercise a refusal is the same false-green shape this plan is trying to avoid elsewhere.

- [ ] **Step 2: Write the failing parameterization**

```python
@pytest.fixture(params=["native", "fake-windows"])
def backend(request, monkeypatch):
    """Run the contract against the live backend and the in-memory model."""
    if request.param == "native":
        if not _identity_io.identity_io_available():
            pytest.fail("no identity-IO backend on this platform")
        return _identity_io.open_identity_directory
    from phenotypic.sdk_ import _identity_io_windows
    from tests.unit.sdk_._identity_io_fake import memory_api_factory

    def _open(path):
        return _identity_io_windows.open_identity_directory(
            path, api=memory_api_factory(path)
        )

    return _open
```

Every test in the file then takes `backend` and calls `backend(tree / "store")` instead of `_identity_io.open_identity_directory(...)`.

- [ ] **Step 3: Run it and watch the fake lane fail**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q`
Expected: the `native` lane passes, every `fake-windows` lane fails — `open_identity_directory` is not defined in the Windows backend yet.

- [ ] **Step 4: Implement the walk, read and reverify**

```python
class _WindowsHeldDirectory:
    """A directory pinned by an NT handle and its FILE_ID_INFO identity."""

    def __init__(
        self,
        path: Path,
        handle: int,
        api,
        owned: bool,
        *,
        parent: "_WindowsHeldDirectory | None" = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self._handle = handle
        self._api = api
        self._owned = owned
        self._parent = parent
        self._name = name
        self._closed = False
        info = api.handle_info(handle)
        if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise IdentityRefused(f"reparse point: {path}")
        if len(info.file_id) != 16 or not any(info.file_id):
            raise IdentityRefused(f"no stable identity: {path}")
        # ``open_anchor`` uses CreateFileW with FILE_FLAG_BACKUP_SEMANTICS
        # (``_windows_metadata_journal.py:614-625``), which SUCCEEDS on a
        # regular file. POSIX refuses that twice (``O_DIRECTORY`` and the
        # ``S_ISDIR`` check), so without this the two backends disagree on
        # ``open_identity_directory(<a file>)``.
        if not api.is_directory(handle):
            raise IdentityRefused(f"not a directory: {path}")
        self._identity = info.identity
        self._children: list[_WindowsHeldDirectory] = []

    def _close(self) -> None:
        """Close children leaf-first, then self. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for child in self._children:
            child._close()
        self._api.close(self._handle)

    def child_directory(self, name: str) -> HeldDirectory:
        validate_component(name)
        handle = self._api.open_directory(
            self._handle, name, create=False, share_delete=True
        )
        try:
            child = _WindowsHeldDirectory(
                self.path / name,
                handle,
                self._api,
                owned=True,
                parent=self,
                name=name,
            )
        except BaseException:
            self._api.close(handle)
            raise
        self._children.append(child)
        return child

    def _open_regular(self, name: str) -> int:
        validate_component(name)
        handle = self._api.open_file(
            self._handle, name, create_new=False, share_delete=True
        )
        try:
            info = self._api.handle_info(handle)
            if info.attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                raise IdentityRefused(f"reparse point: {self.path / name}")
            if self._api.link_count(handle) != 1:
                raise IdentityRefused(
                    f"not a single-link file: {self.path / name}"
                )
        except BaseException:
            self._api.close(handle)
            raise
        return handle

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes:
        payload, _stat = self.read_regular_with_stat(name, max_bytes=max_bytes)
        return payload

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]:
        """Bytes plus the stat CPython would build for the same file.

        The stat comes from ``os.fstat`` on a descriptor adopted from the held
        handle, so it is the *same* ``os.stat_result`` shape the path branch of
        ``store_publication_token`` produces -- the two token branches then
        agree by construction rather than by hope.
        """
        handle = self._open_regular(name)
        descriptor = self._api.adopt_descriptor(handle)  # handle is now the fd's
        try:
            before = os.fstat(descriptor)
            if max_bytes is not None and before.st_size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = stream.read()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
        ) or len(payload) != after.st_size:
            raise IdentityRefused(f"{self.path / name} changed during the read")
        self.reverify()
        return payload, after

    def reverify(self) -> None:
        """Re-resolve the NAME, not the handle. See the POSIX backend's note.

        Reading ``handle_info`` back from ``self._handle`` returns the identity
        that handle was opened with, so a check written that way can never
        fail. ``_windows_metadata_journal.py:146-156`` has that bug today.
        """
        if self._parent is None or self._name is None:
            probe = self._api.open_anchor(
                str(self.path), share_delete=True
            )
        else:
            probe = self._api.open_directory(
                self._parent._handle, self._name, create=False, share_delete=True
            )
        try:
            if self._api.handle_info(probe).identity != self._identity:
                raise IdentityRefused(f"identity changed: {self.path}")
        finally:
            self._api.close(probe)
```

`open_regular_read`/`open_directory` already pass `FILE_OPEN_REPARSE_POINT` and the directory/non-directory flag, so refusals 2 and 3 come from the open itself; the explicit reparse check is belt-and-braces for a filesystem that reports it differently. Opening a *non-existent* entry must surface `FileNotFoundError` — `_nt_open` already maps errors 2 and 3 to it (`_windows_metadata_journal.py:679-684`).

**`_nt_open` needs the same refusal mapping the POSIX backend grew.** It maps
only errors 2 and 3 today, so `STATUS_NOT_A_DIRECTORY` (`ERROR_DIRECTORY`,
267) and a reparse-point refusal surface as a bare `OSError` and the contract
tests expecting `IdentityRefused` fail. Translate 267 and
`ERROR_ACCESS_DENIED` (5) *when the target is a reparse point* into
`IdentityRefused`, leaving everything else as `OSError`. Do this in the
backend's wrapper, not inside `_nt_open`, so the journal's error vocabulary is
untouched.

Add the module-level entry point, mirroring the POSIX one:

```python
BACKEND_NAME = "windows"

# Bind at import, not per call. ``_CtypesWindowsApi()`` raises
# ``WindowsJournalUnavailable`` from ``__init__`` when a symbol fails to bind
# (:471, :479). Constructing it inside ``open_identity_directory`` would leave
# ``identity_io_available()`` True and ``active_backend_name()`` "windows"
# while every call raised -- Task 8's assertion would pass while nothing
# worked, and consumers would see an uncaught RuntimeError instead of the
# designed refusal (``_cli_recompile_recovery.py:446-453`` catches only
# KeyError/OSError/TypeError/ValueError/JSONDecodeError, ``:519`` only
# OSError/ValueError).
try:
    _API: "_CtypesWindowsApi | None" = (
        _CtypesWindowsApi() if os.name == "nt" else None
    )
except WindowsJournalUnavailable:
    _API = None

SUPPORTED = _API is not None


def _extended_length(path: Path) -> str:
    """Return the ``\\\\?\\`` spelling of *path* WITHOUT resolving links.

    Not ``ngff_.long_path``: that calls ``Path.resolve()``
    (``ngff_.py:1659``), which follows a junction. A store root that *is* a
    junction would then open successfully here while POSIX's ``O_NOFOLLOW``
    refuses it -- the two backends would stop refusing identically, which is
    the one property the spec insists on.
    """
    text = os.path.abspath(os.fspath(path))
    return text if text.startswith("\\\\?\\") else "\\\\?\\" + text


@contextmanager
def open_identity_directory(path: Path, *, api=None) -> Iterator[HeldDirectory]:
    """Hold *path* by identity through NT handles."""
    root = Path(os.path.abspath(os.fspath(path)))
    held: _WindowsHeldDirectory | None = None
    # Reuse the module-level instance; it holds only bound function pointers
    # and no per-call state (``_bind``, :485-594). The journal constructs its
    # own (:93), so a Windows process holds two -- wasteful, not a problem,
    # and worth saying so because a reader will wonder.
    resolved_api = api if api is not None else _API
    handle = resolved_api.open_anchor(_extended_length(root), share_delete=True)
    try:
        held = _WindowsHeldDirectory(root, handle, resolved_api, owned=True)
        yield held
    finally:
        # Same rule as the POSIX entry point: closing after the ``yield`` is
        # skipped whenever the ``with`` body raises, leaking every child handle.
        if held is not None:
            held._close()
        else:
            resolved_api.close(handle)
```

Note the sharing mode: every open passes `share_delete=True`, and `_nt_open` already adds `FILE_SHARE_READ | FILE_SHARE_WRITE`. A read-only reader must never lock a store against a running CLI.

- [ ] **Step 5: Add `link_count` and `file_size` to the ctypes API**

Both come from `FILE_STANDARD_INFO` (class `1`) through the already-bound `GetFileInformationByHandleEx`:

```python
class _FileStandardInfo(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_int64),
        ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", ctypes.c_uint32),
        ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


_FILE_STANDARD_INFO_CLASS = 1


def _standard_info(self, handle: int) -> _FileStandardInfo:
    buffer = _FileStandardInfo()
    if not self.GetFileInformationByHandleEx(
        handle,
        _FILE_STANDARD_INFO_CLASS,
        ctypes.byref(buffer),
        ctypes.sizeof(buffer),
    ):
        self._raise_last_error("GetFileInformationByHandleEx(FileStandardInfo)")
    return buffer


def link_count(self, handle: int) -> int:
    return int(self._standard_info(handle).NumberOfLinks)


def file_size(self, handle: int) -> int:
    return int(self._standard_info(handle).EndOfFile)


def is_directory(self, handle: int) -> bool:
    return bool(self._standard_info(handle).Directory)
```

Two more methods, and the first is a correctness fix rather than an addition.
`open_file` (`_windows_metadata_journal.py:714-733`) requests
`FILE_WRITE_DATA | FILE_WRITE_ATTRIBUTES | DELETE` because the journal writes
through it. A viewer that asks for write and delete access is refused on a
read-only store, a read-only share, or a file whose ACL denies writes -- so a
read-only opener is required, and `open_file` must keep its mask unchanged so
the journal is unaffected:

```python
def open_regular_read(self, parent: int, name: str) -> int:
    """Open a member for reading only, sharing it with every other writer."""
    return self._nt_open(
        parent,
        name,
        desired_access=_FILE_READ_DATA | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE,
        disposition=_FILE_OPEN,
        options=(
            _FILE_NON_DIRECTORY_FILE
            | _FILE_OPEN_REPARSE_POINT
            | _FILE_SYNCHRONOUS_IO_NONALERT
        ),
        attributes=_FILE_ATTRIBUTE_NORMAL,
        share_delete=True,
    )


def adopt_descriptor(self, handle: int) -> int:
    """Adopt a handle as a CRT descriptor. Ownership transfers to the fd.

    The caller closes with ``os.close``; calling ``CloseHandle`` as well is a
    double close.
    """
    import msvcrt

    return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
```

`_WindowsHeldDirectory._open_regular` calls `open_regular_read`, not
`open_file`. `open_directory` is already read-only enough for a hold
(`_FILE_LIST_DIRECTORY | _FILE_TRAVERSE | _FILE_READ_ATTRIBUTES`).

- [ ] **Step 6: Run both lanes**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q`
Expected: PASS for `native` and `fake-windows`, with **one** strict xfail.

Only `test_listing_excludes_dot_entries` exists to mark at this point; the stream tests are not written until Task 5 Step 1. And the lane comes from a fixture `params`, so a plain `@pytest.mark.xfail` decorator marks *both* lanes and `pytest.param("fake-windows", marks=...)` marks the *whole* lane. Per-test-per-lane needs `applymarker` inside the test body:

```python
def test_listing_excludes_dot_entries(backend, request, tree: Path) -> None:
    if request.node.callspec.params["backend"] == "fake-windows":
        request.applymarker(pytest.mark.xfail(strict=True, reason="Task 4"))
    with backend(tree / "store") as held:
        assert sorted(held.list_names()) == ["nested", "root.json"]
```

`strict=True` is deliberate: Task 4 Step 1 removes the marker and the test must already be failing for that sequencing to mean anything.

- [ ] **Step 7: Commit**

```bash
uv run ruff check src/phenotypic/sdk_/_identity_io_windows.py tests/unit/sdk_/
uv run mypy src/phenotypic/sdk_/_identity_io_windows.py
git add src/phenotypic/sdk_/_identity_io_windows.py tests/unit/sdk_/_identity_io_fake.py tests/unit/sdk_/test_identity_io_contract.py
git commit -m "feat(sdk): hold, walk and read directories by identity on Windows"
```

---

### Task 4: Directory listing on Windows

**Files:**
- Modify: `src/phenotypic/sdk_/_identity_io_windows.py`
- Modify: `tests/unit/sdk_/_identity_io_fake.py`, `tests/unit/sdk_/test_identity_io_contract.py`

**Interfaces:**
- Consumes: Task 0's confirmed information class; Task 3's `_WindowsHeldDirectory`.
- Produces: `HeldDirectory.list_names()` on Windows.

- [ ] **Step 1: Drop the xfail from `test_listing_excludes_dot_entries`**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -k listing -q`
Expected: FAIL for `fake-windows` — `list_names` not implemented.

- [ ] **Step 2: Implement listing**

Use the class Task 0 confirmed (`FileFullDirectoryRestartInfo`/`FileFullDirectoryInfo` unless the probe said otherwise). Restart class on the first call, continue class thereafter, until the call returns false; `ERROR_NO_MORE_FILES` (18) ends the walk and anything else raises.

```python
_FILE_FULL_DIRECTORY_INFO_CLASS = 14
_FILE_FULL_DIRECTORY_RESTART_INFO_CLASS = 15
_ERROR_NO_MORE_FILES = 18
_LISTING_BUFFER_BYTES = 64 * 1024


def list_names(self, handle: int) -> tuple[str, ...]:
    """Return every entry name in a held directory, excluding . and .."""
    names: list[str] = []
    buffer = ctypes.create_string_buffer(_LISTING_BUFFER_BYTES)
    info_class = _FILE_FULL_DIRECTORY_RESTART_INFO_CLASS
    while self.GetFileInformationByHandleEx(
        handle, info_class, buffer, ctypes.sizeof(buffer)
    ):
        info_class = _FILE_FULL_DIRECTORY_INFO_CLASS
        offset = 0
        while True:
            entry = ctypes.cast(
                ctypes.byref(buffer, offset),
                ctypes.POINTER(_FileFullDirInfo),
            ).contents
            name = ctypes.wstring_at(
                ctypes.byref(buffer, offset + _FileFullDirInfo.FileName.offset),
                entry.FileNameLength // 2,
            )
            if name not in {".", ".."}:
                names.append(name)
            if entry.NextEntryOffset == 0:
                break
            offset += entry.NextEntryOffset
    error = ctypes.get_last_error()
    if error != _ERROR_NO_MORE_FILES:
        raise OSError(error, "GetFileInformationByHandleEx(directory listing)")
    return tuple(names)
```

On the held-directory class:

```python
    def list_names(self) -> tuple[str, ...]:
        self.reverify()
        return tuple(sorted(self._api.list_names(self._handle)))
```

- [ ] **Step 3: Add a multi-buffer test**

A single 64 KiB buffer holds a few hundred entries; the continuation path never runs on a small fixture, and a recompile transition directory can hold thousands. Add a test that writes 2,000 entries and asserts they all come back, so the loop is exercised rather than assumed.

```python
def test_listing_spans_multiple_buffer_fills(backend, tmp_path: Path) -> None:
    directory = tmp_path / "many"
    directory.mkdir()
    expected = {f"entry_{index:05d}.json" for index in range(2000)}
    for name in expected:
        (directory / name).write_bytes(b"x")
    with backend(directory) as held:
        assert set(held.list_names()) == expected
```

- [ ] **Step 4: Run, lint, commit**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q`
Expected: PASS on both lanes.

```bash
git add src/phenotypic/sdk_/_identity_io_windows.py tests/unit/sdk_/
git commit -m "feat(sdk): list a held directory on Windows"
```

---

### Task 5: Streaming a member

**Files:**
- Modify: `src/phenotypic/sdk_/_identity_io_windows.py`
- Modify: `tests/unit/sdk_/test_identity_io_contract.py`

**Interfaces:**
- Consumes: Task 3's `_open_regular`.
- Produces: `HeldDirectory.open_regular_stream(name) -> BinaryIO`, valid after the hold closes.

- [ ] **Step 0: Establish the mypy baseline for Windows-only code**

`msvcrt` and `os.O_BINARY` are guarded behind `sys.platform == "win32"` in typeshed, and the per-commit gate runs mypy on Linux. Find out what the repo already tolerates before writing code that depends on the answer:

Run: `uv run mypy src/phenotypic/sdk_/_windows_metadata_journal.py`
That file calls `ctypes.WinDLL` at `:474-475`, which typeshed guards the same way, and carries no `type: ignore` and no `sys.platform` guard anywhere. So either mypy tolerates the pattern here or that file is already red. Record which in the plan. If it is clean, this task needs nothing; if not, put the Windows-only code under `if sys.platform == "win32":`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_stream_outlives_the_hold(backend, tree: Path) -> None:
    """Browse hands this stream to send_file and returns; the hold is gone."""
    with backend(tree / "store") as held:
        stream = held.child_directory("nested").open_regular_stream("chunk")
    try:
        assert stream.read() == b"0123456789"
    finally:
        stream.close()


def test_a_stream_is_seekable_for_range_requests(backend, tree: Path) -> None:
    with backend(tree / "store") as held:
        stream = held.child_directory("nested").open_regular_stream("chunk")
    try:
        stream.seek(4)
        assert stream.read(3) == b"456"
    finally:
        stream.close()


def test_a_streamed_member_does_not_lock_the_file(backend, tree: Path) -> None:
    """A read-only viewer must never block a concurrent CLI run."""
    with backend(tree / "store") as held:
        stream = held.open_regular_stream("root.json")
    try:
        (tree / "store" / "root.json").write_bytes(b'{"ok": false}')
    finally:
        stream.close()
```

- [ ] **Step 2: Run them and watch the fake lane fail**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -k stream -q`
Expected: FAIL for `fake-windows`.

- [ ] **Step 3: Implement the hand-off**

```python
    def open_regular_stream(self, name: str) -> BinaryIO:
        handle = self._open_regular(name)
        try:
            stream = self._api.stream(handle)
        except BaseException:
            self._api.close(handle)
            raise
        return stream
```

On the ctypes API — ownership of the handle transfers to the descriptor, so this must not also `CloseHandle`:

```python
def stream(self, handle: int) -> BinaryIO:
    """Wrap a handle as a Python binary file; the fd owns it from here."""
    import msvcrt

    descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(descriptor, "rb")
```

- [ ] **Step 4: Run, lint, commit**

Run: `uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q`
Expected: PASS on both lanes, no xfail markers left in the file.

```bash
git add src/phenotypic/sdk_/_identity_io_windows.py tests/unit/sdk_/test_identity_io_contract.py
git commit -m "feat(sdk): stream a held member on Windows"
```

---

### Task 6: Port recompile transition recovery

**Files:**
- Modify: `src/phenotypic/_cli/_cli_recompile_recovery.py:213-305,500-520`
- Modify: `tests/unit/cli/test_cli_recompile_slurm.py:2019`
- Modify: `tests/unit/cli/test_recompile_no_store_writes.py`, `tests/unit/cli/test_finalize_run.py`, `tests/unit/cli/test_slurm_finalize_flushes_trailing.py`, `tests/unit/cli/test_recompile_no_longer_migrates.py` (remove Windows expectations if any were added)

**Interfaces:**
- Consumes: `open_identity_directory`, `identity_io_available`, `IdentityRefused` (Task 1).
- Produces: no new public names; `_open_transition_directory` now yields `(root, HeldDirectory)`.

- [ ] **Step 1: Write the failing test**

```python
def test_transition_recovery_runs_wherever_identity_io_is_available(
    tmp_path: Path,
) -> None:
    """The gate is the facade's capability, not the operating system name."""
    from phenotypic._cli import _cli_recompile_recovery

    assert not hasattr(
        _cli_recompile_recovery, "_IDENTITY_BOUND_DIRECTORY_OPERATIONS"
    ), "the platform gate moved to phenotypic.sdk_._identity_io"


def test_recovery_refuses_when_no_backend_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed survives the port: no backend, no transition reads."""
    from phenotypic._cli import _cli_recompile_recovery
    from phenotypic.sdk_ import _identity_io

    monkeypatch.setattr(_identity_io, "_BACKEND", None)
    with pytest.raises(_identity_io.IdentityIoUnavailable):
        _cli_recompile_recovery.recoverable_recompile_measurement_sources(
            tmp_path, ["ds"]
        )
```

- [ ] **Step 2: Run and watch it fail**

Run: `uv run pytest tests/unit/cli/test_cli_recompile_slurm.py -k identity -q`
Expected: FAIL — the constant still exists.

- [ ] **Step 3: Port the module**

Delete `_IDENTITY_BOUND_DIRECTORY_OPERATIONS` (`:213`) and `_require_identity_bound_directory_operations` (`:226`). Rewrite `_open_transition_directory` to walk with the facade, keeping every existing refusal — the escape check against the output root, `_validate_transition_component`, and the `FileNotFoundError` passthrough:

```python
@contextmanager
def _open_transition_directory(
    output_root: Path,
    dataset_name: str,
) -> Iterator[tuple[Path, HeldDirectory]]:
    """Hold the transition directory by identity. Read-only."""
    canonical_output = Path(output_root).resolve(strict=True)
    root = _transition_root(canonical_output, dataset_name)
    try:
        relative = root.relative_to(canonical_output)
    except ValueError as exc:
        raise ValueError("Transition directory escapes output root") from exc
    with open_identity_directory(canonical_output) as held:
        current = held
        try:
            for component in relative.parts:
                _validate_transition_component(component)
                current = current.child_directory(component)
        except FileNotFoundError:
            raise
        except IdentityRefused as exc:
            raise ValueError("Transition directory is not canonical") from exc
        yield root, current
```

Then `_read_regular_file_at(directory_fd, name)` becomes `held.read_regular_bytes(name)` at both call sites (`:387`, `:415`), and `os.listdir(directory_fd)` (`:514`) becomes `held.list_names()`. Keep `IdentityRefused` mapped to the existing `ValueError` vocabulary so callers' `except (OSError, ValueError)` arms still fire.

- [ ] **Step 4: Run the recompile suites**

Run: `uv run pytest tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_recompile_no_store_writes.py tests/unit/cli/test_finalize_run.py tests/unit/cli/test_slurm_finalize_flushes_trailing.py -q`
Expected: PASS on Linux; these are the suites that were red on Windows.

- [ ] **Step 5: Commit**

```bash
uv run ruff check src/phenotypic/_cli/_cli_recompile_recovery.py tests/unit/cli/
uv run mypy src/phenotypic/_cli/_cli_recompile_recovery.py
git add src/phenotypic/_cli/_cli_recompile_recovery.py tests/unit/cli/
git commit -m "feat(cli): run recompile transition recovery on any identity-IO platform"
```

---

### Task 7: Port the Browse route and `store_publication_token`

This task carries the **breaking public API change**. It is one task because the route and the function must change together.

**Files:**
- Modify: `src/phenotypic/_gui/browse/_tile_routes.py:53-145,209,357-437`
- Modify: `src/phenotypic/sdk_/_io_constants.py:1907`
- Modify: `tests/gui/browse/test_tile_routes.py`, `tests/integration/cli/test_process_store_consumers.py:88`
- Modify: `docs/source/how_to/pages/zarr_storage.md` (or the store-layout reference that documents the function), `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 1's facade.
- Produces: `store_publication_token(store: Path, *, root_directory: HeldDirectory | None = None) -> str | None`. `root_dir_fd` is gone; passing it raises `TypeError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_store_publication_token_takes_a_held_directory(tmp_path: Path) -> None:
    from phenotypic.sdk_ import _identity_io, store_publication_token

    store = _published_store(tmp_path)          # existing helper in this module
    with _identity_io.open_identity_directory(store) as held:
        assert store_publication_token(store, root_directory=held) is not None


def test_both_token_branches_agree_for_one_store(tmp_path: Path) -> None:
    """If they ever disagree, every Browse tile request 409s forever."""
    from phenotypic.sdk_ import _identity_io, store_publication_token

    store = _published_store(tmp_path)
    by_path = store_publication_token(store)
    with _identity_io.open_identity_directory(store) as held:
        by_hold = store_publication_token(store, root_directory=held)
    assert by_path == by_hold is not None


def test_the_removed_posix_only_parameter_is_refused(tmp_path: Path) -> None:
    """The break is deliberate and must be visible, not silently ignored."""
    from phenotypic.sdk_ import store_publication_token

    with pytest.raises(TypeError):
        store_publication_token(tmp_path, root_dir_fd=3)
```

In `test_tile_routes.py`, **delete `requires_safe_store_io` (`:20-23`) and all
8 of its uses.** Retargeting it at `identity_io_available()` would convert a
Windows bind failure into 8 silent skips — a second false-green vector beside
B7, and this time one that goes genuinely green while serving nothing. Both
platforms now have a backend; the gate is Task 8's platform assertion, which
has no skip. (Deleting the constant without deleting the decorator breaks
collection with `AttributeError`, so this is not optional.)

Then replace the `_SAFE_STORE_IO` monkeypatch in
`test_store_member_route_refuses_on_a_platform_without_safe_store_io` (`:204`)
with one that makes the facade report no backend:

```python
    monkeypatch.setattr(_identity_io, "_BACKEND", None)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/gui/browse/test_tile_routes.py -q`
Expected: FAIL — `root_directory` is not a parameter yet.

- [ ] **Step 3: Change the public signature**

In `_io_constants.py:1907`, replace `root_dir_fd: int | None` with
`root_directory: HeldDirectory | None`. The held branch (`:1943-1970`) becomes:

```python
        else:
            try:
                raw, after = root_directory.read_regular_with_stat(
                    STORE_ROOT_JSON
                )
            except (IdentityRefused, OSError):
                # Matches the existing `except OSError: return None` arm: no
                # token means "use the conservative fallback", not "fail".
                return None
            before = after      # the backend already proved they matched
```

**The path branch (`root.lstat()` / `read_bytes()` / `root.lstat()`, `:1944-1949`) is untouched.** Both branches must keep producing the same digest for the same file: the token folds `st_mtime_ns`, `st_ctime_ns` and `st_ino` (`:1994-1997`), the route compares a path-branch token (`_tile_routes.py:347`) against a held-branch token (`:385`, `:405`), and if they ever disagree every Browse tile request returns 409 forever. That is why `read_regular_with_stat` returns a real `os.stat_result` taken through `os.fstat`, on both platforms, rather than a hand-built tuple.

Swallowing `IdentityRefused` here is deliberate and is the one place the contract's "raise" becomes "return None": today a `st_nlink != 1` member yields `None` (`:1948-1950`), and letting it raise instead would turn a 404 into a 409.

Update the docstring's `Args:` block: the descriptor sentence becomes "Optional held directory for the store root. When given, `zarr.json` is read through that held identity so a route can keep validation and serving bound to one directory generation."

- [ ] **Step 4: Port the route**

Replace `_SAFE_STORE_IO` (`:57`) with `identity_io_available()`. Keep `_UnsafeStoreAccess` (`:53`) and its 422 — the message, `"this platform cannot safely serve Zarr store members"`, stays accurate and needs no change.

**Write the guard rather than implying it.** `IdentityIoUnavailable` is a `RuntimeError`, and the route catches `_UnsafeStoreAccess` first but `(OSError, RuntimeError, TypeError, ValueError)` → 404 right after (`_tile_routes.py:377-382`). Letting the facade's exception escape turns the documented 422 into a 404, and the Task 7 Step 1 test fails for a reason nobody reads as "clause order":

```python
def _open_store_root(store: Path):
    if not identity_io_available():
        raise _UnsafeStoreAccess(
            "this platform cannot safely serve Zarr store members"
        )
    return open_identity_directory(store)
``` `_open_store_root` returns the context manager; `_open_regular_store_member` walks `parts[:-1]` with `child_directory` and returns `open_regular_stream(parts[-1])`; `_read_store_json` uses `read_regular_bytes(name, max_bytes=_MAX_STORE_METADATA_BYTES)`.

The route's structure changes in one important way: the hold must wrap everything up to and including the `send_file` call, and the stream must be the only thing that outlives it (`response.call_on_close(handle.close)` already does that).

Cache held children inside the hold — a `dict[tuple[str, ...], HeldDirectory]`, which is what `WindowsJournalSession._directories` already is (`_windows_metadata_journal.py:91`, `:176-179`). `_read_store_json` runs once per series and once per label inside `_image_store_prefixes` (`_tile_routes.py:147`, `:196`, `:209`), each time re-walking from the root. It is bounded per request rather than a leak, but the cache is cheaper and matches the shape this code now sits beside.

- [ ] **Step 5: Run the route and consumer suites**

Run: `uv run pytest tests/gui/browse/test_tile_routes.py tests/integration/cli/test_process_store_consumers.py tests/unit/sdk_/test_io_constants.py -q`
Expected: PASS.

- [ ] **Step 6: Document the break where this repo publishes it**

There is no `CHANGELOG.md`; `docs/source/api_reference/core/store_layout.rst`
renders this function with `.. autofunction::`, so the docstring *is* the
published record. Add a `versionchanged` directive to it:

```python
    .. versionchanged:: 0.20.0
       ``root_dir_fd`` (a POSIX file descriptor) is replaced by
       ``root_directory``, a held directory from
       :mod:`phenotypic.sdk_._identity_io`. Passing ``root_dir_fd`` now
       raises :class:`TypeError`. The descriptor form could not be
       supported on Windows, where the store route needs the same
       identity binding.
```

Use the project's actual next version, from `pyproject.toml`, not `0.20.0`
verbatim. Then check the page renders: build only this page, with notebook
execution off, as a Slurm job per the repo's docs-build rule — never inline.

- [ ] **Step 7: Commit**

```bash
uv run ruff check src/phenotypic/_gui/browse/_tile_routes.py src/phenotypic/sdk_/_io_constants.py tests/
uv run mypy src/phenotypic/_gui/browse/_tile_routes.py src/phenotypic/sdk_/_io_constants.py
git add src/phenotypic/_gui/browse/_tile_routes.py src/phenotypic/sdk_/_io_constants.py tests/
git commit -m "feat(gui): serve OME-Zarr store members wherever identity IO is available

BREAKING: store_publication_token takes root_directory, not root_dir_fd."
```

---

### Task 8: CI lane and the anti-false-green assertion

**Files:**
- Create: `tests/unit/sdk_/test_identity_io_platform.py`
- Modify: `pyproject.toml` (register the `platform_io` marker)
- Modify: `.github/workflows/run-pytest.yml` (add the Windows job)
- Modify: `tests/unit/ci/test_pytest_shard_manifest.py`, `tests/CLAUDE.md`

**Interfaces:**
- Consumes: `active_backend_name()` (Task 1).
- Produces: a `platform_io` marker; a `tests-windows-platform-io` job.

- [ ] **Step 1: Write the platform assertion test**

This is the test that stops the whole port from passing by refusing everything.

```python
"""The backend must be the one this platform is supposed to use.

Every refusal test in the contract suite passes vacuously if the backend
degrades to "unavailable" -- a ctypes symbol that fails to bind would turn the
Windows lane green while shipping nothing. This test has no skip for that
reason.
"""

import os

import pytest

from phenotypic.sdk_ import _identity_io


def test_the_expected_backend_is_active_on_this_platform() -> None:
    """Fails loudly if a ctypes symbol failed to bind at import.

    This works only because the Windows backend binds its API at import time
    and sets ``SUPPORTED`` from the result. Bound lazily per call, this test
    would pass while every call raised.
    """
    expected = {"posix": "posix", "nt": "windows"}[os.name]
    assert _identity_io.active_backend_name() == expected
    assert _identity_io.identity_io_available() is True
```

- [ ] **Step 2: Register the marker**

In `pyproject.toml`'s `[tool.pytest.ini_options].markers`:

```toml
    "platform_io: identity-bound directory I/O; also run on Windows per PR (see tests/CLAUDE.md)",
```

Apply `pytestmark = pytest.mark.platform_io` at the top of
`tests/unit/sdk_/test_identity_io_contract.py`,
`tests/unit/sdk_/test_identity_io_platform.py`,
`tests/unit/sdk_/test_windows_metadata_journal.py`,
`tests/gui/browse/test_tile_routes.py` and
`tests/unit/cli/test_cli_recompile_slurm.py`.

- [ ] **Step 3: Add the Windows PR job**

In `run-pytest.yml`, after the `tests-linux` job:

```yaml
  tests-windows-platform-io:
    name: Windows py3.12 / platform-io
    needs: check-manual-run
    if: needs.check-manual-run.outputs.skip != 'true'
    runs-on: windows-latest
    defaults:
      run:
        shell: bash
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - uses: astral-sh/setup-uv@v8.1.0
        with:
          version: "latest"
      - name: Install dependencies
        run: |
          uv lock
          uv sync --group dev --group test-qt --all-extras
      - name: Run platform-io tests
        run: |
          source .venv/Scripts/activate
          pytest -q --tb=long -r "Efw" --no-header -n auto -m platform_io
```

- [ ] **Step 4: Guard the new job**

Add to `tests/unit/ci/test_pytest_shard_manifest.py`:

```python
def test_the_pr_lane_runs_platform_io_on_windows() -> None:
    """Windows-only I/O regressed for weeks behind a nightly-only lane."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["tests-windows-platform-io"]

    assert job["runs-on"] == "windows-latest"
    commands = [
        line.strip()
        for step in job["steps"]
        if isinstance(step.get("run"), str)
        for line in step["run"].splitlines()
        if line.strip().startswith("pytest ")
    ]
    assert commands and all("-m platform_io" in line for line in commands)
```

Document the marker in `tests/CLAUDE.md`'s marker table: runs everywhere by default, and additionally on a Windows PR job. Add one sentence about the interaction: a command-line `-m platform_io` **replaces** `addopts`' `-m 'not slow'` (`pyproject.toml:222`) rather than composing with it, so a test marked both `slow` and `platform_io` runs on the Windows lane while being excluded on Linux. Harmless today, surprising later.

- [ ] **Step 5: Run, commit**

Run: `uv run pytest tests/unit/ci/ tests/unit/sdk_/test_identity_io_platform.py -q`
Expected: PASS.

```bash
git add tests/ pyproject.toml .github/workflows/run-pytest.yml
git commit -m "ci: run identity-IO tests on Windows per PR; assert the active backend"
```

---

### Task 9: Prove the guards, then verify on every platform

**Files:** none changed unless a guard turns out to be dead.

- [ ] **Step 1: Mutation-check every refusal**

For each of the six refusals, remove its check in one backend, run the contract suite, confirm the matching test fails, then restore. A guard whose test still passes is not a guard — fix the test, not the guard. Back up before mutating and compare afterwards:

```bash
cp src/phenotypic/sdk_/_identity_io_posix.py /tmp/backup.py
# mutate one check, then:
uv run pytest tests/unit/sdk_/test_identity_io_contract.py -q   # expect a failure naming that refusal
cp /tmp/backup.py src/phenotypic/sdk_/_identity_io_posix.py
cmp src/phenotypic/sdk_/_identity_io_posix.py /tmp/backup.py && echo restored
```

Record the six results in the commit message of this task.

- [ ] **Step 2: Run the affected surface on the cluster**

Derive the surface from importers, not directory names:

```bash
grep -rl "_identity_io\|store_publication_token\|_cli_recompile_recovery\|_tile_routes" tests | grep -v "^tests/e2e"
```

Submit that list as a Slurm job (`short`, 16 CPU, 48 GB, 1 h; see the `slurm-job` skill and `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`). Never `-n auto`; use `-n $SLURM_CPUS_PER_TASK`.

- [ ] **Step 3: Full cross-platform run**

```bash
gh workflow run run-pytest-full.yml --ref <branch>
gh run list --workflow run-pytest-full.yml --branch <branch> -L 1
```

Expected: green on Linux 3.11, Linux 3.12, macOS 3.12 and Windows 3.12 — including the `cli-packaging` and `gui-browser` shards that this plan exists to fix.

- [ ] **Step 4: Confirm the two features actually work on Windows**

A green suite is not a working feature. Read the Windows `cli-packaging` job log and confirm `test_recompile_writes_no_store_byte` and `test_a_metadata_tree_recompiles_end_to_end` **ran** rather than skipped, and that `tests/gui/browse/test_tile_routes.py` reports no skips on Windows.

- [ ] **Step 5: Commit the verification record**

```bash
git commit --allow-empty -m "test: record identity-IO mutation results and cross-platform verification"
```

---

## Self-Review

**Spec coverage.** Facade and backend selection → Task 1; Windows mechanism and its three primitives → Tasks 3–5, with the two documented-but-unobserved mechanisms probed in Task 0; refusal contract → Task 1's suite, extended in 4 and 5, proven in Task 9; recompile port → Task 6; Browse port and the breaking `store_publication_token` change → Task 7; testing strategy, the `platform_io` marker, the Windows PR lane and the anti-false-green assertion → Task 8; risks → Task 0 (Win32 behaviour), Task 8 (silent degradation), Task 5 (handle leak, share mode), Task 2 (journal behaviour unchanged).

**Placeholders.** None: every code step carries the code, every test step the test, every run step the command and expected result.

**Type consistency.** `HeldDirectory`, `IdentityRefused`, `IdentityIoUnavailable`, `identity_io_available()`, `active_backend_name()`, `open_identity_directory()`, `validate_component()`, `link_count()`, `file_size()`, `list_names()`, `stream()` are spelled identically in Tasks 1–8. Both backends expose `SUPPORTED` and `BACKEND_NAME`; the Windows entry point alone takes the test-only `api=` keyword.

**One known dependency on an external answer:** Task 4's information-class constants are confirmed by Task 0 rather than assumed. If the probe contradicts them, Task 4 uses what the runner accepted.
