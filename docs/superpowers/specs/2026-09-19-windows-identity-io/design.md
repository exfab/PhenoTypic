# Identity-bound directory I/O on Windows

**Date:** 2026-09-19
**Status:** approved design, not yet implemented

## Objective

Make `--mode recompile` and the Browse OME-Zarr store route work on Windows,
by giving both a platform-neutral way to reach a file through directories whose
identity the process holds, without ever following a link.

Both features refuse to run on Windows today, and both refuse for the same
reason: they are written directly against POSIX directory file descriptors
(`os.open(..., O_DIRECTORY | O_NOFOLLOW, dir_fd=...)`), which Windows does not
provide.

- `_cli_recompile_recovery.py:213` gates every recompile transition read behind
  `_IDENTITY_BOUND_DIRECTORY_OPERATIONS`, and `:226` raises
  `RuntimeError("This platform cannot safely access recompile transition
  directories")` when it is false. Every `--mode recompile` on Windows fails
  there.
- `_gui/browse/_tile_routes.py:57` gates store-member serving behind
  `_SAFE_STORE_IO` and answers `422` (`_UnsafeStoreAccess`, `:53`) instead, so
  Browse cannot display an OME-Zarr store on Windows.

The refusals are correct as written: the guarantee these modules exist to make
cannot be made with path-based opens. This design supplies the missing
mechanism rather than relaxing the guarantee.

## Non-goals

- Weakening any refusal. Every check the POSIX path makes today is made by both
  backends after this change (see **Refusal contract**).
- Porting the metadata-migration journal's *write* path. It already works on
  Windows, and this design only borrows its low-level open.
- A user-visible switch. A platform that can do the work safely does it.
- Supporting a platform that offers neither backend. That case still fails
  closed, exactly as Windows does today.

## Background: what the two consumers actually need

Both needs are **read-only**. The recompile transition writer was removed in an
earlier release; every remaining caller reads evidence a previous run left.

| Operation | Recompile | Browse | POSIX spelling today |
|---|---|---|---|
| Hold a directory by identity, no-follow | yes | yes | `os.open(O_DIRECTORY\|O_NOFOLLOW)` |
| Walk to a child directory, still held | yes | yes | `os.open(name, ..., dir_fd=parent)` |
| Read a whole small regular file | yes (`:287`) | yes (`_read_store_json`, `:147`) | `os.open(..., dir_fd=)` + read |
| List entries | yes (`:514`) | no | `os.listdir(dir_fd)` |
| Open a streamable member | no | yes (`:110`) | `os.fdopen(member_fd, "rb")` |
| Refuse a multi-link file | yes | yes | `st_nlink != 1` |

The Windows journal (`sdk_/_windows_metadata_journal.py`) already performs
handle-relative, no-follow opens (`_CtypesWindowsApi`, `:466`), exposes handle
identity (`handle_info`, `:63`) and whole-file reads (`read_all`, `:67`). It has
no directory listing, no link count, and no way to hand out a Python stream.

## Architecture

### The facade

A new private module `sdk_/_identity_io.py` owns the concept. It exports one
entry point and one protocol:

```python
def open_identity_directory(path: Path) -> AbstractContextManager[HeldDirectory]: ...

class HeldDirectory(Protocol):
    def child_directory(self, name: str) -> HeldDirectory: ...
    def list_names(self) -> tuple[str, ...]: ...
    def read_regular_bytes(self, name: str, *, max_bytes: int | None = None) -> bytes: ...
    def open_regular_stream(self, name: str) -> BinaryIO: ...
    def reverify(self) -> None: ...
```

`open_regular_stream` returns a stream whose lifetime is **independent of the
hold**: Browse registers it with `Response.call_on_close`, so it is read after
the view returns and after the directory handles are closed
(`_tile_routes.py:400-437`).

Two backends implement it:

- `_identity_io_posix.py` — the existing dir-fd code, moved rather than
  rewritten.
- `_identity_io_windows.py` — `NtCreateFile` relative opens, built on the
  ctypes class that moves here out of `_windows_metadata_journal.py`; the
  journal then imports it from this module, so "open a Windows handle safely"
  has one implementation.

`open_identity_directory` selects a backend at import time and raises
`IdentityIoUnavailable` where neither is usable. A module-level
`identity_io_available()` reports the same fact without raising, for callers
that answer with a refusal rather than an exception.

A `Protocol` rather than a base class: the journal's tests already drive this
layer with an in-memory handle model, and the same technique lets the Windows
backend's logic be tested on Linux.

### Windows mechanism

Opens are always relative to a held handle — `NtCreateFile` with
`RootDirectory` set to the parent, `FILE_OPEN_REPARSE_POINT` for no-follow, and
`FILE_DIRECTORY_FILE` / `FILE_NON_DIRECTORY_FILE` to pin the type. This is the
call the journal already makes in production.

Three primitives are added:

| Need | Mechanism | Rationale |
|---|---|---|
| `list_names()` | `GetFileInformationByHandleEx` with `FileFullDirectoryRestartInfo` / `FileFullDirectoryInfo` | The binding already exists. Preferred over `NtQueryDirectoryFileEx`, which requires Windows 10 1709 or later. Iterate the chunked buffer until `ERROR_NO_MORE_FILES`. |
| link count | `FILE_STANDARD_INFO.NumberOfLinks`, same call | The Windows spelling of `st_nlink != 1` |
| `open_regular_stream()` | `msvcrt.open_osfhandle(handle, O_RDONLY)` then `os.fdopen(fd, "rb")` | Yields a real seekable file, so `send_file` and range requests work unchanged. Ownership transfers to the fd; the backend must not also close the handle. |

Three decisions that carry correctness:

- **Sharing mode.** Members open `FILE_SHARE_READ | FILE_SHARE_WRITE |
  FILE_SHARE_DELETE`. A read-only viewer must never lock a store against a
  concurrent CLI run. The journal's stricter sharing guards *its writes* and is
  the wrong default here.
- **Identity.** `FILE_ID_INFO` supplies a 128-bit volume-scoped identity. The
  journal already refuses an absent or all-zero id (seen on some network
  filesystems); that refusal carries over, so no stable identity fails closed.
- **Long paths.** A relative walk takes one component at a time, so `MAX_PATH`
  never applies to it. Only the initial absolute open needs `ngff_.long_path`.

Two mechanisms are asserted here on documentation, not on observation, and are
confirmed by a throwaway Windows probe **before** implementation begins:
`GetFileInformationByHandleEx` listing against an `NtCreateFile` directory
handle, and an `open_osfhandle` stream that Werkzeug can range-serve. The
journal's rename shipped on an assumption of this kind and was wrong (a mocked
API hid it); a five-minute probe is the cheap half of that lesson.

## Refusal contract

Both backends refuse, identically:

1. a non-canonical component — `""`, `.`, `..`, or one containing a separator;
2. a symlink, junction or other reparse point anywhere on the walk;
3. a non-directory where a directory is required, and the reverse;
4. a regular file with a link count other than 1;
5. an entry whose identity changed since it was held (`reverify`);
6. an absent entry — `FileNotFoundError`, which callers already treat as
   "no evidence", not as failure.

A platform providing neither backend refuses everything with
`IdentityIoUnavailable`. **The port widens where these guarantees hold; it never
changes what they are.** A reviewer should read the implementation against that
sentence.

## Consumer changes

### `_cli_recompile_recovery.py`

`_IDENTITY_BOUND_DIRECTORY_OPERATIONS` (`:213`) and
`_require_identity_bound_directory_operations` (`:226`) are deleted; the facade
owns the question. `_open_transition_directory` (`:241`) becomes a thin wrapper
over `open_identity_directory`, `_read_regular_file_at` (`:287`) becomes
`read_regular_bytes`, and `os.listdir(directory_fd)` (`:514`) becomes
`list_names()`. `tests/unit/cli/test_cli_recompile_slurm.py:2019` patches the
deleted constant and is retargeted at the facade's availability, keeping the
fail-closed path covered on every OS.

### `_gui/browse/_tile_routes.py`

`_SAFE_STORE_IO` (`:57`) is replaced by `identity_io_available()`.
`_UnsafeStoreAccess` (`:53`) and its `422` remain — a platform with no backend
must still refuse — but the refusal now means "this platform has no
identity-bound I/O" rather than "this is Windows". `_open_store_root` (`:91`)
and `_open_regular_store_member` (`:110`) become facade calls;
`_read_store_json` (`:147`) uses `read_regular_bytes(max_bytes=...)`, which
preserves the existing `_MAX_STORE_METADATA_BYTES` bound.

### `phenotypic.sdk_.store_publication_token` — breaking public change

`store_publication_token` (`sdk_/_io_constants.py:1907`, exported at
`sdk_/__init__.py:348`) takes `root_dir_fd: int | None`. It becomes:

```python
def store_publication_token(
    store: Path,
    *,
    root_directory: HeldDirectory | None = None,
) -> str | None: ...
```

`root_dir_fd=<int>` then raises `TypeError`. Both in-repo callers are in
`_tile_routes.py`. This is a documented `phenotypic.sdk_` export, so the change
needs a changelog entry and a note in the store-layout reference. The decision
was taken deliberately over an additive alias: one parameter with one meaning,
and the deprecation would otherwise outlive its usefulness.

### User-visible outcome

`--mode recompile` works on Windows, and Browse serves OME-Zarr stores there.
Neither is gated behind a flag.

## Testing

**One contract suite, every backend.** The six refusals above are written once
and parameterized over backends. On Linux the suite runs against the POSIX
backend and an in-memory handle model; on Windows it runs against the real
backend. Passing that suite is the definition of a correct backend.

**The in-memory model** extends `_MemoryWindowsApi`
(`tests/unit/sdk_/test_windows_metadata_journal.py:79`), which already rejects
path-based child operations, so code that forgets to route through a held
handle fails there rather than in production. It gains listing and link counts.
Its limit is explicit: it proves our logic, never Win32's behaviour.

**Every refusal is shown to be load-bearing.** For each, the check is removed
and the corresponding test must fail. A guard test that passes with its guard
deleted is not a test.

**Degradation must fail loudly.** If the Windows backend fails to bind its
symbols, every consumer degrades to "unsupported" and a suite of refusal tests
goes green by refusing everything. One test therefore asserts the *expected
backend* is active per platform, with no skip: the Windows backend on Windows,
the POSIX backend on Linux and macOS.

**CI.** A `platform_io` marker tags the facade, recompile-recovery and
tile-route suites, and a Windows job in `run-pytest.yml` runs `-m platform_io`
(about 3–5 minutes). The marker avoids a second path list that would drift from
`.github/pytest-shards.json` and its coverage test. The nightly full lane is
unchanged. Before merge: one full `run-pytest-full` run, green on all four
OS/Python combinations.

## Risks

| Risk | Mitigation |
|---|---|
| A Win32 call behaves differently than documented | Probe both unverified mechanisms on a runner before implementing, as was done for the journal rename |
| Silent degradation to "unsupported" turns the suite green | Per-platform backend assertion test, no skips |
| Handle leak in the route's error paths | The hold is a context manager; the stream is the one object that outlives it, and it is closed by `call_on_close` |
| A member open locks a store against a running CLI | Full share mode, asserted by a test that opens a member and writes the same file concurrently |
| The moved ctypes class changes journal behaviour | The journal's existing suite runs unchanged against the relocated class; the move is mechanical |
