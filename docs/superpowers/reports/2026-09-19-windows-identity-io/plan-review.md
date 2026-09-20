# Plan review — Windows identity-bound directory I/O

**Reviewed:** `docs/superpowers/specs/2026-09-19-windows-identity-io/design.md`,
`docs/superpowers/plans/2026-09-19-windows-identity-io/plan.md`
**Date:** 2026-09-19 · **Scope:** feasibility and correctness before any code is written.
**Tally:** 7 blocking · 9 should-fix · 5 nits.

All line references were resolved against the worktree
`/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/ci-split-full-suite` at review time.

> **Status note.** B1–B7 are patched into the plan and spec as of commit `85d8c45f`
> plus a follow-up commit covering B6 and B7. They are retained here in full because
> this file is the record of *why* those changes were made. The should-fixes and nits
> below had not been applied when this report was written.

---

## BLOCKING

### B1 — `reverify()` is a tautology on both backends; its contract test cannot pass

**Evidence.** The POSIX backend (plan.md:546-549) compares `os.fstat(self._fd)` against an
identity captured from *that same fd* at construction (plan.md:483-486). A held descriptor's
`(st_dev, st_ino)` cannot change — renaming a directory moves the name, not the inode. The
Windows backend has the identical shape (plan.md:740-743): `handle_info(self._handle).identity`
against the value read earlier from the same handle, whose `FILE_ID_INFO` is fixed for the
handle's life.

Consequences:

- `test_a_swapped_directory_fails_reverification` (plan.md:310-317) fails on both lanes. It
  renames `store/nested` aside and creates a new directory at the same path, then expects
  `nested.reverify()` to raise. The held fd still refers to the moved inode, so it does not.
- `list_names()` and `read_regular_bytes()` calling `self.reverify()` first (plan.md:501, :534,
  :737) is dead code.

**Fix.** `reverify()` must re-resolve the **name in the parent**, not re-read the held object:

- POSIX child: `os.stat(name, dir_fd=parent_fd, follow_symlinks=False)` compared against the
  held fd's `(st_dev, st_ino)`.
- Windows child: a fresh `open_directory(parent_handle, name)` plus `FILE_ID_INFO`, compared
  against the held identity.

`HeldDirectory` therefore needs to carry `(parent, name)`. The root is handled separately — see
**Follow-up 1**.

**Related, separate:** the same defect ships today in `_windows_metadata_journal.py` — see
**Follow-up 3**.

*Applied in `85d8c45f`.*

---

### B2 — `store_publication_token` cannot be built on `read_regular_bytes`; Task 7 Step 3 is not implementable as written

**Evidence.** The token digest folds three stat fields taken from the held descriptor:

- `_io_constants.py:1994-1997` — `digest.update(after.st_mtime_ns…)`, `after.st_ctime_ns`,
  `after.st_ino`.
- `_io_constants.py:1956-1970` — the torn-read guard compares a *before* and an *after*
  `os.fstat` of the same fd and raises `OSError` when they differ or when
  `len(raw) != after.st_size`.

`read_regular_bytes` returns bytes and nothing else (plan.md:369-371), so neither the digest
nor the torn-read guard can be reconstructed by the caller.

**Worse: a silent cross-branch hazard.** The route computes the token twice by two different
routes and compares the results against a stored revision:

- `_tile_routes.py:347` — path branch, `store_publication_token(source)`.
- `_tile_routes.py:385` and `:405` — held branch, `store_publication_token(source, root_dir_fd=root_fd)`.

If the two branches ever disagree on `st_ino` or `st_ctime_ns`, the route returns **409 on
every tile request, forever**. On Windows a `FILE_ID_INFO`-derived identity will not equal
`os.stat().st_ino`, so this is the default outcome of a naive port, and nothing in the plan
tests for it.

**Fix.** See **Follow-up 2** for the exact protocol addition, the contract, which branch changes,
and the missing test.

*Applied in `85d8c45f`.*

---

### B3 — `long_path()` resolves symlinks, so the Windows root open follows junctions the POSIX one refuses

**Evidence.** `ngff_.py:1658-1660`:

```python
if os.name != "nt":
    return str(path)
text = str(Path(path).resolve())
return text if text.startswith("\\\\?\\") else "\\\\?\\" + text
```

`Path.resolve()` follows symlinks and junctions. The plan passes its output straight to the
root open (plan.md:762, `resolved_api.open_anchor(long_path(root), share_delete=True)`). A store
root that *is* a junction resolves to its target and opens successfully, whereas the POSIX
backend's `O_NOFOLLOW` (plan.md:469) refuses it. The `_FILE_ATTRIBUTE_REPARSE_POINT` check in
`_WindowsHeldDirectory.__init__` (plan.md:685-686) cannot catch it, because after `resolve()`
the handle is on the target, not the link.

This directly contradicts design.md:136 ("Both backends refuse, identically") and refusal 2.

**Fix.** Do not resolve. Build the `\\?\` prefix from `os.path.abspath` only — or walk from the
filesystem anchor the way `WindowsJournalSession.__enter__` already does
(`_windows_metadata_journal.py:98-119`), which additionally delivers refusal 2 for intermediate
components. The `\\?\` prefix is only legal on a fully qualified path, which `abspath` already
guarantees; `resolve()` was doing more than the prefix requires.

*Applied in `85d8c45f`.*

---

### B4 — neither backend maps `OSError` to `IdentityRefused`, so three of Task 1's own tests fail

**Evidence.** On Linux:

- `os.open(name, O_RDONLY|O_NONBLOCK|O_NOFOLLOW, dir_fd=…)` on a symlink raises
  `OSError(ELOOP)`, not `IdentityRefused`. Test at plan.md:259-269 expects `IdentityRefused`.
- `os.open(name, O_RDONLY|O_DIRECTORY|O_NOFOLLOW, dir_fd=…)` on a regular file raises
  `OSError(ENOTDIR)`. Test at plan.md:279-283 expects `IdentityRefused`.

The code being replaced *did* perform this mapping —
`_cli_recompile_recovery.py:273-274`:

```python
except OSError as exc:
    raise ValueError("Transition directory is not canonical") from exc
```

Windows has the same hole: `_nt_open` maps only DOS errors 2 and 3 to `FileNotFoundError`
(`_windows_metadata_journal.py:679-684`), so `STATUS_NOT_A_DIRECTORY` → `ERROR_DIRECTORY` (267)
surfaces as a bare `OSError`.

**Fix.** Wrap each relative open in both backends and translate: `ELOOP`, `ENOTDIR`, `EISDIR`
(POSIX) and `ERROR_DIRECTORY` plus `ERROR_ACCESS_DENIED`-on-reparse (Windows) become
`IdentityRefused`; `ENOENT` and DOS 2/3 stay `FileNotFoundError`. Everything else stays `OSError`.

Note this also matters for the consumers' existing exception vocabulary:
`recoverable_recompile_table_transition` catches `ValueError` and returns `False`
(`_cli_recompile_recovery.py:446-453`), which is the behaviour `IdentityRefused(ValueError)`
preserves and a bare `OSError` would too — but the Browse route distinguishes them
(`_tile_routes.py:377-382`), so the mapping is not cosmetic.

*Applied in `85d8c45f`.*

---

### B5 — `open_file` demands write + delete access; a read-only read will fail

**Evidence.** `_windows_metadata_journal.py:715-741` requests

```
_FILE_READ_DATA | _FILE_WRITE_DATA | _FILE_READ_ATTRIBUTES
| _FILE_WRITE_ATTRIBUTES | _DELETE | _SYNCHRONIZE
```

(`_FILE_WRITE_DATA` at `:728`) with options including `_FILE_WRITE_THROUGH` (`:739`). That mask
is correct for the journal, which writes, renames and deletes receipts. The plan reuses it
verbatim for a read-only consumer (plan.md:709-711).

Consequences on Windows: a store on read-only media, on a read-only SMB share, or a member whose
ACL grants read but not write returns `ERROR_ACCESS_DENIED` from `NtCreateFile`. Because
`_nt_open` maps only errors 2 and 3 (`:679-684`), it surfaces as a bare `OSError` →
the route's broad `except` (`_tile_routes.py:399-400`) → **404 on a store the user can plainly
read**; and recompile's `except (OSError, ValueError)` (`_cli_recompile_recovery.py:519`) →
`RuntimeError("Cannot safely enumerate the recompile transition directory")`. Neither message
names the real cause.

It also makes `msvcrt.open_osfhandle(handle, os.O_RDONLY)` (plan.md:986) dishonest about the
handle it wraps.

**Fix.** Add a read-only sibling rather than widening the journal's method — implemented as
`open_regular_read(parent, name, *, share_delete=True)` with
`desired_access = _FILE_READ_DATA | _FILE_READ_ATTRIBUTES | _SYNCHRONIZE` and
`options = _FILE_NON_DIRECTORY_FILE | _FILE_OPEN_REPARSE_POINT | _FILE_SYNCHRONOUS_IO_NONALERT`
(no `_FILE_WRITE_THROUGH`). `open_file` stays untouched so Task 2 remains mechanical.

Add it to Task 0's probe: open a file whose ACL denies write and confirm the read-only mask
succeeds where the journal's mask fails. Otherwise this ships untested.

*Applied in `85d8c45f`, as `open_regular_read`.*

---

### B6 — Task 2's move is circular as specified

**Evidence.** `_CtypesWindowsApi` raises `WindowsJournalUnavailable` in three places:
`__init__` (`_windows_metadata_journal.py:466-480`), `_raise_last_error` (`:601-605`), and
`_nt_open` (`:685`). Moving the class into `_identity_io_windows` while the journal imports it
back (plan.md:605-613) produces the cycle
`_windows_metadata_journal → _identity_io_windows → _windows_metadata_journal`.

Task 2 Step 1's move list (plan.md:601) names the ctypes structures, the `_FILE_*`/`_OBJ_*`/
`_ERROR_*` constants, `WindowsHandleInfo`, the `_WindowsApi` protocol and `_CtypesWindowsApi` —
but not that exception class.

**Fix.** Move `WindowsJournalUnavailable` with them and re-export it from the journal alongside
the rest of the Step 2 re-export block, or redefine it in `_identity_io_windows` as a subclass
of `IdentityIoUnavailable` and alias it in the journal.

One wording change for Step 3: "PASS, same count as before the move" only runs if the modules
import at all. A circular import shows up as a **collection error**, not a count mismatch — say
so, or the step reads as if it would catch this and it would not.

*Applied in the B6/B7 follow-up commit.*

---

### B7 — `SUPPORTED = os.name == "nt"` defeats the anti-false-green test the plan rests on

**Evidence.** plan.md:752 sets `SUPPORTED = os.name == "nt"`, and plan.md:761 constructs the
ctypes API lazily inside `open_identity_directory`. A symbol that fails to bind raises
`WindowsJournalUnavailable` from `_CtypesWindowsApi()` (`_windows_metadata_journal.py:466-480`)
at **call** time, not import time. So:

- `identity_io_available()` stays `True` and `active_backend_name()` stays `"windows"`;
- Task 8's assertion (plan.md:1220-1223) — the one test written specifically to catch this —
  passes while nothing works;
- consumers get an uncaught `RuntimeError` instead of the designed refusal.
  `recoverable_recompile_table_transition` catches only
  `(KeyError, OSError, TypeError, ValueError, JSONDecodeError)`
  (`_cli_recompile_recovery.py:446-453`) and
  `recoverable_recompile_measurement_sources` only `(OSError, ValueError)` (`:519`), so
  recompile crashes rather than degrading. That is the opposite of design.md:215-219.

**Fix.** Bind at import and let the result decide:

```python
try:
    _API: _CtypesWindowsApi | None = (
        _CtypesWindowsApi() if os.name == "nt" else None
    )
except WindowsJournalUnavailable:
    _API = None

BACKEND_NAME = "windows"
SUPPORTED = _API is not None
```

and have `open_identity_directory` reuse `_API` rather than constructing a fresh instance per
call (plan.md:761 becomes `resolved_api = api if api is not None else _API`). A bind failure then
makes Task 8's test fail loudly, which is what it exists for.

Write one consequence into the plan: `_CtypesWindowsApi` becomes a process-wide singleton shared
with the journal, which constructs its own at `:93`. That is harmless — it holds only bound
function pointers and no per-call state (`_bind`, `:485-594`) — but a reader will wonder, so say it.

*Applied in the B6/B7 follow-up commit.*

---

## FOLLOW-UP ANSWERS

### Follow-up 1 — what the root's `reverify()` does, and what it can still catch

`(parent, name)` is available for every directory reached through `child_directory()`. The root
is the one that has neither, and the honest answer for it is a **path-based `lstat`**:

```python
def reverify(self) -> None:          # root only
    current = os.lstat(self.path)    # never used to open anything
    if (current.st_dev, current.st_ino) != self._identity:
        raise IdentityRefused(f"identity changed: {self.path}")
```

The key property is the **direction of use**: the stat result is only ever *compared*, never
opened through. A hostile swap makes the check *fail*; it can never make the facade open the
wrong object, because every actual open still goes through the held descriptor. So
re-introducing a path lookup here does not reopen the TOCTOU window the module exists to close.

**What the root's check still catches:** the held root being renamed, deleted, or replaced by a
different directory between the hold being taken and the read.

**What it cannot catch:** a swap of any **ancestor** of the root, since `lstat` re-walks that
prefix through whatever is there now. A child's check is strictly stronger — it re-resolves one
component inside a held parent, so no ancestor is re-walked.

**The Flask route does not depend on this.** Its real freshness guarantee is
`store_publication_token` compared against `source.store_revision`, taken **twice** — before
`_open_regular_store_member` and again after (`_tile_routes.py:385` and `:405`). The route never
calls `reverify()`. It is `recoverable_recompile_measurement_sources` → `list_names()`
(`_cli_recompile_recovery.py:508-516`) that does.

**Spec consequence.** Refusal 5 as originally worded cannot be honoured for the root, only for
children. Reword it to bind children only.

*Applied in `85d8c45f`; refusal 5 now binds children only.*

---

### Follow-up 2 — the exact protocol addition for `store_publication_token`

Add one method to `HeldDirectory`. Do **not** change `read_regular_bytes`:

```python
def read_regular_with_stat(
    self, name: str, *, max_bytes: int | None = None
) -> tuple[bytes, os.stat_result]: ...
```

**Contract.** Returns the member's bytes and the `os.stat_result` of the *same open file
description*, taken after the read. The backend itself performs the torn-read comparison — it
stats before and after and raises `IdentityRefused` when
`(st_size, st_mtime_ns, st_ctime_ns, st_ino)` differ across the read, or when
`len(payload) != st_size`. The non-regular and `st_nlink != 1` refusals stay exactly where
`read_regular_bytes` already puts them.

- **POSIX backend:** `os.fstat(fd)` before and after — the code already at
  `_io_constants.py:1948-1970`, moved rather than rewritten.
- **Windows backend:** `fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)` then
  `os.fstat(fd)`. This is the point of the whole fix: it is the *same* `os.stat_result` CPython
  builds for `os.stat(path)` on Windows, so the two token branches agree by construction rather
  than by hope. Ownership transfers to the fd — close with `os.close(fd)`, never also
  `CloseHandle`.

**Which branch changes.** Only the held branch (`_io_constants.py:1943-1970`,
`root_dir_fd` → `root_directory`):

```python
        else:
            raw, after = root_directory.read_regular_with_stat(STORE_ROOT_JSON)
            before = after      # the backend already proved they matched
```

The path branch (`root.lstat()` / `read_bytes()` / `root.lstat()`, `_io_constants.py:1944-1949`)
is **untouched**. That asymmetry is what makes cross-branch agreement checkable.

**Keep `IdentityRefused` from escaping.** `read_regular_with_stat` raises on `st_nlink != 1`,
where the function today returns `None` (`_io_constants.py:1951-1955`). Wrap the call so it
returns `None`, matching the existing `except OSError: return None` arm — otherwise the route's
409-vs-404 taxonomy shifts for a hard-linked `zarr.json`.

**The test the plan is missing**, and the one that would have caught B2:

```python
def test_both_token_branches_agree_for_one_store(tmp_path):
    store = _published_store(tmp_path)
    with _identity_io.open_identity_directory(store) as held:
        assert store_publication_token(store) == store_publication_token(
            store, root_directory=held
        )
```

Run it on every platform, not only Linux. Without it, a Windows disagreement surfaces as Browse
returning 409 on every tile request forever, behind a green suite.

*Applied in `85d8c45f`, including the cross-branch test and the `IdentityRefused -> None` wrapper.*

---

### Follow-up 3 — the journal's vacuous identity checks: a gap in depth, not the whole guarantee

**`_verify_directories` is the only identity re-check, and the apparent backstop is vacuous too.**

- `_verify_directories` (`_windows_metadata_journal.py:146-156`) compares
  `handle_info(held.handle).identity` against `held.identity`, both read from the same handle.
  A handle's `FILE_ID_INFO` is fixed for its life, so this can never differ. Same defect as B1.
- `read_bytes` (`:265-277`) takes `handle_info(handle)` before and after the read and compares.
  Both readings come from the same handle. **Second tautology site.**

**The checks that do fire are all open-time:**

- `_validated_directory` (`:136-144`) — rejects a reparse point, and an absent or all-zero
  `FILE_ID_INFO`, when a component is *first* opened.
- `_open_file` (`:252-258`) — the same two checks for a child file.

**Why this is a gap in depth rather than an exploitable hole, and the reason is structural.**
The session caches every component in `self._directories` (`:174`, `:208`, `:233`), and
`_directory` returns the cached entry without re-resolving the name (`:176-179`). Every
subsequent open goes through a cached handle. The journal therefore **never re-looks-up a path
after its initial walk**, so a swap after the walk changes what a path *names* but cannot change
what the session *touches*. The vacuous checks are dead code offering false assurance, not a
live vulnerability.

**How to file it.** The fix is to make the checks mean something — re-resolve the name in the
held parent, exactly as B1 prescribes — not to treat shipped migrations as compromised. The
same reading applies to B1 in this plan: it is a **spec-and-test** defect (refusal 5 as written
cannot be satisfied by the design), not a live vulnerability in the ported code.

*Independently confirmed by the team lead at `:265-277`, `:136-144`, `:252-258`.*

---

## SHOULD-FIX

*None of the findings below had been applied when this report was written.*

### S1 — fd/handle leak on every error path in both entry points

**Evidence.** plan.md:563-568 (POSIX) and plan.md:763-768 (Windows):

```python
    fd = os.open(root, _DIR_FLAGS)
    try:
        held = _PosixHeldDirectory(root, fd, owned=False)
        yield held
        held._close()
    finally:
        os.close(fd)
```

`held._close()` sits inside the `try`, before the `finally`. When the `with` body raises, the
generator receives the exception at the `yield`, `_close()` is skipped, and only the root
descriptor is closed — every child fd or NT handle accumulated in `self._children`
(plan.md:497, :704) leaks. In the Browse route that is one leaked descriptor per *failing*
request, which is the path most likely to be exercised under load.

**Fix.** Move the close into the `finally`, make `_close()` idempotent, and let it close the
root too — drop the separate `owned=False` root close, or guard it with a `_closed` flag.

### S2 — `_WindowsHeldDirectory` has no `_close`, and accepts a regular file as a root

**Evidence.** The entry point calls `held._close()` (plan.md:766), but Task 3's class body
(plan.md:676-744) never defines it; only `_PosixHeldDirectory` has one (plan.md:551-555).

Separately, `open_anchor` uses
`CreateFileW(..., _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT)`
(`_windows_metadata_journal.py:614-625`), which **succeeds on a regular file**. And
`_WindowsHeldDirectory.__init__` checks only the reparse bit and the file id
(plan.md:684-689) — never that the object is a directory. POSIX refuses this twice, via
`O_DIRECTORY` (plan.md:469) and the explicit `S_ISDIR` check (plan.md:484-485). So
`open_identity_directory(<a file>)` succeeds on Windows and fails on Linux.

**Fix.** Define `_close` (children leaf-first, then self when owned), and check
`FILE_STANDARD_INFO.Directory` in `__init__` — the `_standard_info` helper Task 3 Step 5 already
adds (plan.md:791-800) provides it for free.

### S3 — the tile-route module-level skipif is unhandled, and is a second false-green vector

**Evidence.** `tests/gui/browse/test_tile_routes.py:20-23`:

```python
requires_safe_store_io = pytest.mark.skipif(
    not _tile_routes._SAFE_STORE_IO,
    reason="this platform cannot anchor store reads to directory fds",
)
```

It is applied **8 times** in that file. Task 7 mentions only the
`monkeypatch.setattr(_tile_routes, "_SAFE_STORE_IO", False)` at `:204` (plan.md:1127-1133).

Two failure modes: deleting `_SAFE_STORE_IO` breaks collection with `AttributeError`; and
retargeting the skipif at `identity_io_available()` converts a Windows bind failure into 8
silent skips — precisely the degradation design.md:215-219 exists to prevent, caught only by a
human reading a log in Task 9 Step 4 (plan.md:1344-1346).

**Fix.** Delete the decorator and its 8 uses outright. Both platforms now have a backend; the
Task 8 platform assertion is the gate, and it has no skip.

### S4 — say explicitly that the 422 is preserved

**Evidence.** `_tile_routes.py:377-382` catches `_UnsafeStoreAccess` first and only then
`(OSError, RuntimeError, TypeError, ValueError)` → 404. `IdentityIoUnavailable` is a
`RuntimeError` (plan.md:351-352). If `_open_store_root` lets it through instead of keeping its
own guard (`_tile_routes.py:93-96`), the documented 422 silently becomes a 404 — and the Task 7
Step 1 test, which asserts a 422 via `monkeypatch.setattr(_identity_io, "_BACKEND", None)`
(plan.md:1132), fails for a reason nobody will read as "clause order".

**Fix.** Write the guard into Task 7 Step 4 rather than leaving it implied by
"keep `_UnsafeStoreAccess` and its 422":

```python
def _open_store_root(store: Path):
    if not identity_io_available():
        raise _UnsafeStoreAccess(
            "this platform cannot safely serve Zarr store members"
        )
    return open_identity_directory(store)
```

### S5 — the fake-Windows lane is vacuous for three tests and blocked by an assert on a fourth

**Evidence.** `_MemoryWindowsApi` (`tests/unit/sdk_/test_windows_metadata_journal.py:79`) is a
pure in-memory model with no connection to the real filesystem, and asserts
`share_delete is False` in `open_anchor` (`:106`), `open_directory` (`:117`) and `open_file`
(`:139`). The plan's backend always passes `share_delete=True` (plan.md:695, :711, :762), so
every fake-lane test errors on an assertion until that is parameterized.

Beyond that, tests that mutate the real tree *after* the hold is taken cannot be observed by an
in-memory model:

- `test_a_swapped_directory_fails_reverification` — `os.rename` (plan.md:314-315);
- `test_a_streamed_member_does_not_lock_the_file` — `write_bytes` (plan.md:956);
- `test_a_multi_link_file_is_refused` — `os.link` (plan.md:289); the fake has no hard-link
  concept at all, so `link_count` would have to be invented rather than modelled.

Note also that `memory_api_factory(path)` (plan.md:660) is never specified — the fake must be
seeded from the real tree at construction time for `test_listing_spans_multiple_buffer_fills`
(plan.md:896-903) to mean anything, and that only works because those 2,000 files are written
*before* the hold opens.

**Fix.** State per test which lanes it runs in, and specify `memory_api_factory` as a scan of the
tree at construction. Keep `share_delete is False` asserted on the journal's own path — it pins
the journal's stricter sharing, which design.md:117-120 deliberately does not inherit — by making
the fake take the expected value as a constructor argument rather than hard-coding `False`.

### S6 — Task 3's xfail mechanism does not exist as described, and the count is wrong

**Evidence.** The lane comes from a **fixture** `params` (plan.md:648), so a plain
`@pytest.mark.xfail` decorator marks both lanes, and
`pytest.param("fake-windows", marks=pytest.mark.xfail)` marks the *entire* lane rather than the
two named tests. Per-test-per-lane requires `request.applymarker(...)` inside the test body,
guarded on `request.node.callspec.params["backend"]`.

Separately, plan.md:814 says "mark those two tests" — but at Task 3 only
`test_listing_excludes_dot_entries` exists. The stream tests are not written until Task 5 Step 1
(plan.md:930-959), so there is nothing to xfail for `open_regular_stream`.

**Fix.** Use `applymarker`, and correct the count to one test. `strict=True` is the right call and
Task 4 Step 1's "drop the xfail, expect FAIL" sequencing is sound once the mechanism works.

### S7 — refusal 2 overstates what either backend delivers

**Evidence.** `O_NOFOLLOW` and `FILE_OPEN_REPARSE_POINT` / `FILE_FLAG_OPEN_REPARSE_POINT` apply
to the **final** component only; intermediate components of the root path are followed on both
platforms. `_cli_recompile_recovery.py:260` (`os.open(canonical_output, flags)`) and
`_tile_routes.py:95-98` have the same property today, so this is not a regression — but
design.md:139 says "anywhere on the walk" and design.md:147-149 instructs the reviewer to read
the implementation against that sentence. A conscientious reviewer will find the mismatch and
have no way to tell whether it is a defect.

**Fix.** Narrow the wording to "anywhere on the walk **below the held root**" — or walk from the
filesystem anchor on Windows the way `WindowsJournalSession.__enter__` already does
(`_windows_metadata_journal.py:98-119`), which would genuinely deliver the stronger claim on one
platform and make the asymmetry explicit rather than accidental.

### S8 — Task 5 will fail its own mypy gate

**Evidence.** `msvcrt` and `os.O_BINARY` (plan.md:986) are both guarded behind
`sys.platform == "win32"` in typeshed, and `uv run mypy src/phenotypic` — the per-commit gate at
plan.md:21 — runs with the host platform, i.e. Linux.

Establish the baseline first: `_windows_metadata_journal.py` today contains **no** `type: ignore`
and **no** `sys.platform` guard anywhere (verified by grep), while calling `ctypes.WinDLL`
directly at `:474-475`, which typeshed guards the same way. So either mypy already tolerates this
pattern here or that file is already red — and the plan should not discover which during Task 5.

**Fix.** Run `uv run mypy src/phenotypic/sdk_/_windows_metadata_journal.py` before Task 5 and
record the answer in the plan. If it is clean, Task 5 needs nothing; if not, put the Windows-only
code under an `if sys.platform == "win32":` block.

### S9 — the POSIX `SUPPORTED` predicate narrows correctly but silently

**Evidence.** plan.md:459-466 keeps `os.listdir in os.supports_fd` and
`os.open in os.supports_dir_fd`, but drops the `os.mkdir` / `os.stat` / `os.unlink` / `os.rename`
requirements carried by the constant it replaces (`_cli_recompile_recovery.py:219-222`). That is
*correct* for a read-only facade — those four were needed by the transition writer that has since
been removed, as the docstring at `_cli_recompile_recovery.py:246-250` records.

But B1's fix reintroduces a dependency on `os.stat in os.supports_dir_fd` for child
reverification.

**Fix.** Add `os.stat` back to the predicate deliberately, and have Task 6's replacement test
assert the new narrower predicate rather than inherit it by accident.

---

## NITS

### N1 — the Win32 constants check out, with one correction to the spec's rationale

Verified against
`https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ne-minwinbase-file_info_by_handle_class`.
`FILE_INFO_BY_HANDLE_CLASS` is declared without explicit values, so ordinals are positional:

| Member | Value | Used at |
|---|---|---|
| `FileStandardInfo` | 1 | plan.md:788 |
| `FileAttributeTagInfo` | 9 | `_windows_metadata_journal.py:398` |
| `FileIdBothDirectoryInfo` / `…RestartInfo` | 10 / 11 | plan.md:174 (fallback) |
| `FileFullDirectoryInfo` / `…RestartInfo` | 14 / 15 | plan.md:847-848 |
| `FileIdInfo` | 18 | `_windows_metadata_journal.py:399` |

Every value the plan and the existing journal use is correct. The `FILE_FULL_DIR_INFO` and
`FILE_STANDARD_INFO` field orders in the probe (plan.md:61-85) match the headers, and neither
struct is packed, so ctypes' natural alignment and the plan's `FileName.offset` arithmetic
(plan.md:869) are right.

**One correction.** `FileFullDirectory*` is documented as "not supported before Windows 8 and
Windows Server 2012", not the Windows 10 1709 figure design.md:111 contrasts against — 1709 is
the `NtQueryDirectoryFileEx` floor. The named fallback `FileIdBothDirectory*` is Vista+, so the
fallback story holds; just fix the sentence.

### N2 — Task 0 should probe three more things, all load-bearing above

1. **Does `os.fstat` on an `open_osfhandle` fd return the same `st_ino` and `st_ctime_ns` as
   `os.stat(path)`?** B2 turns entirely on this and it is not currently probed. Print both and
   compare in the probe script.
2. **Does a read-only access mask succeed where the journal's write mask fails?** Create a file,
   deny write on its ACL (`icacls`), and try both masks — this is B5's only real test.
3. **Listing against a `FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT` handle.**
   Already covered, because the probe goes through `api.open_directory` (plan.md:58), which is
   the right call. Note it explicitly so nobody "simplifies" the probe to a plain `CreateFileW`
   handle and ends up probing the wrong thing.

### N3 — `_children` grows per call, and the route re-walks the same prefixes

Each `child_directory()` appends unconditionally (plan.md:497, :704), and `_read_store_json` is
called once per series and once per label inside `_image_store_prefixes`
(`_tile_routes.py:147`, `:196`, `:209`), each time re-walking from the root. It is bounded per
request, so not a leak — but a `dict[tuple[str, ...], HeldDirectory]` cache in the hold, which is
exactly what `WindowsJournalSession._directories` already is
(`_windows_metadata_journal.py:91`, `:176-179`), is cheaper and closer to the shape this code is
being merged alongside.

### N4 — every line reference in the plan resolves

Checked and confirmed: `_cli_recompile_recovery.py:213` (`_IDENTITY_BOUND_DIRECTORY_OPERATIONS`),
`:226` (`_require_identity_bound_directory_operations`), `:241` (`_open_transition_directory`),
`:287` (`_read_regular_file_at`), `:387` and `:415` (its two call sites), `:514`
(`os.listdir(directory_fd)`); `_tile_routes.py:53` (`_UnsafeStoreAccess`), `:57`
(`_SAFE_STORE_IO`), `:91` (`_open_store_root`), `:110` (`_open_regular_store_member`), `:147`
(`_read_store_json`); `_io_constants.py:1907` (`store_publication_token`);
`sdk_/__init__.py:348`; `tests/unit/sdk_/test_windows_metadata_journal.py:79`
(`_MemoryWindowsApi`); `_CtypesWindowsApi` at `_windows_metadata_journal.py:466`; constants at
`:370-406`.

Also confirmed out of band: `tests/unit/sdk_/` is a **directory** entry in
`.github/pytest-shards.json` (shard `tune-sdk`), so the two new test modules do not break
`test_each_configured_test_file_belongs_to_exactly_one_shard`
(`tests/unit/ci/test_pytest_shard_manifest.py:92-97`), which asserts
`set(owners) == _configured_test_files()`. And `check-manual-run` (`run-pytest.yml:33`) and
`tests-linux` (`:86`) both exist, as Task 8 Step 3 assumes.

### N5 — Task 8's `-n auto` is correct here, and the marker interaction deserves one sentence

`test_pr_workflow_uses_complete_shards_without_testmon` asserts `"-n auto" in workflow`
(`tests/unit/ci/test_pytest_shard_manifest.py:106`), so leave the plan's `-n auto` alone despite
the repo's usual caution about it — a GitHub runner's core count and its allocation are the same
number.

Note `addopts = "--verbose --capture=no -m 'not slow'"` (`pyproject.toml:222`): the job's
command-line `-m platform_io` **replaces** it rather than composing with it, so a test marked
both `slow` and `platform_io` would run on the Windows lane while being excluded on Linux.
Harmless today, surprising later — put it in the `tests/CLAUDE.md` marker row that Task 8 Step 4
already adds.

---

## TASK ORDERING AND INDEPENDENCE

Tasks 1 and 2 are independent of each other. Task 3 needs both, plus Task 0's answer. Tasks 4 and
5 need 3. Tasks 6, 7 and 8 need only 1. Task 9 is last.

Each is individually implementable and testable — **except Task 1**, whose Step 5
"Expected: PASS" (plan.md:573-574) is wrong as written: B1 and B4 make three of its own contract
tests fail before Windows enters the picture. Fix those inside Task 1, or the entire chain starts
red and every later "Expected: PASS" becomes unreadable.

On Task 3's xfail question specifically (`list_names` / `open_regular_stream` marked
`xfail(strict=True)` for the fake lane until Tasks 4-5): the *intent* is sound — a strict xfail
that Task 4 Step 1 then removes is exactly the right way to sequence a partially implemented
backend. The *mechanism* does not work as described and the count is wrong; see **S6**.

## FALSE-GREEN ASSESSMENT

The question asked was: if the Windows backend fails to bind a symbol, does anything in this plan
still pass while shipping nothing? **Yes — two independent vectors.**

1. **B7.** A bind failure leaves `active_backend_name() == "windows"`, so the single test written
   to catch exactly this (plan.md:1220-1223) passes. Consumers then crash with an uncaught
   `RuntimeError` rather than degrading, so the Windows lane is red *somewhere* — but not at the
   assertion that was supposed to own the question, and the failure names ctypes rather than the
   backend.
2. **S3.** Retargeting the module-level `requires_safe_store_io` skipif
   (`tests/gui/browse/test_tile_routes.py:20-23`, 8 uses) at `identity_io_available()` turns the
   same bind failure into 8 silent skips. That lane goes genuinely green while serving nothing.

Fix both and the Task 8 assertion does the job it was written for. With them unfixed, a Windows
backend that binds nothing can ship behind a green PR lane.

---

## ORCHESTRATOR DISPOSITION (2026-09-20)

Added by the orchestrator, not the reviewer. Every finding above was verified
against the tree before being applied; none was taken on the report's word.

**Applied to the plan and spec** (commits `85d8c45f` and the one carrying this
file): B1–B7, S1–S9, N1, N2, N3, N5.

Notes where the applied form differs from the report:

- **B5 / read-only opener** is spelled `open_regular_read`, not
  `open_file_readonly`. `open_file`'s write mask is untouched, as required.
- **S3** is applied in its strong form: `requires_safe_store_io` and all 8 uses
  are deleted rather than retargeted. Task 8's platform assertion is the gate.
- **S6** is applied with `applymarker` and a corrected count of one test.
- **S7** narrowed the spec's refusal 2 to "below the held root" rather than
  walking from the filesystem anchor. The stronger guarantee on one platform
  only would have re-created the asymmetry the facade exists to remove; the
  honest narrower claim is what the code actually delivers on both.
- **Task-ordering verdict** (Task 1's "Expected: PASS" being wrong) is resolved
  by construction: the B1 and B4 fixes were patched *into* Task 1's own backend
  code, so its contract suite passes at the end of Task 1 as written.

**N4** required no action — it confirms every line reference in the plan
resolves, including that `tests/unit/sdk_/` is a directory entry in
`.github/pytest-shards.json`, so the two new test modules do not break the
shard manifest's coverage test.

**Carried out of this plan as a separate issue:** the vacuous identity checks
in `_windows_metadata_journal.py` (`:153`, `:272`, `:309`, `:329`). Confirmed
independently: each compares two reads of one handle, so none can fire. The
redirection guarantee survives structurally — the session caches every
component and never re-resolves a path after its initial walk — so these are
defence-in-depth checks that can never fire, not an exploitable hole. The fix
is to make them mean something (re-resolve the name in the held parent), which
is the same fix B1 applies here.
