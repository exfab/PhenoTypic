# Tasks 3–5: the Windows identity-IO backend

**Author:** implementation agent (C3)
**Date:** 2026-09-19
**Subject:** `docs/superpowers/plans/2026-09-19-windows-identity-io/plan.md`, Tasks 3, 4 and 5
**Status:** implemented, not committed

This report exists because the code it describes **cannot run on the machine
that wrote it**. Everything below distinguishes what was measured from what was
reasoned, and the central section is the list of code that has never executed.

---

## 1. What landed

All paths relative to
`/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/ci-split-full-suite`.

| File | Change |
|---|---|
| `src/phenotypic/sdk_/_identity_io_windows.py` | Task 3: `_WindowsHeldDirectory`, `_API`/`SUPPORTED` bound at import, `_extended_length`, `open_identity_directory(path, *, api=None)`, `_IdentityWindowsApi` protocol, and the ctypes additions `_standard_info` / `link_count` / `file_size` / `is_directory` / `open_regular_read` / `adopt_descriptor`. Task 4: `list_names` + `_FileFullDirInfo`. Task 5: `stream`. Plus `_FILE_ATTRIBUTE_REPARSE_POINT`, relocated here. |
| `src/phenotypic/sdk_/_windows_metadata_journal.py` | Its own `_FILE_ATTRIBUTE_REPARSE_POINT` deleted; the name added to the existing import-back block, now nine names. `open_file` and its write+delete mask untouched. |
| `tests/unit/sdk_/_identity_io_fake.py` | New. In-memory NT-handle model extending `_MemoryWindowsApi`. |
| `tests/unit/sdk_/test_identity_io_contract.py` | Parameterized over `native` and `fake-windows`; new listing, stream, stat-agreement, permission-vs-refusal, struct-offset and import-order tests. |
| `tests/unit/sdk_/test_windows_metadata_journal.py` | `_MemoryWindowsApi` gained `expect_share_delete` (default `False`). |

### Measured results

Run by the orchestrator, not by me:

- Contract + journal suites: **42 contract + 5 journal, all passing.** The
  journal still reads exactly 5 and still collects.
- `ruff check` on the five files in play: **All checks passed.**
- `uv run mypy` on the four `sdk_` identity-IO sources: **Success: no issues
  found in 4 source files.**

Plan Task 5 Step 0 is therefore answered: mypy tolerates Windows-only ctypes
and CRT usage in this repo (`_windows_metadata_journal.py` already calls
`ctypes.WinDLL` with no guard and no `type: ignore`, and is clean). That is
evidence about this repo's mypy configuration, **not** about whether the code
runs.

One assertion was added after those runs, at the orchestrator's request:
`test_the_directory_entry_name_offset_matches_the_probe`. It needs a re-run
before commit.

---

## 2. Code that has never executed

The `fake-windows` lane is a *model*. It proves this backend's logic; it never
proves Win32's behaviour. The tiers below say exactly how far each piece of
code has been exercised.

### Tier A — no lane reaches it, on any platform available here

The fake overrides every one of these, so `fake-windows` does not touch them
even in simulation. **This is the list the Windows CI lane has to cover.**

- Every ctypes method added to `_CtypesWindowsApi`: `_standard_info`,
  `link_count`, `file_size`, `is_directory`, `open_regular_read`,
  `list_names`, `adopt_descriptor`, `stream`.
- `_FileStandardInfo` and `_FileFullDirInfo`. The classes are built at import,
  so their field offsets exist, but no Win32 buffer is ever parsed through
  them. `_FileFullDirInfo.FileName.offset == 68` now has a mechanical
  assertion; `_FileStandardInfo`'s layout does not, and does not need one as
  urgently, because the probe exercised that class end to end and read back
  correct values (`links=1`, `size=18`, `Directory=0`).
- The listing loop end to end: the restart-class → continue-class switch, the
  `NextEntryOffset` chain walk, and the `ERROR_NO_MORE_FILES` terminator.
  `test_listing_spans_multiple_buffer_fills` proves the *held class*
  aggregates 2,000 names; it proves nothing about the buffer refill.
- `_API = _CtypesWindowsApi()` and its `except WindowsJournalUnavailable` arm;
  `SUPPORTED is True`.
- The `sys.platform == "win32"` arms of `adopt_descriptor` and `stream`.
- `_extended_length`'s already-prefixed arm (on Linux `os.path.abspath` never
  returns a `\\?\` path, so only the prefix-adding arm runs).
- `open_identity_directory`'s `resolved_api is None` → `IdentityIoUnavailable`.
- The bottom-import ordering fix (§4).

### Tier B — logic proven by the fake, Win32 behaviour not

Everything else in `_WindowsHeldDirectory`: the walk, the reparse and
link-count refusals, the read and its stat, the listing filter, `reverify`'s
two branches, `_close`, and both arms of `_refuse_type_mismatch`.

### Tier C — branches no lane reaches, fake or native

These sit inside code the fake lane otherwise exercises, which makes them easy
to mistake for covered.

- `__init__`: the no-stable-identity refusal
  (`len(info.file_id) != 16 or not any(info.file_id)`). The fake always hands
  back a 16-byte non-zero id.
- `__init__`: `not api.is_directory(handle)` — the refusal of a **regular file
  as the root**. This is exactly the probe's S2 finding, the one case it
  identified where the two backends would otherwise disagree, and it is
  untested. It is untested for a reason; see §5.
- `_open_regular`: the `is_directory(handle)` belt-and-braces refusal,
  unreachable behind the fake's `ERROR_ACCESS_DENIED` and behind
  `FILE_NON_DIRECTORY_FILE` on Windows.
- `read_regular_with_stat`: the torn-read `changed` branch.
- `reverify`: the identity-mismatch raise on the **root** branch. The child
  branch is covered, on the native lane only.
- Every `except BaseException: close; raise` cleanup arm —
  `child_directory`, `_open_regular`, the adopt-descriptor guard in
  `read_regular_with_stat`, and `open_regular_stream`.
- `_close`'s idempotent early return.

### The largest unverified assumption is not code — it is two numbers

The backend recognises refusals 2 and 3 by Win32 error code:

- `ERROR_DIRECTORY` (**267**) for `FILE_DIRECTORY_FILE` against a regular file
  (`STATUS_NOT_A_DIRECTORY`), and
- `ERROR_ACCESS_DENIED` (**5**) for `FILE_NON_DIRECTORY_FILE` against a
  directory, on the premise that `RtlNtStatusToDosError` folds
  `STATUS_FILE_IS_A_DIRECTORY` onto the same code a denied ACL produces.

The plan names both. **The 2026-09-20 probe covered neither.** If 267 is wrong,
`test_a_file_is_refused_where_a_directory_is_required` fails on the real
runner; if the 5-folding is wrong, the directory-where-a-file-is-required
refusal fails. The fake models these codes, so the `fake-windows` lane is green
either way — it confirms the *handling*, never the codes.

The orchestrator is extending the Windows probe to resolve both, plus the S2
regular-file-root case, before the CI lane runs. If either observed value
differs, the mapping constants `_ERROR_DIRECTORY` and `_ERROR_ACCESS_DENIED`
in `_identity_io_windows.py` are the only things that change, along with the
fake's two `raise OSError(...)` lines that model them.

---

## 3. Which contract tests run in which lane

42 collected. Parameterized over both lanes: the reads, the child walk, the
stat-agreement test, both listing tests, the five non-canonical components, the
symlink refusal, both type mismatches, the multi-link refusal, the missing
entry, `max_bytes`, the missing root, and both stream tests.

The three tests the plan review flagged as impossible against an in-memory
model:

| Test | Lane | Why |
|---|---|---|
| `test_a_swapped_directory_fails_reverification` | native only | `os.rename` after the hold; the model is a snapshot taken when the hold opens |
| `test_a_streamed_member_does_not_lock_the_file` | native only | what is under test is the operating system's sharing mode, which the model does not have |
| `test_a_multi_link_file_is_refused` | **both** — deviation | see below |

The two native-only tests are **not skips**. They take an unparameterized
`native_backend` fixture, so nothing is reported as skipped and the refusal is
still covered on every platform.

The multi-link deviation: the plan restricted this test to the native lane on
the grounds that "the fake has no hard-link concept, so a link count would be
invented rather than modelled". The fake seeds `link_counts` from the real tree
via `entry.stat(follow_symlinks=False).st_nlink` during its construction scan
(`tests/unit/sdk_/_identity_io_fake.py:78-85`), and the contract test creates
the hard link *before* the hold opens. Refusal 4 is therefore modelled from the
filesystem, not fabricated, and running it on both lanes is strictly more
coverage than the plan assumed was available. Verified independently by the
orchestrator.

Two lane-specific tests were added:

- `test_a_genuine_permission_failure_is_not_reported_as_a_refusal` — fake lane
  only; there is no way to provoke a Win32 error code from the POSIX backend.
- `test_the_windows_backend_binds_supported_before_the_facade_import` and
  `test_the_directory_entry_name_offset_matches_the_probe` — lane-independent,
  both read facts that no lane can execute.

### `expect_share_delete` is not vacuous

`_MemoryWindowsApi` gained `expect_share_delete`, defaulting to `False`. All
four journal call sites construct `_MemoryWindowsApi(root)` bare, so
`assert share_delete is self.expect_share_delete` is byte-for-byte the old
`assert share_delete is False` there; only the identity-IO fake passes `True`.

That is an argument, so it was checked by mutation: flipping one journal call
site to `share_delete=True` fails **4 of the 5** journal tests with
`assert True is False`. The guard still pins the journal's stricter sharing,
which this design deliberately does not inherit.

---

## 4. A Windows-only circular import that Task 3 would otherwise have created

`_windows_metadata_journal` imports `_identity_io_windows` at its top.
`_identity_io` calls `_select_backend()` at its bottom, which reads
`SUPPORTED` off `_identity_io_windows`. On Windows, therefore,
`import phenotypic.sdk_._windows_metadata_journal` re-enters
`_identity_io_windows` through `_select_backend()` **while that module is
still executing its own body**.

With the facade import at the top of the file — the obvious place, and where
the POSIX backend has it — `SUPPORTED` would not yet exist and the whole
package would fail to import on Windows with an `AttributeError`. The POSIX
backend does not have this problem only because nothing but `_identity_io`
imports it.

The fix: `from ._identity_io import ...` sits at the **bottom** of
`_identity_io_windows.py`, after `SUPPORTED` and after
`open_identity_directory`, with `# noqa: E402` and a comment recording why the
placement is load-bearing. Both import orders then resolve, because all four
imported names are defined above `_select_backend()`'s call site in
`_identity_io`, and `SUPPORTED` is defined above the bottom import here.

Nothing on Linux can execute the failure. The only guard is
`test_the_windows_backend_binds_supported_before_the_facade_import`, which
reads the source and asserts the ordering. **If a later reviewer tidies that
import to the top of the file, that test is what stops them.**

---

## 5. Noticed, not acted on

- **The two backends do not refuse a regular-file root identically.**
  `_identity_io_posix.py:239` opens the root with a bare
  `os.open(root, _DIR_FLAGS)`, so a file root surfaces as `NotADirectoryError`
  (an `OSError`); the Windows backend raises `IdentityRefused` (a
  `ValueError`). The spec says both backends refuse identically, and the
  probe's S2 result is precisely the observation that this case needs
  handling. Both named consumers catch `OSError` and `ValueError`, so nothing
  breaks today. No contract test was added, because it would fail the native
  lane: this is a Task 1 fix, with its own test and its own commit.

- **`_refuse_type_mismatch` resolves a clause of the plan that cannot be
  implemented as written.** The plan says to translate `ERROR_ACCESS_DENIED`
  "when the target is a reparse point" — but once the open has failed, the
  target's nature is not knowable from the error. The implementation
  disambiguates by re-asking the *held parent* for the same name as a
  directory: that succeeds only when the entry really is a directory or a
  junction (which `FILE_OPEN_REPARSE_POINT` opens as itself), and a genuine
  ACL denial is left to propagate as the `OSError` it is. The probe open goes
  through the held handle and is only ever compared against, never opened
  through, so it reopens no TOCTOU window. Both outcomes are tested on the
  fake lane.

- **`file_size` is currently dead code.** The plan asks for it (Task 3 Step 5)
  and nothing in Tasks 3–5 calls it; the reader is `store_publication_token`
  in Task 7. Recorded so it is a decision rather than an oversight.

- **Three deviations from the plan's literal code**, each closing a gap:
  1. `read_regular_with_stat` includes `st_ctime_ns` in the torn-read
     comparison. The plan omitted it; the POSIX backend includes it, and
     stricter is permitted where identical is not achievable.
  2. `reverify`'s root probe uses `_extended_length(self.path)` rather than
     `str(self.path)`. Otherwise a long root opens fine and then fails its own
     reverification.
  3. The `.`/`..` filter lives on `_WindowsHeldDirectory.list_names` rather
     than inside the ctypes `list_names`, which returns raw names. One
     implementation of the filter, exercised on both lanes, instead of a
     native-only one that the fake lane would silently not test — the same
     false-green shape the plan guards against elsewhere.

- **`_identity_io_fake.py` imports a test module.** It subclasses
  `_MemoryWindowsApi` from `tests.unit.sdk_.test_windows_metadata_journal`,
  which is what the plan directs, and matches the repo's existing
  `from tests.unit.sdk_._migration_fixtures import ...` convention. The
  consequence is a second module object for that test file under a second
  name; nothing crosses between them. If it ever bites, the fix is to move
  `_MemoryWindowsApi` into a non-test helper — not worth doing speculatively.

- **The fake's `open_file` raises.** `tests/unit/sdk_/_identity_io_fake.py`
  overrides the journal's `open_file` with an `AssertionError`, so "reads go
  through `open_regular_read`, never the journal's write+delete mask" is a
  structural property of the fake lane rather than a claim in a docstring. The
  probe confirmed that mask is refused with `ERROR_ACCESS_DENIED` on a
  read-only-attribute file and on one whose ACL denies `WD,AD`.

---

## 6. What a reviewer should check first

1. Re-run the two suites; the struct-offset assertion was added after the last
   measured run.
2. The two error codes in §2. Nothing on Linux can tell you they are right.
3. The bottom import in `_identity_io_windows.py` (§4) — that it is still at
   the bottom, and that its guard test is still present.
4. The regular-file-root asymmetry in §5, which belongs to Task 1 and is still
   open.
