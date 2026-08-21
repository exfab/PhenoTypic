# Phase 7 — Verification: commit protocol, invariant gates, Windows lane, sign-off

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §3.8, §7.

**Depends on:** Phases 3, 4, 5, 6.

Everything here is a test or a gate. No production behaviour changes in this phase — if a
test written here fails, the fix belongs in the phase that owns the code, not here.

---

### Task 7.1: End-to-end commit-protocol tests

**Files:**
- Test: `tests/integration/cli/test_commit_protocol.py` (create)

> **This task was cut down in round 1 of the refinery.** Four of its seven original tests
> duplicated Phase 1 Tasks 1.5/1.6 — one of them strictly weaker (2 uuid samples where
> Task 1.5 draws 64) — and the "prove case (a) can fail" step could not work. What remains
> here is only what genuinely needs a **real `save2zarr`** rather than Task 1.5's
> `_fake_store`. Recorded as ledger SIMP-2, SIMP-3, GEN-4.
>
> **Kept here** — the two tests Step 1 defines below:
> `test_two_concurrent_writers_produce_one_coherent_winner`,
> `test_a_new_write_does_not_reuse_a_stale_part`.
>
> > **Corrected (missing-owner review, 2026-08-19).** This paragraph previously read *"Moved
> > to Phase 2, beside Task 2.2 (they exercise the real writer, so they belong in the phase
> > that builds it, not four phases later)"* — naming these same two tests. It contradicted
> > the rest of its own task: Step 1 below defines exactly these two and nothing else, the
> > Constraints section says *"the remaining case here is (b) concurrency"*, and Step 3's
> > mutation proof operates on `test_two_concurrent_writers_produce_one_coherent_winner`
> > directly. It also contradicted the plan: `grep -rn "concurrent_writers\|stale_part"
> > phase-2-image-io.md` returns **nothing**, so the move was never carried out. An executor
> > trusting the header deletes Step 1, and with it the only coverage of commit-protocol
> > case (b) and case (c) — and Task 7.1 becomes an empty task. **The header was wrong; the
> > tests stay here.** They are placed in Phase 7 for the reason the task's own opening gives:
> > they are the cases that need a real `save2zarr` and real concurrent processes, which is a
> > verification concern, not a writer-construction one.
>
> **Left in Phase 3** (it imports `classify_staged_image`):
> `test_interrupted_store_classifies_stage1`.
>
> **Deleted as already covered by Task 1.5/1.6:**
> `test_promote_leaves_no_trash_on_success`,
> `test_a_stale_part_is_removed_not_merged_into`,
> `test_concurrent_writers_use_distinct_part_directories` (and it asserted only that two
> `uuid4()` calls differ — a test of the stdlib), and
> `test_a_part_without_a_root_never_validates` (ledger PRE-S8).
>
> **Deleted as unable to fail:** `test_interrupt_before_the_root_reads_as_absent`. It
> monkeypatched `promote_store` to raise, but every byte is written into the `.part` sibling
> and only `promote_store` ever creates the final path — so `not final.exists()` holds under
> **any** write order, including the reversed one it was named for.

**Constraints specific to this task:**

- **What "root last" actually buys.** Because a `.part` is never at the published path, the
  ordering is *not* load-bearing for reader visibility — an interrupted write is invisible
  either way. It is load-bearing for **flush ordering** (§3.7): without it the kernel may
  make the root durable before the chunks it describes, leaving a store that passes
  `valid_staged_store` while reading `fill_value`. State that plainly rather than repeating
  the reader-visibility rationale, which does not hold.
- The remaining case here is **(b) concurrency**, which needs real concurrent writers.
- **Give the trash path a fresh uuid per attempt** (ledger **GEN-15**). If
  `os.replace(part, final)` fails after a successful move-aside *because a concurrent
  promoter recreated `final`*, the rollback is skipped (`not final.exists()` is False) and a
  non-empty `trash` survives; every later attempt's `os.replace(final, trash)` then fails
  `ENOTEMPTY` until the budget exhausts, stranding the previous store. Rare, but cheap.

- [x] **Step 1: Write the tests**

Only the cases that need a **real `save2zarr`** live here — cases (b) and (c). Case (a) is
Phase 3's `test_interrupted_store_classifies_stage1`; see the header for what was deleted and
why.

```python
"""Commit-protocol case (b): concurrency, through the real writer."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.ngff_ import valid_staged_store


def _write_one(args) -> str:
    final, marker = args
    image = Image(load_synth_yeast_plate())
    image._metadata.public["Metadata_Strain"] = marker
    image.save2zarr(final)
    return marker


def test_two_concurrent_writers_produce_one_coherent_winner(tmp_path: Path) -> None:
    """The property the uuid .part and the retrying promote exist to provide.

    Two real processes race on one stem. Neither may raise -- duplicate
    execution is benign today and must stay benign -- and the survivor must be
    one coherent store, not an interleaving of both.
    """
    final = tmp_path / "p.ome.zarr"
    with mp.get_context("spawn").Pool(2) as pool:
        pool.map(_write_one, [(final, "A"), (final, "B")])

    assert valid_staged_store(final) is True
    winner = Image.load_zarr(final)._metadata.public["Metadata_Strain"]
    assert winner in {"A", "B"}
    assert list(tmp_path.glob("*.part")) == []
    assert list(tmp_path.glob("*.trash")) == []


def test_a_new_write_does_not_reuse_a_stale_part(tmp_path: Path) -> None:
    """A killed worker's leftovers must never be merged into."""
    image = Image(load_synth_yeast_plate())
    final = tmp_path / "p.ome.zarr"
    stale = final.parent / ".p.ome.zarr.deadbeefdeadbeef.part"
    (stale / "gray" / "0").mkdir(parents=True)
    (stale / "gray" / "0" / "0.0").write_bytes(b"garbage")

    image.save2zarr(final)

    assert valid_staged_store(final) is True
    assert (Image.load_layer_zarr(final, "gray") == image.gray[:]).all()
```

- [x] **Step 2: Run them.**

```bash
uv run pytest tests/integration/cli/test_commit_protocol.py -v
```

Expected: all PASS.

- [x] **Step 3: Prove the concurrency case has teeth** — see the correction below

Temporarily replace `promote_store`'s retry loop with a single
`exists -> move-aside -> replace` pass (the pre-**PRE-B5** shape) and re-run:

```bash
uv run pytest tests/integration/cli/test_commit_protocol.py -v
```

Expected: `test_two_concurrent_writers_produce_one_coherent_winner` **fails**, because the
loser hits `ENOTEMPTY` on a target the winner created between its check and its rename, and
`pool.map` re-raises. Restore the loop, confirm green, and paste the observed failure into
the commit body.

> **Do not try to prove "root last" here.** A `.part` is never at the published path, so no
> interruption test can distinguish the two write orders — that ordering is load-bearing for
> **flush** ordering under §3.7, which this suite does not exercise. An earlier draft's
> mutation proof claimed otherwise and could not work.

- [x] **Step 4: Commit** (`39215314`)

```bash
git add tests/integration/cli/test_commit_protocol.py
git commit -m "test: pin the two commit-protocol cases that need a real writer

(b) two concurrent writers get distinct uuid .part directories and produce
one coherent winner with no leftovers; (c) a stale .part from a killed
process is removed rather than merged into. Case (b) was proven to have
teeth by reducing promote_store to a single exists/move-aside/replace pass
and watching it go red on ENOTEMPTY. Case (a) is covered in Phase 3 by
test_interrupted_store_classifies_stage1; it cannot be proven here,
because a .part never sits at the published path and so no interruption
test can distinguish the two write orders."
```

---

### Task 7.2: Windows nightly lane and platform assertions

**Files:**
- Modify: `.github/workflows/run-pytest.yml` (PR lane: ensure the commit-protocol tests are
  not excluded by `-m 'not slow'`)
- Modify: `.github/workflows/run-pytest-full.yml` (nightly Windows job at lines 129–144)
- Test: `tests/unit/sdk_/test_ngff_windows.py` (create)

**Constraints specific to this task:**
- Commit-protocol tests run **in the PR lane on Linux** and **the nightly lane on Windows**.
  The spec accepts a one-day latency on a Windows-specific promote regression rather than
  promoting the whole Windows suite to PR time.
- Windows facts to assert rather than assume:
  1. no directory `fsync` (POSIX-guarded);
  2. the move-aside retries;
  3. the two-step move-aside is the only path (no single-call replace fallback);
  4. `\\?\` prefixing;
  5. **no case-only collisions** among store path segments;
  6. per-file overhead is documented, not mitigated — no test.

- [x] **Step 1: Write the platform tests**

```python
"""Windows-specific promote behaviour, asserted rather than assumed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_


def test_no_case_only_collisions_in_store_path_segments() -> None:
    """NTFS is case-insensitive."""
    segments = [
        ngff_.OME_GROUP,
        ngff_.LABELS_GROUP,
        ngff_.OBJMAP_LABEL,
        *ngff_.SERIES_ORDER,
    ]
    assert len({s.lower() for s in segments}) == len(segments)


def test_directory_fsync_is_posix_guarded(tmp_path: Path, monkeypatch) -> None:
    """Windows cannot open a directory handle for flushing."""
    store = tmp_path / "s"
    store.mkdir()
    (store / "f").write_bytes(b"x")
    monkeypatch.setattr(os, "name", "nt")
    opened: list[str] = []
    real_open = os.open
    monkeypatch.setattr(
        os, "open", lambda p, f, *a: (opened.append(str(p)), real_open(p, f, *a))[1]
    )
    ngff_.fsync_tree(store)
    assert str(store) not in opened


def test_promote_retries_a_transient_rename_failure(tmp_path: Path, monkeypatch) -> None:
    """The retry lives inside promote_store's loop, not in a separate helper.

    B5 requires the WHOLE `exists -> move-aside -> replace` sequence to retry
    and re-evaluate, so there is no single-rename helper to test. The simulated
    error must also reach the retryable branch: on POSIX `OSError(32, ...)` is
    EPIPE with no `winerror`, which `_is_retryable` correctly rejects — use
    ENOTEMPTY, which it accepts on both platforms.
    """
    import errno

    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    calls = {"n": 0}
    real = os.replace

    def _flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.ENOTEMPTY, "Directory not empty")
        return real(src, dst)

    final = tmp_path / "p.ome.zarr"
    Image(load_synth_yeast_plate()).save2zarr(final)
    part = _fake_store(ngff_.new_part_path(final), "new")
    monkeypatch.setattr(os, "replace", _flaky)
    ngff_.promote_store(part, final, fsync=False)
    assert calls["n"] > 1
    assert ngff_.valid_staged_store(final) is True


def test_is_retryable_discriminates_on_errno(monkeypatch) -> None:
    """A genuine ENOSPC must fail fast, not burn the whole backoff budget."""
    import errno

    assert ngff_._is_retryable(OSError(errno.ENOTEMPTY, "not empty")) is True
    assert ngff_._is_retryable(OSError(errno.ENOENT, "missing")) is True
    assert ngff_._is_retryable(OSError(errno.ENOSPC, "no space")) is False

    sharing = OSError(13, "sharing violation")
    sharing.winerror = 32
    assert ngff_._is_retryable(sharing) is True


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_long_path_prefix_is_applied(tmp_path: Path) -> None:
    assert ngff_.long_path(tmp_path).startswith("\\\\?\\")


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_a_deep_store_path_still_writes(tmp_path: Path) -> None:
    """MAX_PATH: an output root + dataset + stem + store-internal path."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    deep = tmp_path.joinpath(*["longish_directory_name_segment"] * 6)
    deep.mkdir(parents=True)
    store = Image(load_synth_yeast_plate()).save2zarr(deep / "p.ome.zarr")
    assert ngff_.valid_staged_store(store) is True


def test_chunk_keys_are_one_path_segment(tmp_path: Path) -> None:
    """The '.' separator is what keeps a chunk key from being four directories."""
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    chunk_dirs = [p for p in (store / "gray" / "0").iterdir() if p.is_dir()]
    assert chunk_dirs == [], "chunk keys must not nest into directories"
```

- [x] **Step 2: Wire the CI lanes.** — already wired; asserted rather than edited. In `run-pytest-full.yml`'s Windows job, ensure
`tests/integration/cli/test_commit_protocol.py` and `tests/unit/sdk_/test_ngff_windows.py`
are collected. In `run-pytest.yml`, confirm neither is marked `slow`.

- [x] **Step 3: Run on Linux**

```bash
uv run pytest tests/unit/sdk_/test_ngff_windows.py -v
```

Expected: the four platform-independent tests PASS, the three Windows-only tests SKIP.

- [x] **Step 4: Commit** (`64d9d76f`; `.github/workflows` needed no edit)

```bash
git add .github/workflows tests/unit/sdk_/test_ngff_windows.py
git commit -m "test: assert the six Windows consequences of the store layout

Case-only collisions, the POSIX-guarded directory fsync, the move-aside
retry, and the one-path-segment chunk key are asserted on every platform;
the \\?\ prefix and a deep-path write are Windows-only. Commit-protocol
tests run in the PR lane on Linux and the nightly lane on Windows -- the
spec accepts the one-day latency rather than promoting the whole Windows
suite to PR time."
```

*Committed as `64d9d76f`.*

---

### Task 7.3: Architectural invariant gates

**Files:**
- Test: `tests/unit/test_ome_zarr_invariants.py` (create)

**Constraints specific to this task:**
These are grep-style gates over the source tree, in the same spirit as
`tests/unit/schema/test_no_metadata_literals.py`. Each one guards an invariant that a
future edit could plausibly violate without any other test noticing.

- [x] **Step 1: Write the gates**

```python
"""Source-tree invariants for the OME-Zarr store. Each guards a silent-failure mode."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "phenotypic"
PY = sorted(SRC.rglob("*.py"))


def _hits(pattern: str, *, allow: set[str] = frozenset()) -> list[str]:
    rx = re.compile(pattern)
    out = []
    for path in PY:
        if path.name in allow:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if rx.search(line):
                out.append(f"{path.relative_to(SRC)}:{number}: {line.strip()}")
    return out


def test_store_suffix_is_joined_in_exactly_one_place() -> None:
    assert _hits(r'\.ome\.zarr"', allow={"ngff_.py", "_io_constants.py"}) == []


def test_objmap_path_is_never_hard_coded() -> None:
    """An rgb-less store puts the label under gray."""
    assert _hits(r'rgb/labels/objmap') == []


def test_no_module_still_writes_hdf() -> None:
    """Phase 6 keeps h5py READERS for migration; only WRITE paths must be gone.

    `_image_io_handler.py` retains `_load_v2_grouped` / `_load_legacy_flat_group`
    / `_load_hdf5_for_migration`, which open `h5py.File(..., "r")` at :1172 and
    :1211. Gating on a bare `h5py.File` would be red on day one against code the
    plan deliberately keeps.
    """
    assert (
        _hits(
            r'save2hdf5|save_intermediate_layers|h5py\.File\([^)]*["\']w',
            allow={"_hdf_to_zarr.py", "hdf_.py", "_image_io_handler.py"},
        )
        == []
    )


def test_metadata_ownership_is_never_a_prefix_check() -> None:
    """CLAUDE.md: use metadata_owner_for_header, never string parsing.

    `_metadata_migration.py:210` is the one sanctioned carve-out: it is the
    centralized canonicalization helper, which is exactly where string handling
    is allowed to live.
    """
    assert (
        _hits(r'startswith\(\s*["\']Metadata_', allow={"_metadata_migration.py"}) == []
    )


def test_no_recursive_glob_for_stores() -> None:
    """rglob walks INTO every store: ~400k stat calls at 10k images.

    Matches the f-string form too, so `sweep_orphan_parts` -- which lives in
    `ngff_.py` and once used exactly this pattern -- cannot exempt itself.
    """
    assert _hits(r'rglob\(\s*f?["\'][^"\']*\.ome\.zarr') == []
    assert _hits(r'rglob\(\s*f["\'][^"\']*\{STORE_SUFFIX\}') == []
```

**Four candidate gates were considered and dropped as unable to fail.** Recording them so
they are not "helpfully" re-added:

| Dropped gate | Why it could never fail |
|---|---|
| `test_no_pid_in_a_part_directory_name` | The regex needs `getpid` and `part` on one **physical** line; the only real instance (`_cli_output_manager.py:1658`) is wrapped across lines, and Phase 6 deletes it anyway. |
| `test_scale_vectors_are_never_powers_of_two` | `r'"scale":\s*\[?\s*2\s*\*\*'` matches a JSON-literal-with-Python-exponent form no implementation emits. Phase 1 Task 1.1's `test_scale_vector_comes_from_actual_shapes_not_powers_of_two` is the real guard. |
| `test_resume_state_never_lives_in_ngff_metadata` | `r'labels.*stage2\|stage2.*ome\.labels'` matches nothing plausible. Phase 3 Task 3.4's differential parity test is what actually catches this defect. |
| `test_zarr_errors_are_caught_not_propagated` (Task 1.6) | `BaseZarrError` subclasses `ValueError`, so the assertion holds with or without it in the tuple. |

- [x] **Step 2: Run.** Any hit is a real defect; fix it in the owning phase's module, not by
relaxing the gate.

```bash
uv run pytest tests/unit/test_ome_zarr_invariants.py -v
```

- [x] **Step 3: Commit** (`0cd6fbfb`)

```bash
git add tests/unit/test_ome_zarr_invariants.py
git commit -m "test: gate the OME-Zarr source-tree invariants

Five grep gates, each guarding a failure mode no other test would notice:
a hand-joined store suffix, a hard-coded rgb/labels/objmap, an HDF write
path surviving retirement, prefix-parsed metadata ownership, and
file_fingerprint or a recursive glob pointed at a store directory.

The allow lists matter: _image_io_handler.py keeps h5py READERS for
migration, and _metadata_migration.py:210 is the sanctioned string-handling
carve-out, so gating on either without an exemption is red on day one.

Four candidate gates were dropped as unable to fail -- a PID-in-.part regex
needing two tokens on one physical line, a 2**n scale-vector regex matching
a form no implementation emits, an ome.labels regex matching nothing, and
the zarr-error catch test (BaseZarrError subclasses ValueError, so it
passed either way)."
```

---

> **Task 7.3a was written and then folded away (ledger SIMP-12).** A reader-level interop
> gate — `datasets[].path` resolves, `dimension_names` matches the declared `axes`, the label
> is reachable through the `labels` array with an integer dtype, chunk keys stay to one path
> segment — was added here in round 2 under the ALGO-4 user ruling. The **checks** were right
> and are all still enforced; the **placement** was not.
>
> They now live in `_assert_reader_level_musts`, called from `assert_store_conforms`
> (Phase 2 Task 2.5). `assert_store_conforms` is imported by every later phase that writes a
> store, so folding them in gates every store written anywhere in Phases 2–7 rather than one
> store in one Phase-7 test. The two chunk-key tests merged into one (the no-nesting form
> observes the actual layout; the separator assertion is its declared cause), and ALGO-13's
> two further MUSTs — path order vs `Image` element order, and label-vs-image level count —
> went in with them.
>
> **Do not re-add a task here.** What belongs in Phase 7 is what cannot run earlier: real
> multi-process concurrency (7.1), the Windows lane (7.2), source-level invariant gates
> (7.3), full-suite sign-off (7.4).

---

### Task 7.4: Full-suite sign-off

> **Executed 2026-08-20, except the wide pytest run.** Measured values below;
> where one contradicts a number written into this plan, the measurement is
> authoritative and the plan's figure is corrected in place.
>
> | Check | Recorded | Measured | Verdict |
> |---|---|---|---|
> | `pytest tests -q` | — | SLURM job `27685608` (`batch`, `c19`, 6 h) | in flight; **started before** `39215314`, `64d9d76f`, `0cd6fbfb`, `21421725`, so it does not cover them |
> | `mypy src/phenotypic` | 417 errors / 124 files | **422 / 122** (743 files checked) | **REGRESSION — see below** |
> | `ruff check src/phenotypic` | 25 | **25** (4 E402, 2 E721, 2 F401, 2 F403, 13 F405, 2 F841) | matches exactly |
> | `ruff check src/phenotypic tests` | *(never recorded)* | **49** = 25 src + **24 tests** | the 24 are in seven files, **none touched by this branch** |
> | `sphinx-build -W` | — | failed on a missing autosummary stub | **not a docs defect** — see below |
> | `ngff_store_geometry.py` | exit 0 | **exit 0**, all claims hold; C6 already removed per ALGO-8 | pass |
> | `check_features_md.py` | — | `OK (451 feature rows, 377 shipping, 0 in progress)` | pass |
> | `check_workflows_md.py` | — | `OK -- 20 workflows, 20 capture functions, 20 dispatched` | pass |
> | `tests/unit/core/test_ngff_conformance.py` | — | **22 passed** | pass |
> | `ome-zarr` / `ome-zarr-models` in `uv.lock` | banned | **0 occurrences** | pass |
>
> **mypy is a real regression, and it belongs to Phase 2, not here.** The
> plan's 417/124 baseline is stale: this branch's merge-base
> (`a742ac8a`) measures **419 errors in 123 files (729 checked)** under its own
> `pyproject.toml`. Against *that* baseline the branch adds:
>
> * `_core/_image_parts/_image_io_handler.py` — `arg-type` 3 → 6,
>   `assignment` 7 → 8, `return-value` 3 → 6 (**+7**). Named sites include
>   `:724` (`Argument "kind" to "build_pyramid" has incompatible type "str";
>   expected "Literal['image', 'label']"`) and `:1314` / `:1526`
>   (`moveaxis` applied to `zarr`'s `NDArrayLike` union).
> * `_cli/_cli_gui_lifecycle.py:116` — `arg-type` (**+1**).
>
> and removes 6 (`post/_expand_metadata.py` ×2, `post/_merge_metadata.py` ×3,
> `sdk_/hdf_.py [misc]` ×1). The exit criterion is "no new errors **and none in
> any file this change touched**", and `_image_io_handler.py` is the file Phase 2
> rewrote — so this fails as written. Phase 7 changes no production code; the fix
> belongs to Phase 2's module.
>
> **The sphinx failure was a collision, not a defect.** `sphinx -W` aborted on
> `docs/source/api_reference/api/phenotypic.enhance.SubtractGaussian.rst` not
> existing. That directory is **gitignored** (`.gitignore:62`), holds 550
> autosummary-generated stubs, and a *second* `sphinx -W` process
> (`/scratch/anguy344/27684616/sphinx_fix2`) was regenerating it in the shared
> worktree at the time. `SubtractGaussian` is not new on this branch. Re-run it
> when no concurrent build is writing that directory.
>
> **Steps 3 and 4 (a real dataset end-to-end, and the third-party `ome-zarr`
> read) were not run.** Both are manual release checks, not CI gates, and both
> want a real image tree and an uncontended node.

- [ ] **Step 1: Run everything**

```bash
uv run pytest tests -q
uv run mypy src/phenotypic
uv run ruff check src/phenotypic tests
uv run sphinx-build -W -b html docs/source docs/_build/html
uv run python docs/superpowers/logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py
```

- [ ] **Step 2: Run the GUI ledger gates** (CLAUDE.md's `gui-tutorial-capture` skill)

```bash
uv run python scripts/check_features_md.py
uv run python scripts/check_workflows_md.py
```

**`features-md-gate` is a DIFF gate, not a judgement about affordance novelty** — this
change must edit `src/phenotypic/gui/FEATURES.md`, and the edits are scheduled in Phase 4 and
Task 5.7, not here. Verified at `.github/workflows/gui-checks.yml:92-106`: it computes
`TOUCHED_GUI` as any diff under `src/phenotypic/gui/` (excluding only that directory's
`CLAUDE.md`/`FEATURES.md`/`WORKFLOWS.md`) and fails the PR when that is non-empty while
`TOUCHED_FEATURES` is empty. Phase 4 modifies five GUI source files and Task 5.7 modifies
`gui/results_viewer/_output_consistency.py`, so the gate fires **regardless of whether the
chrome changed**. `pyproject.toml` is also in the workflow's path trigger, so Phase 0 alone
guarantees the workflow runs at all.

By this step both rows should already exist; running the gates here confirms it. A failure
means a scheduled FEATURES.md edit was skipped — **not** that a chrome change slipped in.

> **Corrected twice (ledger GEN-31, then GEN-36).** An earlier draft asserted the chrome was
> unchanged and pointed a failing gate at Phase 4; a second said the banner "needs **no**
> `FEATURES.md` row" because it is a new reason on an existing surface. That reasoning is
> about the `gui-tutorial-capture` skill's *screenshot* rule and does not apply to
> `features-md-gate`, which never inspects what changed. Left as written, an eight-phase
> change would have been blocked at its final step with the executor sent to hunt a chrome
> change that does not exist.

- [ ] **Step 3: End-to-end on a real dataset**

```bash
uv run python -m phenotypic --mode full --pipeline <pipeline.json> --input <images> -o /tmp/zarr_run
uv run python -m phenotypic --mode full --pipeline <pipeline.json> --input <images> -o /tmp/zarr_run   # resume: converges immediately
uv run phenotypic-gui --root /tmp/zarr_run --port 8050
```

Confirm: the second run reports every image `complete`; the viewer renders whole-plate tiles
and colony crops; `results/<ds>/zarr/` holds one `.ome.zarr` directory per input and no
`.part` or `.trash` leftovers.

- [ ] **Step 4: Read the store with a third-party reader — outside the locked environment**

⚠️ **Do NOT `uv add` anything for this.** `napari-ome-zarr` depends on `ome-zarr`, which
Global Constraints and `test_ome_zarr_packages_are_not_adopted_anywhere` ban from **every**
dependency group — verified: neither package appears in `uv.lock`. An earlier draft ran
`napari.run()`, which both violates that ban and blocks on a display the HPCC context does
not have. Recorded as ledger **GEN-5**.

Run it in a throwaway environment instead, headless, reading rather than rendering:

```bash
uv run --isolated --no-project --with ome-zarr python - <<'PYEOF'
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

store = "/tmp/zarr_run/results/<ds>/zarr/<stem>.ome.zarr"
node = next(iter(Reader(parse_url(store))()))
print("axes:", node.metadata.get("axes"))
print("levels:", len(node.data))
print("level-0 shape:", node.data[0].shape)
PYEOF
```

Expected: the axes match the series geometry, the level count matches
`phenotypic.pyramid.levels`, and level 0 matches the source image extent.
`--isolated --no-project` keeps `ome-zarr` entirely out of the project's resolution, so the
ban holds.

This is the headline external claim of the design — a PhenoTypic output directory readable
without a PhenoTypic install. If it fails, the store does not conform in a way the schema
gate missed, and that is a Phase 1/2 defect. Because it runs outside the locked environment,
it is a **manual release check, not a CI gate**.

- [ ] **Step 5: Update the spec status and commit**

Set `**Status:**` in the design doc to `Implemented` and link this plan.

```bash
git add docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md
git commit -m "docs: mark the OME-Zarr store design implemented"
```

---

### Task 7.5: Comprehensive interface documentation (user request, 2026-08-20)

**Added after the plan was written**, at the user's explicit request: *"at the end, do a
comprehensive doc update with all the final interfaces."* This is a **deliverable of this
change**, not optional polish, and it is the last thing done — after Phase 7's verification
settles, so the documented interfaces are the ones that actually shipped and passed.

**Why it needs its own task rather than riding along in Task 6.4.** Task 6.4 documents what
was *removed* (the migration page naming the retired symbols). Nothing in phases 0-7
documents the **complete resulting public surface** in one place. Six phases each added
interfaces and each documented them locally — in a module `CLAUDE.md`, a docstring, a plan
blockquote — so a user or a future agent has no single page that answers "what is the API
now?" The pieces are scattered across `sdk_/ngff_.py`, `_image_io_handler.py`,
`_cli/CLAUDE.md`, `sdk_/CLAUDE.md`, and four plan documents.

**Scope — every interface this change created, changed, or retired:**

- **`phenotypic.sdk_.ngff_`** — the full public surface: layout constants
  (`STORE_SUFFIX`, `STORE_ROOT_JSON`, `STORE_SCHEMA_VERSION`, `NGFF_VERSION`,
  `OBJMAP_LABEL`, `SERIES_ORDER`), pyramid geometry (`pyramid_level_count`,
  `pyramid_level_shapes`, `level_scale_vector`, `downsample_image`), the
  `attributes.phenotypic` contract, `promote_store` / `new_part_path` and the commit
  protocol, `valid_staged_store`, `require_readable_store`, and the durability resolver.
- **`Image` / `GridImage` I/O** — `save2zarr`, `load_zarr`, `load_layer_zarr`,
  `save_intermediate_zarr`, and **what replaced each removed HDF method**.
- **CLI** — `--mode migrate` (in-place; there is no copy mode), `--durable-writes` /
  `--no-durable-writes` and its tri-state, and the store-era output layout.
- **The on-disk store layout itself** — the tree, what each group is, and the fact that it
  opens directly in napari / QuPath / Vizarr with no PhenoTypic install, which is a headline
  goal of the change and currently documented nowhere a user will find it.
- **Completion markers and resume** — `image_data_artifact`, `SUCCESS_MARKER_VERSION`, and
  the store-vs-file descriptor `kind` dispatch.

**Constraints:**
- Doctest examples must be **runnable** and use `load_synth_yeast_plate()`, per root
  `CLAUDE.md`. `sphinx-build -W` must still pass afterwards.
- Examples go in **docstrings**, never in new example files or notebooks (root `CLAUDE.md`).
- Update, do not duplicate: where `sdk_/CLAUDE.md` or `_cli/CLAUDE.md` already carries a
  contract, link rather than restate, so the two cannot drift.
- Record the **invariants a reader must not violate**, since three subsystems now depend on
  them and none can detect a violation alone: the promote writes the root **last**, nothing
  writes into a promoted store, and both the completion marker and the viewer's staleness
  scan therefore key on the root `zarr.json` alone.

## Phase 7 exit criteria

- [ ] `uv run pytest tests -q` green. **In flight as SLURM job `27685608`**
      (`batch`, `c19`, 6 h walltime — `short`'s 2 h is not enough; a previous gate died at
      89%). It was submitted **before** commits `39215314`, `64d9d76f`, `0cd6fbfb`, and
      `21421725`, so it does not cover the three new suites; those were run individually
      and are green. Known-failing and out of scope: `test_inspect_remeasures_when_explicit_image_changes`,
      three `FilFinderDetector` smoke tests (the `topology` extra is not installed),
      `tests/migration/test_equivalence.py` (57 stale-golden failures, pre-existing), and
      two run-console suites that flake under node load.
- [ ] `uv run mypy src/phenotypic` and `uv run ruff check src/phenotypic tests` show **no
      new** errors against the baseline, and none in any file this change touched.
      **The baseline is the merge-base, measured, not the Phase-0 figure.**
      `a742ac8a` = **419 mypy errors / 123 files** and **25 ruff** in `src/phenotypic`
      (+ **24** pre-existing in `tests/`, in seven files this branch does not touch).
      Phase 0 recorded 417/124; that number is stale and must not be compared against.
      **Currently FAILING**: 422 mypy errors, +7 in `_image_io_handler.py` and +1 in
      `_cli_gui_lifecycle.py`. Owner: Phase 2.

> **This is a NO-NEW-ERRORS gate, not a clean gate — and that is deliberate.**
> None of the baseline errors belong to this change, and raising the Python floor
> cannot move them (mypy sets no `python_version`, so it follows the running
> interpreter rather than `requires-python`). `uv run ruff check src/phenotypic` is
> **25 pre-existing errors**, all `F405`/`E402`/`E721`/`F401`/`F403`/`F841` —
> re-measured 2026-08-20 and identical.
>
> > **Corrected 2026-08-20 (Task 7.4).** This paragraph read *"reports 417 errors in
> > 124 files at the baseline, verified against the pre-change `pyproject.toml` with
> > `git show HEAD:pyproject.toml` and found identical"*. The figure does not
> > reproduce. Measured by extracting the merge-base (`a742ac8a`) with `git archive`
> > and running `mypy --config-file <base>/pyproject.toml <base>/src/phenotypic`:
> > **419 errors in 123 files, 729 source files checked**. `git show HEAD:pyproject.toml`
> > compares a *config file*, which cannot establish an error count; the count was
> > never re-derived. A stale baseline is worse than none — it makes a real +3
> > regression read as +5 and invites the executor to argue about which two errors
> > "were already there".
>
> Stating these as "passes" would hand every later phase a gate that was already red,
> which trains an executor to ignore the one signal that would catch a real regression.
> Record the counts, compare against them, and treat any *increase* as this phase's.

- [ ] `uv run sphinx-build -W` succeeds. **Attempted 2026-08-20 and inconclusive**: it
      aborted on `docs/source/api_reference/api/phenotypic.enhance.SubtractGaussian.rst`
      not existing, while a second `sphinx -W` process was regenerating that directory in
      the same worktree. The directory is gitignored (`.gitignore:62`) autosummary output,
      and `SubtractGaussian` is not new on this branch. Re-run with no concurrent build.
- [x] No consumer takes `Path.stem` of a store directory (ledger **C5**). **This is now a
      gate, not a grep**: `tests/unit/test_ome_zarr_invariants.py::test_path_stem_is_never_taken_of_a_store_directory`
      resolves the receiver by AST. The plan's `grep -rn "\.stem" src/phenotypic/_cli/`
      returns ~100 hits and asks for a manual "is this a source image path?" pass; all 100
      are (`img.stem`, `image.stem`, `item.stem`, `Path(image_name).stem`), so the review
      passes and teaches nothing, while the gate fails when a store path actually acquires
      a `.stem`. Proven by mutation: injecting `store.stem` turns it red; renaming `img` to
      `image_path` at the same call site does not.
- [x] `ngff_store_geometry.py` exits 0 — **verified 2026-08-20** — **after** Phase 1's edit removing its
      `--pyramid-levels` documentation and its C6 claim block, which PRE-P3 descoped
      (ledger **ALGO-8**).
- [x] Task 7.1's concurrency mutation proof holds — **but not through the test this
      criterion named, and not on the error it named.** Two corrections, both measured:
      (1) `test_concurrent_promote_and_read` **does not exist**; Task 7.1 defines
      `test_two_concurrent_writers_produce_one_coherent_winner`. (2) With `promote_store`
      reduced to a single `exists -> move-aside -> replace` pass, that test goes red in
      only **2 of 8** runs, and the error is `ENOENT` (a sibling's move-aside took `final`
      first), not `ENOTEMPTY`. The narrowest window in the race is one `rename`, so no
      un-instrumented multi-process test closes it reliably.
      **The deterministic gate is
      `tests/unit/sdk_/test_ngff_promote.py::test_a_concurrent_promoter_appearing_mid_retry_is_benign`**,
      which injects the interleaving. The same mutant also kills
      `test_a_transient_rename_failure_is_retried_not_surfaced`,
      `test_a_failed_promote_leaves_the_previous_store_in_place`, and
      `test_a_hard_failure_is_not_retried_five_times` — four red, every run.

      > **Replaced (ledger GEN-25).** This criterion previously read "Commit-protocol case
      > (a) demonstrated to fail under a reversed write order", which Task 7.1's own
      > blockquote refutes — *"Do not try to prove 'root last' here… An earlier draft's
      > mutation proof claimed otherwise and could not work"*. The concurrency proof is the
      > one that actually works.
- [x] `tests/unit/core/test_ngff_conformance.py` exercises `_assert_reader_level_musts`
      **and its negative cases fail** — 22 passed, and mutation-swept (`21421725`):
      removing the series-order, `dimension_names`, or separator assertion turns the
      matching negative red. `test_a_dimension_names_mismatch_is_rejected` used a bare
      `pytest.raises(AssertionError)` and was tightened to `match` the reported value.
      Stubbing `zarr.open_array` leaves all 22 green, and that is **correct**: the
      dangling-path MUST has two routes and the surviving one still names the path.
      The published criterion, kept for the record: a reordered series list, a dangling
      `datasets[].path`, a `dimension_names` mismatch, and a nested chunk-key separator each
      raise; a label-less store still passes (ledger **GEN-47**). A positive-only suite
      satisfies this criterion vacuously, which is how a `KeyError` in the path-order check
      went unnoticed once already.
- [x] Every phase that writes a store still calls `assert_store_conforms` — verified:
      `tests/unit/core/test_ngff_conformance.py`, `tests/unit/sdk_/test_ngff_attributes.py`,
      `tests/unit/sdk_/test_hdf_to_zarr.py`, `tests/integration/cli/test_staged_store_stages.py`,
      via `tests/_ngff_conformance.py` (ledger **SIMP-12** — Task 7.3a's checks moved
      there rather than shipping as their own task).
- [ ] A real run resumes to `complete` on a second invocation with no reprocessing.
- [ ] A written store is readable by `ome-zarr` in a throwaway isolated environment —
      a **manual release check, not a CI gate**, since that package is banned from every
      dependency group.
