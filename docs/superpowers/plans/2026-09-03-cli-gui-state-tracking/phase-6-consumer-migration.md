# Phase 6 — Consumer migration and the deletions

**Depends on:** P1–P5. **Blocks:** P7.

**Spec:** §11 (consumer migration), §11.1 (~1,400 lines deleted), §11.2 (folded in).

**Goal:** every consumer of the nine evidence sources calls `resolve_run_state` instead,
and the machinery they used is deleted. This is the phase that pays for the previous five.

**The deletions are the deliverable.** A task here that migrates a consumer without
deleting what it replaced has not finished — the failure mode this whole change addresses
is nine sources that each closed a real hole and none of which was ever removed.

---

## Deletion ledger

Track these in the phase's final commit body. Spec §11.1 estimates ~1,400 lines.

| # | Delete | Where | Task |
|---|---|---|---|
| 1 | `classify_output_consistency`, `OutputCompletionEvidence`, `inspect_output_consistency`, `OutputConsistencyReport` | `gui/results_viewer/_output_consistency.py` (617 lines, whole file) | 2 |
| 2 | `RunRegistry._processing_state_conflict`, `_publication_evidence_conflict`, `_orchestration_state_conflict` | `gui/shell/_runs_registry.py:1087,1202,1264` | 4 |
| 3 | `_local_completion_evidence_conflict`'s 8-branch tree | `_runs_registry.py:591` | 5 |
| 4 | `_latest_event_states` | `_runs_registry.py:1172` | 4 |
| 5 | `_read_status_from_manifest`, `_manifest_is_complete` | `_runs_registry.py`, `_slurm_observer.py` | 4, 6 |
| 6 | `DashboardManifestKey.VERSION` — written as `3` at one site, read at **zero** | `_dashboard/_manifest_builder.py:766` | 7 |
| 7 | `sdk_/monitor_slurm_jobs.py` — zero importers in `src/` or `tests/` | whole file (241 lines) | 7 |
| 8 | `browse/_source_render.py`'s `browse_cache_base` / `cache_png_path` / `init_cache` / `wipe_cache` — zero production callers | `_source_render.py:35-38` | 7 |
| 9 | Eight zero-caller resolvers in `_io_constants` | `_io_constants.py:2107` | 7 |
| 10 | Every `_legacy_*` helper and `resolve_*` fallback on the hot path | across `_cli` | P7 (they **move into** migrate, not away) |

Items 1–9 land here. **Item 10 is P7's** — spec §11.1 says legacy paths move *into*
`--mode migrate`, and deleting them before migrate can read them would strand every
existing tree.

---

## Task 1: `_snapshot_status.py` — 101 lines to ~30

**Files:**
- Modify: `src/phenotypic/gui/_snapshot_status.py`
- Test: `tests/unit/gui/test_snapshot_status.py`

`snapshot_refresh_status` currently branches on `inspect_output_consistency` plus
`snapshot_is_current()` plus `refresh_state_is_current()` — the last of which full-content
SHA-256s seven files on every 5–10 s tick (`_output_root.py:559`).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    "completion,refresh_supported,expected_color",
    [
        ("complete",   True,  "success"),
        ("incomplete", True,  "warning"),
        ("failed",     True,  "danger"),
        ("active",     True,  "warning"),
        ("complete",   False, "success"),
    ],
)
def test_the_badge_is_a_pure_function_of_completion(
    completion, refresh_supported, expected_color, fake_output_root
):
    """§11: ~30 lines mapping `completion` -> badge, replacing two fingerprints and
    a full re-hash of 7 files per poll."""
    from phenotypic.gui._snapshot_status import snapshot_refresh_status

    fake_output_root.run_state = _state(completion=completion)
    _label, color, _disabled = snapshot_refresh_status(
        fake_output_root, refresh_supported=refresh_supported
    )
    assert color == expected_color


def test_no_badge_refresh_hashes_a_deliverable(fake_output_root, monkeypatch):
    """The 5-10s tick currently full-content SHA-256s measurements.parquet,
    measurements.csv, pipeline.json, curation_labels.parquet, custom_categories.json,
    qc.duckdb and review_state.json. Per tab."""
    import hashlib

    calls = {"n": 0}
    real = hashlib.sha256
    monkeypatch.setattr(
        hashlib, "sha256", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1]
    )
    snapshot_refresh_status(fake_output_root, refresh_supported=True)
    assert calls["n"] == 0
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`snapshot_refresh_status` takes `resolve_run_state(output_root.layout.output_root,
depth="shallow")` and maps `completion` → `(label, color, disabled)`. Delete
`_completion_evidence_status` entirely.

**Fold in audit S2 while here** — it is the same function and the same tick. `OutputRoot`'s
frozen `consumed_state_fingerprint` (`_output_root.py:882`) and `CurationLabels`'
self-updating `_source_fingerprint` (`_curation_labels.py:760`) hash overlapping path sets
with different lifecycles, so **marking one colony makes the viewer report its own write as
external drift** — the badge flips to `"Changed on disk"` / `danger`. Exclude GUI-owned
mutable paths from the snapshot fingerprint, the way `snapshot_is_current()` already
deliberately does (`_output_root.py:545-548`). Add a test that a curation click leaves the
badge `success`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_snapshot_status.py src/phenotypic/gui/results_viewer/_output_root.py \
        tests/unit/gui/test_snapshot_status.py
git commit -m "refactor(gui): the snapshot badge becomes a map over RunState.completion

Spec §11. Also fixes audit S2: marking one colony no longer makes the viewer report
its own write as external drift, and the tick stops re-hashing seven deliverables."
```

---

## Task 2: Delete `_output_consistency.py`

**Files:**
- Delete: `src/phenotypic/gui/results_viewer/_output_consistency.py` (617 lines)
- Modify: every importer
- Test: `tests/unit/gui/`

- [ ] **Step 1: Find every importer**

```bash
grep -rn '_output_consistency\|inspect_output_consistency\|classify_output_consistency\|OutputConsistencyReport\|OutputCompletionEvidence' src/ tests/
```

Expected from the pre-flight count: 4 in `src/`, 3 in `tests/`.

- [ ] **Step 2: Write the failing test that `contradictory` is unreachable**

```python
def test_contradictory_is_not_a_state_any_more():
    """Spec §4.3: `contradictory` is DELETED as a state.

    It exists today only because derived counts are cross-checked against each
    other. Once counts stop being evidence, two authorities cannot disagree: there
    is exactly one path to each verdict. This test is the thing that stops it coming
    back -- it was the source of 'run flagged read-only for a reason the user cannot
    act on', which is the user-visible bug this whole change is for.
    """
    import typing

    from phenotypic.sdk_._run_state import Completion

    assert set(typing.get_args(Completion)) == {
        "complete", "incomplete", "failed", "active"
    }


def test_no_module_still_imports_the_deleted_classifier():
    import subprocess

    hits = subprocess.run(
        ["grep", "-rn", "_output_consistency", "src/"],
        capture_output=True, text=True,
    ).stdout
    assert not hits, f"dangling importers of the deleted classifier:\n{hits}"
```

- [ ] **Step 3: Migrate the callers, then delete the file**

- `_snapshot_status.py` — done in Task 1.
- `OutputRoot.discover` (`_output_root.py:178`) — `resolve_run_state(depth="deep")`,
  **one** call replacing the current double read.
- `OutputMutationGuard` — `resolve_run_state(depth="deep")`; the mutation gate becomes
  `state.completion != "active"` rather than `report.is_read_only`.
- The processing-inventory cache's `cache_reusable` (`report.state == "coherent"`) becomes
  `state.completion == "complete"`.

Then `git rm src/phenotypic/gui/results_viewer/_output_consistency.py`.

**`core_readable` needs care.** `OutputConsistencyReport.core_readable`
(`_output_consistency.py:106`) is what the live-run test gate asks — a memory note in this
project records `skipif` must ask `core_readable`, not `is_dir`. Its replacement is
`state.completion in {"complete", "incomplete"}` **and** a valid aggregate proof. Grep for
`core_readable` in `tests/` and migrate each site deliberately; a gate that silently starts
returning `True` runs tests against a half-written tree.

- [ ] **Step 4: Run the GUI suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
```

`tests/gui` **is** in `testpaths`; `tests/e2e` is not and needs `PLAYWRIGHT=1`.

- [ ] **Step 5: Commit**

```bash
git rm src/phenotypic/gui/results_viewer/_output_consistency.py
git add -A src tests
git commit -m "refactor(gui): delete _output_consistency.py -- 617 lines, 9 sources, 23 rules

Spec §4.3, §11. `contradictory` is gone as a reachable state, and a test now stops
it coming back. Callers use resolve_run_state(depth=...)."
```

---

## Task 3: `OutputRoot` currency — one shallow verification

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py:542,559,882`
- Test: `tests/unit/gui/results_viewer/`

- [ ] **Step 1: Write the failing test**

```python
def test_one_currency_check_replaces_two(fake_output_root):
    """§11: snapshot_is_current() + refresh_state_is_current() -> one shallow
    verification. Two overlapping fingerprints with different lifecycles is audit
    S2, and the fix is one owner, not two better-synchronised ones."""
    from phenotypic.gui.results_viewer._output_root import OutputRoot

    assert not hasattr(OutputRoot, "refresh_state_is_current")
    assert not hasattr(OutputRoot, "consumed_state_fingerprint")


def test_a_chmod_does_not_report_changed_on_disk(bound_output_root):
    """Audit S3, at the consumer. _inventory_is_current compares st_ctime_ns
    (_processing_inventory.py:462), which moves on chmod, chown, hardlink and
    rsync -a -- all routine on a shared HPC filesystem, and each one makes the whole
    binding report 'Changed on disk'."""
    for path in bound_output_root.root.rglob("*.parquet"):
        path.chmod(0o644)
    assert bound_output_root.is_current()
```

- [ ] **Step 2: Implement**

`snapshot_is_current()` delegates to `resolve_run_state(depth="shallow")`.
`refresh_state_is_current()` and `consumed_state_fingerprint` are deleted. Drop
`st_ctime_ns` from `_inventory_is_current` (`_processing_inventory.py:462`) — audit S3.

**Confirm no test depends on ctime-sensitivity before dropping it**
(`grep -rn 'ctime' tests/`), as audit S3 asks.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): one currency check, and drop ctime from the inventory sweep

Spec §11, audit S2 and S3. A chmod on GPFS no longer makes a binding report
'Changed on disk'."
```

---

## Task 4: `RunRegistry` claimability — 248 lines to one call

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:1087,1172,1202,1264`
- Test: `tests/unit/gui/shell/`

- [ ] **Step 1: Write the failing tests**

```python
def test_claimability_is_one_resolve_call(fake_registry):
    """§11: three conflict predicates -> one resolve_run_state call."""
    from phenotypic.gui.shell import _runs_registry as reg

    for gone in (
        "_processing_state_conflict",
        "_publication_evidence_conflict",
        "_orchestration_state_conflict",
        "_latest_event_states",
        "_read_status_from_manifest",
    ):
        assert not hasattr(reg.RunRegistry, gone) and not hasattr(reg, gone), gone


def test_the_event_log_is_replayed_at_most_once(bound_registry, monkeypatch):
    """Audit S5 / §11: _latest_event_states reimplements aggregate_state_from_events
    with different semantics (stage demotion, no inventory fence). Two parsers of one
    append-only log will drift."""
    calls = {"n": 0}
    _count_event_log_reads(monkeypatch, calls)
    bound_registry.claimability("some-output")
    assert calls["n"] <= 1
```

- [ ] **Step 2: Implement**

Replace all three conflict predicates with `resolve_run_state(output_dir, depth="shallow")`
and a `completion`-based decision. Delete `_latest_event_states` and
`_read_status_from_manifest`.

**Where does the stage-demotion rule go?** Audit S5 proposes folding it into the CLI
aggregator as an option. Under §4.2 the event log is no longer evidence, so the demotion
has no consumer — verify that with a grep before deleting rather than after, and if
something still reads it, fold it in rather than dropping it silently.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): RunRegistry claimability becomes one resolve_run_state call

Spec §11, audit S5. Three conflict predicates and the second event-log replay are
deleted; one append-only log now has one parser."
```

---

## Task 5: `RunRegistry` local exit, plus DEFERRED D-2

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:591,1058,1306`
- Test: `tests/unit/gui/shell/`

**This task includes an explicit scope addition.** DEFERRED **D-2** (stale
`gui_launch_owner.json` has no repair path) is folded in here, for two reasons stated in
the README: this task rewrites `_assert_output_claimable_locked`, the exact predicate that
causes the dead-end; and under the Q2 verdict ladder a stale owner record now masks
`incomplete` as `active`. DEFERRED itself recommends P6.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_eight_branch_refusal_tree_becomes_advisories(fake_registry):
    """§11: '8-branch refusal tree -> resolve_run_state(deep); refusals become
    advisories.' A refusal the user cannot act on is the bug; an advisory they can
    read is the fix."""
    state = fake_registry.local_exit_state("some-output")
    assert state.completion in {"complete", "incomplete", "failed", "active"}
    assert isinstance(state.advisories, tuple)


def test_a_sigkilled_gui_does_not_lock_the_output_forever(tmp_path):
    """DEFERRED D-2 / audit S7 [verified]: nothing in the codebase ever deletes or
    repairs gui_launch_owner.json. A SIGKILLed GUI leaves status: 'running';
    rehydrate_from_sandbox downgrades it IN MEMORY ONLY (_runs_registry.py:773,
    persist=False), and _assert_output_claimable_locked then refuses the output
    forever, with no UI affordance to clear it.

    The record already stores pid and started_at. Use them."""
    _write_owner_record(tmp_path, status="running", pid=_a_dead_pid(), started_at="2020-01-01")
    registry = _registry(tmp_path)
    registry.assert_output_claimable(tmp_path)   # must not raise


def test_a_live_owner_still_refuses_the_claim(tmp_path):
    """The liveness check must not become a rubber stamp -- an owner whose process
    is alive still owns the output."""
    import os

    import pytest

    _write_owner_record(tmp_path, status="running", pid=os.getpid(), started_at="2026-09-03")
    with pytest.raises(RuntimeError):
        _registry(tmp_path).assert_output_claimable(tmp_path)
```

- [ ] **Step 2: Implement**

`_local_completion_evidence_conflict`'s eight refusal strings become advisories on the
`RunState`. `_assert_output_claimable_locked` gains a liveness check: an owner record whose
`status` is non-terminal but whose `pid` is not alive is downgraded **and persisted** — the
`persist=False` at `_runs_registry.py:773` is precisely what makes today's downgrade
useless.

Use `os.kill(pid, 0)` guarded for `ProcessLookupError` / `PermissionError`; a `pid` that
has been recycled is a real but bounded risk, and `started_at` bounds it further — treat a
record older than the boot time as dead regardless.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): local-exit refusals become advisories; repair a stale owner record

Spec §11 plus DEFERRED D-2, folded in deliberately: this task rewrites the exact
predicate that caused the permanent dead-end, and under the Q2 ladder a stale owner
record masks incomplete as active. The record already stored pid and started_at;
nothing read them."
```

---

## Task 6: SLURM observer — call sites only

**Files:**
- Modify: `src/phenotypic/gui/run_console/_slurm_observer.py:536,909,1312`
- Test: `tests/unit/gui/run_console/`

**Scope discipline is the whole of this task.** Spec §2.2 and DEFERRED D-1 put the
observer's decision tree, its 30-second reconciliation grace window and its `squeue`/`sacct`
state ranking **out of scope**. Only ~185 of its lines are filesystem-derived; the rest is
scheduler domain, it is the least testable code in the GUI, and its failure mode ("run
stuck in `reconciling`") is directly user-visible.

**Change exactly three things:**

1. the two `_cli_completion` call sites (`current_success_counts`, `valid_run_completion`) →
   one `resolve_run_state(depth="shallow")`
2. `_all_stage3_markers_exist` → reads `stages.stage3` from the P3 record
3. `_manifest_is_complete` → deleted

- [ ] **Step 1: Write the failing test**

```python
def test_the_observer_tick_does_not_hash_anything(fake_observer, monkeypatch):
    """Audit §4: the 2-second daemon tick currently runs valid_run_completion ->
    current_run_is_complete -> current_success_counts -> _walk_current_success, which
    calls valid_image_success once per image, each re-hashing the embedded
    measurements parquet AND the overlay PNG. Then current_aggregate_is_current walks
    it all AGAIN. On a 10,000-image run that is ~2-3 x 10^4 file hashes every two
    seconds."""
    import hashlib

    calls = {"n": 0}
    _count_sha256(monkeypatch, calls)
    fake_observer.tick()
    assert calls["n"] <= 8


def test_the_decision_tree_is_untouched():
    """Spec §2.2, DEFERRED D-1. This test exists to make scope creep fail CI rather
    than fail review."""
    import inspect

    from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver

    source = inspect.getsource(SlurmLifecycleObserver._observe_record)
    assert "resolve_run_state" in source
    assert "_manifest_is_complete" not in source
    # The grace window and squeue/sacct ranking stay exactly as they are.
    assert "GRACE" in source or "grace" in source
```

- [ ] **Step 2: Implement, run, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/run_console tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): the observer tick asks resolve_run_state once

Spec §11, scoped by §2.2 and DEFERRED D-1: two call sites and the Stage-3 probe
move; the decision tree, grace window and scheduler polling are untouched. The 2s
tick stops walking every image twice and hashing every artifact."
```

---

## Task 7: The pure deletions, and §11.2's fold-ins

**Files:** as listed in the deletion ledger, items 6–9.

Each of these is **evidence-backed** — the audit verified the caller counts by hand. Re-run
each grep before deleting; a claim from 2026-09-03 is not a claim about the tree you are
editing.

- [ ] **Step 1: Verify each claim, then delete**

```bash
grep -rn 'monitor_slurm_jobs'        src/ tests/    # expect 0 outside the file itself
grep -rn 'DashboardManifestKey.VERSION' src/ tests/ # expect 1 write, 0 reads
grep -rn 'browse_cache_base\|cache_png_path\|init_cache\|wipe_cache' src/ tests/
grep -rn 'read_run_manifest\|load_master_measurements\|resolve_best_pipeline_path\|resolve_qc_dir\|recompile_status_dir\|chunk_parquet_path\|checkpoint_lock_path\|chunk_manifest_path' src/ tests/
```

**If a grep disagrees with the audit, stop and record it** — either the tree moved or the
audit was wrong, and both matter more than the deletion.

Two of the eight `_io_constants` resolvers "claim in their docstrings to replace inline
blocks that still exist" (S21). For those, route the inline block through the helper and
keep it, or delete both. Deleting the helper while the inline block survives is the worse
of the three outcomes.

- [ ] **Step 2: Fold in §11.2 — inside files already being rewritten**

Only these. Everything else in DEFERRED's churn table stays deferred:

- Hand-joined `.phenotypic/aggregate_publication.json` in the GUI → use
  `aggregate_publication_marker_path()` (audit S8, `_output_consistency.py:380` — the file
  Task 2 deleted, so this is now wherever its caller moved).
- The ~17 shadow state filenames into `_io_constants` (S9). Two of them —
  `staged_orchestration.json` and `staged_finalization_complete.json` — are **double-spelled
  across the CLI/GUI boundary**; those two are the ones that matter.
- `DIR_PROGRESS` at the two literal sites in `phenotypicCLI.py:839,841`, in the same file
  that already imports and uses it at `:943` (S15).
- The recompile `task_manifest.json` and `job_metadata.json` naive writers made atomic
  (S11) — these **are** polled by concurrently launched SLURM workers, including the
  unlocked write at `phenotypicCLI.py:3385`.

- [ ] **Step 3: Collapse the three `_canonical_digest` copies**

P1 Task 4 left three, pinned by a test. Now that `_run_state.py` exists and `sdk_` is the
shared layer, move it to `sdk_/_io_constants.py` (or a small `sdk_/_digests.py`) and have
all three import it. Delete the agreement test — it exists only while there are copies to
disagree.

- [ ] **Step 4: Phase gate — the full suite**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix <every path this phase touched>
```

Then the full `tests/unit` **and** `tests/gui` suites, as a Slurm job via the
**`run-phenotypic-test`** and **`slurm-job`** skills. This is the phase most likely to break
something distant.

Compare against the recorded baseline: four failures are known pre-existing, three of which
fail only on compute nodes. **A fifth failure is this phase's**, not the baseline's.

---

## Task 8: Record what the GUI tracks, in `gui/CLAUDE.md`

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`

**This task is not optional and is not a docs-polish afterthought.** The change deletes
nine evidence sources and four classifiers from the GUI; a module guide that still
describes them is worse than no guide, because the next reader will trust it. The
distinction that matters and is nowhere written down today is **what the GUI *owns* versus
what it merely *reads*** — and the change moves that line.

- [ ] **Step 1: Add a "State the GUI tracks" section**

Place it after `### Flask app.server.config keys` (`gui/CLAUDE.md:176`), which already
enumerates the process-wide singletons. Three tables, and the split between them is the
content:

**(a) GUI-owned durable state — the GUI is the writer.**

| Artifact | Path | Written by | Read back by | Notes |
|---|---|---|---|---|
| Launch ownership | `.phenotypic/gui_launch_owner.json` | `_persist_record_locked` (`shell/_runs_registry.py:1306`) | the CLI's freshness guard, and `resolve_run_state` rule 2 | **A §4.1 liveness authority.** Carries `pid` + `started_at`; P6 Task 5 added the liveness check that makes rule 2 sound. |
| Curation labels | `deliverables/qc/curation_labels.parquet` | `_curation_labels.py` | the CLI re-emits from it | GUI is the primary writer; keyed on intrinsic identity, so §7's inversion does not touch it |
| Custom categories | `deliverables/qc/custom_categories.json` | GUI only | GUI only | — |
| Review state | `deliverables/qc/review_state.json` | GUI | GUI | **The CLI deletes it at finalize** (`_cli_output_manager.py:1238`) — a fresh run resets review progress |
| Verified rows | `deliverables/verified.parquet` | GUI only | GUI only | finalize never writes it |
| Error categories | `deliverables/errors/<category>.parquet` | **both** | both | dual-owned, documented |

**(b) GUI-owned ephemeral state — dies with the process or the tab.** The 136 `dcc.Store`s
(only 35 declare `storage_type`), the 14 `dcc.Interval`s, the `app.server.config`
singletons already listed above, and the sandbox caches. State the rule the audit found
and P6 did not change: **one bound output per process, shared by every browser tab.**

**(c) State the GUI reads and must never write.** Everything under `.phenotypic/` except
the owner record. Say it once, plainly, and name the one function that answers questions
about it: `resolve_run_state(output_dir, depth=...)`.

- [ ] **Step 2: Delete what is no longer true**

Grep `gui/CLAUDE.md` for the deleted machinery and remove or rewrite each mention:

```bash
grep -n 'consistency\|coherent\|contradictory\|manifest\|_output_consistency\|snapshot_is_current\|refresh_state_is_current' src/phenotypic/gui/CLAUDE.md
```

The four-state vocabulary (`coherent`/`active`/`incomplete`/`contradictory`) is gone;
`contradictory` no longer exists at all. Replace with the four verdicts and a pointer to
`resolve_run_state`.

- [ ] **Step 3: State the import rule that replaces the 25-symbol seam**

Audit §7: the GUI imports 25 private `phenotypic._cli` symbols across 9 modules, and that
is how the O(N)-hashing completion predicate ended up on a 2-second timer. After this
change the rule is: **the GUI imports readers from `phenotypic.sdk_`, never from
`phenotypic._cli`.** List the remaining legitimate `_cli` imports (the SLURM lifecycle
helpers the observer needs, per DEFERRED D-1) so a reader can tell the survivors from a
regression.

- [ ] **Step 4: Verify every claim you just wrote**

Each path and function name gets a `grep`. A module guide asserting a `file:line` that
moved is the failure this whole change is about.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md
git commit -m "docs(gui): record what state the GUI owns, and what it only reads

The change deletes nine evidence sources and four classifiers; the module guide
still described them. Adds the owned/ephemeral/read-only split, which was nowhere
written down, and the sdk_-not-_cli import rule that replaces the 25-symbol seam."
```

---

- [ ] **Step 5: Commit with the deletion ledger**

```bash
git add -A
git commit -m "refactor: delete the nine-source completion machinery

Spec §11.1, §11.2. Net -<N> lines.

Deleted:
  _output_consistency.py                        -617
  RunRegistry three claimability predicates     -<n>
  _local_completion_evidence_conflict tree      -<n>
  _latest_event_states, _read_status_from_manifest, _manifest_is_complete  -<n>
  sdk_/monitor_slurm_jobs.py (0 importers)      -241
  browse/_source_render.py dead cache API       -<n>
  eight zero-caller _io_constants resolvers     -<n>
  DashboardManifestKey.VERSION (1 write, 0 reads) -<n>

Folded in (§11.2): aggregate_publication_marker_path at the GUI site; 17 shadow
filenames into _io_constants, including the two double-spelled across the CLI/GUI
boundary; DIR_PROGRESS at phenotypicCLI.py:839,841; the four naive control-manifest
writers made atomic.

Every caller count was re-grepped before deletion, not taken from the audit."
```
