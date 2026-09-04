# Phase 6 — Consumer migration and the deletions

**Depends on:** P1–P5. **Blocks:** P7.

**Spec:** §11 (consumer migration), §11.1 (~1,400 lines deleted), §11.2 (folded in).

**Goal:** every consumer of the nine evidence sources calls `resolve_run_state` instead,
and the machinery they used is deleted. This is the phase that pays for the previous five.

**The deletions are the deliverable.** A task here that migrates a consumer without
deleting what it replaced has not finished — the failure mode this whole change addresses
is nine sources that each closed a real hole and none of which was ever removed.

> **The first draft of this phase migrated the GUI and forgot the CLI (CAN-8).** Spec §9's
> caller table names two CLI depths — "CLI finalize, before publishing proofs → deep" and
> "CLI resume, deriving the worklist → deep, cache-assisted" — and §9.2's entire headline
> scenario (10 images added to 6,000) **is** the resume worklist. Migrating only the GUI
> leaves the O(N)-hashing readers on every CLI path, so the double walk is never removed,
> §11's last row ("split: readers → `sdk_/_run_state.py`") is not delivered, and **two
> completion predicates ship permanently** — the exact drift hazard this phase cites when
> deleting `_latest_event_states`. Task 0 fixes that, and it goes first because everything
> else in the phase assumes the split has happened.

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
| 5b | `local_manifest_completion_problem` — **the third manifest consumer** (M6) | `_cli_gui_lifecycle.py:41-65`, gating `publish_run_completion_evidence` at `:130-135` | 4 |
| 6 | `DashboardManifestKey.VERSION` — written as `3` at one site, read at **zero** | `_dashboard/_manifest_builder.py:766` | 7 |
| 7 | `sdk_/monitor_slurm_jobs.py` — zero importers in `src/` or `tests/` | whole file (241 lines) | 7 |
| 8 | `browse/_source_render.py`'s `browse_cache_base` / `cache_png_path` / `init_cache` / `wipe_cache` — zero production callers | `_source_render.py:35-38` | 7 |
| 9 | Eight zero-caller resolvers in `_io_constants` | `_io_constants.py:2107` | 7 |
| 10 | Every `_legacy_*` helper and `resolve_*` fallback on the hot path | across `_cli` | P7 (they **move into** migrate, not away) |

> **M6: `manifest.json` is still evidence after P6 unless this site is converted.**
> `local_manifest_completion_problem` branches on `DashboardManifestKey.COMPLETED`, `FAILED`
> and `TOTAL_IMAGES` (`completed != total`, `failed != 0`) and **gates run-proof
> publication**. It is in neither `_output_consistency.py` nor `_slurm_observer.py`, so the
> round-1 note claiming every manifest-count reader lives in those two files was wrong.
>
> This does **not** disturb U-5 — that ruling was about `RunState`-mediated consumers, and
> this one is not — but it makes two plan claims false unless fixed: P7 Task 6's register
> says *"`manifest.json` as evidence"* was deleted, and §4.2 demotes it. **Convert this
> site**, or state in the register that the GUI-local publication path is the one surviving
> manifest consumer, and why it is allowed to be.

Items 1–9 land here. **Item 10 is P7's** — spec §11.1 says legacy paths move *into*
`--mode migrate`, and deleting them before migrate can read them would strand every
existing tree.

---

## Task 0: Split `_cli_completion.py` and migrate the CLI's own readers (CAN-8)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py` — readers out, writers stay
- Modify: `src/phenotypic/phenotypicCLI.py:2394,2428,2439,2874,3725`
- Modify: `src/phenotypic/_cli/_cli_checkpoint_handler.py:291,348,401` **(gen-r4 N-1)**
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py:643,653` **(gen-r4 N-2)**
- Modify: `src/phenotypic/_cli/_cli_gui_lifecycle.py:90` **(gen-r4 N-1)**
- Modify: `src/phenotypic/_cli/_dashboard/_manifest_builder.py:729`
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py:728`
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py:203-213` — `valid_image_success`, not the three above
- Modify: `src/phenotypic/_cli/_cli_migrate.py:88-89` — likewise
- Test: `tests/unit/cli/test_completion_split.py` *(new)*

> **The file list was short by three files and the count was wrong (gen-r4 N-1/N-2, open
> three rounds).** Measured on `c9d1fbfc`: the three predicate names have **13 invocations
> across 6 files**, not ten across four. `_cli_checkpoint_handler.py` (3 — the in-array
> `__PHENOTYPIC_CHECKPOINT__` dispatch), `_cli_recompile_worker.py` (2) and
> `_cli_gui_lifecycle.py` (1) were named nowhere in this task. P4 and P5 touch two of those
> files but for other reasons — P4 rewrites `_cli_recompile_worker.py:764` only, and P5's
> publisher table marks `_cli_checkpoint_handler.py` **not** a publisher — so nothing else
> in the plan removes these reads. Regenerate the list rather than trusting it:
>
> ```bash
> grep -rn 'current_run_is_complete\|current_success_counts\|current_aggregate_is_current' \
>   src/phenotypic --include=*.py | grep -v _cli_completion.py
> ```

**This task goes first.** Every later task assumes `resolve_run_state` is the only
completion predicate; while a second one survives on the CLI side, the phase's premise is
false.

- [ ] **Step 1: Write the test that keeps the split split**

```python
def test_only_one_completion_predicate_survives():
    """CAN-8 / §11's last row. Two parsers of one question drift -- this phase
    deletes _latest_event_states for exactly that reason, and would ship a new
    instance of it on the CLI side."""
    import subprocess

    from pathlib import Path

    # Scoped to src/phenotypic/_cli + sdk_, NOT all of src/ (gen-r4 N-1). The GUI's three
    # holders -- _runs_registry.py, _slurm_observer.py, and _output_consistency.py -- are
    # migrated by Tasks 1-6 of this phase, so a whole-tree grep here is red by construction
    # at the end of Task 0. The whole-tree assertion is Task 7's, where it can pass.
    root = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    hits = subprocess.run(
        ["grep", "-rn",
         "current_run_is_complete\\|current_success_counts\\|current_aggregate_is_current",
         str(root / "_cli"), str(root / "sdk_"), str(root / "phenotypicCLI.py")],
        capture_output=True, text=True,
    ).stdout.strip()
    assert not hits, f"the old O(N)-hashing readers survive CLI-side:\n{hits}"


def test_the_resume_worklist_uses_the_cache_assisted_path():
    """§9's caller table, row 2 -- and §9.2's headline scenario IS this call."""
    import inspect

    from phenotypic import phenotypicCLI

    source = inspect.getsource(phenotypicCLI)
    assert "resolve_run_state" in source
```

- [ ] **Step 2: Move the readers**

Out of `_cli_completion.py`, into `sdk_/_run_state.py` (already built in P1):
`current_success_inventory`, `_walk_current_success`, `current_success_counts`,
`_current_success_work_ids`, `current_aggregate_is_current`, `current_run_is_complete`,
`valid_aggregate_snapshot`, `valid_run_completion`, `valid_image_success`.

Staying CLI-side because they **write**: `publish_image_success`,
`publish_aggregate_snapshot`, `publish_run_completion_evidence`, `image_data_artifact`,
`refresh_success_markers_after_metadata_migration`, `authorized_measurement_sources`
(a reader, but of run-authorization for the writer path — keep it beside its caller and say
so in a comment).

**INV-LAYER still binds.** The moved readers must not reach back into `_cli`; that is why
P1 Task 4 established the plain-JSON state read. If a mover needs
`load_processing_state`, it has taken the wrong function. The **record** reader they call is
in `sdk_/_image_record.py`, which P3 puts there for exactly this reason (N-3).

> **The move silently drops a relocation the readers depend on (M3).**
> `load_processing_state` calls `migrate_legacy_machine_state(output_dir)` on **every** read
> (`_cli_state_management.py:106`) — relocating `progress/`, `processing_state.json` and
> `processing_events.log` from the output root into `.phenotypic/`
> (`sdk_/_io_constants.py:1006-1052`). The readers being moved use the **non-resolving**
> `progress_dir` (`:903-909`; see `_cli_completion.py:27,785` and
> `image_completion_marker_path`), so on a **pre-relocation** tree they work today *only
> because the state read relocated first.*
>
> Removing the trigger without replacing it makes those readers silently find nothing on
> such a tree — an empty inventory, which is a *valid* result. Two options, and the second is
> better: have the moved readers use the **resolving** path helpers, so relocation stops
> being a precondition; or have P1's `requires_conversion` classify a pre-relocation tree as
> `CONVERT` and let migrate own the move. Decide in this task, and state which — a read path
> that depends on a write side effect is the thing this phase exists to remove.

- [ ] **Step 3: Convert the thirteen CLI call sites**

Each becomes one `resolve_run_state(output_dir, depth="deep")` — `deep` on the CLI, per
§9's table, because the CLI publishes proofs and derives worklists and must not act on a
stat-only answer. Re-grep before editing; these line numbers are from `c9d1fbfc`.

Three of the six files carry a call site the earlier draft of this task never named, and two
of them are the ones a resume actually runs through — so convert by grep output, not by the
list:

| File | Invocations | What the call gates |
|---|---|---|
| `phenotypicCLI.py` | 5 | startup counts, aggregate currency, the two early-exit checks |
| `_cli_checkpoint_handler.py` | 3 | the in-array `__PHENOTYPIC_CHECKPOINT__` dispatch (gen-r4 N-1) |
| `_cli_recompile_worker.py` | 2 | whether recompile may skip re-derivation (gen-r4 N-2) |
| `_cli_gui_lifecycle.py` | 1 | gates `publish_run_completion_evidence` — see item 5b above |
| `_dashboard/_manifest_builder.py` | 1 | the manifest's completion field |
| `sdk_/_hdf_to_zarr.py` | 1 | migration's own progress read |

All thirteen are `deep`, including `_cli_checkpoint_handler.py`'s three. Check that
conclusion rather than assuming it, because the file's name invites the opposite one: it is
the `__PHENOTYPIC_CHECKPOINT__` handler, which sounds per-task and is not. The trigger is a
**reserved entry in the array task list** (root `CLAUDE.md`, *SLURM array auxiliary work*) —
one index, not one per image — and all three call sites sit on a publication path:
`:291` decides whether an empty aggregate is a `RuntimeError` or a legitimate
terminal-incomplete close; `:348` chooses between `deactivate_orchestration` and
`mark_staged_complete`, then gates `publish_run_completion_evidence`; `:401` gates the
completion marker itself. That is §9's first row — *CLI finalize, before publishing proofs →
`deep`* — not its two `shallow` rows, both of which are read-only pollers. No `deep` read
here is per-image, so no O(N²) arises.

> **§9's table has no row for a worker process.** Its six rows are two CLI paths, two
> binding/guard paths, and two pollers. If a *genuine* per-task reader turns up during this
> task — one that runs once per array index — it has no assigned depth and this plan must
> not invent one: raise it, because `deep` there is O(N) per task on the exact walk this
> change exists to make cheap. None of the thirteen is such a reader.

- [ ] **Step 4: Confirm the double walk is gone**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -q
```

Then count: a single completion query must walk the images **once**. Instrument
`valid_image_success` with a counter in a throwaway patch, run one `--mode full` resume
over a 6-image fixture, and assert the count equals 6 rather than 12. That doubling is
audit §4's finding, and it is the thing §9.2's number depends on.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic tests/unit/cli/test_completion_split.py
git commit -m "refactor: split _cli_completion.py -- readers to sdk_, writers stay

Spec §11's last row, §9's two CLI depth rows. Ten CLI call sites moved onto
resolve_run_state; the double walk (audit §4) is gone. A test now fails if a second
completion predicate reappears."
```

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

**But `completion` alone cannot replace it (CAN-18).** The function answers **two**
questions today (`_snapshot_status.py:17-63`): run activity/completion, *and* whether the
**bound in-memory snapshot** still matches disk — "Current" versus "Changed on disk"
(`:38-44`, `:55-62`). `completion` answers only the first. A re-finalize over an unchanged
inventory rewrites `measurements.parquet` while `completion` stays `complete`, so a badge
driven by `completion` alone reads **"Current" over a stale snapshot** — and Task 3 then
deletes `refresh_state_is_current` and `consumed_state_fingerprint`, the two things that
answered the second question.

Keep both axes. The badge is a function of `(completion, snapshot_is_current)`:

```python
def test_a_refinalize_over_an_unchanged_inventory_shows_changed_on_disk(bound_output_root):
    """CAN-18. completion stays `complete` across a re-finalize, because the
    inventory did not change -- but the mirror the viewer is holding is now stale."""
    _refinalize(bound_output_root.root)          # rewrites measurements.parquet
    label, color, _ = snapshot_refresh_status(bound_output_root, refresh_supported=True)
    assert color == "danger" and "Changed" in label
```

The parametrized badge test therefore takes both inputs, not just `completion`, and the
`("complete", True, "success")` row asserts `snapshot_is_current` is also true.

**What replaces the deleted fingerprint** is the verification cache's stat sweep over the
deliverables the viewer actually consumed — the same `(size, mtime_ns)` tuples P1 already
records, not a second full-content hash. That is the win: the question survives, the
7-file SHA-256 per tick does not.

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
- The processing-inventory cache's `cache_reusable` (`report.state == "coherent"`) becomes
  `state.completion == "complete"`.

Then `git rm src/phenotypic/gui/results_viewer/_output_consistency.py`.

### Two predicates that are NOT `completion` in disguise (CAN-17)

The first draft replaced both with one-line `completion` tests. Neither is equivalent, and
**both errors fail open in the dangerous direction.**

**`core_readable`** (`_output_consistency.py:109-114`) is
`not marker_authority_required or aggregate_marker_valid`. Two cases the proposed
`completion in {"complete","incomplete"} and a valid aggregate proof` gets wrong:

- a **legacy** tree is core-readable today *with no aggregate proof at all* — the first
  disjunct carries it;
- an **`active`** output with a valid proof **is** core-readable, and `active` is excluded.

This is the predicate the live-run test `skipif` asks. A false `False` **skips** tests
rather than failing them, and a skip is invisible in a summary line — so this error hides
itself. Keep the disjunction:

```python
def core_readable(state: RunState) -> bool:
    """Whether the canonical aggregate bytes are authorized to read.

    NOT `completion`. A legacy tree with no aggregate proof is readable, and so is
    an ACTIVE run whose previous finalization published one. Both are excluded by a
    naive completion test, and because this gates a `skipif`, the failure is a
    silent skip rather than a red test (CAN-17).
    """
    return not state.marker_authority_required or state.aggregate_proof_valid
```

**`is_read_only`** is `state != "coherent"` (`:93-96`), so today `incomplete` prohibits
mutation. The proposed `completion != "active"` would make **every `incomplete` output
GUI-mutable** — a widening with no spec authority; §4.3 says only that `incomplete` is
"safe to read, safe to resume", which is not "safe to write". Keep the current meaning:
mutation requires `complete`.

```python
def test_an_incomplete_output_is_not_mutable(fake_output_root):
    """CAN-17. `is_read_only` is `state != coherent` today. `completion != "active"`
    would silently grant write access to every incomplete output."""
    fake_output_root.run_state = _state(completion="incomplete")
    with pytest.raises(RuntimeError):
        OutputMutationGuard(fake_output_root).require_mutable()


def test_an_active_run_with_a_valid_proof_is_still_core_readable(fake_output_root):
    fake_output_root.run_state = _state(completion="active", aggregate_proof_valid=True)
    assert core_readable(fake_output_root.run_state)
```

Grep `core_readable` in `tests/` and migrate each site deliberately.

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

**This is not an optional fold-in — Q2 rule 2 requires it (CAN-24).**

The first draft justified pulling DEFERRED D-2 in as "cheap and adjacent". Round 1 showed
it is a **correctness requirement of the verdict ladder**. Spec §4.1 makes
`gui_launch_owner.json` one of the three liveness authorities, and Q2 rule 2 reads it.
Audit S7 **[verified]**: nothing in the codebase ever deletes or repairs that record, and
`rehydrate_from_sandbox` downgrades it in memory only (`_runs_registry.py:773`,
`persist=False`). So a SIGKILLed GUI pins `status: "running"` **forever**, and rule 2 is
unsound as written — it reports `active` for a run nothing is working on.

The ladder's obligation therefore lives in **P1 Task 5** (added there: a verdict-matrix row
asserting a dead `pid` does not yield `active`). The **repair** lives here, where
`_assert_output_claimable_locked` is rewritten. Both are required; neither substitutes for
the other.

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

- [ ] **Step 3: Delete the three tests that assert on text, not behaviour (CAN-31)**

`_canonical_digest`'s collapse **moved to P1 Task 4** (CAN-29): hoisting a pure function
into `sdk_` up front is less total work than adding a third copy plus a keeper test here and
then deleting both. Nothing to do for it in this phase.

Instead, delete three tests this plan proposed that cannot fail for the right reason:

| Test | Why it goes |
|---|---|
| `test_run_state_exports_no_writer` (P1 T1) | asserts `__all__` name **prefixes**. A writer called `record_stage` or `persist_x` passes, and the prefix list is itself a tracked list needing sync with naming fashion. |
| `test_no_module_still_imports_the_deleted_classifier` (P6 T2) | a CWD-relative `subprocess` grep. A deleted module with a live importer is an `ImportError` the suite already raises; the relative path makes this pass or fail on where pytest was invoked. |
| `test_the_decision_tree_is_untouched` (P6 T6) | `"GRACE" in source`. Fails on a harmless rename, passes if you gut `_observe_record` and leave the word in a comment. Its goal — "make scope creep fail CI rather than fail review" — is a review concern, and this is the weakest possible enforcement of it. |

**Keep the INV-LAYER AST test exactly as written.** It is structural, it can fail, and P1
Task 1 Step 6 proves both the module-scope and lazy-in-function forms trip it. If a
write-side structural guarantee is still wanted for `_run_state.py`, check the AST for
`open(..., "w")`, `Path.write_*` and `os.replace` — a real check, where the prefix list was
a spelling check.

The two advisory tests (`any("migrate" in advisory)`, `any("metadata" in advisory)`)
substring-match human prose, which makes advisory wording a de-facto API. Give advisories a
small closed set of codes plus optional detail, and assert on the code.

- [ ] **Step 4: The whole-tree predicate assertion — and commit**

Task 0's gate test was scoped to `_cli` + `sdk_` because the GUI's holders were still live
at that point (gen-r4 N-1). Tasks 1–6 have now migrated them, so the unrestricted form can
finally pass. Add it here, in the same file, beside the scoped one:

```python
def test_no_completion_predicate_survives_anywhere():
    """The unrestricted form of test_only_one_completion_predicate_survives.
    Scoped to _cli + sdk_ in Task 0 because gui/ had not been migrated yet; by the
    end of this phase the whole tree must be clean."""
    import subprocess

    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "src"
    hits = subprocess.run(
        ["grep", "-rn",
         "current_run_is_complete\\|current_success_counts\\|current_aggregate_is_current",
         str(src)],
        capture_output=True, text=True,
    ).stdout.strip()
    assert not hits, f"a completion predicate survives:\n{hits}"
```

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_completion_split.py -q
git add -A src/phenotypic tests
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

Three text-asserting tests removed (CAN-31). The completion-predicate gate is now
unrestricted -- Task 0 could only scope it to _cli + sdk_ because gui/ had not been
migrated yet.

Every caller count was re-grepped before deletion, not taken from the audit."
```

- [ ] **Step 5: Phase gate — the full suite**

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
