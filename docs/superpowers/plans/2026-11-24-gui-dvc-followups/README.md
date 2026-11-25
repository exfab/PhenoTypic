# GUI Follow-ups Implementation Plan (#283 PR A, #282 GUI items PR B)

**Status:** revision 3 · 2026-11-25 · revised per `reports/2026-11-24-gui-dvc-followups/plan-review.md`
(the review, then its "Re-review of revision 2") and the user's decisions D15–D18; see
"Revision 2" and "Revision 3" at the end
**Spec:** `docs/superpowers/specs/2026-11-24-gui-dvc-followups/design.md`, as committed with this
revision (decisions D1–D18, no open question)
**Reviews behind the spec:** `docs/superpowers/reports/2026-11-24-gui-dvc-followups/spec-review.md`
**Bases:**
- **PR A** is built on branch `feat/gui-dvc-followups`, worktree
  `.claude/worktrees/gui-followups`, cut from `origin/feat/dvc-embeddings-refactor` @
  `36b9e840e` (#281). Every PR A `file:line` below is read there. PA7 rebases onto
  `origin/main` after #281 merges (D9), then re-verifies every anchor by name.
- **PR B** is cut only after #282 is rebased or merged onto a `main` that contains PR A
  (spec §1 step 4, PB0). PR B anchors are read at `origin/feat/gpu-embedding-staging` @
  `0993c8e06`; PB0 re-finds them on the new base by name.

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The GUI reads #281's contract (exit codes 3/4, per-pipeline table folders, TSV,
`.pht-*.json`), `measurements.csv` is retired, a run that finished with failed images is
shown, curated and relaunched as such, and the run console reaches #282's Stage 4 and
`--mode measure`.

**Architecture:**
- **One verdict.** `run_finished_with_failures(state, output_dir)` in `sdk_/_run_state.py` is
  the failed-run twin of rule 1. The waiter, the registry, the observer, the badge, the claim
  gate and the mutation guard all ask it, each with the `RunState` it already resolved (PA1).
- **Exit codes by name.** `submit_slurm` and `observe_local_exit` classify by `RunExitCode`.
  A new terminal status `complete_with_failures` carries `failed_images` (PA1).
- **TSV everywhere a writer exists**; the CSV mirror is retired by finalize and by curation
  after the TSV lands, never before (PA3).
- **PR B** factors `resolve_embed_stage_for_pipeline` out of `_cli_embed_stage.py`, so the
  form's visibility, hint and refusals are the CLI's own functions (PB1–PB2), and adds a
  `cli_mode` selector for `--mode measure` (PB3).

**Tech Stack:** Python 3.12, Dash/Flask, polars, pyarrow, zarr v3, click, pytest
(+ Playwright under `PLAYWRIGHT=1`), Slurm. Renderer: `~/.claude/tools/render_md_artifact.py`.

## Global Constraints

- **Exit codes are `phenotypic.sdk_.RunExitCode`** (`sdk_/_exit_codes.py:42-47`): `0`
  COMPLETE, `1` ERROR, `3` COMPLETE_WITH_FAILURES, `4` SUBMITTED, `5` ENDED_UNPUBLISHED,
  `6` WAIT_TIMEOUT; `2` is Click's usage error. No GUI code compares against a literal.
- **Readers rule** (`_gui/CLAUDE.md`, "Importing readers"): a GUI module asking "did this
  run finish (with failures)?" imports from `phenotypic.sdk_` only. The single exception is
  the SLURM observer's `finished_run_outcome` (`_cli/_cli_wait.py`). Every new reader gets a
  row in `docs/source/contrib_guide/tracked_state.md`'s Known consumers table (`:433-450`) in
  the same commit (`:18, 525`).
- **`sdk_` never imports `_cli`** (`tests/unit/sdk_/test_run_state_layering.py`).
- **The verdict never asks `valid_aggregate_snapshot`**; it uses `_valid_aggregate_proof`
  (GUI-tolerant, `sdk_/_run_state.py:1397-1433`).
- **Status vocabulary:** the new literal is exactly `complete_with_failures`; label
  `complete · N failed` (Recent Runs, banner), badge `Run complete · N images failed`
  (warning), colour `--oi-orange-text`. `incomplete` is never persisted (D13).
- **The claim gate** allows only `complete` and a confirmed `complete_with_failures`; it
  refuses the five states in spec PA11's table, and a process tree with failures (N4).
- **CSV retirement:** `measurements.csv` is unlinked only **after** its TSV is published, and
  in finalize inside `commit_guard` when one is given. `_KEPT_CSV_STEMS == {"error_analysis",
  "metadata"}`: `error_analysis.csv` stays CSV (D18) and `metadata.<ext>` is never swept.
  `MEASUREMENTS_CSV`, `measurements_csv_path`, `BundleLayout.mirror_csv` stay defined as legacy
  read names. `_GUI_WRITTEN_PROOF_DESCRIPTORS` keeps `measurements_csv`.
- **The verdict** is `run_finished_with_failures(state, output_dir)`; its config comes from
  `RunState.finalization_digests` and `.process_only_layer`, filled by `resolve_run_state`'s
  one read (spec PA4). An all-failed run is confirmed through the no-proof arm (D16), which
  tests the marker **by path**, refuses a master or mirror Parquet left by another generation,
  and requires a finalizer-closed fence when a fence exists.
- **`--accept-failures` maps only `3` to `0`** (D15). The exit-5 variant prints only when this
  generation published an aggregate proof, and names a store only through
  `excluded_store_names` (the per-image unprojectable predicate, shared with `_advisories`),
  never from "verified minus sources". No marker field is added.
- **D17:** the claim gate cannot see a local CLI run; the docs say so, and no task adds a lock.
- **Suffixes:** canonical `.pht-pipe.json`, `.pht-op.json`, `.pht-cc.json`, `.pht-tune.json`;
  legacy `.json.pht-*` read-only; a plain `pipeline.json` is never removed (#281 A4).
- **Unmounted tune app** (`_gui/CLAUDE.md:20-26`): only reds #1–6's sites and D8's pickers.
- **TSV inputs:** `sdk_.INPUT_TABLE_SUFFIXES` (`{".csv", ".tsv"}`), delimiter by
  `sdk_.delimiter_for`, reading by `sdk_.read_delimited`, writing by `sdk_.write_tsv`.
- **No CLI exit-code value changes.** D14 tightens rule 3 only; its exit-5 line gains the
  variant text fixed in spec PA4 (N3).
- **No edit outside the repository** (D12): the ucr_029/ucr_033 scripts are the user's.
- **Lazy entry points** stay lazy: `tests/unit/ci/test_startup_imports.py`,
  `test_deferred_imports.py`, `tests/unit/gui/shell/test_hub_startup_imports.py`.
- **`MeasurementInfo`**: untouched. **`FilteredMeasurements`**: one-writer swap only.

### The 13 inherited reds (PR A turns them green)

Spec §3.3 is authoritative. #1–6 green in PA4 T1; #7 in PA3 T1 (set becomes
`{"master_parquet", "measurements_tsv", "measurements_parquet"}`); #8–13 in PA2 T1. No
`xfail`, skip, deselect or `-k`. A PR A gate that runs a GUI lane before its owning task has
landed treats the not-yet-fixed members of this set as expected-red; any other red is real.

## Review Focus

Inputs the spec implies but no spec test names. Each is pinned in the task named.

1. **A run whose every image failed** (verified = 0, no aggregate proof possible). It is
   `complete_with_failures` (D16): the verdict returns N, the waiter exits `3`, Recent Runs reads
   "complete · N failed", and the GUI can relaunch it. Any marker, an old master from before a
   `--restart`, a non-finalizer fence or an unresolved image keeps it `failed`. Pinned in PA1 T1
   (`test_an_all_failed_run_is_finished_with_failures`,
   `test_an_all_failed_run_beside_any_marker_returns_none`,
   `test_an_all_failed_restart_beside_an_old_master_returns_none`,
   `test_an_all_failed_run_reads_its_fence`,
   `test_an_all_failed_run_with_an_unresolved_image_returns_none`), PA1 T2
   (`test_an_all_failed_run_exits_3`) and PA1 T3
   (`test_an_all_failed_cli_run_is_claimable_for_retry_failures`).
2. **A #281-era tree curated by the old GUI**: a curated `measurements.csv`, an uncurated
   `measurements.tsv`, a curated `.parquet`. No reader may show the uncurated TSV's rows; the
   next curation rewrites the TSV from the Parquet and retires the CSV. Pinned in PA3 T3
   (`test_an_old_gui_curated_tree_is_repaired_by_the_next_curation`).
3. **A metadata TSV written by Excel or R**: a BOM, a quoted field holding a tab. Browse's
   panel and the run console's preflight read the same rows as the equivalent CSV. Pinned in
   PA4 T2 (`test_an_excel_tsv_reads_like_its_csv`).
4. **A relaunch that turns `complete_with_failures` into `complete`** on the same output
   (`--retry-failures` fixes the last image). The record's `failed_images` becomes `None`,
   the badge reads `Current`, and Recent Runs shows `complete`. Pinned in PA1 T3
   (`test_a_retry_that_fixes_every_image_clears_failed_images`).
5. **A form switched from Measure back to Full** keeps the input folder the user had typed
   and sends `--input` again; switching to Measure never deletes it from state. Pinned in
   PB3 T1 (`test_switching_cli_mode_keeps_the_input_slot`).

## Conventions for every task

- **Shared-env test command** (`run-phenotypic-test` skill: headless Qt, never `-n auto`, no
  streamed output, no `-x`), from the worktree root:

  ```bash
  PT='env QT_QPA_PLATFORM=offscreen uv run pytest -q --no-header -p no:randomly -o addopts= -m "not slow"'
  ```

  The env is synced with `uv sync --group dev --group test-qt --group docs --extra gui
  --extra napari --extra tune --extra topology` (root `CLAUDE.md`); without `tune`/`topology`
  a gate shows 24 false failures.
- **Lanes:** `tests/unit`, `tests/integration`, `tests/gui` (all in `testpaths`); Playwright
  `PLAYWRIGHT=1 $PT tests/e2e/gui -m "not slow and not ci_flaky"`; live CPU SLURM
  `PHENOTYPIC_RUN_LIVE_SLURM=1` (`tests/integration/gui/test_run_console_slurm_live.py`);
  live GPU (PB4, new flag `PHENOTYPIC_RUN_LIVE_SLURM_GPU=1`).
- **Lint:** `uv run ruff check --fix <the paths you changed>`, never bare. `uv run mypy
  src/phenotypic` at each phase gate; no new errors over the branch's baseline.
- **Commits:** explicit paths, never `git add -A`; end with the session's attribution lines.
- **Fixtures are built with the real publishers**, never hand-written JSON (the
  `tests/_output_layout.py` rule, `:448-470`). New shared helpers live in
  `tests/_output_layout.py`; the first task that needs one creates it under
  **Create (helpers)**, later tasks reuse it.
- **Mutation checks** (the `porting-a-reference-algorithm` discipline), by **deletion** of one
  conjunct at a time, never by inversion (plan review C2): the verdict (PA1 T1), waiter rule 3
  (PA1 T2), the claim-gate condition and the N reset (PA1 T3), the cancel/bind paths (PA1 T4),
  the CSV-after-TSV order and the pre-PA8 descriptor (PA3 T1), and the guard's three
  `run_permits_mutation` calls (PA5 T1). The task names each mutant and the test that must fail.
  Back up by full path, mutate, run, **restore from the saved bytes** and hash-check; never
  `git checkout --` (it once wiped a subagent's uncommitted fixes). A mutant that survives
  stops the cluster.
- **Independent-reader check** where a deliverable's bytes change (PA3): read the TSV with
  `pandas.read_csv(sep="\t", dtype=str, keep_default_na=False)` and the Parquet with
  `pyarrow.parquet.read_table`, never with the writer's own reader. No task changes a store.
- **Gates** (root `CLAUDE.md`, "Focused between phases"):
  - per task: the task's test files, seconds to a minute;
  - per phase: the affected surface once, from importers of the changed modules
    (`grep -rl <module> tests/`), on Slurm when over a couple of minutes;
  - end of each PR: one full sharded regression, **after the rebase** for PR A.
- **Gate shape** (precedent `/bigdata/exfab/anguy344/gate-results/final/make_gate.sh`): a
  **detached worktree** at the gated SHA under
  `/bigdata/exfab/anguy344/gate-worktrees/<cluster>-<sha9>`; on `batch` or `intel` (never
  `preempt`; `short` only for an array whose tasks fit 2 h), explicit `--mem`/`--time`,
  `-n $SLURM_CPUS_PER_TASK`; logs under `/bigdata/exfab/anguy344/gate-results/guiplan/<cluster>/`,
  never `/scratch`. The job script `cd`s to the worktree path it was given, **never
  `cd "$SLURM_SUBMIT_DIR"`**. Each shard prints `dirty=0` at start and end. An `afterany`
  cleanup job removes the worktree.
- **Docs builds** are Slurm jobs: `sphinx-build -j "$SLURM_CPUS_PER_TASK" -D
  nbsphinx_execute=never`; read the rendered HTML of each changed page afterwards.
- **GPU jobs** (PB4 only): `slurm-job` skill; measure `sbatch --test-only` on `exfab`
  (`--account=exfab`) and `short_gpu --constraint=gpu_latest` and take the earlier start.
- This plan's Markdown is the source of truth. Re-render after any edit (command at the end).

## Phases

| Phase | Document | Depends on | Tasks | Gate |
|---|---|---|---|---|
| **PR A** | | | | |
| PA1 | [pa1-verdict-and-exit-codes.md](pa1-verdict-and-exit-codes.md) | — | 6 | mutation checks; **deep review** |
| PA2 | [pa2-viewer-tables.md](pa2-viewer-tables.md) | — | 1 | reds #8–13 green |
| PA3 | [pa3-tsv-and-csv-retirement.md](pa3-tsv-and-csv-retirement.md) | PA1 (main line) | 3 | importers of `tests/_output_layout.py`; independent reader; **deep review** |
| PA4 | [pa4-suffixes-and-pickers.md](pa4-suffixes-and-pickers.md) | PA3 (main line) | 2 | reds #1–6 green |
| PA5 | [pa5-curation-with-failures.md](pa5-curation-with-failures.md) | PA4 (main line; needs PA1 T1, T3 and PA3 T3) | 1 | **deep review PA4–PA5** |
| PA6 | [pa6-docs-ledgers-simplify.md](pa6-docs-ledgers-simplify.md) | PA1–PA5 | 3 | docs build; ledger gates; e2e; simplify |
| PA7 | [pa7-rebase-and-merge.md](pa7-rebase-and-merge.md) | PA6; **#281 merged (stop 1)** | 2 | re-verified anchors; reds; live CPU SLURM; full regression |
| **PR B** | | | | |
| PB0 | [pb0-cut-point.md](pb0-cut-point.md) | PR A merged; **#282 on a main with PR A (stop 2)** | 1 | #282 rebase-gate tests |
| PB1 | [pb1-embed-stage-control.md](pb1-embed-stage-control.md) | PB0 | 2 | |
| PB2 | [pb2-stage4-refusals.md](pb2-stage4-refusals.md) | PB1 | 1 | |
| PB3 | [pb3-measure-mode.md](pb3-measure-mode.md) | PB2 | 3 | **deep review PB1–PB3** |
| PB4 | [pb4-ledgers-live-gpu-regression.md](pb4-ledgers-live-gpu-regression.md) | PB3 | 3 | ledger gates; **live GPU (stop 3)**; full regression |

28 tasks in all: 18 in PR A, 10 in PR B (D18 removed PA3's error-analysis task).

## Execution

### Dependency DAG

Arrows are must-finish-before. Shared files force an edge even without an interface edge.

```
Main line (one worktree, sequential):
PA1.T1 (verdict) ─► PA1.T2 (waiter) ─► PA1.T3 (status, rehydration, claim gate)
  ─► PA1.T4 (submit_slurm, callback) ─► PA1.T5 (observer hook) ─► PA1.T6 (badge)
  ─► PA3.T1 (CLI CSV retirement) ─► PA3.T2 (test sweep) ─► PA3.T3 (curation TSV)
  ─► PA4.T1 (suffixes, pickers' predicate) ─► PA4.T2 (TSV pickers)
  ─► PA5.T1 (curation with failures)
  ─► PA6.T1 ─► PA6.T2 ─► PA6.T3 ─► [stop 1] ─► PA7.T1 ─► PA7.T2

Lane V (own worktree, merged after PA1's gate):
PA2.T1 (viewer tables, reds #8–13)

PR B (after PR A merged, [stop 2: #282 on main + PR A]):
PB0.T1 ─► PB1.T1 ─► PB1.T2 ─► PB2.T1 ─► PB3.T1 ─► PB3.T2 ─► PB3.T3 ─► PB4.T1 ─► [stop 3] PB4.T2 ─► PB4.T3
```

**Shared files** (force sequencing):

| File | Tasks |
|---|---|
| `sdk_/_run_state.py`, `sdk_/__init__.py` | PA1 T1, PA4 T1 (`is_pipeline_config_path` goes in `_io_constants.py`, export in `__init__`) |
| `_cli/_cli_wait.py` | PA1 T2 |
| `sdk_/_state_types.py` | PA1 T1 |
| `_cli/_cli_completion.py` | PA3 T1 (`required_paths`) |
| `_cli/_cli_slurm_lifecycle.py` | PA1 T1 (`FENCE_CLOSED_BY_FINALIZER` re-export) |
| `tests/unit/sdk_/test_run_state.py` | PA1 T1 (field set; moved `authorize_measurements`) |
| `_gui/shell/_runs_registry.py` | PA1 T3, T4, PB3 T2 |
| `_gui/run_console/_slurm.py` | PA1 T4 |
| `_gui/run_console/_callbacks.py` | PA1 T3 (`_TERMINAL_RUN_STATUSES`), T4, PA4 T1, T2, PB1 T2, PB2 T1, PB3 T1 |
| `_gui/run_console/_slurm_observer.py` | PA1 T5, PB3 T3 |
| `_gui/_snapshot_status.py` | PA1 T6 |
| `_gui/results_viewer/_output_root.py` | PA3 T3, PA5 T1 |
| `_gui/results_viewer/_curation_labels.py` | PA3 T3 |
| `_cli/_cli_output_manager.py`, `sdk_/_delimited.py` | PA3 T1 |
| `sdk_/_io_constants.py`, `tests/unit/sdk_/test_io_constants.py` | PA3 T1 (docstrings), T2 (sweep), PA4 T1 |
| `tests/unit/gui/test_snapshot_status.py` | PA1 T6, PA3 T2 |
| `tests/_output_layout.py` | PA1 T1 (helper), PA3 T2 (default flip) |
| `_gui/run_console/_state.py`, `_form.py`, `_ids.py` | PB1 T2, PB3 T1 |
| `_cli/_cli_embed_stage.py` | PB1 T1 |
| `phenotypicCLI.py`, `_cli/_cli_reference.py` | PB3 T1 (moved reference helpers) |
| `_gui/FEATURES.md`, `WORKFLOWS.md`, capture script | PA6 T2, PB4 T1 |

**Parallel lanes** (zero file overlap with the line they run beside, each in its own
worktree off the branch tip, merged back by the orchestrator):
- **Lane V:** PA2 T1 runs alongside PA1. Its files (`results_viewer/_measurement_source.py`,
  `_zarr_routes.py`, `_shared/tiles.py`, and four test files: `test_measurement_source.py`,
  `test_measurement_routes.py`, `test_measurement_join_migration_run.py`, `test_zarr_routes.py`)
  touch nothing on the main line. Its new helpers stay in its own test modules (R2-M1).
- **No Lane T** (plan review I8). PA3 runs on the main line after PA1: it shares
  `tests/_output_layout.py`, `tests/unit/gui/test_snapshot_status.py`, `sdk_/_io_constants.py`
  and `tests/unit/sdk_/test_io_constants.py` with PA1 and PA4. Before PA3 T1 lands,
  `publish_aggregate_snapshot` still resolves `measurements.csv` strictly, so PA1 T1's helper
  must write it; it does, through `write_measurements_mirror`'s default.

### Shapes and clusters

"Opus" is the frontier tier, "Sonnet" the mid tier (`execute-plan-orchestration`
references). A reviewer is never weaker than its implementer.

| # | Tasks | Shape | Model | Gate after |
|---|---|---|---|---|
| C1 | PA1 T1, T2 — verdict, waiter | Keystone | Opus, high | light + **mutation check** (5 comparisons, rule 3) |
| C2 | PA1 T3 — registry status, rehydration, claim gate, CSS | Keystone | Opus, high | light |
| C3 | PA1 T4 — `submit_slurm` classification, callback, 0/3 arm | Seam | Opus, high | light + **mutation check** (cancel/bind paths) |
| C4 | PA1 T5, T6 — observer hook, snapshot badge | Seam | Opus, high | **deep: PA1** (`implementation-test-reviewer`) |
| C5 | PA2 T1 — viewer tables, reds #8–13 (Lane V) | Leaf | Sonnet, medium | light |
| C6 | PA3 T1 — CLI CSV retirement, red #7 | Seam | Opus, high | light + independent reader |
| C7 | PA3 T2 — test sweep over the CSV mirror | Sweep | Sonnet, medium; **Opus verify pass** | the importer surface |
| C8 | PA3 T3 — curation TSV | Keystone | Opus, high | **deep: PA3** |
| C9 | PA4 T1 — suffix predicate, reds #1–6, RecipeState, capture-script names | Keystone | Opus, high | light |
| C10 | PA4 T2 — TSV pickers and readers | Sweep | Sonnet, medium | light |
| C11 | PA5 T1 — curation of a run with failures | Seam | Opus, high | **deep: PA4–PA5** |
| C12 | PA6 T1 — docs and exit-5 surfaces | Leaf | Sonnet, medium | docs build (Slurm) |
| C13 | PA6 T2 — ledgers, tutorial text, screenshots | Sweep | Sonnet, medium | `check_features_md.py --strict`, `check_workflows_md.py -v`, e2e |
| C14 | PA6 T3 — simplify | code-simplifier, Opus | — | affected surface |
| C15 | PA7 T1, T2 — rebase, re-verify, final gates, PR | orchestrator; **user approves push/PR** | — | full sharded regression |
| C16 | PB0 T1 — cut point and rebase gate | orchestrator | — | rebase-gate tests |
| C17 | PB1 T1, T2 — resolver refactor, control, visibility, hint | Keystone | Opus, high | light |
| C18 | PB2 T1 — refusal probe | Seam | Opus, high | light |
| C19 | PB3 T1–T3 — measure mode | Keystone | Opus, high | **deep: PB1–PB3** |
| C20 | PB4 T1 — ledgers, tutorial, screenshots | Sweep | Sonnet, medium | ledger gates, e2e |
| C21 | PB4 T2 — live GPU launch | orchestrator; **user approves (stop 3)** | — | the live assertions |
| C22 | PB4 T3 — simplify, full regression, PR | code-simplifier (Opus), then orchestrator | — | full sharded regression |

**Rules for every cluster.**
- The agent runs its own tests and lint, and commits its own work with explicit paths.
- It reports each commit SHA, the failing-test names it saw before each fix, and the exact
  commands it ran.
- The orchestrator re-reads the disk at each gate and never verifies while an agent is
  mid-edit.
- Reviewers write their report to
  `docs/superpowers/reports/2026-11-24-gui-dvc-followups/<what-it-reviewed>.md`; that file is
  their only write.
- Before C1 is dispatched, a `plan-reviewer` reads this plan against the spec and the code;
  its critical findings are fixed first.

### Stop points that need the user

1. **The D9 rebase step (PA7).** PR A is done and reviewed on #281's branch. #281 is held
   until then; the user merges #281, and PR A is rebased and merged back to back. No push, no
   PR, no merge without the user's go.
2. **PR B's cut point (PB0).** Only after #282 is rebased or merged onto a `main` that holds
   PR A, and #282's rebase-gate tests pass.
3. **The D7 live GPU launch (PB4 T2).** It spends GPU time; it runs only on the user's
   go-ahead.
4. **Any design question** a task meets that the spec does not settle: the cluster stops
   before the task that would consume the answer.

## Spec coverage

| Spec item | Task |
|---|---|
| §1 sequencing, D5, D9 | PA7 T1–T2, PB0 T1 |
| G1, PA1 exit codes (0/3/4/2/timeout; `cancel_inactive`; `SlurmFinishedWithoutSubmission`; cancel race) | PA1 T4 |
| G2, PA4 status `complete_with_failures`, `failed_images`, CSS, Recent Runs, banner, no-dashboard text | PA1 T3 |
| PA4 verdict `run_finished_with_failures(state, output_dir)`, shared helper, N4 | PA1 T1 |
| PA4 waiter composition, `RunSnapshot.finished_with_failures`, D14, N3 exit-5 variant | PA1 T2 |
| PA4 observer hook, `finished_run_outcome`, terminal_incomplete phase (Minor 16) | PA1 T5 |
| PA4 badge arm through the currency check (D3, I4) | PA1 T6 |
| PA11 claim gate (D10), N7 message | PA1 T3 |
| G3, PA2 `declared_measurement_table_path`; route v2 refusal | PA2 T1 |
| G4, PA3 the 13 reds | PA2 T1 (#8–13), PA3 T1 (#7), PA4 T1 (#1–6); PA7 T2 re-run |
| G5, PA8 `_cli/` and `sdk_/` changes, retirement under the guard | PA3 T1 |
| PA8 test sweep (56 files: 30 naming the CSV ∪ 28 callers; 62 importers run) | PA3 T2 |
| PA5 curation TSV, failed-write unlink, fingerprints, `FilteredMeasurements`, `_schema_cache`, import-walk (N6) | PA3 T3 |
| PA5 error analysis TSV v2 | **deferred (D18)**: spec §8, no task |
| D15 `--accept-failures` strict; the exit-5 line names stores through `excluded_store_names` | PA1 T1 (helper), T2 (line) |
| D16 the all-failed no-proof arm | PA1 T1, T2, T3 |
| D17 the local-CLI window | PA6 T1, T2 (docs, tutorials) |
| G6, PA6 suffixes, reds #1–6, RecipeState, capture script, e2e fixtures | PA4 T1 |
| G5b, PA9 pickers, readers, `ExpectedVsDetectedCount`, reference requirement, MergeMetadata | PA4 T2 |
| G2b, PA10 curation of a run with failures (D2, D11) | PA5 T1 |
| PA7 ledgers, tutorial text (D4, D11, N7) | PA6 T2 |
| PA8 docs list, exit-5 surfaces (N3), Known consumers rows | PA6 T1 (rows also land with each reader's task) |
| §6 simplify, gates, regression | PA6 T3, PA7 T2, PB4 T3 |
| §1 step 4, I2, R13 #282 rebase obligation | PB0 T1 |
| G7, PB1 control | PB1 T2 |
| PB2 resolver refactor | PB1 T1 |
| PB2 visibility, no hidden flags (I10), hint with adoption (I9), wording | PB1 T2 |
| G8, PB3 refusals | PB2 T1 |
| D6, PB5 measure mode: selector, form rules, validate, static refusals | PB3 T1 |
| PB5 registry `cli_mode`, local exits, N (I3) | PB3 T2 |
| PB5 observer outcomes, D13 terminal `failed`, `terminal_status` read; CliRunner argv test | PB3 T3 |
| PB4 ledgers, frames, tutorial | PB4 T1 |
| D7 live GPU launch | PB4 T2 |
| §8 follow-up issue | drafted in the spec; not a task |

## Spec findings from planning

Recorded here for the orchestrator.

1. **D14 and `--accept-failures`.** Settled by **D15**: it stays strict, and the exit-5 line
   names the excluded stores (PA1 T2).
2. **Local measure N (PB5).** The spec's [call] defers where a local pass's own failure count
   lives on disk. PB3 T2 Step 1 locates it first and stops the cluster if no per-pass count
   exists (the spec's fallback wording then applies).
3. **The re-run test needs a Refresh** (re-review C4 note, already in spec PA10). PA5 T1
   writes it that way.

## Revision 2 (plan review)

Revised 2026-11-25 against `reports/2026-11-24-gui-dvc-followups/plan-review.md` and the user's
decisions D15–D18 (spec §10). Each finding, and where it is fixed:

| Finding | Fix |
|---|---|
| **C1** An all-failed run decided by accident; tests that accept either answer | D16: spec PA4 step 4, the no-proof arm. PA1 T1's helper skips the aggregate when nothing verified; T1 pins three all-failed outcomes, each a single answer; T2 pins exit `3`; T3 pins claimability. The exit-5 variant needs a published proof (T2 `test_the_variant_needs_a_published_proof`) |
| **C2** Inversion mutants; tests that could not isolate step 2 or the inventory digest | PA1 T1: deletion mutants, one per conjunct, in a table; `unresolved_stems=`; `test_a_new_image_that_failed_returns_none`; the source-set/count pair named as inseparable |
| **I1** No config source; one helper cannot serve rule 1 | `RunState.finalization_digests` and `.process_only_layer` from `resolve_run_state`'s one read (spec PA4, why: one read, no torn state); two helpers, `_identity_digests_match` and `_source_set_matches`; step 2 over `state.images`. PA1 T1, `test_the_verdict_reads_neither_state_nor_config` |
| **I2** No slot for N in the observer and the CAS | PA1 T3 (`compare_and_set` keyword, reset to `None`), PA1 T5 (`_Observation`, `_apply`) |
| **I3** Rehydration and dead-owner release lose N | PA1 T3: `_rehydrated_status` returns N; both consumers persist it; two tests |
| **I4** The run console's pipeline picker | PA4 T1: `run_console/_form.py:58-73`, both builder wrappers, one parametrised test |
| **I5** More consumers of the 0/3 result | PA1 T4: `_SlurmCompletion.finished`, the banner (`:2672`), `shell/_app.py:215, 681`, a rendered-banner assertion |
| **I6** The error-analysis v2 early return | D18: PA3 T4 removed; the sites go to spec §8 |
| **I7** A vacuous pre-PA8 proof test | PA3 T1: inject a real `measurements_csv` descriptor; mutation drops it from `_GUI_WRITTEN_PROOF_DESCRIPTORS` |
| **I8** Lanes V/T overlap the main line | Lane T removed; PA3 and PA4 run on the main line; the DAG and the shared-files table name the four files |
| **I9** The flip test passes for the wrong reason; no guard mutants | PA5 T1: flip only the verdict (a success record, no re-finalize); the message names the completion; deletion mutants for the three calls; PA3 T1 mutant for CSV-after-TSV |
| **I10** Review Focus 4 cannot fail | PA1 T3: the same record, through `compare_and_set` and `observe_local_exit(0)` |
| **I11** The rebase step | PA7: `git rebase --onto origin/main 36b9e840e feat/gui-dvc-followups`; `gh pr view 281`; main has moved (#284); push from the root after `ExitWorktree(keep)`; the plan is committed before C1 (this revision) |
| **I12** R14 | D17: spec R14 corrected; PA6 documents it |
| **Minor 1** Excluded-store fixture | PA1 T1: the omitted stem leaves the master and mirror too |
| **Minor 2** `clear_terminal_failure` | Removed: a success record outranks a failure |
| **Minor 3** Process fixture | PA1 T1: no master, mirror or aggregate |
| **Minor 4** Which generation exit 4 binds | PA1 T4: asserts `THIS_GENERATION` |
| **Minor 5** Call counts | PA1 T4: every mock-based test asserts `cancel_generation`'s count |
| **Minor 6** Render coverage | PA1 T3 `test_recent_runs_renders_every_status` |
| **Minor 7** The `commit_guard=None` path | PA3 T1: both paths pinned |
| **Minor 8** Sweep size; `run_is_complete` test refs | PA3 T2: 56 files; PA5: four test refs, 11 lines |
| **Minor 9** Review Focus 2 asserted columns | PA3 T3: rows and keys; `_curation_labels.py:284`, `_filtered_state.py:255` |
| **Minor 10** `MergeMetadata` default | PA4 T2: `_instantiate(..., output_dir=)` (`analysis/_callbacks.py:1070`) |
| **Minor 11** `RunMode`, notes type | PB1 T1: `_cli_preflight.RunMode`, `list[str]` |
| **Minor 12** Test API names | PB1/PB3: `run_state_from_json`, `tracked_generation_counts` |
| **Minor 13** `measure_reference_refusal` placement; survivor rows | PB3 T1 moves the helpers into `_cli/_cli_reference.py`; PB4 T1 adds the `_gui/CLAUDE.md` rows; shared-files table corrected |
| **Minor 14** Measure details | PB3: the log reader for exit 1; both-snapshot and empty-output tests; a fence-free CliRunner fixture |
| **Simplicity** `TUNING_SPEC_TYPED_SUFFIXES` | PA4 T1: derived from `TUNING_CONFIG_SUFFIXES` |
| **Simplicity** PA5's rename | Kept: the spec asks for `run_permits_mutation`; the four test refs are listed |
| **D15–D18** | Spec §10; PA1 T1–T3 (D15, D16), PA6 (D17), PA3 (D18) |

## Revision 3 (plan re-review)

Revised 2026-11-25 against `plan-review.md`, "Re-review of revision 2" (`9a31a5361`). Each
finding, and where it is fixed:

| Finding | Fix |
|---|---|
| **R2-C1** `excluded_work_ids` measured a gap, not an exclusion | Dropped from the spec and plan; `publish_aggregate_snapshot` is unchanged. PA1 T1: `excluded_store_names(images)` factored out of `_advisories` (`_run_state.py:1680-1695`), shared by both; PA1 T2: the line names a store only through it, otherwise the gap line. The fixture's `omit_from_proof` is replaced by `unprojectable_stems` (a real unprojectable store, `authorize_measurements(projectable=False)`) and `late_stems` (the rolling-input gap, which must name no store). `test_a_master_that_merely_predates_an_image_claims_no_exclusion` stays green unedited |
| **R2-I1** The pinned `RunState` field set | PA1 T1 Files: `test_run_state.py:56-75` and `tracked_state.md:43-53` change deliberately, with the docstring line |
| **R2-I2** The no-proof arm | Spec PA4 step 4 and PA1 T1: the marker tested by path (`[valid]`, `[wrong_version]`, `[empty_outputs]`); no master or mirror Parquet from another generation (`test_an_all_failed_restart_beside_an_old_master_returns_none`, and its confirmed twin with no deliverables); the killed-retry case pinned both ways (`test_a_killed_retry_that_verified_nothing_stays_confirmed`, `test_a_retry_that_verified_one_image_returns_none`); spec N7 and PA7's tutorial line corrected |
| **R2-I3** The PA5 flip test | PA5 T1: three stems, completion asserted `failed`, only the verdict flips; `test_a_fresh_verdict_whose_n_differs_refuses` added. PA1 T1's retry test relabelled and given three stems, so it isolates the source set |
| **R2-I4** The waiter keeps no `RunState` | PA1 T2: `RunSnapshot.excluded_stores` from the same pass; `_unpublished_line` takes it; `test_the_excluded_names_come_from_the_same_pass` |
| **R2-M1** Lane V helpers; the count | PA2: its helpers stay in its own test modules; README: four test files |
| **R2-M2** Fixture names | PA1 T1: the inventory is `FIXTURE_STEMS` ∪ named stems in `FIXTURE_DATASET` (`"plate"`); tests use `a`/`b`/`c` and expect `plate/c` |
| **R2-M3** The SLURM all-failed waiter case | PA1 T2 `test_an_all_failed_slurm_run_exits_3`, through the real finalizer path |
| **R2-M4** One cancelled all-failed run, two answers | Spec PA4 step 4: the no-proof arm requires a finalizer-closed fence when one exists; `FENCE_CLOSED_BY_FINALIZER` moves to sdk_; `test_an_all_failed_run_reads_its_fence` |
| **R2-M5** "under `commit_guard`" | Spec PA8: "when one is given" |
| **R2-M6** PB3's helper move | PB3 T1: grep for monkeypatch targets on the rebased tree first; re-export only if one exists |
| **I7 note** `inject_descriptor` hand-edits the proof | PA3 T1's test says so (no writer produces a pre-PA8 proof after T1); PA1 T1's `write_marker` invalid kinds carry the same note |

## Render

From this folder, with the files in phase order:

```bash
uv run --no-project python ~/.claude/tools/render_md_artifact.py \
  README.md pa1-verdict-and-exit-codes.md pa2-viewer-tables.md \
  pa3-tsv-and-csv-retirement.md pa4-suffixes-and-pickers.md \
  pa5-curation-with-failures.md pa6-docs-ledgers-simplify.md pa7-rebase-and-merge.md \
  pb0-cut-point.md pb1-embed-stage-control.md pb2-stage4-refusals.md \
  pb3-measure-mode.md pb4-ledgers-live-gpu-regression.md \
  -o ../../artifacts/2026-11-24-gui-dvc-followups/plan.html \
  --title "GUI Follow-ups Plan" \
  --eyebrow "Build plan · PhenoTypic #283 / #282 · 2026-11-24"
```
