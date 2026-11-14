# DVC and Embeddings Implementation Plan

**Status:** draft for review · 2026-11-12 · revised per plan-review.md
**Spec:** `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`
**Base:** `main` @ `3248a9173`, branch `feat/dvc-embeddings-refactor` (PR #281)

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a PhenoTypic CLI run usable as one DVC stage by automation, with
per-pipeline tables in every OME-Zarr store, and DINOv2 crop and grid embeddings
stored under the pipeline that produced them.

**Architecture:** Six phases, each shippable on its own:
- P0 and P1 rename files: typed config suffixes, and CSV to TSV.
- P2 adds the run contract automation reads: exit codes, `phenotypic-wait`,
  `metrics.json`, `schema.json`, byte-stable deliverables, and a DVC how-to page.
- P3 moves a store's tables into one folder per pipeline behind a single
  resolver, and every reader goes through it.
- P4 adds the embedding producer.
- P5 lets process mode run measurements, adds the in-place process run, and
  rolls the result out to AutoConvertRaw-GC.

**Tech Stack:** Python 3.12, pydantic v2, polars/pandas/pyarrow, zarr v3 (OME-Zarr
0.5), click, torch + transformers (the `foundation` extra), pytest, Slurm. The
render tool is `~/.claude/tools/render_md_artifact.py`.

## Global Constraints

- Canonical typed suffixes are `.pht-pipe.json`, `.pht-op.json`, `.pht-cc.json`
  and `.pht-tune.json`. The old `.json.pht-*` forms are **read-only legacy**.
  Plain `.json` stays legacy-readable for pipelines and tune specs.
- A file rename never changes file bytes, so `pipeline_sha256` is unchanged.
- TSV is UTF-8, one header row, `\t` separator, `\n` line ends, with `"`
  quoting for fields containing tab, newline or quote. Generated files only;
  inputs accept `.csv` or `.tsv`.
- Two generated files stay CSV until #283, because GUI code writes them
  (spec B9): `error_analysis.csv`, and the curated mirror `measurements.csv`,
  which finalize writes beside `measurements.tsv`.
- `deliverables/metadata.<ext>` is a byte-for-byte copy of the user's input,
  keeping its extension. Nothing ever rewrites it.
- DVC outs are `<out>/deliverables`, `<out>/results` and `<out>/.phenotypic`,
  all `persist: true`. `cache.type` must be `reflink,copy`; `hardlink` and
  `symlink` are unsupported.
- Exit codes, fixed here and used everywhere (`phenotypic.sdk_.RunExitCode`):

  | Code | Name | Meaning |
  |---|---|---|
  | `0` | `COMPLETE` | finished, no terminal failures |
  | `1` | `ERROR` | a runtime or config error (today's code) |
  | `2` | — | a refusal raised as a Click usage error; Click's own, unchanged |
  | `3` | `COMPLETE_WITH_FAILURES` | finished, some images failed terminally |
  | `4` | `SUBMITTED` | submitted to SLURM, not finished |
  | `5` | `ENDED_UNPUBLISHED` | `phenotypic-wait`: no job alive, and this submission published nothing |
  | `6` | `WAIT_TIMEOUT` | `phenotypic-wait` gave up after `--timeout-seconds` |

- Store tables are `tables/<n>-<sha12>/{measurements,metadata,embeddings/<model>/{crops,grid}}/`.
  - `<n>` is a **store-local call-order counter**: a new folder gets 1 + the
    highest number among the store's run folders and its `upstream` entries.
    Numbers are never reused, so two folders never share one.
  - `<sha12>` is the first 12 hex of the pipeline file's sha256, or `unknown`
    when no digest is known.
  - The descriptor is `attributes.phenotypic.tables` with `schema_version: 2`;
    each run entry carries `order`.
  - The same pipeline replaces its own folder; a different one adds a folder.
- `embedding` columns are Arrow `FixedSizeList<float16>[D]`. They are never in
  master, mirror, splits or any TSV.
- `PROCESS_LAYER_SEMANTICS_REVISION` goes from 4 to 5 (`_cli_failure_tracker.py:214`).
- `MeasureDinoEmbedding` fields, in order, with defaults: `size="base"`,
  `version=2`, `revision=None`, `regions="objects"`, `grid=None`, `pooling="crop_cls"`,
  `crop_px=224`, `crop_pad_frac=0.25`, `mask_background=False`,
  `dtype="float16"`, `batch_size=64`, `device="auto"`.
- `MeasurementInfo` members get `label`/`desc` only, never `bio_desc`.
- Heavy imports (torch, transformers) stay inside functions. The startup-import
  guards in `tests/unit/ci/` must stay green.
- `sdk_` never imports `_cli`. `tests/unit/sdk_/test_run_state_layering.py`
  enforces this.
- **No file under `src/phenotypic/_gui/` changes** (GUI deferred to issue #283).
  - Every name the GUI imports from `phenotypic.sdk_` stays defined, and every
    `sdk_` function the GUI calls keeps its name and signature. P1 T1's AST
    guard (`tests/unit/sdk_/test_gui_sdk_imports.py`) enforces this.
  - Other GUI breakage is accepted until #283. The GUI tests it turns red
    **stay red**: no `xfail`, no skip, no edit to a GUI test.
  - **Expected-red GUI tests** are listed below. A phase gate treats exactly
    that list as expected-red. Any other red GUI test is a real failure.
- **Merge gate:** #281 does not merge to `main` until #283 has fixed every
  known GUI break (spec §2.2), including the run console's reading of exit code
  `4` (`_gui/run_console/_slurm.py:826`) and every expected-red test below.

### Expected-red GUI tests

| Phase | Test | Why it goes red |
|---|---|---|
| P0 | `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable` (:98-99) | `_gui/tune/_validation.py:61` accepts only `CONFIG_SUFFIX_TUNING`, so a legacy `.json.pht-tune` spec is refused |
| P0 | `tests/unit/gui/tune/test_setup_authoring.py::test_write_authored_setup_spec_preserves_existing_spec_search_space` (:147, spec at :171) | `_gui/tune/_setup_authoring.py:356` loads a `.json.pht-tune` file as a pipeline once the canonical suffix changes |
| P0 | `tests/unit/gui/tune/test_setup_authoring.py::test_existing_spec_replaces_scorer_only_when_requested` (:190, spec at :213) | same cause |
| P0 | `tests/integration/gui/tune/test_setup_view.py::{test_existing_spec_controls_preserve_seed_budget_and_storage, test_existing_spec_trial_override_preserves_authored_strategy, test_existing_grid_spec_never_emits_trial_override}` | `_gui/tune/_validation.py:61`, as above (found at the C1 gate) |
| P1 | `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact` | asserts the proof's `required_outputs` is exactly the three pre-TSV names; spec B9 adds `measurements_tsv` (found at the C4 gate) |
| P3 | `tests/unit/gui/results_viewer/test_measurement_routes.py`: every test whose `route` fixture store (`:100`, written with `save2zarr(measurement_table=…)`) is read through `_measurement_source`, and `test_an_undeclared_column_never_opens_the_parquet` (`:205` unlinks the v1 path) | `_gui/results_viewer/_measurement_source.py:107-108` stats `store / MEASUREMENT_TABLE_RELATIVE_PATH`, which a v2 store does not have |
| P3 | `tests/unit/gui/results_viewer/test_measurement_source.py`: every test over the `run` fixture (`:88`), and `test_a_rewritten_table_is_read_fresh` (`:249`) | same cause |
| P3 | `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py`: every test over the migrated-run fixture (`:105` reads the v1 path), and `test_the_four_table_less_stores_contribute_nothing_and_raise_nothing` (`:209`) | same cause |

The P3 rows name files and the fixture that makes them red, because almost every
test in those files reads a store through the v1 path. A test in those files
that stays green is fine. A red test anywhere else in `tests/unit/gui`, `tests/integration/gui` or
`tests/gui` is a real failure.

Checked and expected to stay green:
- `tests/unit/gui/tune/test_setup_authoring.py:60`, because the GUI writes the
  literal `.setup.json.pht-tune` (`_setup_authoring.py:535`).
- The GUI SLURM tests (`tests/unit/gui/run_console/test_slurm*.py`), because
  they mock the submitter's return code.
- The zarr-route tests that hand-write `tables/measurements/table.parquet` and
  expect `404`.
- The GUI tests that read `measurements.csv` (`test_schema_cache.py`,
  `test_snapshot_status.py`, `tests/gui/results_viewer/test_complete_run_fixture.py`),
  because finalize keeps writing it (B9) and P1 T2 updates the shared
  `tests/_output_layout.py` helper to the new proof keys.

## Review Focus

Inputs and conditions the spec implies but no feature test would hit by
accident. Each one is pinned by a test in the task named.

1. **A tree written before this change is opened by the new code.** That covers
   old `.json.pht-pipe` names, `measurements.csv` and v1 flat `tables/`.
   Readers must still work; forward modes must refuse a v1 store with a pointer
   to `--mode migrate`; and a v1 tree that read `complete` must still read
   `complete` after migrate. Pinned in P0 T1, P1 T4 and P3 T5.
2. **A staged GPU run re-runs Stage 1 over a store that already holds crop
   embeddings.** The all-zeros Stage-1 objmap must not permanently mark the
   entry stale, and the table-less Stage-1 save must not drop the folder.
   Pinned in P3 T2 (`test_stale_is_recomputed_not_latched`,
   `test_table_less_save_keeps_its_own_folder`).
3. **The same `--output` is run again with an edited pipeline, or measured
   again.** A full run with A then A′ must give folders `1-` and `2-`; measure
   with the same pipeline must leave one folder and the same row count; measure
   with another pipeline must add a folder and leave the first byte-identical,
   and a later recompile must keep the second pipeline's measurements. Pinned
   in P3 T2, T3 and T4.
4. **The SLURM finalizer dies (OOM or cancel) and the lifecycle fence stays
   `active`, or a measure resubmission leaves the previous run's proof
   current.** `phenotypic-wait` must keep waiting while any job is live, then
   exit `5` unless this submission published. Pinned in P2 T2.
5. **An in-place process run meets a store whose root changes under it** (an
   AutoConvertRaw-GC reprocess, or a second writer). The run must rebuild once
   from the new root or fail the image, never swap a root built from stale
   bytes, and never rewrite a pixel. Pinned in P5 T3
   (`test_root_changed_before_swap_rebuilds_once_then_fails`).

## Conventions for every task

- **Test command** (`run-phenotypic-test` skill: headless Qt, no `-n auto`, no
  streamed output):

  ```bash
  PT='env QT_QPA_PLATFORM=offscreen uv run pytest -q --no-header -p no:randomly -o addopts= -m "not slow"'
  eval $PT tests/unit/sdk_/test_io_constants.py
  ```

- **Lint:** `uv run ruff check --fix <the paths you changed>`, never bare.
- **Commits** use explicit paths, never `git add -A`, and end with the
  attribution lines from the session's system reminder.
- **Gates** follow root CLAUDE.md:
  - per task: the task's own test files;
  - per phase: the affected surface once, derived from importers of the changed
    modules (`grep -rl <module> tests/`);
  - end of plan: one full sharded regression on Slurm, via
    `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`
    or the 24×8 gate recipe.
  - Every gate that runs `tests/unit/gui` or `tests/gui` compares its red set
    with the **Expected-red GUI tests** table for the phases landed so far.
- **Store changes** also get an **independent reader** check: zarr-python
  `zarr.open_group` + `pyarrow.parquet.read_table`, never only PhenoTypic's own
  reopen (`working-with-ome-zarr` skill).
- **Shared test helpers.**
  - Plan tests call helpers from `tests/_tiny_run.py`. Fixtures defined there
    are re-exported by `tests/conftest.py` with an explicit import list.
  - Each task's Files list names the helpers it adds under **Create
    (helpers)**. The first task that uses a helper creates it; later tasks
    reuse it.
  - Existing fixtures reused as they are: `synth_one_level_input`,
    `simple_pipeline_json`, `make_exec_config` (`tests/unit/cli/conftest.py`);
    the `submissions` SLURM stub (`tests/unit/cli/test_cli_slurm_submission_summary.py:67-118`,
    moved to `tests/unit/cli/conftest.py` by P2 T1); `tests/_output_layout.py`;
    and `file_sha256` (`_cli/_cli_failure_tracker.py:86`).
- **This plan's own Markdown is the source of truth.** Re-render the plan
  artifact after any edit (global CLAUDE.md, "Specs and plans").

## Phases

| Phase | Document | Depends on | Tasks |
|---|---|---|---|
| P0 | [p0-config-suffix-flip.md](p0-config-suffix-flip.md) | none | 3 |
| P1 | [p1-csv-to-tsv.md](p1-csv-to-tsv.md) | none | 4 |
| P2 | [p2-dvc-run-contract.md](p2-dvc-run-contract.md) | P0, P1 | 6 |
| P3 | [p3-tables-v2.md](p3-tables-v2.md) | none | 6 |
| P4 | [p4-measure-dino-embedding.md](p4-measure-dino-embedding.md) | P3; P4 T4 also P2 T3 (`build_run_metrics(embeddings=)`) | 4 |
| P5 | [p5-process-measurements-acr.md](p5-process-measurements-acr.md) | P0, P3, P4 | 5 |

P3 is the largest and the riskiest phase, because 27 modules read today's flat
table layout. "none" means no task dependency; the phases still share files
(see Execution), so they run in sequence on one branch.

## Execution

**The phases are not file-disjoint.** These hot files are shared:

| File | Phases |
|---|---|
| `sdk_/_io_constants.py` | P0 P1 P2 P4 |
| `sdk_/__init__.py` | P0 P1 P2 P3 |
| `sdk_/_run_state.py` | P1 P2 P3 |
| `_cli_output_manager.py` | P0 P1 P3 P4 |
| `_cli_readme_generator.py` | P0 P1 P2 P4 |
| `phenotypicCLI.py` | P0 P1 P2 P3 P5 |
| `_cli_migrate.py` | P0 P1 P3 |
| `_cli_finalize_run.py` | P2 P3 P4 |
| `_measurement_tables.py` | P3 P4 P5 |
| `_image_io_handler.py` | P3 P4 |
| `_cli_completion.py` | P1 P2 |
| `_cli_preflight.py` | P2 P5 |
| `_cli_process_single.py` | P3 P4 |
| `_cli_process_only.py` | P3 P5 |
| `_cli/CLAUDE.md` | P1 P2 P5 |
| `tests/_tiny_run.py`, `tests/conftest.py` | every phase |
| `tests/_output_layout.py` | P1 |

So the main line runs **sequentially in one worktree**. Two side lanes have zero
overlap and may run alongside it:
- **Lane A:** P0 T3, in the AutoConvertRaw-GC repo.
- **Lane B:** P2 T6 Step 1, the C8 DVC timing on a bounded subset of the
  maresca copy. It is a read-only Slurm job whose numbers land in the how-to
  page later.

**Clusters.** One agent per cluster. "Opus" is the frontier tier and "Sonnet"
the mid tier. A reviewer is never weaker than its implementer.

| # | Tasks | Shape | Model | Gate after |
|---|---|---|---|---|
| C1 | P0 T1, T2 steps 1–4 | Keystone | Opus | light |
| C2 | P0 T2 step 5, the literal sweep | Sweep | Sonnet | light; the Opus phase review covers it |
| C3 | P0 T3 (Lane A) | Leaf | Sonnet | light |
| C4 | P1 T1, T2 | Keystone | Opus | light |
| C5 | P1 T3, the snapshots (provenance invariant) | Seam | Opus | light |
| C6 | P1 T4 | Sweep | Sonnet | **deep: P0+P1** (implementation-test-reviewer) |
| C7 | P2 T1, exit codes | Seam | Opus | light |
| C8 | P2 T2, `phenotypic-wait` | Keystone | Opus | light |
| C9 | P2 T3, metrics and schema | Keystone | Opus | light |
| C10 | P2 T4, T5: byte stability and layout guards | Seam | Opus | light |
| C11 | P2 T6, the how-to page (uses Lane B's numbers) | Leaf | Sonnet | light |
| C11a | P2 T2a, the waiter's launch form (`--`), and the page's one-command stage | Seam | Opus | **deep: P2** |
| C11b | P2 T2b: waiter and page fixes from the P2 review | Seam | Opus | light |
| C11c | P2 T2b: CLI exits, fence retirement, test gaps from the P2 review | Seam | Opus | light |
| C11d | P2 T2b: measure failures journalled; metrics.json agrees with the exit code | Seam | Opus | light |
| C12 | P3 T1, T2: v1 fixtures, resolver, writer, carry, stale | Keystone | Opus | light |
| C13 | P3 T3, the measure rewrite | Keystone | Opus | light |
| C14a–d | P3 T4 reader sweep, by subsystem: (a) finalizer/aggregation, (b) per-image writers and staged, (c) recompile, (d) migrate, state and the `sdk_` readers | Sweep, consistency-critical | Opus | light each |
| C15 | P3 T5, v1 refusal and migrate conversion | Seam | Opus | light |
| C16 | P3 T6, upstream reference | Keystone | Opus | **deep: P3**, then a code-simplifier pass on the P3 layer before P4 builds on it |
| C17 | P4 T1, T2 | Keystone | Opus | light |
| C18 | P4 T3, `MeasureDinoEmbedding`; a fact-checker on the `crop_px` and `crop_pad_frac` tune ranges | Keystone | Opus | light |
| C19 | P4 T4, CLI wiring and deliverables | Seam | Opus | **deep: P4** |
| C20 | P5 T1, T2, process mode | Seam | Opus | light |
| C21 | P5 T3, in place | Seam | Opus | light, then **deep: P5**, then the full sharded regression on Slurm, the simplify pass, and the regression again |
| C22 | P5 T4, the AutoConvertRaw-GC rollout | Seam | Opus | light. **Blocked on the user's rig1 grid edges** |
| C23 | P5 T5, pilot and backfill decision | orchestrator and user | — | — |

**Rules for every cluster.**
- The agent runs its own tests and lint, and commits its own work with explicit
  paths.
- It reports each commit SHA and the failing-test names it saw before the fix.
- The orchestrator re-reads the disk at each gate, and never runs verification
  while an agent is mid-edit.
- An open design question found at any gate stops the run until the user
  answers.
