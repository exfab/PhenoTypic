# DVC and Embeddings Implementation Plan

**Status:** draft for review · 2026-11-12
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
- P5 lets process mode run measurements and rolls the result out to
  AutoConvertRaw-GC.

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
- `deliverables/metadata.<ext>` is a byte-for-byte copy of the user's input,
  keeping its extension. Nothing ever rewrites it.
- DVC outs are `<out>/deliverables`, `<out>/results` and `<out>/.phenotypic`,
  all `persist: true`. `cache.type` must be `reflink,copy`; `hardlink` and
  `symlink` are unsupported.
- Exit codes, fixed here and used everywhere (`phenotypic.sdk_.RunExitCode`):

  | Code | Name | Meaning |
  |---|---|---|
  | `0` | `COMPLETE` | finished, no terminal failures |
  | `1` | `ERROR` | refused, or a usage/config/runtime error (today's code) |
  | `2` | — | Click usage error; Click's own, unchanged |
  | `3` | `COMPLETE_WITH_FAILURES` | deliverables published, some images failed terminally |
  | `4` | `SUBMITTED` | submitted to SLURM, not finished |
  | `5` | `ENDED_UNPUBLISHED` | `phenotypic-wait`: no job alive and nothing published |
  | `6` | `WAIT_TIMEOUT` | `phenotypic-wait` gave up after `--timeout-seconds` |

- Store tables are `tables/<seq>-<sha12>/{measurements,metadata,embeddings/<model>/{crops,grid}}/`.
  The descriptor is `attributes.phenotypic.tables` with `schema_version: 2`.
  `<sha12>` is the first 12 hex of the pipeline file's sha256.
- `embedding` columns are Arrow `FixedSizeList<float16>[D]`. They are never in
  master, mirror, splits or any TSV.
- `PROCESS_LAYER_SEMANTICS_REVISION` goes from 4 to 5 (`_cli_failure_tracker.py:214`).
- `MeasureDinoEmbedding` fields, in order, with defaults: `size="base"`,
  `version=2`, `regions="objects"`, `grid=None`, `pooling="crop_cls"`,
  `crop_px=224`, `crop_pad_frac=0.25`, `mask_background=False`,
  `dtype="float16"`, `batch_size=64`, `device="auto"`.
- `MeasurementInfo` members get `label`/`desc` only, never `bio_desc`.
- Heavy imports (torch, transformers) stay inside functions. The startup-import
  guards in `tests/unit/ci/` must stay green.
- `sdk_` never imports `_cli`. `tests/unit/sdk_/test_run_state_layering.py`
  enforces this.
- **No file under `src/phenotypic/_gui/` changes** (GUI deferred to issue #283).
  Every name the GUI imports from `phenotypic.sdk_` stays defined, including
  `MEASUREMENTS_CSV` and `MEASUREMENT_TABLE_RELATIVE_PATH` as legacy names.
  Every `sdk_` reader the GUI calls keeps its signature. The existing GUI tests
  must stay green.
- **Merge gate:** #281 does not merge to `main` until #283 has fixed the run
  console's reading of exit code `4` (`_gui/run_console/_slurm.py:826`).

## Review Focus

Inputs and conditions the spec implies but no feature test would hit by
accident. Each one is pinned by a test in the task named.

1. **A tree written before this change is opened by the new code.** That covers
   old `.json.pht-pipe` names, `measurements.csv` and v1 flat `tables/`. Readers
   must still work, and writers must refuse the v1 store with a pointer to
   `--mode migrate`. Pinned in P0 T2, P1 T4 and P3 T5.
2. **A staged GPU run re-runs Stage 1 over a store that already holds crop
   embeddings.** The all-zeros Stage-1 objmap must not permanently mark the
   entry stale. Pinned in P3 T2 (`test_stale_is_recomputed_not_latched`).
3. **`--mode measure` with a different pipeline over a store from a full run.**
   It must add a second run folder and leave the first byte-identical. Pinned in
   P3 T3.
4. **The SLURM finalizer dies (OOM or cancel) and the lifecycle fence stays
   `active`.** `phenotypic-wait` must exit `5`, not wait forever. Pinned in
   P2 T2 (`test_dead_finalizer_with_active_fence_ends_unpublished`).
5. **A grid cell extends past the image edge on a different rig.** The image
   must fail with the cell index and the image size named, and no table may be
   written for it. Pinned in P4 T3.

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
- **Store changes** also get an **independent reader** check: zarr-python
  `zarr.open_group` + `pyarrow.parquet.read_table`, never only PhenoTypic's own
  reopen (`working-with-ome-zarr` skill).
- **This plan's own Markdown is the source of truth.** Re-render the plan
  artifact after any edit (global CLAUDE.md, "Specs and plans").

## Phases

| Phase | Document | Depends on | Tasks |
|---|---|---|---|
| P0 | [p0-config-suffix-flip.md](p0-config-suffix-flip.md) | none | 3 |
| P1 | [p1-csv-to-tsv.md](p1-csv-to-tsv.md) | none (parallel with P0) | 4 |
| P2 | [p2-dvc-run-contract.md](p2-dvc-run-contract.md) | P0, P1 | 6 |
| P3 | [p3-tables-v2.md](p3-tables-v2.md) | none (parallel with P0–P2) | 6 |
| P4 | [p4-measure-dino-embedding.md](p4-measure-dino-embedding.md) | P3 | 4 |
| P5 | [p5-process-measurements-acr.md](p5-process-measurements-acr.md) | P0, P3, P4 | 4 |

P0, P1 and P3 touch disjoint files and can be worked in parallel. P3 is the
largest and the riskiest, because 27 modules read today's flat table layout.
