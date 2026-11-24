# CLI Execution (`_cli/`)

The batch-processing engine behind `python -m phenotypic`. `../phenotypicCLI.py`
(at the package root, not in `_cli/`) parses options into an `ExecutionConfig`
(`_cli_types.py`, a mutable `@dataclass`), then
`create_execution_strategy(config, output_manager)`
(`_cli_execution_strategies.py`) dispatches to one strategy.

## Execution strategies (the dispatch)

`create_execution_strategy` picks by `(is_slurm_mode, measure_only,
process_only_layer, pipeline_requires_gpu, embed_stage)`:

| Strategy | When | File |
|---|---|---|
| `LocalParallelStrategy` | local run with no `GpuDetector` (joblib), `--mode measure` included; under `--embed-stage gpu` it runs the local Stage 4 after its loop (`run_local_stage4`), in republish mode for measure | `_cli_execution_strategies.py` |
| `AutonomousSLURMStrategy` | SLURM run that embeds inline and has no `GpuDetector` (array + drip-feed chunk chain), measure and process included | `_cli_execution_strategies.py` |
| `StagedGpuStrategy` | **local** forward run with a `GpuDetector`, or `--mode process --layer objmap`; under `--embed-stage gpu` it runs the local Stage 4 after Stage 3 | `_cli_staged_strategy.py` |
| `StagedSlurmStrategy` | **SLURM**: `uses_staged_controller` — a forward run with a `GpuDetector` (`process_only_layer is None`), or any run whose `embed_stage` resolved `gpu` (#282), with or without a detector, `--mode measure` included | `_cli_staged_slurm.py` |

A "staged GPU run" (`uses_staged_gpu_strategy`) = the pipeline contains a
`GpuDetector` (`pipeline_requires_gpu`), **at any depth in the operation tree**,
outside `--mode measure`. On SLURM that is no longer the only way onto the
staged controller: `uses_staged_controller` (`_cli_execution_strategies.py`) is
`uses_staged_gpu_strategy(config) or (is_slurm_mode and embed_stage == "gpu")`,
so a SLURM run with no detector that embeds in a GPU Stage 4 runs the
controller with phases `["stage3", "stage4"]` (see **Stage 4 on SLURM** below).
Locally a run with no detector stays on `LocalParallelStrategy`, which runs
Stage 4 itself. Non-objmap process-layer exports always stay on the
Local/Autonomous strategies, since `--mode process` embeds inline (R1).

**`--mode process --layer objmap` runs the post-detector op chain.**
`_export_objmap_layer` (`_cli_staged_strategy.py`) runs after Stages 1–2 and
substitutes a `ReplayDetector` at `plan.gpu_path` inside `post_pipeline`, then
applies that residual pipeline — so the export is *the objmap the pipeline
produces*, not the detector's raw output. For a **nested** detector the raw
array is one branch of a composite and not an objmap at all, so the previous
behaviour exported the wrong thing rather than merely an unrefined one. Three
details that are load-bearing rather than incidental:

- It reads `load_stage2_raw`, **never the store** — Stage 2 writes nothing into
  the store, so the store's objmap here is still Stage 1's zeros and a store
  read would silently export an all-zeros PNG for every image.
- The apply is wrapped in `continuing_provenance_application` and **no**
  `provenance_success_sink`. Stage 1 left the application `"staged"`, which is
  not terminal, so a plain apply at owner-depth 0 raises; and a success sink
  would write to the store *after* the image success marker, invalidating the
  descriptor the marker just recorded. Do not "restore" the sink, and do not
  reach for `set_provenance_status(image, "in_progress")` — that status is also
  non-terminal and raises the very error it looks like it prevents.
- The change of meaning is fenced in the work id:
  `PROCESS_LAYER_SEMANTICS_REVISION` (`_cli_failure_tracker.py`, currently `5`;
  2 → 3 is the per-image figures, see **Per-image figures** below; 3 → 4 is
  single-channel integer inputs normalised to float32 `[0, 1]` at construction;
  4 → 5 is process mode running `meas` into the store, see **Process mode runs
  `meas`** below)
  rides beside `process_format` in the process-only branch of
  `processing_configuration_digest_from_values`, **not** in the base payload —
  a base placement would cold-start every in-flight `full` and `measure`
  continuation for no correctness gain. It is one integer for all layers, so a
  bump also invalidates `--layer gray` continuations; invalidating too much is
  safe, so that is a cost, not a bug.

## The run preflight, and the read-only/mutating split

Spec `docs/superpowers/specs/2026-09-24-cli-preflight/design.md`.

**`phenotypic_cli` has a read-only half and a mutating half, in that order.**
The read-only half parses and validates options, scans the inputs, runs the
refusals that cannot be skipped (the GPU placement refusal, the grid-preset
refusal, the process-mode lost-table refusal in **Process mode runs `meas`**
below, the in-place refusals in **In-place process runs** below, the refusal of a
`--restart`/`--overwrite` that would delete the run's own inputs, the
`--metadata`, `--bit-depth` and `--gpu-slurm` parses, and the two
reference-metadata refusals in **Reference metadata** below), loads the pipeline,
runs the **run preflight**, and ends at the `--dry-run` exit. Only then does
the mutating half run: the `--restart` clear, the `--overwrite` rmtree,
`mint_run_identity`, and everything after. **A new step that writes, deletes
or submits goes below the `--dry-run` exit**, or `--dry-run` (and the GUI's
Validate) stops being a promise that nothing under `--output` changed. Pinned
by `tests/unit/cli/test_cli_preflight_ordering.py`.

**The run preflight** (`_cli_preflight.py`) is a tuple of checks, `CHECKS`,
each `(PreflightContext) -> list[PreflightFinding]`. A check reads; it never
writes, never decodes a pixel, and never imports `torch` or `micro_sam` in the
submitting process. Severity follows reach: an error when every image is
affected, a warning listing the subset otherwise. A check that raises becomes a
`PF-CHECK-CRASHED` warning and never blocks a run. `--skip-validation` skips
the whole validation block (execution-config validation, the pipeline load
check, and the run preflight) and none of the refusals listed above. Every `FindingCode` needs a `HINTS` entry
and a row in the table in `docs/source/tutorials/pages/cli_batch_processing.md`;
`tests/unit/test_docs_preflight_codes.py` fails on drift. Stage 4 (#282) adds one
code, `PF-EMBED-RESOURCES` (`check_embed_loader_memory`: the Stage-4 GPU
profile's memory against its loader workers' need for the largest plate, a
warning, run only for a SLURM Stage-4 run). `PF-EMBED-SHARD-AMORTIZE` was built
and then dropped with its hint, docs row and tests (U13): it assumed a 750 s
cold load and fired falsely on the live smoke, whose loads took 39.7–297.9 s.
The R1–R3 `--embed-stage` refusals are refusals, not findings, and
`--skip-validation` does not skip them.

**Mode scoping.** A requirement check iterates `operations_in_scope(context)`,
never the whole operation tree: `MODE_SLOTS` says which root slots each mode
executes (`full`: all; `process`: `ops`, `meas`; `measure`: `meas`, `post`,
`filters`, `model`), and a pipeline nested in `ops` contributes only its own
`ops`. In `process` mode the walk also drops each root `meas` entry that is
not a no-objects embedder (`_skipped_in_process_mode`), because process mode
skips it: the walk is what runs. Walking everything would refuse a `process`
run over a measurer it never runs.

**Operations declare their own requirements** through
`BaseOperation.preflight_requirements()` (`abc_/_requirements.py`), not
through a checker-side table, so a new operation cannot fall out of date with
the checks.

## Process mode runs `meas` (`_cli_process_meas.py`)

#281 spec E4, user decision 2026-11-18: **run what fits in a store, skip the
rest.** `MODE_SLOTS["process"]` is `ops` + `meas`, and process mode never runs
`post`/`filters`/`model` or builds deliverables.

- **Runs (v1: embedders only):** each `meas` entry that is a
  `MeasureEmbedding` with `requires_objects` false, i.e. a grid embedder
  (`regions="grid"`, named or not), per `runs_in_process_mode`.
  `MeasureFeatures.requires_objects` is `True`; `MeasureEmbedding` overrides
  it. `process_single_apply_only_core` calls `embed_in_process_mode` after
  `apply()`, inside the same provenance and reference context. It passes the
  payloads through `write_process_only_layer(embeddings=)` to `_save_store`,
  so the run folder lands in the `.part` before `_consolidate_store_part` and
  the store stays write-once and consolidated. `upstream=` is passed as
  before. `measure()` is never called.
- **Skipped, never silently:** every other entry, which is what process mode
  always did with `meas`. One warning line names each, with its reason:
  - it needs objects (an ordinary measurer, an objects embedder): `meas:size
    skipped: process mode stores no objects`. A process store holds no
    object map, so a table keyed on `Object_Label` would point at labels the
    store cannot resolve;
  - it needs no objects but is not an embedder: `meas:foo skipped: process
    mode stores only embedding tables`. `measure()` is never called in
    process mode, so it has nowhere to go. None ships today; the wording
    keeps a future one from being dropped silently.

  Both groups share one line, separated by `; `. The CLI prints it once, in
  the read-only half, as `Warning: <line>.`; the core logs it per image.
  Measurer-backed `PlotImage` figures in a process store still work, because
  the figure binding recomputes in `inspect()`.
- **Refused:** a runnable entry with a flat output (`--process-format tiff`,
  `--layer objmap`, `--layer detect_mat`), because its table would be lost.
  `_refuse_process_meas_it_would_lose` (`phenotypicCLI.py`) raises it as a
  `click.UsageError` (exit 2), beside the grid-preset refusal: un-skippable,
  above `--restart`, `--overwrite`, `mint_run_identity` and `--dry-run`.
  `process_single_apply_only_core` runs the same check,
  `process_meas_refusal`, before it reads the image, and raises
  `ProcessMeasRefusedError` (a `ValueError`, never wrapped as a per-image
  failure), because AutoConvertRaw-GC calls the core directly.
  `write_process_only_layer` refuses `embeddings=` with a flat format too.
- **Continuation:** `PROCESS_LAYER_SEMANTICS_REVISION` went 4 → 5, so a
  process tree resumed across the change re-derives its stores rather than
  reusing stores that lack the tables it now writes.
- Pinned by `tests/unit/cli/test_process_mode_measurers.py` and
  `tests/unit/cli/test_process_store_tables.py`.

## In-place process runs (`_cli_process_in_place.py`)

#281 spec E7. `--mode process` with `--output` resolving to the same
directory as `--input` (`is_in_place_process`, `_cli_types.py`;
`ExecutionConfig.in_place`) adds the pipeline's run folder to each input
store where it is. Any other `--output` writes new stores as before. Only the
**equality** case of the process overlap refusal is lifted: an output
strictly inside the input, or an input strictly inside the output, stays
refused.

- **The write is the measure rewrite's transaction**, `_rewrite_store_tables`
  (`sdk_/_measurement_tables.py`), with two options measure mode never
  passes. `expected_root_sha256` refuses a root whose bytes are not the ones
  the write was built from, both when the rewrite reads it and again
  immediately before `promote_store`, raising `StoreRootChangedError` and
  removing the part. `consolidate` re-consolidates the part after the root,
  dropping the stale `consolidated_metadata` first; a root that already
  carries `consolidated_metadata` is re-consolidated without being asked,
  measure mode included, and an unconsolidated one given `False` is written
  as before (P5 review I1). A third option,
  `clear_tables=False`, keeps every run folder as carried, for a pipeline
  that only draws figures. The plan (`_in_place_plan`) writes this pipeline's
  folder through `write_run_entry` (the next number, or its own folder
  replaced when stale), the figure run, and the journal, append-only
  (`_appended_journal`): the root's earlier applications byte-for-byte, plus
  the new `process` application without its wall-clock fields (I1).
  Pixel chunks stay hard links, so they keep their inodes, and every other
  run folder and the `upstream` block are carried byte for byte.
- **`process_in_place_core`** is what both process callers dispatch to in
  place: `LocalParallelStrategy._process_single_local_apply_only` and the
  SLURM per-image worker (`_cli_process_single.py` `main`). It runs
  `embed_in_process_mode`, never `measure()` (spec E4), and checks the pixels
  after `apply()`: the exported layer must have the `level0_digest` (the
  `objmap_digest` recipe) it had right after the read, so a legacy integer
  `gray` store, which `imread` normalises to float32, is supported (user
  decision 2026-11-18, Q3). A grid payload bound to the read layer is then
  re-bound to the stored array's digest (`_bind_to_stored`), which is what
  the entry certifies.
  Otherwise it raises `InPlacePixelsChangedError`, a terminal failure that
  writes nothing, which catches a pixel-changing op the `ImageCorrector`
  refusal misses. If the root changes before the swap, it rebuilds once from
  the new root. A second `StoreRootChangedError` is raised **unwrapped**, so
  the image fails this invocation (exit `3`) without a terminal record, and a
  plain re-run retries it.
- **Continuation.** A store already holding this pipeline's folder with
  each of its embedders' payload keys present and `stale: false` does no
  work (`_owned_run_is_current`; an empty folder is not current, M1). A root
  unreadable at that check is a per-image `PerImageScientificError`, not an
  error retried for ever (M3). The skip warning is logged once per process
  (M5). The work
  id's input digest is `in_place_input_digest`: the `file_sha256` tree digest
  (`store_tree_sha256`) minus the root `zarr.json`, `tables/` and `figures/`,
  under its own domain tag, so the run's own write never changes its work id.
  `input_sha256_for` is its one dispatcher. Selection, the SLURM worklist
  (`_cli_slurm_array_scripts.py`) and the SLURM worker's identity and
  expected-input check all call it. `processing_configuration_digest_from_values`
  adds `in_place: true` to the process payload, and only when true, so a
  copy run's digest is unchanged and an in-place run never reuses its
  continuation.
- **Refused in the read-only half** by `_refuse_in_place_run`
  (`phenotypicCLI.py`), above `_refuse_run_inputs_the_run_would_delete`,
  `--restart`/`--overwrite`, `mint_run_identity` and `--dry-run`, and
  outside `--skip-validation`. Each refusal is one usage-error line, from
  `in_place_refusal`, in this order: `--overwrite`; a layer or format that
  writes no store; an `ObjectDetector` anywhere (any slot, any
  depth, via `walk_operations`); an `ImageCorrector` in `ops` at any depth
  (`ops` only, never a slot); an `--input` naming one store or file; a
  legacy root-level machine-state name at the input root (`progress/`,
  `processing_state.json`, `processing_events.log`, read from
  `machine_state_restart_targets`), which the legacy migration would move
  into `.phenotypic/` for `--restart` to delete (P5 review C1, Q2); an
  orphaned `.<stem>.ome.zarr.<token>.trash` with no live store beside it,
  naming the `mv` that restores it (I5, `transaction_debris`); then, per
  store, reading one root each and stopping at the first failure, a store
  that is a symlink or resolves outside `--input` (Q4), a non-PhenoTypic
  input, a full run's bundle store whose root carries `image_class` (Q1), a
  v1 tables layout (pointing at `--mode migrate`), and a `--layer` that is
  not the primary series. The per-store checks see the images the run would
  process, `--image-manifest` applied. Leftover `.part` directories are
  named in one warning and never deleted (M6).
- **No mode deletes an input through a restart target.**
  `_refuse_run_inputs_the_run_would_delete` also scans `--input` when
  `--restart` has targets, and refuses an image inside one: with `--input`
  equal to `--output`, a dataset folder named `progress` is the legacy
  root-level `progress/` although `--input` itself lies in no target (C1). The "already contains
  files" check is skipped in place. Pinned by
  `tests/unit/cli/test_process_in_place.py` and
  `test_an_in_place_refusal_comes_before_the_mutating_half`.
- **`--restart` is allowed in place** (user ruling 2026-11-18). It clears
  only the non-preserved `.phenotypic/` machine state under the input root,
  keeps the terminal-failure records, and never touches a store:
  `machine_state_restart_targets` lists only `.phenotypic/` entries, so
  `_refuse_run_inputs_the_run_would_delete` admits `--input` there and
  refuses only a run input stored inside the machine state, as anywhere.
  It is the remedy for a **different pipeline over the same stores**, which
  the normal resume check refuses (`Cannot continue - Pipeline contents
  changed`, exit `1`); in place that message is followed by a line naming
  `--restart`. Pinned by `test_restart_in_place_touches_no_store` (every
  store file keeps its inode and bytes) and
  `test_another_pipeline_in_place_needs_restart`.
- **Only store exports are in place for a worker.** `ExecutionConfig.in_place`
  and the SLURM worker also require `writes_process_store` (`--layer
  rgb|gray --process-format zarr`). The CLI refuses any other export with
  equal paths before a worker exists, so this changes nothing a CLI run can
  reach; it keeps a hand-built config with equal paths (the staged objmap
  tests) on the copy path.
- **`.phenotypic/` lives under the input root.** `scan_directory_structure`
  does not skip dot-directories by name. It finds no input there because
  `.phenotypic/` holds no image file at its top level, and a transaction's
  `.<stem>.ome.zarr.<uuid>.part` is a dotted name and holds none either.
  `test_machine_state_lives_under_the_input_root_and_is_not_an_input` pins
  this.

## The grid-preset refusal (`_refuse_grid_preset_unlike_cli_grid`)

In `--mode full`, `--nrows`/`--ncols` shape each `GridImage`
(`resolve_grid_shape`), while a pipeline that presets **both** `nrows` and
`ncols` and holds no `GridFinder` in `meas` makes `measure()` add
`CenteredAutoGridFinder` with the *preset* shape. Two shapes mean two grids
that disagree on colonies' `Grid_*` values, and `measure()` raises
`MeasurementConflictError` for every image (before `887e2533e`, 2026-11-04,
it silently dropped those colonies instead). So flags unlike such a preset are a `click.UsageError`
beside the GPU placement refusal: un-skippable, above `--overwrite` and the
`--dry-run` exit. A pipeline that fails to load is left to the validation.
Not refused, because `measure()` adds no finder for them: a preset of only
one of `nrows`/`ncols`, or none. Not covered, and reported per image by the
failure tracker instead: an explicit grid finder in `meas` (then the preset
lays out nothing, and the finder is what must match the image), and
`--mode measure`, whose stores carry the grid the forward run recorded.
Pinned, both halves, by `tests/unit/cli/test_cli_grid_preset_refusal.py`.

## Reference metadata (`_cli_reference.py`)

Spec `docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md` §5.2. An
operation mixing in `RefMetadata` (`SubtractBlank`) reads per-image values from
the `--metadata` table through a `ReferenceContext`. Workers never receive the
table as an argument; they read a run-level plan.

- **Only the operations the mode runs count.** `reference_operations_in_scope`
  is the preflight's mode walk (`operations_run_in_mode`, the function behind
  `operations_in_scope`). A `SubtractBlank` inside a measurer that needs
  objects needs the table in `full` mode and not in `process` mode, which
  skips such measurers. One inside a `meas` entry process mode runs (an
  embedder that needs no objects, #281 P5) needs it in both.
- **Process mode uses `--metadata` only when those operations exist.** It is
  then snapshotted byte-for-byte to `.phenotypic/reference_metadata.<ext>`
  (the table's own suffix, `.csv` or `.tsv`)
  (preserved across `--restart`), a continuation may omit it, and a table
  inside `--output` is refused like any run input. Otherwise the old
  "`--metadata` is ignored in --mode process" warning still prints.
- **Two un-skippable refusals in the read-only half**, above `--restart`,
  `--overwrite`, `mint_run_identity` and `--dry-run`:
  - `_refuse_unusable_reference_table`: `--overwrite` without `--metadata`
    (the snapshot it would fall back to is deleted); no `--metadata` and no
    snapshot (`[PF-REF-NO-TABLE]`); a `--metadata` that fails
    `metadata_csv_payload` (not a `.csv`/`.tsv`, or bytes pandas cannot parse --
    the very check `snapshot_reference_metadata` runs after `--overwrite`,
    so the two cannot disagree); or a table, given or snapshotted, that
    `ReferenceContext` cannot read. `--metadata` is a `.csv` or `.tsv` table
    in every CLI mode (the `--metadata` Click callback refuses any other
    suffix with exit 2), although a notebook `ReferenceContext` also reads
    `.parquet`.
  - `_refuse_measuring_reference_pipeline`: `--mode measure` over a pipeline
    whose measurers read reference metadata. Measure mode plans nothing, never
    touches the manifest and carries no reference digest in its work-ids, so
    re-measuring such a pipeline is not supported (user decision, 2026-10-06);
    the message points at `--mode full`.
- **Startup publishes the plan before any work-id.** `publish_reference_inputs`
  runs in `_prepare_incremental_startup` and writes
  `.phenotypic/reference_manifest.json` (table path and sha, reader kwargs,
  each dataset's resolved reference images, each image's digest, and —
  schema 2 — `reference_files`, each planned file's sha). The worker context
  is narrowed with `planned_digests=reference_files`, so
  `ReferenceContext._load` refuses a blank whose bytes changed since planning
  (`ReferenceImageChangedError`, raised inside the apply and wrapped there);
  `worker_reference_context` converts exactly that cause, and nothing else,
  into a bare `ReferencePlanStaleError` at its exit. A schema-1 manifest has
  no `reference_files`, so every file load is refused and the same command
  re-plans. Only the root `zarr.json` of an OME-Zarr blank is hashed. An image
  whose planning failed gets `"unplanned:<reason>[:<values digest>]"`, so a
  changed cause re-derives it rather than matching a terminal failure recorded
  for the old one. Measure mode returns without touching the manifest. A
  pipeline that reads no reference metadata removes a stale one.
- **A blank value is a file name or a file path.** A bare name (no `/`, not
  absolute) is searched for in the image's input folder, as before. A path
  (`is_file_path_value`) names that file, relative to the **cwd** when not
  absolute; startup resolves it once in the submitting process, and the
  manifest's `images`/`reference_files` hold only the absolute path, so a
  SLURM worker (another cwd) never re-resolves it. The GUI builder preview
  confines path blanks to its image root (`file_path_guard`); Python and the
  CLI read any path.
- **The reference plate is no special case** (user decision, 2026-10-07). A
  blank naming or resolving to the image's own file is planned, hashed into
  `reference_files` and digested like any blank, and the op subtracts it, so
  the plate's `detect_mat` is all zero. In full mode that plate then fails
  measurement with an ordinary `NoObjectsError` terminal record, like any plate
  with no colonies. There is no `PF-REF-SELF`, no `unplanned:self`, and no
  `ReferenceLookupError(reason="self")`. An empty blank cell is still
  `unplanned:ambiguous` / `PF-REF-AMBIGUOUS` and the op's `reason="null"`.
- **Every apply site enters `worker_reference_context`.** Seven call sites:
  `process_single_image_core` (`apply_and_measure`),
  `process_single_apply_only_core`, Stage 1's `pre_pipeline.apply`, Stage 2's
  `_apply_stage2_prefix` (inside its `try`), Stage 3's replay `apply` **and**
  its `measure` (a measurer's private op reads the table too), and
  `StagedGpuStrategy._export_objmap_layer`. A new site added bare fails every
  image with `RefMetadataUnavailableError`. Guards:
  `test_every_worker_core_enters_the_reference_context` (a source tripwire)
  and the behavioural runs in `tests/unit/cli/test_cli_reference_e2e.py`.
- **One identity per image, before the apply.** The manifest is the first
  work-id input that a later invocation rewrites in place, so a worker reads
  the image's digest **once** (`work_identity_for_image`, or `image_reference_digest`
  in the SLURM worker), publishes and records failures under that identity,
  and hands the same digest to the context as a `ReferencePin`. The context
  refuses a manifest whose digest for the image has since changed. Never
  recompute a work-id after the apply on these paths. **The staged engine
  holds one identity per image for the whole run, not one per stage**:
  Stage 3 certifies Stage 1's pixels, so a stage pinned to a freshly computed
  identity would still apply under one plan and certify under another. The
  local `StagedGpuStrategy` computes it in Stage 1 (`_stage1` returns it) and
  hands it to Stage 2, Stage 3 and `_export_objmap_layer`; the staged SLURM
  submitter computes it before submission (`staged_manifest_entry`) and
  stores `reference_digest` beside `work_id` in each `StagedManifestEntry`,
  which every stage worker pins (`_reference_pin`). An entry written before
  that field existed loads with `None`, which pins "no reference plan": under
  a reference manifest such an entry is refused, never applied unpinned. Each
  staged core takes the pin as `reference_pin=`. The staged controller's
  `_classify_stage2` lists an entry whose pin no longer matches the manifest
  as `terminal`, not `retryable`: another GPU round would refuse the same pin.
  That list records nothing, so the next invocation re-plans the image.
- **`ReferencePlanStaleError` is never terminal.** It is raised for a pin
  mismatch, a table whose bytes no longer match the manifest, or a reference
  file whose bytes no longer match `reference_files`. It is a
  `RuntimeError`, not a `ReferenceContextError`, and every scientific wrapper
  re-raises it unwrapped (beside `MemoryError`), so no terminal record is
  written and the next run re-attempts the image without `--retry-failures`.

## Staged GPU engine

When a CLI pipeline contains a `GpuDetector`, the **CLI** (not `ImagePipeline`)
splits it at the detector boundary (`split_pipeline_at_gpu` in
`_cli_pipeline_split.py` →
`StagePlan{pre_pipeline, gpu_path, gpu_detector, stage2_prefix, post_pipeline}`)
and runs three content-defined stages. The per-image stage cores live in
`_cli_staged_workers.py` and are shared by both staged strategies:

1. **Stage 1** `stage1_preprocess_core` — apply pre-detector ops → publish the
   staged OME-Zarr store `results/<ds>/zarr/<stem>.ome.zarr/` (objmap included,
   as zeros, because `valid_staged_store` requires it).
2. **Stage 2** `stage2_detect_core` — load the input layer (store **read-only**),
   apply `plan.stage2_prefix` to a provenance-detached copy when the detector is
   nested, run the resident detector, and drop its **Stage-2 signal**: the
   retained **raw** detector output at
   `.phenotypic/progress/stage2_raw/<ds>/<slot>/<stem>.npy` — the name is
   unchanged but the file is a **compressed** archive (`np.savez_compressed`),
   because the dataset-wide barrier keeps every image's raw output live at
   once: ~32 MB each uncompressed is ~1 TiB over a 33,923-image run.
   `load_stage2_raw` reads a pre-compression bare `.npy` too, so a run
   interrupted across that change replays instead of re-inferring — then a
   consumable
   **token** at
   `.phenotypic/progress/stage2_done/<ds>/<slot>/<stem>.json`
   (`_cli_stage2_token.py`,
   both atomic temp+`os.replace`; raw first, so a crash between them leaves no
   "done" signal). **Stage 2 does not write into the store** — only the final
   store needs third-party interop, and an in-store write would be visible to
   the uncached crop route as raw pre-`drop_frame_background` labels.
3. **Stage 3** `stage3_merge_measure_core` — substitute a `ReplayDetector`
   (`_cli_replay_detector.py`) at `plan.gpu_path` inside `post_pipeline`, so the
   **raw** array is written through the real detector's `_write_object_output`
   and the enclosing operation runs normally. Never the store's own objmap:
   Stage 3 re-promotes over it, so a retry would refine already-refined labels.
   Then apply post-ops + `measure(apply_post=False)`, re-promote the store, and
   consume the signal — **token first, then the raw array** (mandatory).
4. **Stage 4** (#282, only when `--embed-stage` resolved `gpu`) — every
   `MeasureDinoEmbedding` of the run, on a GPU, after the store is final. Stage
   3 (or the single-pass core, in a run with no `GpuDetector`) then **defers**
   `embed()` and the per-image record: it writes the store with its
   measurements and `stages.stage3`, and no embeddings and no record. See
   **Stage 4: GPU embedding** below.

**Stage 1 reads an input exactly as a single-pass run does.** Every forward
reader builds its kwargs from the same two helpers in `_cli_utils.py`: the
caller's `input_read_settings(bit_depth, detect_mode)`, then, inside the core
that holds the pipeline, `image_read_kwargs(...)`, which adds a `GridImage`'s
shape from `resolve_grid_shape` (flags, then the preset, then 8 x 12). The
callers are `LocalParallelStrategy` (full and process), the single-pass SLURM
worker (`_cli_process_single.py` `main`), `StagedGpuStrategy`, and the staged
SLURM worker, which reads the four flags from `job_metadata.json`
(`JobMetadataKey.NROWS`/`NCOLS`/`BIT_DEPTH`/`DETECT_MODE`, written by
`_write_staged_job_metadata` before Controller 0 is submitted) rather than from
its command line, and refuses to run when that file is missing or
unreadable (a key an older submitter never wrote still reads as unset).
Before `fe8863c81` (2026-11-04), Stage 1 applied no grid shape at all (every
staged `GridImage` was 8 x 12) and the SLURM worker also dropped `--bit-depth`
and `--detect-mode`. **Nothing fences a staged run resumed across that fix**:
the work id carries the same flags before and after, so its Stage-1 stores and
completed images are reused as they were read. A staged `GridImage` run whose
grid is not 8 x 12, or a staged SLURM run given `--bit-depth` or a
`--detect-mode` other than `gray`, must be re-run with `--restart` (or
`--overwrite`). Pinned by
`tests/unit/cli/test_staged_stage1_reads_like_single_pass.py`.

### Stage 4: GPU embedding (#282)

Spec `docs/superpowers/specs/2026-11-12-gpu-embedding-staging/design.md`;
user-facing guide `docs/source/how_to/pages/embeddings.md`.

- **Where it runs is resolved once** (`_cli_embed_stage.py`,
  `resolve_embed_stage_for_config`), in the read-only half, from
  `--embed-stage {auto,gpu,inline}`. `auto` is `gpu` only on SLURM with a GPU
  profile (a `GpuDetector` in `--mode full`, or `--gpu-slurm`); every local
  `auto`, `--mode process`, an in-scope embedder pinned to `device="cpu"` (U9)
  and an embedder that is not a `MeasureDinoEmbedding` (O3, `_has_stage4_encoder`)
  resolve `inline`. The refusals R1 (`gpu` with `--mode process`), R2 (`gpu` over
  a CPU pin or an unstageable embedder) and R3 (a continuation's explicit
  `--embed-stage` that differs from the recorded one) are one `click.UsageError`
  line each, outside `--skip-validation`, above `--restart`, `--overwrite`,
  `mint_run_identity` and `--dry-run`. A continuation under `auto` **adopts** the
  recorded stage (`adopt_recorded_embed_stage`, U12), and refuses a SLURM `auto`
  that would adopt `gpu` with no GPU profile, since its Stage 4 would go out on
  the CPU profile. The value is `processing_state.json` config `embed_stage` and
  `JobMetadataKey.EMBED_STAGE`; it is never in the work-id digest, and a state
  without it reads `"inline"`. `defers_embeddings(config)` is the one predicate.
- **Startup** (`prepare_stage4_environment`, after the refusals and before any
  strategy): `CUBLAS_WORKSPACE_CONFIG` is `setdefault` to `:4096:8` (a preset
  `:16:8` is kept; any other value is exit `1` locally, and anywhere once CUDA is
  already initialised in the process), and a **local** `gpu` run checks CUDA
  (`stage4_device` → `check_cuda_usable`, which also checks the torch build has
  kernels for the device, `has_kernels_for`) and exits `1` with one line before
  Stage 1. That check imports torch, so a `--dry-run` with the flag pays it.
- **The signal.** One atomic Parquet per embedder key,
  `.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet`
  (`_cli_stage4_signal.py`, `DIR_STAGE4_PAYLOAD`), whose key-value footer
  carries `{work_id, run_id, root_sha256, key, descriptor}`. A signal is valid
  only at the **live** store root and for its own key (`classify_stage4_signals`);
  a stale one is deleted and that key re-embedded. Validity is per key, so a
  round killed after one backbone keeps that backbone's keys.
- **The commit order, per image, never another** (`_cli_stage4_worker.py`):
  signals → the store transaction `replace_run_embeddings(store, run_id,
  payloads, expected_root_sha256=<the signals' root>)` (root last, only that
  run's `embeddings` block rewritten, `stale` recomputed from the live store's
  digests) → the record, published once → the signals consumed. One commit
  thread overlaps the GPU work. A `StoreRootChangedError` leaves the image
  pending, never failed.
- **Backbone-major** (`plan_stage4`): the `(meas key, embedder)` pairs group by
  `BackboneKey(version, size, revision, compute_dtype)`; one model is resident at
  a time, and embedders sharing a backbone share one load. One DataLoader per
  backbone (`spawn`, not persistent, identity `collate_fn`) cuts and resizes the
  crops on the CPU through the embedder's own `prepare_regions`, with
  `resolve_local_worker_count(config.n_jobs, n) - 1` workers locally and
  `resolve_local_worker_count(-1, n) - 1` on a SLURM shard.
- **Determinism** (`configure_determinism`, `_batch_encoder.py`): cuDNN
  deterministic and no benchmark, TF32 off for matmul and cuDNN,
  `set_float32_matmul_precision("highest")`,
  `use_deterministic_algorithms(True, warn_only=True)`.
- **The physical batch** `m = min(batch_size, pow2floor((0.8·total GPU memory −
  weights − 1 GiB) / per_sample))`, from `physical_batch_for` only, per queue
  geometry, never from free memory. A GPU OOM halves that queue's `m` once for the
  shard; a second OOM, or `m = 1`, raises `Stage4OutOfMemory`. Any failed forward
  pass poisons the `BatchEncoder` and fails the shard. Each entry's `compute`
  block records every pass size its rows ran at (`physical_batch`, or `null` plus
  `physical_batches`). Batch invariance holds **within one GPU model and
  physical batch** (S1), measured bit-identical on Ada, A100 and Blackwell
  (`reports/2026-11-12-gpu-embedding-staging/batch-invariance.md`).
- **Node or model state is never an image's failure:** no CUDA, a GPU or host
  OOM, a poisoned encoder, a refused licence (`Stage4DeviceError`,
  `Stage4OutOfMemory`, `Stage4ShardError`, `MemoryError`) journal nothing. A
  SLURM shard returns `1` and the next round retries; a local Stage 4 exits `1`
  with "run the same command again". Neither is a transient `OSError` reading
  the store at commit, or a changed root: the image stays pending with its
  signals. A per-image failure (a store that does not decode, a grid cell
  outside the image, a non-finite vector, a payload bound to another image) is
  journalled at stage `stage4` (`MEASURE_FAILURE_STAGE` in republish
  mode) and gives exit `3`.
- **`PHENOTYPIC_STAGE4_DEVICE=cpu` is test-only**: it runs Stage 4 on the CPU so
  a CPU node can exercise it. Inside Stage 4 an embedder's device is otherwise
  `cuda`, and a missing CUDA never falls back to the CPU.
- **Measure mode** runs Stage 4 in *republish* mode (U4, U8): the measure core
  carries each image's readable embeddings forward
  (`measure_embedding_carry`), defers `embed` and the record republish, and
  Stage 4 rewrites the measure pipeline's run folder's embeddings and then
  republishes the existing record (identity kept). A per-image Stage-4 failure
  still republishes, certifying what the core carried, and journals
  `MEASURE_FAILURE_STAGE` (exit `3`). Republish has one carrier, the staged
  manifest's flag (version 5; a forward manifest stays version 4). Locally
  `LocalParallelStrategy` runs it over the stores its measure loop measured; a
  measured image Stage 4 cannot take ends the invocation with exit `1`, named
  (`_refuse_unfinished_measure`, U16's local half).
- **Revisions** (U5, U17): `pin_embed_revisions` resolves every unpinned
  `MeasureDinoEmbedding`'s hub commit once, at a Stage-4 run's first
  submission (`resolve_embed_revisions`), records it (`embed_revisions` in the
  processing state and in job metadata), and every shard, round and
  continuation reuses it (`recorded_embed_revisions`); a recorded `None`
  (resolved offline) is resolved again. `--mode measure` reuses the run's
  recorded commits and resolves only a model with none.

### Nested detectors: found tree-wide, cut at the top-level ancestor

**`GpuDetector`s are found tree-wide**, not at the top level.
`find_gpu_detectors` (`_cli_validation.py`) walks the whole operation tree via
`sdk_._operation_tree.walk_operations` and returns `(path, detector)` pairs,
where `path` is a tuple of non-empty strings (`"ops[0]"` for a list entry). That
same path is the detector's recorded `pipeline_step_path`, the argument to
`substitute_at_path`, and the input to `detector_slot` — **one addressing
scheme, not three.**

That identity is only true because container operations now push a per-branch
step: `CompositeDetector`, `CompositeEnhance`, `FilamentousFungiDetector` and
`TwoKFilamentousDetector` drive their children through
`_core/_provenance.apply_child(child, image, segment=...)` instead of calling
`.apply()` directly, so each branch records its own segment rather than
inheriting the container's path. **Recorded `pipeline_step_path`s are therefore
deeper for every pipeline using a container op**, not only for staged ones.
`apply_child` is deliberately *not* used for a nested operation run by a
`MeasureFeatures` — those are private probes that stay out of the plate's
provenance (`measure/CLAUDE.md`).

- **The split cuts at the detector's top-level ancestor.** `pre_pipeline` holds
  the root ops before that ancestor; `post_pipeline` holds the ancestor onward
  **including the detector itself**. Stage 3 never applies `post_pipeline`
  as-is — it substitutes a `ReplayDetector` at `gpu_path` first, or the run
  re-executes live GPU inference on a CPU node.
- **`stage2_prefix` is the ops ahead of the detector *inside its own branch*.**
  `_branch_prefix` dispatches on `_child_contract`, never on `isinstance`: a
  `"sequence"` container contributes the ops preceding the step taken from it, a
  `"parallel"` one contributes nothing, and the **root** contributes nothing
  because Stage 1 already ran and stored its ops. Empty for a bare leaf and for
  a top-level detector.
- **Those prefix ops run twice and must be deterministic.** Stage 2 applies them
  to a provenance-detached copy purely to produce the detector's model input,
  then discards it; Stage 3 runs the same ops again inside the enclosing
  operation and records *those* in the journal. A non-deterministic prefix means
  Stage 2 infers from an image Stage 3 never reconstructs, and nothing fails to
  say so.
- **Unstageable placements are refused, on the production path.** All refusals
  live in `find_gpu_detectors`, which `pipeline_requires_gpu` calls — a refusal
  reachable only from `split_pipeline_at_gpu` fires after the run has already
  been routed, which is how this narrowing was silently lost once already. They
  are: a container that is not a composition primitive; a subclass of a listed
  composite that **overrides `_operate`** (the contract was verified against the
  base's `_operate`, so a replacement has been verified against nothing); a
  detector reached through **any** pipeline's `meas`/`post`/`filters`/`model`,
  root or nested, at any depth (`refuse_cpu_only_slot`); and, under
  `strict=True` (only `split_pipeline_at_gpu`), more than one detector. All
  raise `UnstageableGpuDetectorError`, a `ValueError` subclass.
- **`_CHILD_CONTRACT` is the closed table**, `{CompositeDetector: "parallel",
  CompositeEnhance: "parallel"}`, with `ImagePipeline` handled separately as
  `"sequence"` in `_child_contract`. It is populated lazily by
  `_populate_child_contract` in a **single** `dict.update` — read it through
  that function, never directly, and never populate it key-by-key: the guard is
  `if _CHILD_CONTRACT: return`, so a half-populated table makes a thread refuse
  a `CompositeEnhance` the design permits, and the GUI (threaded Werkzeug) would
  show a refusal alert and disable Run for a valid pipeline. Entries match by
  `isinstance`. There is no `_UNSUPPORTED_CONTAINERS` map. Coverage is asserted
  on the table itself (`tests/unit/cli/test_gpu_detection_tree_wide.py`) and
  each `"parallel"` entry carries a behavioural probe
  (`tests/unit/detect/test_container_child_contracts.py`); a new container needs
  no entry, because refused-by-default is the right answer for it.

#### Pipeline slots, and how a refusal reaches the user

- **Every pipeline's four slots are walked, root and nested.**
  `iter_child_operations` yields an `ImagePipelineCore`'s `ops` *and* its
  `meas`/`post`/`filters`/`model` entries (`_iter_slot_children`). Slot
  children are yielded **unconditionally**, never filtered through
  `_is_operation`: that predicate rejects `ModelFitter`, `SetAnalyzer` and
  `PostMeasurement`, so filtering would stop the root `model` slot being found
  at all. `find_gpu_detectors` then refuses any hit whose path passes through a
  slot at any depth. There is no `_CPU_ONLY_SLOTS` any more; the root-only loop
  it drove is gone, and so is its doubled `model/model/...` path.
- **Slot segments are spelled as a colon namespace:** `meas:<key>`,
  `post:<key>`, `filters:<key>`, and `model:<ClassName>` for the single model
  (`sdk_/_operation_tree.py`, `PIPELINE_SLOTS`). A bare `meas` segment would
  collide with an `ops` key a user is free to choose, and the `ops[0]` bracket
  form admits only integers. `_child` checks the model's class name, so a
  recorded `model:` path stops resolving once the model it addressed is
  replaced. **Never classify a segment by parsing it:** a user may key an op
  `"meas:X"`, so `pipeline_slot_of` decides against the live node, and a
  segment that is both an `ops` key and a resolvable slot entry raises
  `KeyError` as ambiguous rather than picking one.
- **Why a slot placement is refused.** The slots run after the op chain, which
  in the staged engine is Stage 3 on a CPU node, after Stage 2's inference has
  finished; and a measurement may hand its nested detector a derived input
  (possibly once per object) that Stage 2, which records one array per image
  per slot, cannot reproduce. **Do not reason "unrefused would be silently
  wrong" for ordinary pipelines — that was measured false.** Every shipped slot
  type (`MeasureFeatures`, `PostMeasurement`, `SetAnalyzer`, `ModelFitter`) is
  not a composition primitive, so `validate_ancestor_contracts` refuses those
  shapes anyway, just blaming the entry's class ("cannot be nested inside
  MeasureSymZones"). The slot check runs *first* so the message names the real
  reason. **It is the only guard** for a class that is both a slot type and a
  composition primitive — `class MeasuringComposite(CompositeDetector,
  MeasureFeatures)` inherits `_operate`, passes `_child_contract`, and without
  the slot refusal would be staged and run by Stage 2 on the stored image with
  no error. Pinned by `test_a_slot_entry_that_is_also_a_composite_is_refused`,
  which asserts the ancestor check admits the entry so it cannot pass on that
  check alone.
- **CLI: a usage error, before anything touches `--output`.** Right after
  `ExecutionConfig` is built, `phenotypic_cli` calls
  `uses_staged_gpu_strategy(config)` and converts `UnstageableGpuDetectorError`
  into `click.UsageError`; any *other* exception is ignored there so a corrupt
  pipeline still reaches the existing "Pipeline loading failed" line. This
  **GPU placement refusal** (it was once called "the preflight"; that name now
  belongs to the run preflight below) sits above the manifest load,
  `--overwrite` clearing,
  `mint_run_identity`, the `--dry-run` exit and `create_execution_strategy`.
  Both placements matter: below the clearing, a refused pipeline pointed at an
  existing run directory **deleted that run's results** before refusing, and
  `--dry-run` (the GUI's Validate) never reached the check, so Validate passed
  what Run refused. The outer handler converts a later refusal (e.g. from the
  splitter) the same way. Pinned by `tests/unit/cli/test_cli_gpu_refusal.py`.
  Two consequences of routing through `uses_staged_gpu_strategy`:
  `--mode measure` returns before scanning and is never refused, while
  `--mode process --layer rgb|gray` **is** refused although those layers come
  from pre-detector ops — `pipeline_requires_gpu` raises before the layer is
  consulted. The second predates this change and is unresolved.
- **GUI: the message, in either mode, and no Run.**
  `_gui/run_console/_callbacks.py:_staged_gpu_capability(path)` returns
  `(uses_gpu, refusal | None)` and catches `UnstageableGpuDetectorError`
  *before* its generic
  `(OSError, ValueError, TypeError, AttributeError, ImportError)` clause — the
  refusal **is** a `ValueError`, so clause order is the whole fix. The generic
  clause also takes `AttributeError` and `ImportError` so that an unknown
  operation class (`UnknownOperationClassError`, an `AttributeError`) reaches
  the CLI's own message instead of surfacing here. The
  `rc-staged-gpu-refusal` alert sits outside the staged-GPU section, so it
  shows in Local mode too; `update_run_disabled`, the sole owner of Run's
  `disabled`, takes the alert's `is_open`; and `click_action` refuses Validate
  and Run before `registry.allocate`, covering a click that races the
  pipeline-change repaint. An unreadable pipeline returns `(False, None)` — no
  alert; the CLI's validation reports it.
- **The previous text here was wrong about the mechanism.** It said the GUI
  swallowing the refusal routed a refused run to CPU; in fact the old
  `_pipeline_uses_staged_gpu` had one consumer, which only toggled the GPU form
  section's `display`, and the run itself is a `python -m phenotypic`
  subprocess, which raised — a hidden section, then a raw traceback.

### Stage-2 signals are keyed by detector slot

Both halves of the signal live under `<ds>/<slot>/`, where `slot` is
`detector_slot(plan.gpu_path)` — each path segment with runs of
non-alphanumerics collapsed to `-`, joined by `__`, plus 8 hex of the **exact**
path so `ops[0]` and `ops-0` cannot collide
(`CompositeDetector__ops-0__dddf3676`). Every path helper in
`_cli_stage2_token.py` takes `slot` with **no default**: a default would let a
caller reach a shared path without being handed one, which is the structural
hole the keying closes. Sites holding a pipeline path rather than a `StagePlan`
use `staged_detector_slot(pipeline_path)`; where a `StagePlan` is in scope, call
`detector_slot(plan.gpu_path)` directly.

At the shipped one-detector limit the slot level changes no behaviour. It exists
so that supporting N detectors is an additive feature rather than a migration of
the on-disk layout of a signal a 33,923-image run depends on.

Two deliberate exceptions to "always pass a slot":

- `find_stage2_token(output_dir, dataset, stem)` — the one legitimate slot-free
  **read**, for `--mode migrate`, which has no pipeline and therefore no
  `StagePlan`. It searches the legacy flat path, then each slot directory, and
  never moves or unlinks anything.
- `relocate_legacy_stage2_signal(...)` — a one-time move of a pre-slot-keying
  signal into its slot directory, so a run interrupted before the change and
  resumed after it does not repay Stage 2 in GPU time. Moves **both** halves,
  raw first, never overwrites a current signal, and **refuses** when the caller
  declares `slot_count != 1` (a per-image legacy signal is one detector's output
  and cannot be attributed among several).

**Continuation is automatic and content-defined.** Run the same command again;
there is no `--resume` flag. Exact terminal failures remain skipped unless
`--retry-failures` is supplied.
A missing or invalid store selects Stage 1; a valid store without a complete
Stage-2 signal selects Stage 2; and a valid store with one selects Stage 3.
**In a run that embeds on the GPU, an image that has finished Stages 1–3
selects Stage 4** (`ResumeStage` `"stage4"`): `stages_1_to_3_done`
(`_cli_staged_resume.py`) is `stages.stage3` in the record file, a store bound
to the image's work id, and a measurement table in its journal run, and
`stage4_status` reads it `"pending"` until its record is valid (`"done"`).
That one predicate is the "done" of every recovery shortcut under deferral:
each returns early on it and publishes nothing (local `_terminal_output_exists`,
SLURM `run_stage3_step`, Stage 3's own publish), since publishing would certify
an image with no embeddings; and every core that re-runs an image drops
`stages.stage3` and its Stage-4 signals first (`drop_stages_1_to_3_done`), as a
Stage-1 re-run does (`clear_downstream_artifacts_for_stage1`). The three
interrupted Stage-4 states resume from disk (`_classify_image`,
`_cli_stage4_worker.py`): signals valid for every key but no commit → commit
only; store promoted with fresh embeddings but no record → publish only; record
published but signals left → swept as orphans
(`sweep_orphan_stage4_signals`, locally before Stage 4 and in the controller
before each classification). **`--retry-failures` re-runs only Stage 4 for an
image whose Stage 4 failed**: the failure kept `stages.stage3`, so the image
reads `"pending"` again. Pinned locally
(`tests/integration/cli/test_local_stage4.py::test_retry_failures_reruns_only_stage4`).
On SLURM it holds by trace, not by an end-to-end test: for a run with no
detector, `get_remaining_images_for_datasets(..., retry_failures)` admits the
image, `_plan_slurm_no_detector_continuation` → `split_stage4_pending` reads it
`"pending"` (Stage 4 only; the `single_pass` role returns early on
`stages_1_to_3_done`), and `_classify_stage4` treats a failure journalled under
an earlier epoch as retryable (`test_a_failure_from_another_epoch_is_retryable`);
without the flag the image is left out of the manifest.
**Every prereq probe tests BOTH halves of the signal** —
`stage2_result_replayable()` is the one function all **seven call sites in three
modules** call (`_cli_staged_strategy.py` ×3, `_cli_staged_slurm_worker.py` ×2,
`_cli_staged_controller.py` ×2: `_classify_stage2`, and `_dead_stage3_indices`
since #282 U15) — a count a `grep` can confirm or refute, which
"five sites" could not, because it never said sites of what. The token
is only a flag; Stage 3's actual input is the raw `.npy`, so a
token-present/raw-missing image is routed back to Stage 2, not into a Stage 3
that would raise `FileNotFoundError` and be recorded as a terminal *scientific*
failure. Stage 3 embeds the authoritative Parquet inside the final store transaction,
publishes the image marker over both the store root and table, then consumes the
signal. **A Stage-3 scientific failure keeps both halves**:
`reconcile_stage3_publications` consumes them only for an image that is complete
or has a Stage-3 marker, so `--retry-failures` replays Stage 3 from the retained
raw array with no GPU (#282 P5 smoke: 44 files for 22 failures, review M4). The output is byte-identical to a
single-pass run, with one exception: a `PlotImage` bound to a pre-GPU `ops`
entry goes to `pre_pipeline`, so Stage 1 draws it on the image after the
pre-GPU ops only, where a single-pass run draws it after the whole op chain.
Stage 3 keeps that figure (see **Per-image figures**). Today the only shipped
`PlotImage` that is also an operation is `CalibrateColorRpcc`, whose overlay is
the same either way because it draws from its own `apply()`.

**Stage 4 (local, #282 P4) is fenced exactly as the local strategies are.**
`run_stage4` holds the local epoch ownership (`_local_epoch_ownership`: the
processing-state lock plus a `processing_generation` compare) around each
record and each terminal-journal entry, as `_publish_local_image_success` and
`_record_local_terminal_failure` do. A new generation (`--restart`,
`--overwrite`) therefore fences an older process at its next record. Two
concurrent invocations of the **same** generation are not fenced, in Stage 4
or in the cores, because a continuation keeps its content-derived generation:
the store write stays safe (the loser meets `StoreRootChangedError` and leaves
the image pending), but both can publish one image's record. Concurrent local
invocations over one `--output` are unsupported; a local run lease would be a
separate change (P3–P4 review M1). **The same holds for local `--mode measure`
(#282 U19), by decision, with no lease:** its republish Stage 4 records and
failure journal entries (`_local_record_lock(republish=True)`), and since the
P6 review the measure core's own failure journal
(`_record_local_terminal_failure` under `measure_only`, review M3), take only
the processing-state lock, with no generation compare, because measure mode
keeps no run state and its content-derived generation is the forward run's
only when its pipeline is. A concurrent rewrite still leaves the image pending
(exit `1`): `replace_run_embeddings` is root-fenced.

**Progress events.** Stages emit stage-tagged events via the `stage` field on
the event log (`_cli_update_state.py`: `append_event(..., stage="stage1|2|3|4")`,
`_stages.py` `VALID_STAGE_TAGS`). `status` stays the closed
`{started, completed, failed}` set. **The terminal stage is per run** (#282):
`aggregate_state_from_events(..., terminal_stage=)` counts overall completion at
`run_terminal_stage(state.config)`, which is `"stage4"` for a run whose recorded
`embed_stage` is `gpu` and `"stage3"` (`STAGED_TERMINAL_STAGE`) otherwise,
including a state recorded before the key existed. Every caller passes it, so a
deferring run never reads complete at Stage 3. Under deferral the single-pass
core's events are tagged `stage="stage3"`; an inline run's stay untagged.
`aggregate_stage_state_from_events` gives the per-`(dataset, stage)` breakdown.
Use the shared `stage_event` context manager + `emit_missing_prereq` helpers in
`_cli_staged_workers.py` rather than hand-writing the started/completed/failed
trio.

## SLURM chaining (`_cli_staged_slurm.py` + `_cli_staged_slurm_worker.py`)

### Exit codes (`phenotypic.sdk_.RunExitCode`)

The CLI's exit code is a contract (#281 spec C10): an orchestrator such as a
DVC stage acts on it without parsing logs. Use the enum member at every exit
site, never a bare literal.

| Code | Name | Meaning |
|---|---|---|
| `0` | `COMPLETE` | finished, no terminal failures |
| `1` | `ERROR` | a runtime or config error |
| `2` | — | a refused option: a `click.UsageError`, Click's own code. Includes an option value the CLI checks after parsing (`--slurm`/`--gpu-slurm` `KEY=VALUE` pairs and `time=`, `--ext`) |
| `3` | `COMPLETE_WITH_FAILURES` | finished, some images failed terminally |
| `4` | `SUBMITTED` | submitted to SLURM, not finished |
| `5` | `ENDED_UNPUBLISHED` | `phenotypic-wait`: no job alive, and the run ended without finishing (nothing published, or accepted images never attempted: a Stage-4 stall, #282 U7; Stage-1/3 tasks that kept dying, U15; a measure pass that left images neither republished nor journalled, U16); re-run the same command (spec C3/C10, fork F5) |
| `6` | `WAIT_TIMEOUT` | `phenotypic-wait` gave up after `--timeout-seconds` |

- **Every SLURM submission exits `4`**: full, measure, staged and process
  runs, a SLURM `--mode recompile`, a forward run whose only remaining work is
  a SLURM recompile (a new metadata snapshot), a submitted `--mode migrate`
  and a SLURM migrate `--dry-run`. When nothing needed SLURM,
  `_handle_recompile_slurm` runs the recompile locally and returns the
  finished run's code (`0` or `3`, below) instead, because then nothing was
  submitted.
- **"Failed" is the run's, not the invocation's.** A local exit asks
  `_run_failure_count` (`phenotypicCLI.py`), which is
  `resolve_run_state(...).diagnostics.failed` when this invocation recorded no
  failure itself. A continuation skips recorded terminal failures, so a run
  that exited `3` exits `3` again on a plain re-run, whether that re-run is a
  "nothing left" no-op or a re-finalization. The same count gates
  `publish_local_gui_completion`, which raises on a run with failed images.
- **A recompile exits `3` over a run that carries a terminal failure**, local
  or through either of `_handle_recompile_slurm`'s local fallbacks (user
  decision 2026-11-13, P2 review I2/F4). A recompile re-publishes the
  aggregate and leaves every failure record where it was, and on SLURM the
  waiter reads the same run as `failed`, so the two agree.
- **A local staged run whose only Stage-3-less images are current terminal
  failures exits `3`, not `1`** (user decision 2026-11-13, P2 review I3/F3).
  It is the local twin of the SLURM finalizer's `terminal_incomplete` close:
  publication succeeded, every image without a Stage-3 marker has a `failed`
  verdict in `resolve_run_state(depth="deep")`, the run is **not** marked
  staged-complete, and a plain re-run (which skips those images)
  re-finalizes and exits `3` again. It prints how to retry them
  (`--retry-failures`), not "Run the same command again". A real publication
  failure, or an unfinished image with no terminal record, still prints
  `STAGED FINALIZATION FAILED ... Run the same command again` and exits `1`.
  **Under `--embed-stage gpu` the predicate is record-based**
  (`_unpublished_images`, in place of `_stage3_less_images`): Stage 3 writes
  `stages.stage3` and no record, so a Stage-3 marker is not completion. A
  Stage-4 per-image failure is a terminal failure, so the same `3` follows.
- **A local Stage 4 exits `1`, one line, journalling no image** (#282): no usable
  CUDA at startup, before Stage 1 (`prepare_stage4_environment`, a
  `click.ClickException`, an environment error rather than a refused option);
  node or model state during Stage 4 (`run_local_stage4`: "Stage 4 stopped …
  No image was recorded as failed"); any image left pending ("N image(s) still
  need Stage 4 (<why>)"); and a local `--mode measure` that left a measured
  image neither re-embedded nor journalled (`_refuse_unfinished_measure`, U16).
  Each says "run the same command again". On SLURM the same cases are a shard's
  exit `1` and a next round, and, if the rounds stall, the waiter's `5`.
- A forward `--dry-run` submits nothing and exits `0`; an interrupt exits
  `130`.
- **`phenotypic-wait --output <out>` (`_cli_wait.py`) turns a `4` into the
  run's own code** once the run ends, for a DVC stage or any orchestrator.
  **It returns only once every job of the bound generation's ledger has
  ended** (#282 U15, smoke F1): every controller pre-arms a recovery
  controller, the one after the finalizer included, and the finalizer's fence
  close makes `live_slurm_job_ids` answer empty while that controller is
  still queued. So once a poll decides the outcome, the waiter keeps it and
  polls `ledgered_jobs_of_generation` (the same classification, without the
  fence short-cut) until nothing is live, unverified or unresolved, under the
  same interval and `--timeout-seconds`, and then returns it. A teardown the
  waiter triggers therefore cannot race that controller.
  Read-only: it never aggregates, publishes or cancels, and submits only the
  command given after `--` (below). Each poll
  reads `live_slurm_job_ids` **before** `resolve_run_state(depth="shallow")`.
  Any `live`/`unverified`/`unresolved` job keeps it waiting, whatever the
  completion says. With none live, it exits `0`/`3` only when **this
  submission** published. It reads the fence generation once, at the first
  poll. Published means one of: that fence was closed with
  `closed_by: "finalizer"` (`FENCE_CLOSED_BY_FINALIZER`) and no
  `terminal_status`; the run proof's `execution_epoch` is that generation; or
  the current aggregate proof's `execution_epoch` is that generation. **A
  fence of that generation carrying a `terminal_status` overrides all three**
  (#282 U16): a SLURM `--mode measure` finalizer closes so, in one
  `mark_generation_failed(..., closed_by=FENCE_CLOSED_BY_FINALIZER)` write,
  when its pass left any forward-published manifest image neither
  republished under its epoch nor journalled (`unfinished_measure_entries`),
  and logs each such image by name. Those images keep forward records that
  verify, so the run state alone reads `complete`; the waiter exits `5`. The
  pass's measure steps write a per-epoch measure-done marker
  (`progress/measure_done/<epoch>/<ds>/<stem>.json`, #282 U18) after their
  table rewrite, and an image without one is never republished. The
  finalizer is the only writer of both fields. It passes `execution_epoch=`
  through `aggregate_measurements`/`finalize_run` to
  `publish_aggregate_snapshot`, and `closed_by=` to the fence close. A cancel
  leaves `closed_by` unset. **Matching digests are not enough:** a
  `--mode measure` resubmission keeps no run state, so the earlier run's
  proofs still match after its finalizer dies. **`3` also needs every
  accepted image resolved** (`RunSnapshot.unresolved == 0`, i.e.
  `verified + failed == accepted`); an image never attempted makes it `5`.
  A SLURM `--mode measure` resubmission's failures sit outside the forward
  inventory, so `_generation_failures` counts distinct work ids in the
  terminal journal whose `lifecycle_epoch` is the bound generation; published
  plus any such row gives `3` (review I5). **Those rows exist only since
  C11d:** before it, measure mode journalled nothing (its core raises the
  bare exception), so the count was always `0` for a real failure. Both
  measure workers now wrap the core in `measure_scientific_boundary`
  (`_cli_process_single.py`), which raises `PerImageScientificError` with
  stage `MEASURE_FAILURE_STAGE` (`"measure"`, not the staged `"stage3"`), and
  journal it under the measure work id. One helper,
  `terminal_failure_work_ids_under_epoch` (`_cli_failure_tracker.py`), is
  the count for both the waiter and `metrics.json`: on SLURM `finalize_run`
  counts its `stage=MEASURE_FAILURE_STAGE` rows under `execution_epoch` (the
  stage filter keeps a forward generation's own rows, already counted through
  the inventory, from counting twice); locally the CLI passes the
  invocation's own count as `measure_failed=` and mints no epoch, because a
  local measure's `lifecycle_epoch` is the forward run's content-derived
  generation, shared by every measure invocation. `failed` adds the count and
  `succeeded` drops by it, although such an image's previous table usually
  stays aggregated. The rows are keyed by measure work ids, so a forward
  continuation never skips on them; its "N recorded failure(s) skipped" line
  therefore counts `_run_failure_count`, not the whole journal. A process run that reads
  `failed` exits `3` anyway. Everything else exits `5`, and `--timeout-seconds` exits
  `6`. It does not serve `--mode migrate`, whose lifecycle lives under an
  external control root.
- **`phenotypic-wait --output <out> -- <submit…>` launches the submit
  itself** (spec C3 "Launch", P2 T2a), so a SLURM DVC stage is one `cmd`
  (DVC aborts a stage on the submit's `4`). `_run_submit` runs the argv with
  `subprocess.run` (no shell; the waiter's cwd, env and stdio), mapping a
  signal death to `128 + N` and a missing/unexecutable program to `127`/`126`.
  Child `4` → `wait_for_run`; any other child code is returned unpolled.
  `check_submit_argv` refuses (usage error, `2`) before launching: an empty
  command after `--`, a child `--mode migrate` (also `--mode=`, `-m`, `-m…`),
  and a child `-o`/`--output`/`--output=`/`-o…` resolving elsewhere than
  `--output` (cwd-relative). Only the tokens after the program `phenotypic`
  (`-m phenotypic`, `-mphenotypic`, or a token whose file name is
  `phenotypic`) are read, so a wrapper's own `-o` (`srun -o log …`) is not the
  run's; a command naming no such program, or no output, is not checked. The
  command must be introduced by `--` (`_WaitCommand` keeps the raw args to
  tell), and `allow_interspersed_args=False`, so a child's `-o` can never be
  parsed as the waiter's. `--output` need not exist in the launch form (the
  submit creates it); the wait-only form still requires it. `--timeout-seconds`
  bounds the wait, never the submit.
- **Before launching, an active job means attach** (fork F2): if
  `LiveSlurmJobs.live` is non-empty for `--output`, the waiter skips the
  submit, prints the generation and jobs, and waits on that run, which is the
  recovery after the job running `dvc repro` hits its walltime.
  `unverified`/`unresolved` alone still launch, and the child's own guard
  refuses. Ctrl-C during the submit waits for the child and returns its code
  unpolled; an interrupted wait exits `130`. Up to
  `MAX_CONSECUTIVE_POLL_ERRORS` (5) failed polls in a row are printed and
  retried; the fifth exits `1` with the reason.
- **`--accept-failures`** (fork F1) maps a final `3`, waited or from a local
  child, to `0` after printing the terminal-failure count, so a DVC stage
  over a dataset with one empty plate completes. Every other code is left
  alone. `_live_jobs` passes a per-process `_InactiveJobMemo` as
  `live_slurm_job_ids(..., is_active=)`, since a job answered not-active
  stays so. The seams tests replace are `_run_snapshot`, `_live_jobs`,
  `_ledgered_jobs`, `_fence_generation`, `_published`, `_cancelled`, `_generation_failures`
  and `_is_process_run`.
- **The GUI run console reads `4` as a failed submission**
  (`_gui/run_console/_slurm.py`), a known break #283 fixes before #281 merges.

### Array trigger routing, not scheduler sidecar jobs

Do not add a **scheduler sidecar job**, meaning an extra `sbatch` job intended
to run in parallel beside an already active ordinary array. The cluster's
allocation and submission bounds are consumed by the array cohort, so an
outside sidecar may remain pending, starve the work it is meant to accompany,
or exceed the bounded submission topology.

Route ancillary work that must run with an ordinary array **through the array
itself**. Insert a reserved trigger token into the array task-entry list and
dispatch that token inside the generated array script. Follow the existing
`_CHECKPOINT_SENTINEL = "__PHENOTYPIC_CHECKPOINT__"` and
`_MANIFEST_SENTINEL = "__PHENOTYPIC_MANIFEST__"` pattern in
`_cli_slurm_array_scripts.py`; do not submit a parallel helper job. A new
trigger must:

- use a collision-resistant reserved `__PHENOTYPIC_<ROLE>__` token;
- be routed by an explicit array-worker branch rather than treated as an image;
- be included when calculating the task-entry count and chunk size, so the
  final `#SBATCH --array` length remains within `MaxArraySize`; and
- have tests proving both the trigger routing and the absence of a standalone
  parallel submission.

This scheduler rule is unrelated to the staged GPU Stage-2 **signal files**
(the retained raw `.npy` and its token) and to the Stage-4 signal files and
round worklist (#282). It also does not convert a terminal `afterany` finalizer into an array
entry: a finalizer runs after the array becomes terminal and is not a parallel
sidecar. The existing staged-GPU controller topology is a specialized,
explicitly capacity-reserved design; do not generalize it into new ordinary
array sidecars.

> **Queue ordinary SLURM work through the drip-feed dispatcher, and staged GPU
> work through its recoverable controller.** The CPU autonomous strategy and
> `--recompile` funnel their
> ordered chunk scripts through `submit_slurm_script_chain`
> (`_cli_slurm_submission.py`) → `generate_dispatcher_chain` +
> `submit_drip_feed_start` (`sdk_/slurm/_dispatcher.py`). The dispatcher submits
> **only chunk 0 + a tiny dispatcher job** up front; when chunk N ends, its
> dispatcher submits chunk N+1. Peak queue occupancy stays at ~1 chunk + 1
> dispatcher, so a run's full task count (which for the staged path is
> ~2 × n_images) never trips the per-user `MaxSubmitJobs`. A new SLURM
> submission site MUST reuse this helper, not loop `submit_script` over all
> chunks — eager submission is what caused the `AssocMaxSubmitJobLimit` failures.

`StagedSlurmStrategy` writes per-stage SBATCH scripts plus a controller script.
The dispatcher paragraph above applies only to ordinary CPU/recompile chains;
the staged GPU path no longer flattens its whole lifecycle into that dispatcher.
It creates an orchestration UUID, versioned image manifest, atomic controller
state, and append-only job ledger before submitting Controller 0. That controller
pre-arms its recovery controller before submitting Stage-1 chunk 0. Each controller
thereafter pre-submits its recovery
controller, then either launches the next Stage-1/Stage-3 chunk, a retry of a
chunk's dead tasks, a Stage-2 round or a Stage-4 round, or launches the
finalizer (see **Stage 4 on SLURM** below for the phases a run has). Deterministic SLURM comments let a
successor discover a job when its predecessor died after `sbatch` but before
persisting the returned ID.

- Stage 1 / Stage 3 = arrays over **images**; Stage 2 and Stage 4 = arrays over
  **shards** (`--gpu-shards`, `partition_shards`), each a resident-model
  `run_stage2_shard` / `run_stage4_shard`. Controller jobs run on
  `config.slurm_args` (CPU).
- **Array chunking (`min(MaxArraySize, MaxSubmitJobs - 2)`):** the image count is
  split into `ceil(n_images / chunk_limit)` chunk scripts. The two reserved slots
  are for the running controller and its dependent recovery controller.
  (`calculate_optimal_array_chunks` → `_write_image_stage_chunks`), where
  `get_slurm_max_submit_jobs()` conservatively uses the smallest configured QoS
  or user-association limit, and `chunk_limit` reserves one submission slot for
  the dependent dispatcher queued alongside the active array. A single chunk
  must fit **both** the array-index cap and the remaining per-user submit
  capacity. Known limits below three are rejected.
  Each chunk is a 0-based `--array=0-(k-1)` whose `TASK_INDICES` window holds the
  **absolute** manifest indices and whose worker reads
  `--index $CURRENT_TASK_INDEX`, so no array index ever reaches the limit. A
  single chunk keeps the plain `stage1.sh`/`stage3.sh` name; multiple become
  `stageN_chunk{i}.sh`. Stage 2 is **never chunked** (a shard worker streams its
  whole shard on one GPU); `--gpu-shards > chunk_limit` raises. `generate_staged_scripts`
  returns stage arrays plus controller, finalizer, config, and manifest paths.
  Image stages are always lists. The finalizer is a
  one-task CPU job that reloads the canonical pipeline and runs the same
  aggregate/finalize path as ordinary SLURM, including named analysis and plot
  publication. The controller records every dynamically submitted ID and keeps
  only one work array plus its recovery controller active at a time.
- Per-stage resources: Stages 1 & 3 use `config.slurm_args` (CPU); Stages 2 and
  4 use `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)` — inherit/delta over the
  CPU profile, auto-add `slurm_gpus_per_node=1` (explicit `=0` **omits** the
  directive so a CPU partition can run the GPU stage, e.g. tests).
- **Walltime survival:** neither Stage 2 nor Stage 4 installs a signal handler or self-requeues.
  After each array reaches a terminal scheduler state, its dependent controller
  reclassifies the manifest from valid stores, complete Stage-2 signals,
  Stage-3 markers, and terminal failures. Remaining images launch another array round. One unchanged
  retryable-set round is retried; a second unchanged round terminalizes the
  remainder and advances to Stage 3. Stage 4's rounds follow the same rule on
  their own counters, except that a stall terminalizes nothing (U7, below).
- **A Stage-1 or Stage-3 task killed by the host is resubmitted** (#282 U15,
  smoke F2: OUT_OF_MEMORY). After a stage's last chunk, `_resubmit_dead_tasks`
  (`_cli_staged_controller.py`) re-classifies: an image with no stage result
  (no valid Stage-1 store; no `stages.stage3`, and for a detector run a
  replayable Stage-2 signal), no record and no journal row under the epoch is
  retryable. Only those tasks are resubmitted, one chunk at a time, by
  narrowing the chunk's `--array` on the `sbatch` command line
  (`submit_with_lifecycle(extra_args=)`), as `<stage>-retry-<round>-chunk-<i>`.
  The same zero-progress rule as Stage 2, on the stage's own counters (read
  with defaults, so a #281 state works): the second unchanged round stops,
  the leftovers stay unrecorded, and the run exits `5`. The finalizer's log
  repeats every recorded stall (`recorded_stall_lines`), since the waiter's
  `5` points there. A trailing controller that finds the run
  `terminal_incomplete` exits 0, like `complete` (review I3).
- Every worker checks the active epoch immediately before publishing the store,
  the Stage-2 or Stage-4 signal, parquet, plot, or deletion changes. Restart and cancellation fence
  stale workers before clearing or cancelling ledgered jobs.
- Staged submission returns `PROCESSING SUBMITTED` and exits `4`
  (`SUBMITTED`), like every SLURM run (0.21.0 removed `--wait`). The CLI writes the README at submission; remote
  finalization owns aggregation, the dashboard and the completion marker.
  Nothing writes `processing_report.html` automatically.
- Per-image isolation: a missing prereq (S6) is recorded and skipped, never an
  unhandled raise that aborts a shard.

### Stage 4 on SLURM (#282 P5, P6)

- **Controller config version 2** (`staged_controller.json`,
  `generate_staged_scripts`) adds a `phases` list and a `stage3_role`;
  `_load_controller_config` reads a version-1 config as `["stage1", "stage2",
  "stage3"]`, so a run submitted before the bump drives as before. The plan is
  `_staged_plan` (`_cli_staged_slurm.py`):

  | Run | `phases` | `stage3_role` |
  |---|---|---|
  | `GpuDetector`, embeds on the GPU | `["stage1", "stage2", "stage3", "stage4"]` | `staged` |
  | `GpuDetector`, embeds inline | `["stage1", "stage2", "stage3"]` | `staged` |
  | no `GpuDetector`, embeds on the GPU | `["stage3", "stage4"]` | `single_pass`: the single-pass core with `defer_embeddings` |
  | `--mode measure`, embeds on the GPU (U10) | `["stage3", "stage4"]` | `measure`: the measure core, embeddings carried and deferred |

  A run with no `stage1` phase writes no `stage1.sh`/`stage2.sh`, records no
  `detector_slot` (`recorded_detector_slot` answers `None`), and its controller
  jumps from `submitted` to `stage3_starting`. A `"stage4"` config must carry
  `stage4_script`, `stage4_worklist` and `n_shards`.
- **After the last Stage-3 chunk** (and its dead-task retries), a `"stage4"`
  config goes to `_advance_stage4`: sweep orphan signals (in republish mode,
  `sweep_republished_stage4_signals`, which asks of this pass's epoch rather than
  the forward record), then `_classify_stage4`: **done** when `stage4_status` is
  `"done"`; **terminal** on a failure journalled under this epoch, on
  `"not_ready"` (no `stages.stage3`, a store that does not match the work id, no
  measurement table), or on a changed reference digest; **retryable** otherwise.
  A failure from an earlier epoch is retryable: the continuation chose to retry
  it. `_log_stage4_terminal` says how many terminal entries are journalled and
  how many never finished Stages 1–3.
- **Rounds** (`_submit_stage4_round`): Stage 2's zero-progress rule on Stage 4's
  **own** counters in `staged_orchestration.json`: `stage4_round`,
  `stage4_last_progress_digest`, `stage4_zero_progress_rounds`. The progress
  digest (`_stage4_progress_digest`) covers each retryable image's identity and
  each of its signal files' `st_mtime_ns`, one `stat` per key and no Parquet
  read, so a round that wrote signals and committed nothing still progressed.
  The ledger role is `stage4` and the token `stage4-round-<n>`, so
  `live_slurm_job_ids` sees the round.
- **The worklist** (`.phenotypic/slurm_scripts/stage4_worklist.json`,
  `write_stage4_worklist`, version 1): one per round, the retryable set only,
  sorted by identity and split contiguously over `n_shards`, so after a
  walltime the leftover work spreads over every GPU. A shard reads its slice by
  `$SLURM_ARRAY_TASK_ID` (`load_stage4_worklist`); an empty shard exits `0`. A
  round whose token the ledger already holds (`intent`, `blocked`, `submitted`,
  `recovered`) keeps its worklist rather than re-partitioning under a running
  job (review M12), and only the filled shards get a GPU: the round is submitted
  with `--array=0-<filled-1>` on the command line, overriding the script's
  `#SBATCH --array` (review M13).
- **`stage4.sh`** is one array over shards on the GPU profile, never chunked,
  with no trigger entries; it adds no scheduler sidecar job beside the array. Its
  prelude (`STAGE4_PRELUDE_BASH`) is the thread pins plus
  `export CUBLAS_WORKSPACE_CONFIG=:4096:8`, before Python starts. The worker is
  `--stage 4 --worklist <path> --index $SLURM_ARRAY_TASK_ID`
  (`run_stage4_shard`), which prints `<n> published, <n> failed, <n> left for the
  next round, <n> signal(s) written`.
- **A stall ends unfinished (U7).** At two unchanged rounds the controller logs
  `stage4_stall_line` and submits the finalizer; the leftover images keep no
  record and no journal row, so `phenotypic-wait` exits `5` and a plain re-run
  retries them. GPU OOM and a changed root are never per-image failures. The
  finalizer repeats the stall line (`recorded_stall_lines`), since the waiter's
  `5` points at its log.
- **Measure mode** (U10, U16–U18). The `measure` role's steps
  (`run_measure_step`) write a per-epoch **measure-done marker**,
  `progress/measure_done/<epoch>/<ds>/<stem>.json` (`write_measure_done_marker`),
  after the table rewrite. `_republish_stage4_status` reads an image without one
  `not_ready`, so the controller resubmits it as a dead task
  (`_dead_measure_indices`), Stage 4 never lists it, and the worker refuses it
  (`PENDING_NOT_MEASURED`). "Done" is a record this pass republished: it carries
  the measure epoch as its `lifecycle_epoch`. The submitter clears earlier
  passes' markers (`clear_measure_done_markers`) before it publishes the epoch,
  and passes `initialize_orchestration(keep_completion_marker=True)`, since the
  staged-completion marker is the forward run's. The measure pipeline's sha256
  has one producer, the CLI's `pipeline_identity`, carried in the controller
  config (`pipeline_sha256`, `recorded_measure_pipeline_sha256`); the measure
  step and Stage 4 refuse a file that no longer hashes to it. The
  **measure-mode finalizer** publishes no forward run proof, counts
  `measure_failed` from the journal under the epoch, and, when
  `unfinished_measure_entries` finds a forward-published image neither
  republished under the epoch nor journalled, logs each by name
  (`unfinished_measure_line`) and closes the fence failed, `closed_by:
  finalizer` and a `terminal_status` in one write (U16).

### One live SLURM run per output

A forward invocation (`full`, `process`, `measure`, with or without
`--restart`/`--overwrite`/`--dry-run`) and `--mode recompile` refuse while an
earlier run's SLURM jobs are still live over the same `--output`.
`_refuse_live_slurm_run` (`phenotypicCLI.py`) is the one check. It sits above
the first write of each path: before the continuation-state load on the
forward path, and before `_snapshot_metadata_csv` in the recompile branch. It
asks `live_slurm_job_ids` (`_cli_staged_orchestration.py`). `--mode migrate`
keeps its own lease and lifecycle guard.

**Ordinary chains are ledgered, not only staged ones.** Every chunk,
dispatcher and finalizer goes through `submit_with_lifecycle`, including the
chunks a dispatcher submits from inside SLURM (`dispatch_continuation`). That
call writes an `intent` row, runs `sbatch --comment phenotypic:<generation>:<token>`,
then writes a `submitted` row, all into the one lifecycle ledger
(`slurm_jobs.jsonl`, which *is* `staged_job_ledger_path`). Grepping direct
`append_lifecycle_entry(` callers misses this; the Phase 2 review of the
reference-metadata change did, and concluded wrongly that ordinary arrays were
invisible to the guard.

How `live_slurm_job_ids` answers, and when it asks the scheduler. It returns
a `LiveSlurmJobs(live, unverified, unresolved)`, falsy only when all three are
empty:

- A closed lifecycle fence (`active is False`) answers empty with **no**
  scheduler query, so a cancelled or finalized run whose history SLURM has
  purged never blocks.
- **Only the fence's own generation is consulted.** A new generation is
  published only when the previous one is inactive (`initialize_slurm_lifecycle`
  refuses to replace an active one) or after this guard passed for it, so an
  older generation is either fenced or already judged not live. That keeps the
  per-job `squeue`/`sacct` calls bounded by one run, not the output's whole
  history. A ledger with no readable fence (a legacy staged tree) is walked
  across every generation instead.
- Each unterminated ledgered job ID is classified by `scheduler_job_is_active`:
  truthy is `live`, `None` is `unverified`, and `False` is dropped. The walk is
  shared with `active_ledger_job_ids`, whose own return is unchanged because
  cancellation reads it.
- A token whose latest ledger row is `intent` or `blocked` may own a live job
  whose `submitted` row was never written (the submitter died after `sbatch`).
  Only then does it make one `squeue`-only `query_scheduler_comments` call and
  match the exact `phenotypic:<generation>:<token>` comment. A match is `live`.
  If the query raises `SchedulerQueryUnavailable`, the comment stays in
  `unresolved`. A clean ledger, and every local run (no lifecycle, no ledger),
  never queries.

**It fails closed, but never calls an unknown job active.** Any non-empty
group refuses. `_refuse_live_slurm_run` prints one sentence per group:
`SLURM jobs are active: <ids>` with `scancel <ids>` for `live`;
`Ledgered SLURM jobs are unverified: <ids>` (may be running or long finished)
for `unverified`; and "the scheduler could not be queried from this host"
naming each `<generation>:<token>` for `unresolved`. When either unknown group
is non-empty it adds the one remedy that works: run the same command on a
cluster node where `squeue` works. It then lists `sacct -j <id>` and
`squeue --noheader --format='%i|%k' | grep 'phenotypic:<gen>:<token>'` for
checking by hand. There is no override flag. **A restored fence whose job ids
SLURM has purged** (`squeue` says invalid, `sacct` has no rows) stays
`unverified` forever; after confirming with `sacct` that nothing runs, set the
fence's `"active"` to `false` in `.phenotypic/slurm_lifecycle.json`. Deleting
the file does **not** work: with no fence, every generation's ledger is
walked and the same jobs stay unverified (measured, P2 review M7). A host that cannot reach the
scheduler cannot clear a stuck fence.

**A stuck `active: true` does not refuse on its own, deliberately.** A
finalizer that is OOM-killed or cancelled outside PhenoTypic leaves the fence
active forever. Refusing on the flag would block the plain re-run that is the
recovery procedure. So liveness always comes from the scheduler. Once every
ledgered job is terminal and no unresolved token matches a queued job, the run
is allowed.

**Allowed also means the stale fence is closed.** Until #281 P2 T5 it was not:
the guard let a SLURM re-run through, and `initialize_slurm_lifecycle` then
refused to publish a new generation over the still-`active` one, so the
documented recovery died with `Unexpected error` after the mutating half had
begun (a SLURM recompile refused the same way in
`_initialize_recompile_slurm_attempt`). Now the first step of the mutating
half, and the recompile attempt before its own check, call
`retire_dead_lifecycle_fence` (`_cli_staged_orchestration.py`), which closes
the fence on the guard's own evidence (`live_slurm_job_ids` empty) for the
modes it owns, `ordinary`, `recompile` and `staged-*` (a staged fence through
`deactivate_orchestration`). It leaves `closed_by` unset, so
`phenotypic-wait` never reads the close as the old generation's publication.
A `dvc checkout` that restores an old `.phenotypic/` (spec C13) is the same
case. `migrate` (own lease and reconcile), `tune` and owned fences are never
retired. Pinned by `tests/unit/cli/test_dvc_outs_at_submission.py`.

**A fence with no ledger row for its generation is never retired** (P2 review
I4). `live_slurm_job_ids` answers empty for it, because there is nothing to
ask the scheduler about, yet it is exactly the state of a submitter between
publishing its fence (`initialize_slurm_lifecycle`, or
`initialize_orchestration` for staged runs) and writing its first `intent`
row, a window that grows with a staged run's manifest. Retiring it made that
submitter die with `SlurmGenerationInactiveError` after its mutating half had
begun. `unledgered_active_fence` names such a fence; `_retire_dead_fence`
then refuses a **SLURM** invocation with exit `1`, naming the generation and
the fence file, before anything else is written (the refusal
`initialize_slurm_lifecycle` made before the retire existed, moved up). A
local run is not refused, as it never was. A submitter that died before its
first `sbatch` leaves the same state, which is rare: the message says to
delete the fence file once no invocation is running.

The check is a read-only `squeue` from the invoking process, not a scheduler
job, so it is not a sidecar. Pinned by `TestLiveSlurmRunGuard` in
`tests/unit/cli/test_cli_v2.py`.

## Legacy-tree migration (`--mode migrate`)

Migration runs locally without `--slurm`. With one or more `--slurm key=value`
arguments, the public CLI creates a fresh generation-scoped
`MigrationSlurmPlan` and submits its ordered metadata → image → seal → optional
reclaim → finalizer chain through `submit_migration_slurm_plan`. An explicit
`--njobs` is invalid on that path — including an explicit value equal to the
Click default — because the scheduler controls worker parallelism; determine
that distinction at the Click boundary with `ParameterSource`, never by
inspecting the numeric value.

Validate incompatible migration options before generating a plan or initializing
lifecycle state. A SLURM `--dry-run` writes its manifest, config, scripts,
statuses, lifecycle, and logs only below the external cache control root, never
inside the scientific output tree. It submits the metadata preflight and
throttled per-image validation chain, and reports its durable job IDs,
generation, control references, and the terminal status file its finalizer
writes. That file is where a SLURM attempt's verdict is read: 0.21.0 removed
`--wait`, which used to wait for the lifecycle to close and turn a failed
terminal status into a Click error, so a submitted migration exits `4`
(`SUBMITTED`) whatever its jobs later do, and so does a SLURM `--dry-run`,
which submits its validation chain. A local migrate keeps `0`/`1`. **If the finalizer itself never finishes** (out of memory,
time limit, node failure, or cancelled while pending), no terminal status file
appears. Once no job of the attempt is queued, re-run `--mode migrate`:
`reconcile_interrupted_migration_attempt` terminalizes the dead attempt as
failed, writing its terminal status file, before it starts a new attempt.
Each rerun is a new generation/attempt, rather than reuse of a prior failed
scheduler token.

Local mutation and the complete plan/initialize/submit critical section hold a
shared-filesystem migration attempt lease. Recovery must acquire that lease
nonblocking before declaring an owner dead or a submission abandoned; hostname
and PID fields are diagnostic only across hosts. For a ledgered SLURM attempt,
only an explicit known terminal scheduler state permits interrupted
terminalization. Every active/held/requeue/signaling state preserves the attempt,
and an unknown state fails closed.

**Tables v1 -> v2 (#281 spec D10).** Migrate converts a store still in the v1
flat tables layout into run folder `tables/1-<sha12>/` (`1-unknown` when no
digest is known; `v1_table_pipeline_sha`: latest figure run, else latest
`full`/`legacy` journal application) with `convert_tables_v1_to_v2`. The step
runs in the per-image task (`_cli_migrate_image`, local and SLURM image array)
after the table stage, and republishes a record that was current before it via
`_republish_table_marker`; the run proof follows at finalization. **Why
in the image task, not a separate pass after it:** the conversion rewrites the
store root, so the record stops verifying. Inside the task it is republished
*before* the marker stage re-validates it, with its identity kept verbatim.

**"Identity verbatim" means these fields, and it holds by construction on
both paths** (#281 P3 review I1, I2). `_republish_table_marker` re-reads the
record and passes back its `work_id`, `relative_image_path`, `mode`,
`attempt_id`, `provenance` and `source_provenance`
(`publish_image_success(provenance=...)`; before the fix it had no such
parameter, and a `"migrated"` record came back `"forward"`). When the record
is *not* current -- a crash between the conversion and its republish leaves
a v2 store and a record naming the vanished flat path -- the marker stage
republishes it from `_existing_marker_identity`, which reads **the record
first** and the legacy `image_complete/` marker only when no record carries
an identity; it keeps the same fields, and borrows nothing from a record for
another image or another non-migrated `work_id`. The one field not kept is
`lifecycle_epoch`: both paths stamp the live migration generation, because
under SLURM `publish_image_success` refuses any other epoch. Pinned by
`tests/unit/cli/test_migrate_rerun_v2.py`, which also checks that the crash
path and the clean path publish the same identity.

**Which table migrate certifies** (review C1, M2): the run folder the image's
record certifies, when it certifies one of this store; otherwise, for an
image with a legacy external Parquet, the run its import owns
(`import_table_run_id`, the one rule the importer
`replace_embedded_measurement_table` and migrate's contract and exactness
checks share); otherwise the record-less default, the latest run holding
`measurements`. Re-certifying the latest instead silently moved a tree
measured A, B, A to pipeline B, and judging the latest while importing into
A's folder kept a multi-run store from ever converging. Spec D10 needs only
conversion, then the record, then the run proof. Every forward mode refuses a v1 tree before
`--overwrite` clears it and before `--dry-run` exits (pinned in
`test_migrate_tables_v2.py`). `--overwrite` does not bypass it; migrate or
delete the tree (parity with the `.h5` refusal). A
provenance-only target gets it in its store array
(`convert_provenance_store_tables`). The legacy-Parquet import,
`replace_embedded_measurement_table`, writes v2 directly, so no tree is left
mixed.

## Per-image figures

Every `PlotImage` binding's figures are written **into the image's store**, in
one folder per run, `figures/<date>-<pipeline sha[:12]>/`. `deliverables/plots/`
is then filled by copying that folder out. The store layout and descriptor are
in the `working-with-ome-zarr` skill. The build is
`build_image_figures` (`plotting/_pipeline/_store_figures.py`). It writes
nothing, and no figure error can fail the image. The store write is
`write_image_figures` (`sdk_/_image_figures.py`), and the copy-out is
`publish_store_figures` (`plotting/_pipeline/_store_copyout.py`).
`PlotCoordinator.emit_image` is **retired**. Nothing renders an image plot
straight to `deliverables/` any more.

| Mode | Build | Store write | Copy-out |
|---|---|---|---|
| Full (`_cli_process_single.py`) | after `apply_and_measure` | `save_image_store(figures=)`, inside the root-last transaction | yes |
| Staged Stage 1 | only the bindings `split_pipeline_at_gpu` gave to `pre_pipeline`, after the pre-GPU ops | Stage 1's `save_image_store(figures=)` | no; Stage 3 publishes |
| Staged Stage 3 | `keep_image_figures` for Stage 1's bindings, merged with `build_image_figures(post_pipeline, keep_from=<Stage-1 store>)` | `save_image_store(figures=)` | yes |
| Measure | after `measure()` and **before** the table replace | `replace_image_tables(figures=)`: the same root-last transaction as the tables | yes |
| Process, `--process-format zarr` | after `pipeline.apply` (and `embed`), before the provenance status is closed | `write_process_only_layer(figures=, embeddings=)` | no `deliverables/` |
| Process, `tiff` (and the `--layer objmap` PNG) | none for the exported file | none | none |

- **Copy-out runs after the store is promoted and before anything marks the
  image complete.** Full mode: promotion → copy-out → the caller's completion
  record. Stage 3: promotion → copy-out → Stage-3 marker → token → raw. Measure:
  the table-and-figures transaction → copy-out → marker refresh. A crash between
  promotion and copy-out therefore re-runs the image. The copy-out is
  best-effort. It publishes only **this run's** folder. It verifies each file
  against its `sha256`. It writes an `.html` for each stored `plotly-json`, and
  it never renders a PNG. Every failure, including each descriptor `failed`
  entry, becomes a `.failures.jsonl` line; only `PlotPublicationBlocked`
  propagates.
- **One run date for the whole run.** The CLI captures the initial call once
  (`mint_run_initiation`): its UTC date, UTC timestamp and pid. It records them
  in `state.config` as `figures_run_date`, `initiated_at_utc` and
  `initiated_pid` before any worker starts, and a resume reuses them
  (`_run_initiation` in `phenotypicCLI.py`). `--restart` and `--overwrite` mint
  a new call. Measure mode keeps no run state. Locally it mints its own call; on
  SLURM its submitter writes the same three keys into `job_metadata.json` before
  fan-out. In-process workers get the call on the `OutputManager`. A worker in
  another process reads it back with `recorded_run_initiation` (state) or
  `metadata_run_initiation` (measure on SLURM). **It never travels on a command
  line**: no worker argument and no generated script carries it. It is not in
  `processing_configuration_digest`, so a new day never invalidates
  continuation, and a run that crosses midnight stays in one folder.
- **Measure mode reuses a folder.** If the store already has a run with this
  pipeline's sha256, measure mode takes the most recent such run's date
  (`latest_run_date`). It then overwrites that folder's measurer figures and
  keeps its §3a figures. Otherwise it writes a folder dated by its own call. The
  digest is read from the pipeline file measure mode runs, not from the store's
  journal.
- **§3a figures (`FigureInputUnavailable`).** A binding whose `inspect()` raises
  it is kept from the same run's folder, sha-verified and all or nothing, when
  the store being replaced has one. Otherwise it is listed in the run's
  `unavailable`, not in `failed`. It is never copied between run folders.
- **Nothing is wiped.** Every store rewrite carries the other runs' folders
  (`carry_figure_runs` in `_write_store_part`; the measure rewrite keeps them as
  hard links). Only `--overwrite`, which deletes the output folder, removes
  them.
- **A run folder that cannot be named** (no pipeline digest, a malformed date)
  gives the image no figures. It records one `.failures.jsonl` line with
  binding id `"<run>"` (`name_figure_run`), and the image still completes.
- **Continuation.** Process mode bumped `PROCESS_LAYER_SEMANTICS_REVISION` from
  2 to 3, so a process tree resumed across the upgrade re-derives, `tiff`
  included. Full mode has no revision: a full run resumed across the upgrade
  reuses figure-less stores, and `--overwrite` fills them in. `--mode migrate`
  never adds a figure.

## Durability (`--durable-writes` / `--no-durable-writes`)

Whether each per-image store is `fsync`ed before its promote. The flag is a
**tri-state**: unset means *auto-detect* — on under SLURM, off locally — and
`None` is carried end to end, resolved to a bool in exactly one place,
`ngff_._resolve_durability`, which also produces the sentence logged at run
start so the flag and its description cannot drift. **Do not resolve it
earlier**: that would freeze the submitting node's environment into a value a
worker on a different node then reuses.

Durability lives on the **`OutputManager`**, not on each call.
`save_image_store(durable=None)` defers to `self.durable_writes`, which is what
makes it structurally impossible for a write site to be silently inert. There
are exactly **four** forward write sites — `_cli_process_single.py`,
`_cli_staged_workers.py` Stage 1 and Stage 3, all through `save_image_store`,
and Stage 4's `replace_run_embeddings` (`_cli_stage4_worker.py`
`_commit_image_once`, #282). The fourth is an `sdk_` function, not an
`OutputManager` method, so it cannot defer to `self.durable_writes` and is
handed `durable=ctx.output_manager.durable_writes` explicitly, the same
unresolved tri-state (P3–P4 review I2); the Stage-4 SLURM shard builds its
`OutputManager` from the transported flag. Everything else that appears in a
grep is a **transport** site: a fresh process that must be handed the flag on
its command line, namely the staged SLURM worker, the ordinary per-image SLURM
array (`_cli_slurm_array_scripts.py` → `python -m phenotypic._cli._cli_process_single`),
and the staged script generator. A new spawn site needs the flag threaded, or
the option is inert on that path alone.

**That is the rule for this one flag, not the pattern for a new per-run
value.** A value that must be the same for every worker, stage and resume
belongs in the run's recorded state, and workers read it from there. The figure
run date and the initial call's timestamp and pid work this way (see
**Per-image figures**). They are recorded in `state.config`, or in
`job_metadata.json` for measure mode on SLURM, and they are never handed to a
worker on its command line.

`durable_writes` is deliberately **not** part of
`processing_configuration_digest` (`_cli_failure_tracker.py`, an explicit
allowlist). Durability is a storage guarantee, not a scientific parameter —
folding it into the work id would make `--no-durable-writes` restart a finished
run from zero.

Rejected with `--mode recompile` and `migrate`: those
modes write no image store from a pipeline, so the flag could only mislead.

## The record is the last write; one producer per derived value

**The rule the state-tracking design rests on, and it is load-bearing in two
different places.** Both were learned the expensive way in the
`cli-gui-state-tracking` change; the reasoning is in
`docs/superpowers/reports/2026-09-03-cli-gui-state-tracking/`.

### Nothing is rewritten after the record that certifies it

**Each worker owns exactly one image, and within a pass nothing it certifies is
written after the per-image record that certifies it.** The record is the
pass's last write for its image. Retries, `--restart` and `--mode measure` are
separate invocations, seconds to hours apart.

This section used to read "one writer per artifact, per pass", and #282 is why
it was restated (plan review). In a run that embeds on the GPU the store root
is written **twice** in one pass: Stage 3 (or the single-pass core) promotes
it with the measurements, Stage 4 promotes it again through
`replace_run_embeddings`, and only then is the record written. After a
walltime between Stage 4's commit and its record, the next round may write the
image's embeddings a second time (P6 review M2, a recorded follow-up). Both
rewrites happen **before** the record exists, and the record digests the bytes
they left, so neither can leave a stale verdict. A `--mode measure` pass does
the same over an already-certified store (measure core, then Stage 4, then the
republished record). There the forward record may go on verifying throughout,
since a same-pipeline pass can leave the root byte-identical (#282
`live-smoke.md`, M1); that is why a measure pass's "done" is a record stamped
with its own epoch, never a record that merely verifies.

That is not a convention — **it is what makes the verification cache correct.**
The cache fences on `(size, mtime_ns)`, which cannot detect a same-size rewrite
landing inside one filesystem `mtime` tick. It does not have to, because no
rewrite lands after the record whose digests it caches. Measured on GPFS `/bigdata`: 0 of 200 back-to-back same-size
writes shared an `mtime_ns` (~81 µs resolution); node-local scratch is 181/200 at
~1 ms, which is one more reason run output may never live there.

**So a change that makes any component rewrite a tracked artifact after its
record within a pass silently invalidates the cache's soundness argument** — and nothing will
fail, because the cache will simply keep answering from a stale verdict. If you
need a second write, say so where the cache is defined, not only where you write.

### One producer per derived value

**A value derived from state has exactly one producer: the reader of that
state.** Do not compute it a second time somewhere earlier and carry it.

`inventory_digest` is the worked example. It is `canonical_digest(work_ids)` —
a pure function of `processing_state.json` — and four sites compute it that way
and agree: `sdk_/_run_state.py` and the three proof writers in
`_cli_completion.py`. A fifth producer in `_cli_identity.py` derived it from the
`--image-manifest` digest instead, which is `None` by default. So the field whose
job is answering *"did the accepted scope change?"* answered **"no"**
unconditionally, and could never equal the reader's value for any input.

**The tell was the ordering.** `mint_run_identity` runs before
`state.config["work_ids"]` exists. A value that cannot be computed at the point
you are standing is a value you do not own — the fix was to leave it empty and
document the reader as its owner, not to invent a stand-in.

> **Before adding a producer, ask what already computes this and whether you can
> reach it.** If you cannot reach it because of ordering, you are not its
> producer. If you cannot reach it because of layering, the shared definition
> needs a home in `sdk_` — the layer the CLI, the GUI and the readers may all
> import. It is never a reason to restate it.

## Per-image completion markers

`publish_image_success` certifies the artifacts an image produced;
`valid_image_success` is the first conjunct of resume classification. Per-image publications are **versioned**, and there are **two constants, not
one renamed**: `RECORD_VERSION` (`sdk_/_image_record.py:87`, currently **1**)
stamps the per-image record `images/<ds>/<stem>.json`, and
`SUCCESS_MARKER_VERSION` (`sdk_/_io_constants.py:741`, currently **2**) is the
legacy `image_complete/` marker's, still written by the HDF→Zarr migrator
(`sdk_/_hdf_to_zarr.py:607,645`). A version mismatch invalidates rather than
migrates. Do not document either as having replaced the other; both have live
writers.

Never hand-declare the per-image data artifact. Call
**`image_data_artifact(output_dir, output_manager, dataset, image_stem)`**,
which returns `("store", <store dir>)` when a store exists and `("hdf", ….h5)`
otherwise. Three separate clusters each broke a run by declaring `.h5`
directly after the writer had moved to the store — `publish_image_success`
resolves every artifact `strict=True`, so the image fails *after* completing
all of its work.

Descriptors dispatch on `kind`. A **store** is fingerprinted by its root
`zarr.json` alone, not recursively: the root is written **last** by the promote
protocol, so a valid root implies a complete store. An absent `kind` reads as
`"file"` (v1 shape); an unknown `kind` **fails closed**.

> **"Nothing writes into the store after publication" is FALSE, and root-only
> fingerprinting is sound anyway.** The stronger claim was written here and is
> withdrawn — the table replacers have **three** call sites, not the one
> (`_cli_migrate_image.py:281`) an earlier note named. The third,
> `OutputManager.replace_image_store_measurements` ←
> `_cli_process_single.py`, runs on the ordinary **`--mode measure`** path and
> writes into a store that is already published. *(P4 split the replacer in
> two: that path now calls `replace_image_tables`, which moves the measurement
> table, the metadata table and the root's `metadata_table` block together;
> `replace_embedded_measurement_table` survives for `--mode migrate` until its
> producer is repointed. Both go through the same root-last transaction.
> `--mode recompile` was the third writer until 2026-09-11; its per-store
> rewrite was removed rather than repointed, so recompile is now aggregate +
> finalize and writes no store byte.)*
>
> What actually holds is narrower and is what the fingerprint needs: **every
> table replacement is a root-last store transaction**, so the root is rewritten
> last exactly as at promote time.
>
> **There used to be a second write shape, and P4 deleted it.** When the
> descriptor was unchanged, `replace_embedded_measurement_table` took a
> same-directory atomic file replacement and returned **without touching the
> root** at all. That preserved *"a valid root implies a complete store"* —
> which is all the fingerprint needs — but it also meant the per-image proof's
> store digest still matched after the table's bytes had changed underneath it,
> and it left the root's recorded metadata snapshot stale. Both failures were
> silent. After P4's inversion the descriptor became a pure function of the
> measurement schema and the objmap target, so *every* metadata-driven
> re-measure would have taken that branch. The fast path is gone; the cost is
> that a `--mode measure` re-promote now runs on every image it touches.
>
> So the root never observes a partial store. The root fingerprint is
> a **completeness** check, not a content-version one, and it was never asked to
> distinguish two tables carrying the same columns — the descriptor holds
> `schema_version`, `type`, `format`, `path`, `measurement_columns` and `target`,
> and deliberately no row count or content digest.
>
> The measure path then re-publishes the marker through
> `_republish_table_marker` (`_cli_process_single.py:462`), which rehashes every
> artifact and writes the marker last, so no store write outlives the
> publication that certifies it.
>
> **Why the wording mattered even though the conclusion did not.** A reader
> checking the guarantee would have found a counterexample on the first grep and
> had no way to tell whether the fingerprint was unsound or merely
> mis-described. Overstating a true invariant costs the next reader the ability
> to verify it — and the overstatement came from a citation that named one call
> site when there were three, which is the third time in this change a dismissal
> rested on an under-counted citation.

The `"hdf"` branch is **not** dead code, though no forward path writes an
`.h5` any more. `_migrate_legacy_success_evidence` promotes trees from older
releases, which have `results/<ds>/hdf/<stem>.h5` and no store; dropping the
branch would make it name a nonexistent store and silently refuse to promote
the very trees it exists to rescue — reprocessing all of them.

---

## Tracked state, and how everything else is derived

**The full reference now lives in the contributor guide:
[`docs/source/contrib_guide/tracked_state.md`](../../../docs/source/contrib_guide/tracked_state.md)**
— *Run state: what is tracked, and how to read it*.

It is the single source for the four tracked states and their writers, the
three content proofs and their publication order, the derived-value table
naming the function behind each fact, the two retained-but-unread artifacts,
the table of **known consumers** and what each decides, and the two consumers
that deliberately ask a different question.

Read it before adding any counter, flag, marker or cached count to
`.phenotypic/`.

The three things worth carrying in your head without opening it:

- **Four things are written down**, and every other answer is computed from
  them plus the artifacts on disk. The organising principle is *move state that
  is tracked to state that is checked*. **A fifth tracked value appearing is a
  design regression** — the page's last section is the checklist that decides
  whether a proposed one is really derived.
- **`resolve_run_state(output_dir, depth=...)` is the one call.** It never
  raises; an unreadable or foreign tree degrades toward `incomplete`. Use
  `depth="deep"` for anything that writes, `depth="shallow"` for listing and
  polling.
- **Readers live in `sdk_`; writers stay in `_cli`** (INV-LAYER: `sdk_` may
  never import `phenotypic._cli`).

> **If you add, remove or change a tracked state, a proof, or a consumer,
> update that page in the same change.** It carries a checklist for exactly
> this, including adding a row to the consumer table and adding an
> `sdk_/_io_constants.py` path helper rather than hand-joining a name. A
> reference updated one change later is a reference nobody can trust in
> between, and a row that describes something no longer true is worse than a
> missing row — it carries the authority of documentation while being wrong.

---

## What `--mode migrate` asserts, and what it does not

**It converts schema. It does not finish a run.** Everything below follows from
that one sentence, and most of the surprises in this area come from expecting
otherwise.

- **A migrated-but-incomplete tree reports `incomplete`, and that is correct.**
  `publish_run_completion_evidence` refuses to write a run proof when some
  accepted image has no success record, and migrate does not override it.
  Writing one there would certify a run that never finished.
- **`stage2_done/` is read, never moved.** It holds a *consumable token* that
  Stage 3 replays and then `unlink`s. Migrate reads it to enrich a record and
  leaves it exactly where it is; renaming it into `legacy-v2/`, where nothing
  reads it, would orphan the work of any staged run live across the migrate.
  Only `image_complete/` and `stage3_complete/` move.
- **`deliverables/metadata.<ext>` is byte-exact provenance**, with no exception for
  migrate. The canonical view is emitted *alongside* it as
  `metadata.canonical.tsv`; `metadata.original.csv` does not exist and must not
  be created.
- **The master is parquet-only** (D8). `master_measurements.csv` is deleted on
  sight, not rewritten.
- **`work_id`s are not re-minted** (D-C), so every existing one stays valid.
  `record_rejection` (`sdk_/_image_record.py:207`) skips the `work_id`
  comparison for `PROVENANCE_MIGRATED` records — a migrated tree's identity
  cannot be re-derived, so it is marked unavailable rather than fabricated.
  **Absent means `"forward"`**, so a writer that forgets the field produces a
  fenced record rather than an accepted one, and `resolve_run_state` advises
  when the relaxation is in effect.
- **Not every record written during a migration carries `"migrated"`**, and the
  difference is the code path rather than the mode: the state migrator stamps
  it (`_cli_migrate_state.py:350`, `:1080`), while migrating the image artifact
  itself goes through `publish_image_success` (`migrate_image_task`), whose
  `provenance` defaults to `PROVENANCE_FORWARD`: it keeps an existing record's
  provenance (review I2) and is otherwise `forward`. **Do not use record
  provenance to ask whether a tree was migrated** — `_output_was_migrated` keys on the migration manifest
  for exactly this reason. Full table in
  [`tracked_state.md`](../../../docs/source/contrib_guide/tracked_state.md)
  under *Provenance has two writers*.

### A migrated tree is not continuable, and `--restart` is the remedy

User ruling (2026-09-09). Migration rebuilds the admitted image set from the tree
rather than from the original run's record of it, so continuation compares
against a set the inputs no longer match. **That is accepted rather than
repaired**: such a tree *"should be considered as if not ran then, and do a full
restart"*.

The refusal names the command — keyed on `.phenotypic/migration_manifest.json`,
which is a declared fact rather than an inference. **Told, never done**: clearing
machine state and reprocessing every image is hours of compute and a destructive
step, and firing it automatically from a condition the user did not ask about is
the hidden state transition this change exists to remove.

Not every migrated tree is affected — a legacy tree **with markers** carries real
filenames in `work_ids`, so nothing is rebuilt and it continues normally.

---

## Known limits — read these before trusting the above

A register that records only what works is the shape this change exists to
remove.

- **`work_id` does not fold in `restart_epoch`** (`_cli_failure_tracker.py:331-339`).
  So a pre-migration run's journalled failures still key-match after a
  `--restart`, and a later plain re-run reports *"N recorded failure(s)
  skipped"* for images the restart just reprocessed successfully. Inherited, not
  created by migration; `--retry-failures` is the escape.
- **Migration pollutes `initial_images` on a pre-markers tree**, adding bare
  stems beside the real filenames, because `_ensure_migration_processing_state`
  falls back to the stem when `work_ids` carries no filename for it. **Tolerated,
  not fixed**, and pinned by an `xfail(strict=True)` in
  `tests/integration/cli/test_migrate_end_to_end.py` so a future repair is loud.
  It is inert only because a migrated tree refuses continuation *by rule* — if
  that refusal is ever relaxed, this becomes live again.
- **O-3 and O-4** are recorded in
  `docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/OPEN-QUESTIONS.md`.
- **A walltime kill between `promote_store`'s two renames can lose a store**
  (`sdk_/ngff_.py` `promote_store`, "Known weakening"): the old store has been
  moved aside and the new one not yet renamed into place, so the image is absent
  and an orphaned `.trash` is left; absence reclassifies the image to the stage
  that rebuilds it, and `sweep_orphan_parts` clears the leftovers. Pre-existing,
  and a fix belongs to the store writer. **A Stage-4 run crosses that window
  once more per image than an inline run**, since `replace_run_embeddings` is a
  second promote after Stage 3's (#282 plan review, Minor).
- **#282 Stage 4's own limits** (spec §4.8, U14): there is no round cap and no
  per-image strike count. An image whose crops are huge (sprawling objects;
  smoke F3: up to about 35 min of preparation for one image) commits slowly
  but every round progresses, so the stall rule never fires; and a "poison"
  image that crashes its shard every round (a host OOM) ends the run with exit
  `5` once only it remains, and a plain re-run repeats that.

### Process trees, and the one target kind that is not a tree

**Machine-state conversion follows a shared predicate, not the dispatch.**
`target_kind_owns_machine_state` (`_cli_migrate_provenance.py:56`) is true for
`full_run`, `process_tree` and `pre_markers_process`, and **false for
`direct_store`**. The older `kind != "full_run"` test asks *is this not a full
run?* and was read as *is this per-store provenance work?* — which is false for
two of the three provenance-only kinds.

`direct_store` is excluded **by kind, not by luck**: a single store keeps its
lifecycle state in a hashed sibling, never inside itself, so converting there
would write `.phenotypic/` where that kind's own contract forbids it.

**A pre-markers process tree used to migrate to nothing, successfully.**
`execute_provenance_migration` iterates `target.stores`, and such a tree has
`stores=()` by construction — its outputs are process layers, not stores — so
the local arm did no work, reported `provenance_upgraded=0`, and **exited 0**.
The user saw success and got none. It now converts at
`_cli_migrate.py:1647`, after a clean store upgrade and only when there are no
failures.

**`--slurm` on a storeless provenance-only target is refused, by design.** It is
a `click.UsageError` naming the remedy, not a missing feature: the SLURM chain
has no vocabulary for such a tree. `seal_provenance_migration` barriers store
statuses and the finalizer reports upgrade counts, so the tree would come back
*"0 upgraded, succeeded"* with the only real work — the machine-state conversion
— invisible in its own terminal report. **Run it without `--slurm`.** Two
independent guards already refuse the shape, which is what makes it a topology
mismatch rather than an oversight in one place: the manifest writer raises on
zero tasks, and the worker config loader's `target_kind` cannot even parse a
config naming this kind.

---

## Readers live in `sdk_`; writers stay in `_cli`

`resolve_run_state` and the run-state readers are in
`phenotypic.sdk_._run_state`. Writers — anything that publishes a proof, bumps an
epoch or appends to a journal — stay in `phenotypic._cli`, because `sdk_` may not
import `_cli`. `tests/unit/sdk_/test_run_state_layering.py` enforces the
direction with an AST walk, so a violation fails there rather than at import.

**There is no migration version floor.** An earlier draft named v0.17.3 and it
was withdrawn (U-6): `state.version` is a *state-schema* version, not a package
one — `"2.0.0"` is its value both at v0.17.3 and immediately before `"3.0.0"`
arrived — so there is no version string to refuse on. Detection is by **shape**,
the pre-markers signal is an absent `work_ids` key, and a pre-markers tree is
supported however old. `ConversionVerdict` has deliberately no `BELOW_FLOOR`
member (`sdk_/_schema_shape.py:192`).

## Environment variables (important for future work)

- `PHENOTYPIC_STAGE4_DEVICE` — **test-only**: `cpu` runs Stage 4 on the CPU
  (`stage4_device`, `_cli_stage4_worker.py`), so a CPU node can exercise it;
  `stage4_cpu` and the subprocess crash runner set it. Without it Stage 4 is
  `cuda` and never falls back.
- `CUBLAS_WORKSPACE_CONFIG` — deterministic cuBLAS needs `:4096:8` or `:16:8`
  (`DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS`, `_batch_encoder.py`). The CLI
  `setdefault`s `:4096:8` right after `--embed-stage` resolves `gpu`
  (`prepare_stage4_environment`), before any strategy can create a cuBLAS
  handle, and keeps a preset `:16:8`; `stage4.sh`'s prelude exports `:4096:8`
  before Python starts; `configure_determinism` refuses any other value on CUDA
  (`check_cublas_workspace`).

- `PHENOTYPIC_PRELOAD_MODULES` — comma list of self-registering modules that
  **every process that deserializes a pipeline** imports:
  `SerializablePipeline._find_class_in_phenotypic` calls
  `preload_custom_operation_modules_once` (`sdk_/_preload.py`) before its first
  lookup, hit or miss, and again on a miss. That reaches the main CLI, staged
  and ordinary SLURM workers, local loky workers (fresh processes that never run
  `main`) and the finalizer by construction, rather than by a list of call
  sites. The main CLI also imports them at startup, so a broken name is one
  error line naming the variable. `sbatch --export=ALL` propagates it. Spec
  `2026-09-24-cli-preflight` §10.2. (Tests use
  `tests/_fakes/register_fake_gpu.py` and
  `tests/_fakes/register_custom_threshold_detector.py`.)
- `PHENOTYPIC_SLURM_PYTHONPATH` — internal submission snapshot of the caller's
  `PYTHONPATH`. Generated batch scripts restore it before invoking Python. This
  keeps custom-operation modules and the reviewed source checkout importable on
  clusters that filter raw `PYTHONPATH` even when `sbatch --export=ALL` is used.
  Callers set `PYTHONPATH`; PhenoTypic owns the namespaced snapshot.
- `PHENOTYPIC_ACCEPT_MODEL_LICENSE` — comma list of model names accepted for
  gated-weight downloads; checked by `require_license_acceptance`
  (`detect/nn/_checkpoint_manager.py`). SAM2/micro-sam are ungated and never call
  it; the hook exists for Spec 2's gated models (SAM3, DINOv3). Licensing
  scaffolding: root `NOTICE` + `licenses/*.txt` + `MANIFEST.in`.

## Gotchas

- **GPU detectors stage automatically in the CLI** — `op.apply(image)` in a
  notebook is unchanged; a `GpuDetector` in a *CLI* run triggers the staged
  engine, not per-image processing.
- **Output layout** — see the **Output layout & deliverables** section below for
  the full inventory and master-vs-mirror rules.
- **HPCC SLURM heterogeneity (polars build)** — the cluster has pre-AVX2 nodes where the
  stock `polars` wheel SIGILLs ("Illegal instruction"). The project depends on
  `polars[rtcompat]` (a runtime-CPU-dispatch build that runs on pre-AVX2 nodes without a
  per-node wheel swap); numpy/scipy use runtime SIMD dispatch and are unaffected. See
  `docs/source/how_to/pages/polars_cpu_build.md`. Stage 2's GPU work runs on GPU nodes.

---

## Output layout & deliverables

User-facing run outputs live under `<output>/deliverables/` (hard cutover):
`master_measurements.parquet` (**parquet only** since D8), `measurements.{tsv,csv,parquet}`,
`measurements_by_feature/<feature>.{tsv,parquet}`,
`measurements_by_category/<label>.{tsv,parquet}` (one per `CATEGORIES` member
with a present column; written beside the feature split by
`finalize_post_master_outputs`),
`<AnalysisClass>.{tsv,parquet}`, `analysis_manifest.json`,
`metrics.json` and `schema.json` (#281 spec C4/C12: the run summary and the
column schema of every published table, written by `finalize_run`'s
`publish_run_summaries` after the splits and before the aggregate proof, which
declares and enforces both; byte-stable canonical JSON, never read back, so
not tracked state; `metrics.json`'s `pipeline_sha256` is the digest every
aggregated table shares and is `null` for v1 flat tables, which record none),
`embeddings/<model>/<layer>/<region>.parquet` and `embeddings/manifest.json`
(the key and the path are `<model>/<layer>/<region>`: the layer is `rgb` or
`gray`, its own folder, so an rgb and a gray copy of one embedder coexist; the
region is `objects` or `grid`, or `objects-<name>` / `grid-<name>` for an
embedder given a `name`, so one pipeline can publish several embedders of one
model, layer and region, e.g. rig1's `rgb/grid-wells` and `rgb/grid-agar`; #281 spec E5, `_cli_embedding_deliverables.py`, in the finalizer task
itself, never in the `0-K` aggregate shards. **Two phases, so a failure
changes no deliverable** (P4 review I1): `stage_embedding_deliverables`
runs *before the master is written*, reads every non-stale entry of each
image's **certified** run folder once (present, its sha256, its type, one
schema per key) and streams each key to a hidden temporary, one row group
per store; `promote` runs after the splits and before `metrics.json`,
replacing each target and writing `manifest.json` last. A `stale` entry
(`stale is not False`, the one predicate) is skipped with a warning and
counted; a non-stale entry it cannot read, or any entry whose key is not
`<model>/<layer>/<region>` (`split_embedding_key`, the one rule for keys, which
`EmbeddingPayload` refuses on; nothing else restates the region set),
**raises**, failing finalization
like the master, naming every such store, and the image stays `verified`
(D9), so the remedy is a same-pipeline `--mode measure` over the whole run,
which re-embeds every image (user decision Q2: no scoped remedy; on SLURM,
`phenotypic-wait`'s exit `5` points at `slurm_logs_dir`, where the
finalizer's log says this). A key whose stores carry **different entry
descriptors** (entry minus `binds`/`files`/`stale` and the `compute` block
(#282), and minus `model.device`/`model.device_name`: an unpinned `revision` resolving to
different commits, a partly re-measured tree) still publishes,
with one warning per key, and every row carries an `int16`
`descriptor_index` into the key's manifest `descriptors`, listed in order
of first appearance over the sorted stores, so bytes stay stable; it is
`0` everywhere for one descriptor, so the schema never depends on the
data (Q1). **The device is provenance, not descriptor** (user decision
2026-11-19, measured: CPU vs GPU ~1e-6 cosine distance, two GPU models ~one
float16 bit): each store entry records `model.device` and, on CUDA,
`model.device_name`, and `_shared_descriptor` drops both before grouping, so a
run mixing CPU and GPU, or several GPU models (#282 U6), publishes one
descriptor with no warning. `compute_dtype` is **not** dropped: tables computed
in different dtypes never group (an entry without it reads `"float32"`,
`_ABSENT_DESCRIPTOR_DEFAULTS`). The preflight's `PF-EMBED-REVISION-UNPINNED` warns on an
in-scope `MeasureDinoEmbedding(revision=None)`. On the `shard_paths` branch only the
sources the projection kept contribute. Paths are compared unresolved, so a
symlinked `results/` and a DVC symlink cache work. Parquet only, never
joined to metadata. The aggregate proof declares `manifest.json` and each
published table (no GUI writer touches them). A finalization deletes only
the `*.parquet` and `manifest.json` a previous one wrote and it does not,
then empty directories. Every per-image writer passes `pipeline.embed(image)`
beside `measure()`, measure mode included, unless the run defers its
embeddings to Stage 4 (`defers_embeddings`, #282), when Stage 4's
`replace_run_embeddings` writes them instead:
`replace_image_tables(embeddings=None)` drops the run's embeddings; an
embedder's `MemoryError` or torch `OutOfMemoryError` propagates unwrapped,
so it is never a terminal failure),
`plots/<plot-id>/...`,
`dashboard.html`, `README.md`, the on-demand `processing_report.html`
(written only by `python -m phenotypic.sdk_.generate_report`, since 0.21.0),
`pipeline.pht-pipe.json`, and `overlays/<ds>/<stem>.png` (detection overlay PNGs). The
pipeline config is **seeded** from the `--pipeline` bytes when a full or
staged run starts (`_seed_pipeline_config`, only when no config -- canonical
or legacy -- exists, under `pipeline_publication_lock`) and rewritten from
the loaded pipeline at finalize; nothing is copied to the output root any
more. The `--restart --image-manifest` freshness check treats that seeded
file and its lock file as scaffolding. The
dashboard is progress-only: local runs render progress directly, while SLURM
runs add Progress and Download tabs. Use the Results Viewer or the GUI
`/analysis/` app for interactive exploration. Each per-image **OME-Zarr store** stays at
`results/<ds>/zarr/<stem>.ome.zarr/`. Its authoritative object measurements
are embedded per measuring pipeline, at
`tables/<n>-<sha12>/measurements/table.parquet`, described by
`attributes.phenotypic.tables.runs.<n>-<sha12>.measurements` in the store
root (#281 spec D1-D3). A store written before that layout keeps one flat
`tables/measurements/table.parquet`; every mode but migrate refuses such a
tree (`_cli_tables_v1.first_v1_store`), and `--mode migrate` converts it. Forward,
staged Stage 3, and measure runs do not create
`results/<ds>/measurements/<stem>.parquet`; that directory is legacy migration
input only. There is no per-image `.h5` on any forward path: `results/<ds>/hdf/` appears only in a tree written by a pre-store
release, or in one migrated with the default `keep_source=True`, and the only
things that read it are `--mode migrate`, `datasets_needing_migration` (the
predicate every writing mode refuses on), and the `"hdf"` completion-marker
fallback described above. Machine state lives under
`.phenotypic/`: `progress_dir(output)` resolves
`<output>/.phenotypic/progress/` and `processing_state_path(output)` resolves
`<output>/.phenotypic/processing_state.json`; the corresponding `resolve_*`
helpers retain legacy root-level reads. The durable
**QC + curation state** lives under `deliverables/qc/` (`qc.duckdb`,
`review_state.json`, `curation_labels.parquet`, `custom_categories.json`) so a
`deliverables/` bundle is self-contained and GUI-openable standalone;
`BundleLayout.qc_dir` / `migrate_legacy_qc` still read/move a pre-relocation
root `qc/`. `run_qc` writes the
single `deliverables/qc/qc.duckdb` (one self-describing table per QC module plus a
`qc_modules` catalog, atomic full rebuild). Resolve these paths via the
`phenotypic.sdk_` helpers (`deliverables_dir`, `master_measurements_parquet_path`,
`qc_dir`, `qc_duckdb_path`, …), never by hand-joining names.

**Master vs. mirror.** *(Rewritten by P4 — spec §7.3. The paragraph this
replaced described the pre-inversion contract in every particular, and each
particular is now false.)*

Each embedded table carries **measurements alone**; the image's own rows of the
run's metadata snapshot sit beside it in the same run folder, at
`tables/<n>-<sha12>/metadata/pht-metadata.parquet`, and the run's entry in
the store root records which snapshot it was built against in
`tables.runs.<n>-<sha12>.metadata_table.snapshot_sha256`.
`master_measurements.parquet` is the pre-post concatenation of
marker-authorized embedded tables, each **projected onto its own store's
descriptor** first, so it is **un-joined**: intrinsic identity
(`Metadata_Dataset`, `Metadata_ImageName`, the `IMAGE`-owned provenance block)
plus measurements, and no user metadata at all.

**Not every store is inverted, and the projection is what makes that safe**
(P7 Task 4). `--mode migrate` leaves each store's embedded table as the
pre-inversion producer wrote it: the metadata snapshot right-joined in, each
measured row repeated once per matching metadata row. Aggregated as-is, those
tables make a v1-shaped master, and the one global join below then joins a
second time on the user columns — on a real 6,657-image migration that dropped
every measured row from the mirror. So every read path into a master —
`build_master_frame`, the P5 fan-out shards, and the recompile shards — goes
through `project_embedded_measurement_table` (`_cli_parquet_agg.py`): keep the
descriptor's `measurement_columns`, collapse join fan-out on `target.column`
for a table recorded `joined`, and **exclude** a store with no descriptor or
whose same-label rows disagree. An excluded store is logged and left out of the
source set the aggregate proof certifies, so the run reads `incomplete` rather
than certifying an image the master does not carry. Both shard producers report
the set they merged to their finalizer as `planned_work_ids` — the P5 fan-out
through `resolve_finalizer_shard_inputs`, and recompile through the
`source_work_ids` its measurement statuses record — so an exclusion in a shard
is not counted by the proof.

**The join happens once, at finalization**, in `finalize_run`
(`_cli_finalize_run.py`) → `finalize_post_master_outputs` →
`join_metadata(master_df, metadata_csv, how="left")`. That one call identifies
its own common columns, so **nothing reads the stores' recorded join keys** —
which is why D-A is free to leave them inconsistent across snapshot
generations. Finalization no longer rejects mixed metadata digests; divergence
is an advisory, and an advisory is never a gate. It *does* still refuse mixed
**authority** — a tree holding both embedded tables and legacy external
Parquets (`refuse_mixed_measurement_authority`).

**Two generated CSVs and one input snapshot stay CSV** (#281 P1, spec B2/B9):
`measurements.csv` stays because the GUI's curation rewrites it and the
`.parquet` — never the `.tsv` — until #283, so until then a curated edit
reaches `measurements.csv`/`.parquet` and not `measurements.tsv`;
`error_analysis.csv` is written by the GUI; `metadata.<ext>` is the user's
byte-exact snapshot. **Finalize's stale-CSV sweep** is
`remove_stale_csv_sibling(tsv)` (`sdk_/_delimited.py`), called right after a
`.tsv` is published: it deletes only that file's `.csv` twin (the splits in
`_write_split`, `<AnalysisClass>` in `_emit_analysis_outputs`, and
`metadata.canonical.csv` in `emit_canonical_metadata_view`), and raises on the
stems `measurements`, `error_analysis` and `metadata` directly in
`deliverables/` (the same stem in a split folder is an ordinary split). A reader of a possibly
pre-cutover tree resolves the mirror text file with `measurements_table_path` /
`BundleLayout.measurements_table_path()` (`.tsv`, else `.csv`); analysis tables
go through the manifest, whose schema-1 entries map `csv` onto `tsv`.

`measurements.{tsv,csv,parquet}` is the post-applied, metadata-joined mirror the GUI
reads and curates. Metadata is the **left** frame, deliberately: a metadata
identity that matched no measured object survives as a phantom row with
`QC_MetadataOnly=true`, its metadata values kept and its measurement/info
columns null, while a measured object whose key appears in **no** metadata row
is dropped — an object outside the described experiment. The master keeps that
object; the mirror does not. That asymmetry is the master/mirror distinction the
"feed analysis and dashboards from the mirror" rule rests on. Per-feature and per-category
splits and named analysis artifacts derive from the mirror. Analysis consumers resolve
tables through `analysis_manifest.json`, never by constructing filenames.

**Reading a master written before the inversion.** Nothing stamps the file — a
v1 master carries **user** metadata because the join happened per image, a v2
carries only **intrinsic identity** metadata, and that *is* the discrimination.
Ask `phenotypic.sdk_.master_carries_user_metadata(frame)`, which is its one home
and carries its retirement condition; never re-derive the check at a reader.

**Ownership decides, never the `Metadata_` prefix** — both shapes carry it. A v2
master carries `Metadata_Dataset` and `Metadata_ImageName`, and `Metadata_Strain`
is itself a schema member (`GENETIC.STRAIN`), so neither the prefix nor
"is it in the schema" separates the two. A header is user metadata when
`is_metadata_header` accepts it and `metadata_owner_for_header` returns neither
`IMAGE` nor `EXPERIMENT.DATASET`. This is the general rule in root `CLAUDE.md`
("Metadata queries use schema ownership, never string prefixes"), and the master
is the case where getting it wrong classifies *every* v2 master as v1.

The limit that leaves: column provenance is not recoverable from a column name,
so a master carrying a non-`IMAGE` metadata column that a **custom operation**
produced reads as v1 though no CSV was joined into it. The forward pipeline emits
no such column, which is what keeps this a limit rather than a defect.

Ordinary SLURM checkpoints read only marker-authorized embedded tables and write
rolling cache state below `.phenotypic/progress/`. They do not recreate
visible per-image Parquets, dataset aggregates, or a partial deliverable master.

**Metadata snapshot authority.** Before local processing or SLURM submission,
full runs and recompile atomically copy the configured `--metadata` bytes to
`deliverables/metadata.<ext>` (the input's own `.csv` or `.tsv` suffix), verify the copy, and use that stable path for
finalization. The snapshot is source provenance, not a generated schema table:
legacy headers normalize only in memory, while finalization, recompile, and
explicit `--mode migrate` must leave its bytes unchanged. Recompile performs
no metadata migration preflight or mutation. All generated measurement,
analysis, QC, and REMBI outputs use canonical flat `Metadata_<Label>` headers.

`QC_MetadataOnly` is a **user-facing output column, not internal machinery** — it is how a
user filters the mirror for "which strains went undetected". Analysis and post code must
not branch on it. **QC checks (`QualityCheck` subclasses) and plotting code may**: their
verdicts and figures depend on the metadata join the user supplies (for example,
`ExpectedVsDetectedCount` with no layout derives its expected positions from the join's
phantoms). Code that reads the flag still defines what it does when the column is absent,
since a notebook frame never has one.
Analysis and post ops are public API (a notebook calls them on frames that never
saw the CLI and carry no flag), so they detect a phantom the same way they detect any
missing value: **drop/ignore NaN**. A phantom row is null in every measurement/info column,
so NaN-native math (`notna()`, `nanpercentile`, `np.isfinite`, `dropna`) handles it, needs
no flag column to exist, and is automatically a no-op on frames that have none.
Feed configured analysis and GUI result exploration from `measurements.parquet`, not
`master_measurements.*`.

**There is one FINAL master writer, and it is `finalize_run`.** Every mode
reaches it: the forward CLI and `--mode measure` through `aggregate_measurements`
(which is `finalize_run` under the publication lock), and the recompile SLURM
finalizer through `_run_post_master_steps`, which hands its per-shard Parquets
in as `shard_paths` rather than merging and writing a master of its own. That
collapse is the point of §7.4 — recompile is *"call `finalize_run` again"*, not
a second implementation to keep in sync — and it is also what keeps **one
writer per artifact, per pass** true of `master_measurements.parquet`. A new
code path that needs a final master calls `finalize_run`; it does not write the
file and then call `finalize_post_master_outputs` itself.

Mid-run checkpoint writers (`_aggregate_chunks_locked` in
`_cli_chunk_writer.py`) intentionally bypass it and keep their rolling state
under `.phenotypic/progress/`; post, per-feature and per-category splits, analysis, and
`pipeline.json` persistence are deferred to final aggregation. Do not add
`finalize_post_master_outputs` to the chunk writer.
