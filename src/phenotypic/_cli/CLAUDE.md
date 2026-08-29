# CLI Execution (`_cli/`)

The batch-processing engine behind `python -m phenotypic`. `../phenotypicCLI.py`
(at the package root, not in `_cli/`) parses options into an `ExecutionConfig`
(`_cli_types.py`, a mutable `@dataclass`), then
`create_execution_strategy(config, output_manager)`
(`_cli_execution_strategies.py`) dispatches to one strategy.

## Execution strategies (the dispatch)

`create_execution_strategy` picks by `(is_slurm_mode, measure_only,
process_only_layer, pipeline_requires_gpu)`:

| Strategy | When | File |
|---|---|---|
| `LocalParallelStrategy` | local CPU run (joblib) | `_cli_execution_strategies.py` |
| `AutonomousSLURMStrategy` | SLURM CPU run (array + drip-feed chunk chain) | `_cli_execution_strategies.py` |
| `StagedGpuStrategy` | **local** forward GPU run, or `--mode process --layer objmap` | `_cli_staged_strategy.py` |
| `StagedSlurmStrategy` | **SLURM** forward GPU run (`process_only_layer is None`) | `_cli_staged_slurm.py` |

A "GPU run" = the pipeline contains a `GpuDetector` (`pipeline_requires_gpu`).
The staged strategies are forward-oriented; measure-only and non-objmap
process-layer exports stay on the Local/Autonomous strategies.

## Staged GPU engine

When a CLI pipeline contains a `GpuDetector`, the **CLI** (not `ImagePipeline`)
splits it at the detector boundary (`split_pipeline_at_gpu` in
`_cli_pipeline_split.py` → `StagePlan{pre_pipeline, gpu_detector, post_pipeline}`)
and runs three content-defined stages. The per-image stage cores live in
`_cli_staged_workers.py` and are shared by both staged strategies:

1. **Stage 1** `stage1_preprocess_core` — apply pre-detector ops → publish the
   staged OME-Zarr store `results/<ds>/zarr/<stem>.ome.zarr/` (objmap included,
   as zeros, because `valid_staged_store` requires it).
2. **Stage 2** `stage2_detect_core` — load the input layer (store **read-only**),
   run the resident detector, and drop its **Stage-2 signal**: the retained
   **raw** detector output at
   `.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, then a consumable **token**
   at `.phenotypic/progress/stage2_done/<ds>/<stem>.json` (`_cli_stage2_token.py`,
   both atomic temp+`os.replace`; raw first, so a crash between them leaves no
   "done" signal). **Stage 2 does not write into the store** — only the final
   store needs third-party interop, and an in-store write would be visible to
   the uncached crop route as raw pre-`drop_frame_background` labels.
3. **Stage 3** `stage3_merge_measure_core` — replay the **raw** array through the
   accessor (never the store's own objmap: Stage 3 re-promotes over it, so a
   retry would refine already-refined labels), apply post-ops +
   `measure(apply_post=False)`, re-promote the store, then consume the signal —
   **token first, then the raw array** (mandatory).

**Continuation is automatic and content-defined.** Run the same command again;
there is no `--resume` flag. Exact terminal failures remain skipped unless
`--retry-failures` is supplied.
A missing or invalid store selects Stage 1; a valid store without a complete
Stage-2 signal selects Stage 2; and a valid store with one selects Stage 3.
**Every prereq probe tests BOTH halves of the signal** —
`stage2_result_replayable()` is the one function all five sites call. The token
is only a flag; Stage 3's actual input is the raw `.npy`, so a
token-present/raw-missing image is routed back to Stage 2, not into a Stage 3
that would raise `FileNotFoundError` and be recorded as a terminal *scientific*
failure. Stage 3 embeds the authoritative Parquet inside the final store transaction,
publishes the image marker over both the store root and table, then consumes the
signal. The output is byte-identical to a
single-pass run.

**Progress events.** Stages emit stage-tagged events via the `stage` field on
the event log (`_cli_update_state.py`: `append_event(..., stage="stage1|2|3")`).
`status` stays the closed `{started, completed, failed}` set;
`aggregate_state_from_events` counts overall completion at Stage 3 only, and
`aggregate_stage_state_from_events` gives the per-`(dataset, stage)` breakdown.
Use the shared `stage_event` context manager + `emit_missing_prereq` helpers in
`_cli_staged_workers.py` rather than hand-writing the started/completed/failed
trio.

## SLURM chaining (`_cli_staged_slurm.py` + `_cli_staged_slurm_worker.py`)

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
(the retained raw `.npy` and its token). It also does not convert a terminal `afterany` finalizer into an array
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
controller, then either launches the next Stage-1/Stage-3 chunk, launches a
Stage-2 round, or launches the finalizer. Deterministic SLURM comments let a
successor discover a job when its predecessor died after `sbatch` but before
persisting the returned ID.

- Stage 1 / Stage 3 = arrays over **images**; Stage 2 = an array over
  **shards** (`--gpu-shards`, `partition_shards`), each a resident-model
  `run_stage2_shard`. Controller jobs run on `config.slurm_args` (CPU).
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
- Per-stage resources: Stages 1 & 3 use `config.slurm_args` (CPU); Stage 2 uses
  `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)` — inherit/delta over the
  CPU profile, auto-add `slurm_gpus_per_node=1` (explicit `=0` **omits** the
  directive so a CPU partition can run the GPU stage, e.g. tests).
- **Walltime survival:** Stage 2 does not install a signal handler or self-requeue.
  After each array reaches a terminal scheduler state, its dependent controller
  reclassifies the manifest from valid stores, complete Stage-2 signals,
  Stage-3 markers, and terminal failures. Remaining images launch another array round. One unchanged
  retryable-set round is retried; a second unchanged round terminalizes the
  remainder and advances to Stage 3.
- Every worker checks the active epoch immediately before publishing the store,
  the Stage-2 signal, parquet, plot, or deletion changes. Restart and cancellation fence
  stale workers before clearing or cancelling ledgered jobs.
- Without `--wait`, staged submission returns `PROCESSING SUBMITTED` and remote
  finalization owns aggregation, reports, README, and the completion marker.
  With `--wait`, the CLI monitors that marker and never duplicates publication.
- Per-image isolation: a missing prereq (S6) is recorded and skipped, never an
  unhandled raise that aborts a shard.

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
are exactly **three** write sites — `_cli_process_single.py`,
`_cli_staged_workers.py` Stage 1 and Stage 3. Everything else that appears in a
grep is a **transport** site: a fresh process that must be handed the flag on
its command line, namely the staged SLURM worker, the ordinary per-image SLURM
array (`_cli_slurm_array_scripts.py` → `python -m phenotypic._cli._cli_process_single`),
and the staged script generator. A new spawn site needs the flag threaded, or
the option is inert on that path alone.

`durable_writes` is deliberately **not** part of
`processing_configuration_digest` (`_cli_failure_tracker.py`, an explicit
allowlist). Durability is a storage guarantee, not a scientific parameter —
folding it into the work id would make `--no-durable-writes` restart a finished
run from zero.

Rejected with `--mode recompile` (and `migrate`, when Phase 5 lands): those
modes write no image store from a pipeline, so the flag could only mislead.

## Per-image completion markers

`publish_image_success` certifies the artifacts an image produced;
`valid_image_success` is the first conjunct of resume classification. Markers
are **versioned** (`SUCCESS_MARKER_VERSION`, currently **2**) and a
version mismatch invalidates rather than migrates.

Never hand-declare the per-image data artifact. Call
**`image_data_artifact(output_dir, output_manager, dataset, image_stem)`**,
which returns `("store", <store dir>)` when a store exists and `("hdf", ….h5)`
otherwise. Three separate clusters each broke a run by declaring `.h5`
directly after the writer had moved to the store — `publish_image_success`
resolves every artifact `strict=True`, so the image fails *after* completing
all of its work.

Descriptors dispatch on `kind`. A **store** is fingerprinted by its root
`zarr.json` alone, not recursively: the root is written **last** by the promote
protocol and nothing writes into the store after publication, so a valid root
implies a complete store. An absent `kind` reads as `"file"` (v1 shape); an
unknown `kind` **fails closed**.

The `"hdf"` branch is **not** dead code, though no forward path writes an
`.h5` any more. `_migrate_legacy_success_evidence` promotes trees from older
releases, which have `results/<ds>/hdf/<stem>.h5` and no store; dropping the
branch would make it name a nonexistent store and silently refuse to promote
the very trees it exists to rescue — reprocessing all of them.

## Environment variables (important for future work)

- `PHENOTYPIC_PRELOAD_MODULES` — comma list of modules staged SLURM **workers and
  the finalizer** import before `ImagePipeline.from_json`
  (`_cli_preload.py:preload_custom_operation_modules`). Fresh remote processes
  can't see op classes defined outside the `phenotypic` namespace; list a
  self-registering module here so a pipeline with **custom operations**
  deserializes on compute nodes and during final publication. `sbatch
  --export=ALL` propagates it. (Tests use
  `tests/_fakes/register_fake_gpu.py`.)
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
`master_measurements.{csv,parquet}`, `measurements.{csv,parquet}`,
`measurements_by_feature/<feature>.{csv,parquet}`,
`<AnalysisClass>.{csv,parquet}`, `analysis_manifest.json`,
`plots/<plot-id>/...`,
`dashboard.html`, `processing_report.html`, `README.md`,
`pipeline.json`, and `overlays/<ds>/<stem>.png` (detection overlay PNGs). The
dashboard is progress-only: local runs render progress directly, while SLURM
runs add Progress and Download tabs. Use the Results Viewer or the GUI
`/analysis/` app for interactive exploration. Each per-image **OME-Zarr store** stays at
`results/<ds>/zarr/<stem>.ome.zarr/`. Its authoritative object measurements
are embedded at `tables/measurements/table.parquet`, described by
`attributes.phenotypic.tables.measurements` in the store root. Forward,
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
`deliverables/` bundle is self-contained and GUI-openable standalone; `resolve_qc_dir`
/ `migrate_legacy_qc` still read/move a pre-relocation root `qc/`. `run_qc` writes the
single `deliverables/qc/qc.duckdb` (one self-describing table per QC module plus a
`qc_modules` catalog, atomic full rebuild). Resolve these paths via the
`phenotypic.sdk_` helpers (`deliverables_dir`, `master_measurements_parquet_path`,
`qc_dir`, `qc_duckdb_path`, …), never by hand-joining names.

**Master vs. mirror.** Each embedded table is built by right-joining the stable
metadata snapshot (metadata left, baseline measurements right), so it contains
every measured row, excludes metadata-only rows, and records ordered join keys
and snapshot SHA-256 in Parquet schema metadata.
`master_measurements.{csv,parquet}` is the exact pre-post concatenation of
marker-authorized embedded tables; it is already metadata-joined measured data.
Finalization rejects mixed metadata digests or join keys.

`measurements.{csv,parquet}` is the post-applied mirror the GUI reads and
curates. Before post, finalization appends the external metadata anti-join once
using the recorded keys. Measured rows receive `QC_MetadataOnly=false`;
appended phantoms receive `QC_MetadataOnly=true`, keep their metadata values,
and have null measurement/info values. Per-feature splits and named analysis
artifacts derive from the mirror. Analysis consumers resolve tables through
`analysis_manifest.json`, never by constructing filenames.

Ordinary SLURM checkpoints read only marker-authorized embedded tables and write
rolling cache state below `.phenotypic/progress/`. They do not recreate
visible per-image Parquets, dataset aggregates, or a partial deliverable master.

**Metadata snapshot authority.** Before local processing or SLURM submission,
full runs and recompile atomically copy the configured `--metadata` bytes to
`deliverables/metadata.csv`, verify the copy, and use that stable path for
finalization. The snapshot is source provenance, not a generated schema table:
legacy headers normalize only in memory, while finalization, recompile, and
explicit `--mode migrate` must leave its bytes unchanged. Recompile performs
no metadata migration preflight or mutation. All generated measurement,
analysis, QC, and REMBI outputs use canonical flat `Metadata_<Label>` headers.

`QC_MetadataOnly` is a **user-facing output column, not internal machinery** — it is how a
user filters the mirror for "which strains went undetected". Analysis/QC/post code must
**not** branch on it. Those ops are public API (a notebook calls them on frames that never
saw the CLI and carry no flag), so they detect a phantom the same way they detect any
missing value: **drop/ignore NaN**. A phantom row is null in every measurement/info column,
so NaN-native math (`notna()`, `nanpercentile`, `np.isfinite`, `dropna`) handles it, needs
no flag column to exist, and is automatically a no-op on frames that have none.
Feed configured analysis and GUI result exploration from `measurements.parquet`, not
`master_measurements.*`.

**Finalize for FINAL master writes.** Any code path that writes
`deliverables/master_measurements.{csv,parquet}` *as the run's final output* must
immediately call `phenotypic._cli._cli_output_manager.finalize_post_master_outputs(
output_dir, master_df, pipeline)` (it writes into `<output>/deliverables/` and emits the
per-feature splits + analysis chain). The `aggregate_measurements` (forward CLI) and
`--recompile` worker (`_run_post_master_steps`) callers already do this. Mid-run checkpoint writers (`_aggregate_chunks_locked` in
`_cli_chunk_writer.py`) intentionally bypass it and keep their rolling state
under `.phenotypic/progress/`; post, per-feature splits, analysis, and
`pipeline.json` persistence are deferred to final aggregation. Do not add
`finalize_post_master_outputs` to the chunk writer.
