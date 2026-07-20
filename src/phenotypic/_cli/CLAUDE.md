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

1. **Stage 1** `stage1_preprocess_core` — apply pre-detector ops → save the
   normal `results/<ds>/hdf/<stem>.h5`.
2. **Stage 2** `stage2_detect_core` — load the input layer (HDF **read-only**),
   run the resident detector, write a per-image `.npy` objmap **sidecar**
   (`_cli_sidecar.py`, atomic temp+`os.replace`).
3. **Stage 3** `stage3_merge_measure_core` — merge the sidecar via the accessor,
   apply post-ops + `measure(apply_post=False)`, atomically re-save the HDF,
   **delete the sidecar** (mandatory).

**Resume is content-defined.** Stage 1 skips when the HDF exists; Stage 2 skips
when the sidecar **or** the terminal artifact (parquet, or the objmap layer in
process mode) exists — Stage 3 deletes the sidecar, so the terminal artifact is
the durable "done" marker; Stage 3 skips when the parquet exists. The output is
byte-identical to a single-pass run.

**Progress events.** Stages emit stage-tagged events via the `stage` field on
the event log (`_cli_update_state.py`: `append_event(..., stage="stage1|2|3")`).
`status` stays the closed `{started, completed, failed}` set;
`aggregate_state_from_events` counts overall completion at Stage 3 only, and
`aggregate_stage_state_from_events` gives the per-`(dataset, stage)` breakdown.
Use the shared `stage_event` context manager + `emit_missing_prereq` helpers in
`_cli_staged_workers.py` rather than hand-writing the started/completed/failed
trio.

## SLURM chaining (`_cli_staged_slurm.py` + `_cli_staged_slurm_worker.py`)

> **Queue all SLURM work through the drip-feed dispatcher — never submit array
> chunks eagerly.** Every SLURM submission path in this package (the CPU
> autonomous strategy, the staged GPU strategy, `--recompile`) funnels its
> ordered chunk scripts through `submit_slurm_script_chain`
> (`_cli_slurm_submission.py`) → `generate_dispatcher_chain` +
> `submit_drip_feed_start` (`sdk_/slurm/_dispatcher.py`). The dispatcher submits
> **only chunk 0 + a tiny dispatcher job** up front; when chunk N ends, its
> dispatcher submits chunk N+1. Peak queue occupancy stays at ~1 chunk + 1
> dispatcher, so a run's full task count (which for the staged path is
> ~2 × n_images) never trips the per-user `MaxSubmitJobs`. A new SLURM
> submission site MUST reuse this helper, not loop `submit_script` over all
> chunks — eager submission is what caused the `AssocMaxSubmitJobLimit` failures.

`StagedSlurmStrategy` writes its **own** per-stage SBATCH scripts via
`format_sbatch_directives` (the array generator's fixed path + per-image
`_cli_process_single` body cannot be reused per stage), flattens them into one
ordered list `[*stage1_chunks, stage2, *stage3_chunks]`
(`flatten_staged_scripts`), and hands that to the shared drip-feed dispatcher via
`submit_staged_chain` → `submit_slurm_script_chain`. The linear order encodes the
stage dependencies (a chunk is submitted only after the prior one ends), so
Stage 2 starts after the last Stage-1 chunk and Stage 3 after Stage 2:

- Stage 1 / Stage 3 = arrays over **images**; Stage 2 = an array over
  **shards** (`--gpu-shards`, `partition_shards`), each a resident-model
  `run_stage2_shard`. The tiny dispatcher jobs run on `config.slurm_args` (CPU).
- **Array chunking (`min(MaxArraySize, MaxSubmitJobs - 1)`):** the image count is
  split into `ceil(n_images / chunk_limit)` chunk scripts
  (`calculate_optimal_array_chunks` → `_write_image_stage_chunks`), where
  `get_slurm_max_submit_jobs()` conservatively uses the smallest configured QoS
  or user-association limit, and `chunk_limit` reserves one submission slot for
  the dependent dispatcher queued alongside the active array. A single chunk
  must fit **both** the array-index cap and the remaining per-user submit
  capacity.
  Each chunk is a 0-based `--array=0-(k-1)` whose `TASK_INDICES` window holds the
  **absolute** manifest indices and whose worker reads
  `--index $CURRENT_TASK_INDEX`, so no array index ever reaches the limit. A
  single chunk keeps the plain `stage1.sh`/`stage3.sh` name; multiple become
  `stageN_chunk{i}.sh`. Stage 2 is **never chunked** (a shard worker streams its
  whole shard on one GPU); `--gpu-shards > chunk_limit` raises. `generate_staged_scripts`
  returns `{"stage1": [Path…], "stage2": Path, "stage3": [Path…],
  "finalizer": Path}` — image stages are always lists. The finalizer is a
  one-task CPU job that reloads the canonical pipeline and runs the same
  aggregate/finalize path as ordinary SLURM, including named analysis and plot
  publication. The shared dispatcher drip-feeds the ordered scripts, including
  the finalizer, so only one chunk array and one tiny dispatcher are queued at a
  time while each chunk's array still fans out.
- Per-stage resources: Stages 1 & 3 use `config.slurm_args` (CPU); Stage 2 uses
  `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)` — inherit/delta over the
  CPU profile, auto-add `slurm_gpus_per_node=1` (explicit `=0` **omits** the
  directive so a CPU partition can run the GPU stage, e.g. tests).
- **Walltime survival:** the Stage-2 script carries `#SBATCH --signal=B:TERM@<g>`
  + `#SBATCH --requeue`; the worker catches the SIGTERM and **`scontrol
  requeue`s its own array task** (NOT a new job). Because the requeued task keeps
  its job id and `afterany` waits for terminal state, the dispatcher that submits
  the first Stage-3 chunk (it depends `afterany` on the Stage-2 job) naturally
  waits for the continuation. An `attempted` set stops a deterministic failure
  from requeuing forever.
- Per-image isolation: a missing prereq (S6) is recorded and skipped, never an
  unhandled raise that aborts a shard.

## Environment variables (important for future work)

- `PHENOTYPIC_PRELOAD_MODULES` — comma list of modules a SLURM **worker** imports
  on startup (`_cli_staged_slurm_worker.py:_preload_custom_op_modules`) before
  `ImagePipeline.from_json`. A fresh worker process can't see op classes defined
  outside the `phenotypic` namespace; list a self-registering module here so a
  pipeline with **custom operations** deserializes on the compute node. `sbatch
  --export=ALL` propagates it. (Tests use `tests/_fakes/register_fake_gpu.py`.)
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
`dashboard.html`, `analysis.html`, `processing_report.html`, `README.md`,
`pipeline.json`, and `overlays/<ds>/<stem>.png` (detection overlay PNGs). The
**per-image** parquets in `results/<ds>/measurements/` (and the rest of `results/`,
`progress/`, `processing_state.json`) stay at the output-dir **root**. The durable
**QC + curation state** lives under `deliverables/qc/` (`qc.duckdb`,
`review_state.json`, `curation_labels.parquet`, `custom_categories.json`) so a
`deliverables/` bundle is self-contained and GUI-openable standalone; `resolve_qc_dir`
/ `migrate_legacy_qc` still read/move a pre-relocation root `qc/`. `run_qc` writes the
single `deliverables/qc/qc.duckdb` (one self-describing table per QC module plus a
`qc_modules` catalog, atomic full rebuild). Resolve these paths via the
`phenotypic.sdk_` helpers (`deliverables_dir`, `master_measurements_parquet_path`,
`qc_dir`, `qc_duckdb_path`, …), never by hand-joining names.

**Master vs. mirror.** `master_measurements.{csv,parquet}` is a clean, pre-post,
metadata-free archive of what per-image runs measured; `measurements.{csv,parquet}` is
the post-applied mirror the GUI reads/curates. Per-image parquets in
`results/<ds>/measurements/` are also clean — the CLI calls
`pipeline.measure(image, apply_post=False)` on the per-image path. Post is applied once
at the end of aggregation against the merged master, and the post-applied frame is what
the class-named analysis artifacts and `measurements_by_feature/<feature>.{csv,parquet}`
derive from. Analysis consumers resolve tables through `analysis_manifest.json`, never
by constructing filenames. The external `--metadata` CSV **left-join** also lands on
the post-applied frame
(inside `finalize_post_master_outputs`), so the mirror, per-feature splits, and
named analysis artifacts carry metadata while the master archive stays post-free and
metadata-free.
The join is **left** so metadata rows matching no measured object survive as **phantom
rows** — metadata + join keys populated, every measurement/info column null, and
`QC_MetadataOnly=true` (`schema.METADATA_MATCH`) — which is how a user sees the strains
that were never detected. Measurement rows with no metadata are still dropped
(measurements are the join's right frame). `join_metadata(df, csv, *, how=...)` defaults
to `how="inner"`: **only** `finalize_post_master_outputs` passes `how="left"`. The
mid-run `_cli_chunk_writer` and the dashboard `_analysis_data` sidecar keep `inner` on
purpose — they join against a **partial** frame, where a left join would flag every
not-yet-processed strain as missing.

`QC_MetadataOnly` is a **user-facing output column, not internal machinery** — it is how a
user filters the mirror for "which strains went undetected". Analysis/QC/post code must
**not** branch on it. Those ops are public API (a notebook calls them on frames that never
saw the CLI and carry no flag), so they detect a phantom the same way they detect any
missing value: **drop/ignore NaN**. A phantom row is null in every measurement/info column,
so NaN-native math (`notna()`, `nanpercentile`, `np.isfinite`, `dropna`) handles it, needs
no flag column to exist, and is automatically a no-op on frames that have none.
Feed analysis plugins/dashboards from `measurements.parquet`, not
`master_measurements.*`.

**Finalize for FINAL master writes.** Any code path that writes
`deliverables/master_measurements.{csv,parquet}` *as the run's final output* must
immediately call `phenotypic._cli._cli_output_manager.finalize_post_master_outputs(
output_dir, master_df, pipeline)` (it writes into `<output>/deliverables/` and emits the
per-feature splits + analysis chain). The `aggregate_measurements` (forward CLI) and
`--recompile` worker (`_run_post_master_steps`) callers already do this. Mid-run
intermediate writers (`_aggregate_chunks_locked` in `_cli_chunk_writer.py`)
intentionally bypass it — chunks publish partial results for mid-run download, but the
post pipeline, per-feature splits, analysis chain, and `pipeline.json` persistence are
deferred to final aggregation. Don't add `finalize_post_master_outputs` to the chunk
writer — it would re-run expensive finalize work on every checkpoint.
