# CLI Execution (`_cli/`)

The batch-processing engine behind `python -m phenotypic`. `phenotypicCLI.py`
parses options into an `ExecutionConfig` (`_cli_types.py`, a mutable
`@dataclass`), then `create_execution_strategy(config, output_manager)`
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

`StagedSlurmStrategy` writes its **own** per-stage SBATCH scripts via
`format_sbatch_directives` (the array generator's fixed path + per-image
`_cli_process_single` body cannot be reused per stage) and submits a **3-link
`afterany` chain** (`submit_script(stage, dependency_job_id=prev)`):

- Stage 1 / Stage 3 = arrays over **images**; Stage 2 = an array over
  **shards** (`--gpu-shards`, `partition_shards`), each a resident-model
  `run_stage2_shard`.
- Per-stage resources: Stages 1 & 3 use `config.slurm_args` (CPU); Stage 2 uses
  `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)` — inherit/delta over the
  CPU profile, auto-add `slurm_gpus_per_node=1` (explicit `=0` **omits** the
  directive so a CPU partition can run the GPU stage, e.g. tests).
- **Walltime survival:** the Stage-2 script carries `#SBATCH --signal=B:TERM@<g>`
  + `#SBATCH --requeue`; the worker catches the SIGTERM and **`scontrol
  requeue`s its own array task** (NOT a new job) so Stage 3's `afterany`
  dependency naturally waits for the continuation. An `attempted` set stops a
  deterministic failure from requeuing forever.
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
- **Output layout** — per-image parquets/HDF stay at the output root
  (`results/`); user-facing deliverables live under `deliverables/`. Resolve
  paths via the `phenotypic.tools_` helpers, never by hand-joining names.
- **HPCC SLURM heterogeneity** — on a mixed partition, pre-AVX CPU nodes can
  SIGILL ("Illegal instruction") the modern numpy/scipy wheels (this affects ALL
  phenotypic SLURM runs, not just the staged engine). Pin to a modern partition
  or constraint. Stage 2's GPU work runs on GPU nodes, which are consistent.
