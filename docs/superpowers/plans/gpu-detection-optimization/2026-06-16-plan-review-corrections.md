# Plan Review Corrections — Spec 1 Plans 1–3 (apply during implementation)

**Source:** plan-reviewer pass, 2026-06-16. The architecture and 20+ reuse claims were **verified correct**; the items below are API-name/signature fixes and resolved open questions. **This file supersedes the as-written plans wherever they conflict** — implementation agents must apply it alongside Plans 1–3.

---

## Blocker fixes (verified API corrections)

- **`create_structure`, not `create_dataset_directories`.** `OutputManager` provisions the output tree via `create_structure(self, datasets: List[Dataset])` (`src/phenotypic/_cli/_cli_output_manager.py:1009`). Replace **every** `om.create_dataset_directories([...])` in Plans 2 & 3 with `om.create_structure([...])`.
- **`OutputManager.from_config` requires `ext`.** Signature is `from_config(cls, base_dir, ext, ...)` — `ext` is a required positional. In all staged tests call `OutputManager.from_config(out, ".tiff", save_overlays=False)`. (`ext` governs legacy per-layer/overlay outputs, **not** the `.h5` HDF path, so `.tiff` is harmless.)
- **`ExecutionConfig` requires `ext`.** `ExecutionConfig.ext: str` has no default — every `_config(...)` test helper in Plans 2 & 3 must pass `ext=".tiff"`.
- **`format_sbatch_directives`, not `generate_sbatch_directives`.** `tools_/slurm/_sbatch.py` exports `format_sbatch_directives(job_name, slurm_args, output_log, error_log) -> str`. Use that name **and all four args** when building per-stage SBATCH headers; it reserves `--array`, so add the `#SBATCH --array=…` line separately (as the existing generator does).

## Resolved open questions

- **OQ1 — Stage events use a separate `stage` field (NOT new status values).** `ProcessingStatus` is a hard-validated closed set `{started, completed, failed}` (`_cli_update_state.py:155`, `tools_/typing_.py:67`). Do **not** add `stage1_completed`-style values. Instead extend the event-log line with a `stage` field (`"stage1"|"stage2"|"stage3"`, default `None` for legacy lines): update `append_event`/`append_completion_event` to accept+write `stage`, `parse_event_line` to read it, and `aggregate_state_from_events` to bucket per `(image, stage)`. Status stays the 3-value set; the existing dashboard logic is untouched. All Plan 2 / Plan 3 stage-event calls become e.g. `append_event(log, ds, img, "started", stage="stage1")`.
- **OQ5 — Aggregator extended in Plan 2.** Extend `aggregate_state_from_events` for per-stage buckets in **Plan 2 Task 7**, so the local staged run shows stage progress immediately (no emit-but-unused events).
- **OQ6 — Stage 3 is apply-then-measure.** `post_pipeline.apply(image, inplace=True)` (runs post-detector refiners incl. watershed) **then** `post_pipeline.measure(image, apply_post=False)`. `measure()` runs only the measurement queue, **not** `_ops`, so refiners do not run twice. (Plan 2 Task 3 — keep both calls; the concern was unfounded.)
- **OQ7 — SLURM script gen clones `ExecutionConfig` per stage.** Build per-stage scripts via `dataclasses.replace(config, slurm_args=stage_args)` and drive the existing `generate_all_array_job_scripts` once per stage (Stage 1&3 args = `config.slurm_args`; Stage 2 args = `resolve_stage_slurm_args(config.gpu_slurm_args, config.slurm_args)`), then wire the three jobs with `--dependency=afterany:<prev>`. Reuses the chunked-array machinery; new code = the thin stage layer. (Plan 3 Task 6.)
- **OQ4 — Live SLURM test runs Stage 2 on a CPU partition via `gpus_per_node=0`.** No Spec-1 test needs a real GPU (the `FakeGpuDetector` is CPU; SAM functional tests use `device="cpu"`). Extend `resolve_stage_slurm_args` to **omit the GPU directive when `slurm_gpus_per_node == 0` is explicitly set**; Plan 3 Task 10's live test passes `gpu_slurm_args={"slurm_partition": <cpu_test_partition>, "slurm_gpus_per_node": 0}` and runs on the CPU partition. **Convention:** any test that genuinely needs a real GPU (Spec 2 model functional tests) targets the **public GPU partition, not exfab.**

## Should-fix items to fold in

- **S6 — Stage-2 missing-HDF guard.** In the Stage-2 loop and the SLURM shard worker, skip + log a structured failure when the staged `.h5` is absent (Stage 1 failed for that image) rather than letting `load_hdf5` raise `FileNotFoundError`.
- **S7 — drop the redundant `_ensure_model_loaded`.** `GpuDetector._operate` need not call `_ensure_model_loaded()` itself (its `infer_batch` already does, idempotently). Keep the explicit call only in the resident-model engine, once before a stream. (Plan 1 Task 5.)
- **S10 — infinite-resubmit guard test.** Add a test asserting `resubmit_stage2_continuation` is **not** called when all of a shard's sidecars already exist at SIGTERM time. (Plan 3 Task 7.)
- **S1 — objmask-after-refactor assertion.** Add a Plan-1 test asserting `out.objmask[:].any()` after the refactored `Sam2Detector` writes `objmap`, guarding the shared-backend invariant.
- **C5 — live-test cleanup.** The live SLURM test should `scancel` its submitted job IDs in a finalizer on timeout, so a failure doesn't orphan jobs. (Plan 3 Task 10.)

## Confirmed correct (do NOT "fix")

The reviewer verified these against source — leave them as the plans state: `pipeline.get_ops()` ordering + `get_meas/get_post/get_filters/get_model/get_qc` + the `ImagePipeline(...)` construction; `save_image_hdf` atomic temp+rename; `get_output_path(ds,"measurements",stem)` → `.parquet`; `dataset_hdf_dir`/`results_dir`/`event_log_path`; `load_hdf5`; `process_single_hdf_measure_core`; `pipeline_requires_gpu`; `create_execution_strategy` + its `phenotypicCLI.py:1365` call site; `generate_dispatcher_chain` (already `afterany` internally) / `generate_all_array_job_scripts` / `submit_slurm_script_chain` / `parse_slurm_args` / `get_slurm_array_limit`; the `GpuDetector` abstract-method change not breaking SAM2/micro-sam; the `objmap[:]`/`objmask[:]` shared-`sparse_object_map` semantics; the corrected sidecar atomic-write form.
