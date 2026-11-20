# GPU Embedding Stage 4 Implementation Plan

**Status:** revision 3 · 2026-11-19 · revised per `reports/2026-11-12-gpu-embedding-staging/plan-review.md`,
the spec as amended in `c33b58193` (U5, U7–U10, S1), and the re-review of
revision 2 (`671665959`); see "Revision 3" at the end
**Spec:** `docs/superpowers/specs/2026-11-12-gpu-embedding-staging/design.md` (approved 2026-11-19)
**Parent:** #281, `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`
**Base:** `feat/gpu-embedding-staging` (PR #282), after the orchestrator merges #281
(`worktree-dvc-embeddings-refactor` at `26d7f86a6` or later) into it. Every
`file:line` below is read from the #281 worktree at `26d7f86a6`, not from the
#282 branch as it stands today. (`d3c54f723` → `26d7f86a6` changed one test
file only, so revision 1's anchors carried over; every anchor revisions 2 and
3 added or touched was re-read at `26d7f86a6`, with `git show`, because the
#281 worktree holds uncommitted edits.) **`detect/nn/_helper/_dino_support.py`
is being edited on #281** (gridname's snapshot-path fallback in
`resolved_revision`, `:257-272`): once it lands, that file's anchors below
`:257` shift. Re-find them by name; they are not questions.

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every `MeasureDinoEmbedding` of a staged run on GPUs, in a Stage 4
after the store is final. One resident model per GPU shard, batched across
images, with vectors that, within one GPU model and physical batch size, do not
depend on sharding or batching (spec goal 3 as restated, S1).

**Architecture:**
- P1 splits the embedder into a CPU `prepare_regions`, a backbone-only
  `encode`, and `payload_from`, so the inline path and Stage 4 share one crop
  geometry.
- P2 adds the large models, `compute_dtype`, the determinism settings and the
  `BatchEncoder`, and measures batch invariance on three GPU models.
- P3 adds the store transaction (`replace_run_embeddings`) and the Stage-4
  signal. P4 wires a local Stage 4 with its refusals, deferral and resume.
- P5 puts Stage 4 behind the recoverable SLURM controller (config v2). P6 runs
  `--mode measure` through it in republish mode. P7 is docs and the full
  regression.

**Tech Stack:** Python 3.12, pydantic v2, numpy, pyarrow, zarr v3 (OME-Zarr 0.5),
click, torch 2.11 + transformers 5.x (the `foundation` extra), pytest, Slurm.
Renderer: `~/.claude/tools/render_md_artifact.py`.

## Global Constraints

- **#281's device-provenance change has landed** (`b6714d28d`, `d3c54f723`):
  `model.device` and `model.device_name` are recorded per entry and ignored by
  descriptor grouping (`_cli_embedding_deliverables.py:108`,
  `_PROVENANCE_MODEL_FIELDS`). This plan builds on it and leaves those two
  fields where they are; `compute_block` is their one producer and copies them
  into `compute` (O4, accepted with that condition).
- **#281 API this plan starts from, used unchanged:**
  - keys `<model>/<layer>/<region>`;
  - regions `objects | grid | objects-<name> | grid-<name>`;
  - `MeasureEmbedding.name` and `.layer`;
  - `split_embedding_key` (`sdk_/_table_runs.py:205`) is the one key parser;
  - `MeasureEmbedding._layer_pixels` (`abc_/_measure_embedding.py:279`) is the
    one pixel source.
- `--embed-stage {auto,gpu,inline}`, default `auto`. Resolution table, spec §4.1:

  | Mode | `auto` resolves to |
  |---|---|
  | full, SLURM, `GpuDetector` or `--gpu-slurm` given | `gpu` |
  | full, SLURM, neither | `inline` |
  | full, local | `inline` |
  | `--mode measure`, SLURM with `--gpu-slurm` | `gpu`, on the staged controller (U10) |
  | `--mode measure`, otherwise | `inline` |
  | `--mode process` (copy or in place) | `inline`; an explicit `gpu` is refused (R1) |
  | an in-scope embedder pinned to `device="cpu"` | `inline`, with a note; an explicit `gpu` is refused (U9, R2) |
  | a pipeline with no in-scope embedder | `inline`, with a note (O5) |

- The resolved value is recorded as `processing_state.json` config
  `"embed_stage"` and `job_metadata.json` `JobMetadataKey.EMBED_STAGE`
  (`"embed_stage"`). It is **never** in the work-id digest
  (`processing_configuration_digest_from_values`, `_cli_failure_tracker.py:276`).
  A state without the key reads as `"inline"`.
- **Refusals R1–R3** are each one `click.UsageError` line (exit `2`). They sit in
  the read-only half, outside `--skip-validation`, above `--restart`,
  `--overwrite`, `mint_run_identity` and `--dry-run`. Pinned by
  `tests/unit/cli/test_cli_preflight_ordering.py`.
- `compute_dtype: Literal["float32", "bfloat16", "float16"] = "float32"` on
  `MeasureDinoEmbedding`. It is part of the descriptor (top-level
  `compute_dtype`, absent reads `"float32"`). Storage stays `float16`
  (`EmbeddingDType`, `sdk_/typing_.py`).
- The per-entry **`compute`** block is
  `{device, device_name, capability, torch, cuda, physical_batch, stage}`. It is
  excluded from descriptor grouping. `compute_dtype` is not.
- **Determinism**, set once per Stage-4 process before the first load:
  - `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`;
  - `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`;
  - `torch.set_float32_matmul_precision("highest")`;
  - `torch.use_deterministic_algorithms(True, warn_only=True)`;
  - `CUBLAS_WORKSPACE_CONFIG`: exported as `:4096:8` by the Stage-4 SLURM
    script prelude, and `setdefault` to `:4096:8` by the CLI right after
    `--embed-stage` resolves to `gpu`, before any strategy can create a cuBLAS
    handle (local Stage 4 runs in the CLI process; plan review I4).
    `configure_determinism` accepts `:4096:8` or `:16:8`, the two deterministic
    values, so a user's own `:16:8` is kept.
- **Physical batch:**
  `m = min(batch_size, pow2floor((0.8·GPU_mem − weights − 1 GiB) / per_sample))`.
  It depends on the GPU model (total memory), the backbone and `crop_px` only,
  never on free memory. On OOM, m halves once for the shard and is recorded in
  `compute.physical_batch`. At m = 1, or on a second OOM, the shard exits
  non-zero. Constants live in P0's logic-validation script. Because m differs by
  GPU model and halving changes it, batch invariance is promised **within one GPU
  model and physical batch** (S1); across them the distance is measured
  (P2 T4).
- **Inside Stage 4, an embedder's device is `cuda`.** A missing or unusable CUDA
  fails the shard, never falls back to CPU. The test-only escape is
  `PHENOTYPIC_STAGE4_DEVICE=cpu`.
- **Nothing is rewritten after the record that certifies it.** Stage 3 (and
  the single-pass core in a Stage-4 run) writes the store with its measurements
  and `stages.stage3`, and writes **no** record and no embeddings. Stage 4 writes
  the embeddings (one store transaction), then the record, then consumes its
  signals, in that order.
- **One "done" predicate.** Under deferral, `stages_1_to_3_done` (a Stage-3
  marker, a store that matches the work id, a measurement table) means Stages
  1–3 are finished: every recovery shortcut returns early on it and publishes
  nothing, and every core that re-runs an image deletes that image's Stage-4
  signals first (P4 T2).
- **Signals:** `.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet`,
  atomic, with `work_id`, `run_id` and `root_sha256` in the Parquet key-value
  metadata. A signal is valid only at the **live** root; a stale one is deleted
  and its key re-embedded. Signals are reused **per key**, and a round that wrote
  signals has made progress (P3 T2, P4 T3, P5 T1).
- **A stalled Stage 4 ends unfinished** (U7): after two unchanged rounds the
  finalizer runs, the leftover images keep no record and no journal row, and the
  run exits `5`. GPU OOM and a changed root are never terminal per-image
  failures.
- **Measure mode** (U8, U10): the measure core carries the image's existing
  embeddings forward; a Stage-4 failure keeps them and a valid record, and is
  journalled as `MEASURE_FAILURE_STAGE` (exit `3`). SLURM measure runs on the
  staged controller with a measure-mode finalizer that writes no forward run
  proof. Republish has **one carrier**, the manifest flag (manifest v5).
- **U5:** an unpinned revision is resolved once, at a run's first submission,
  for Stage-4 runs only (O9); a continuation reuses the recorded commit.
- **Commit pool and loader:** one commit thread; one DataLoader per backbone,
  `spawn` context, not persistent, identity `collate_fn`, processors loaded in
  the main process and pickled into the dataset. Workers:
  `resolve_local_worker_count(config.n_jobs, n) − 1` locally (`--njobs 1` runs
  it in-process) and `resolve_local_worker_count(-1, n) − 1` on a SLURM shard
  (the allocation), `_cli_utils.py:88`.
- **The encoder's in-flight bound is in bytes**
  (`MAX_IN_FLIGHT_OUTPUT_BYTES`), so crop pipelines keep full batches and only
  `masked_dense` feature grids flush early.
- **No CUDA for a local `--embed-stage gpu` is exit `1`**, one line, before
  Stage 1: an environment error, not a usage error.
- **The revision an entry records comes only from `resolved_revision(model)`**;
  no #282 code reads `config._commit_hash` (transformers ≥ 5.18 stopped setting
  it; the #281 fallback lives in `resolved_revision`).
- **Process mode** (copy and in place) and AutoConvertRaw-GC's direct
  `process_single_apply_only_core` calls stay inline on CPU, byte for byte (H5).
- No new resource flags: Stage 4 runs on `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)`
  (`_cli_staged_slurm.py:110`) and `--gpu-shards` (H6).
- Heavy imports (torch, transformers, huggingface_hub) stay inside functions.
  `tests/unit/ci/test_startup_imports.py` and `test_deferred_imports.py` stay
  green. `sdk_` never imports `_cli` (`tests/unit/sdk_/test_run_state_layering.py`).
- **No file under `src/phenotypic/_gui/` changes.** The run-console control for
  `--embed-stage` is a GUI follow-up (spec §6); see "Follow-ups".
- `MeasurementInfo` members get `label`/`desc` only, never `bio_desc`.

### Expected-red GUI tests (#283, inherited from #281)

These 13 are red on #281 and stay red here: no `xfail`, no skip, no edit. A phase
gate treats exactly this set as expected-red. Any other red GUI test is a real
failure. Source: #281's README table and `gate-results/final/classified.tsv`.

| Test | Why |
|---|---|
| `tests/unit/gui/tune/test_validation.py::test_tuning_spec_path_is_deployable` | #281 P0 suffix flip |
| `tests/unit/gui/tune/test_setup_authoring.py::test_write_authored_setup_spec_preserves_existing_spec_search_space` | #281 P0 |
| `tests/unit/gui/tune/test_setup_authoring.py::test_existing_spec_replaces_scorer_only_when_requested` | #281 P0 |
| `tests/integration/gui/tune/test_setup_view.py::test_existing_spec_controls_preserve_seed_budget_and_storage` | #281 P0 |
| `tests/integration/gui/tune/test_setup_view.py::test_existing_spec_trial_override_preserves_authored_strategy` | #281 P0 |
| `tests/integration/gui/tune/test_setup_view.py::test_existing_grid_spec_never_emits_trial_override` | #281 P0 |
| `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact` | #281 P1 (TSV proof keys) |
| `tests/unit/gui/results_viewer/test_measurement_routes.py::test_an_undeclared_column_never_opens_the_parquet` | #281 P3 (v1 path) |
| `tests/unit/gui/results_viewer/test_measurement_source.py::test_a_rewritten_table_is_read_fresh` | #281 P3 |
| `tests/unit/gui/results_viewer/test_measurement_source.py::test_an_unmeasured_image_contributes_no_keys` | #281 P3 |
| `tests/unit/gui/results_viewer/test_measurement_source.py::test_one_unmeasured_image_does_not_empty_the_picker` | #281 P3 |
| `tests/unit/gui/results_viewer/test_measurement_source.py::test_the_picker_offers_the_stores_own_numeric_columns` | #281 P3 |
| `tests/unit/gui/results_viewer/test_measurement_source.py::test_values_are_keyed_by_dataset_image_and_object_label` | #281 P3 |

## Review Focus

Inputs the spec implies but names no test for. Each is pinned in the task named.

1. **An image with no colonies, or an image whose every embedder is objects-only
   on an empty objmap.** Inline gives an empty table (`_embed_vectors`,
   `abc_/_measure_embedding.py:485`). Stage 4 must submit nothing to the GPU for
   it, still commit an empty table, and publish its record; it must not stall a
   batch waiting for rows. Pinned in P2 T3
   (`test_an_empty_input_completes_without_a_forward_pass`) and P4 T3
   (`test_an_image_with_no_colonies_commits_an_empty_table`).
2. **A per-image error raised inside a DataLoader worker process**
   (`GridCellOutsideImageError` on one plate, a corrupt store). It must cross the
   process boundary as that image's `stage4` failure; the loader, the shard and
   the other images go on. Pinned in P4 T3
   (`test_a_loader_worker_error_fails_only_its_image`).
3. **`--embed-stage gpu` on a machine with no CUDA, locally.** The run must stop
   with one error line and exit `1` before Stage 1 writes anything, and a Stage 4
   that meets no CUDA anyway must journal no per-image failure and leave every
   image pending, so a re-run on a GPU node finishes it. Pinned in P4 T1
   (`test_no_cuda_stops_before_stage1`) and P4 T3
   (`test_no_cuda_fails_the_stage_without_journalling_images`).
4. **More GPU shards than retryable images** (`--gpu-shards 4`, one image left
   after a walltime). Empty shards exit `0`, are not progress and not failure, and
   the next classification sees the one image. Pinned in P5 T1
   (`test_more_shards_than_work_leaves_empty_shards_harmless`).
5. **The same command run again after the run finished.** No image re-embeds, no
   GPU array is submitted (the controller goes straight to the finalizer), and
   every store root is byte-unchanged. Pinned in P4 T4
   (`test_a_finished_run_rerun_embeds_nothing`) and P5 T1
   (`test_all_done_goes_straight_to_the_finalizer`).

## Conventions for every task

- **Shared-env test command** (`run-phenotypic-test` skill: headless Qt, never
  `-n auto`, no streamed output, no `-x`):

  ```bash
  PT='env QT_QPA_PLATFORM=offscreen uv run pytest -q --no-header -p no:randomly -o addopts= -m "not slow"'
  ```

- **Torch lane.** Tests that use `stub_backbone` (`tests/_tiny_run.py:1234`, it
  needs torch) or P0 T2's `param_stub_backbone`, `BatchEncoder` or the Stage-4
  worker run in the foundation venv, from the worktree root:

  ```bash
  TPT='env QT_QPA_PLATFORM=offscreen PYTHONPATH=src:. /bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/python -m pytest -q --no-header -p no:randomly -o addopts= -m "not slow"'
  ```

  Each task marks its torch tests **[torch]**. A torch test in the shared env
  must skip through `pytest.importorskip("torch")`, never fail. A test whose
  embedder has `device="auto"` runs under `stage4_cpu`, which also sets
  `CUDA_VISIBLE_DEVICES=""` **and patches `torch.cuda.is_available`** (the
  variable alone does nothing once CUDA is initialised), so a GPU node gives the
  same bytes as a CPU node.
- **The inline golden is CPU-specific.** Its float32 bytes differ by CPU family
  and torch thread count, so its byte tests run only in the recorded capture
  environment (`tests/unit/measure/golden/inline_embeddings_281.environment.json`):
  a gate puts `test_embedding_inline_golden.py` on a Broadwell node
  (`--constraint=broadwell`, `--partition=short` or `intel`) with
  `--cpus-per-task=4` and `OMP_NUM_THREADS=4`. The test never calls
  `torch.set_num_threads`, which moves the real rows' bytes even at 4. In any
  other environment they skip, naming both, and an always-on cosine check
  (`>= 1 - 1e-6` per row) runs.
- **Per-image faults** in Stage 4 are injected through an on-disk trigger
  (`FaultyDinoEmbedding` and `$PHENOTYPIC_TEST_FAULT_DIR`, P4), never a
  monkeypatch, which a spawned loader worker would not see.
- **Crash tests** run the CLI in a real subprocess and kill it with
  `os._exit(137)` right after the named durable write (`run_cli_subprocess`,
  P4). An exception raised inside the test process is not a crash.
- **Lint:** `uv run ruff check --fix <the paths you changed>`, never bare.
- **Commits:** explicit paths, never `git add -A`; end with the session's
  attribution lines.
- **GPU jobs** use the `slurm-job` skill. Before each, measure the predicted start
  on both `exfab` (`--account=exfab`) and `short_gpu --constraint=gpu_latest`
  with `sbatch --test-only`, and take the earlier one. Never `preempt`. Logs go
  under `/bigdata/exfab/anguy344/gate-results/<cluster>/`, never `/scratch`.
- **Store changes** also get an independent-reader check: `zarr.open_group` plus
  `pyarrow.parquet.read_table` (`working-with-ome-zarr` skill).
- **Shared test helpers** go in `tests/_tiny_run.py` and are re-exported through
  `tests/conftest.py`'s explicit import list. The first task that uses a helper
  creates it under **Create (helpers)**; later tasks reuse it. Reused as they are:
  `stub_backbone`, `install_stub_backbone`, `toy_embedder`, `toy_grid_embedder`,
  `plate`, `detected_plate`, `run_cli`, `tiny_inputs`, `one_store`, `run_entry`,
  `pipeline_file_a`, `with_dino_objects`, `make_exec_config` and `submissions`
  (`tests/unit/cli/conftest.py:153,944`); the fake-scheduler pattern of
  `tests/unit/cli/test_staged_controller.py:108-129`.
- **Gates** follow root `CLAUDE.md`:
  - per task: the task's test files;
  - per phase: the affected surface once, from importers of the changed modules
    (`grep -rl <module> tests/`), on Slurm when over a couple of minutes;
  - end of plan: one full regression (P7 T2).
  - A gate that runs `tests/unit/gui`, `tests/integration/gui` or `tests/gui`
    compares its red set with the expected-red table above.
- **Gate shape** (precedent: `/bigdata/exfab/anguy344/gate-results/final/make_gate.sh`):
  a **detached worktree** at the gated SHA under
  `/bigdata/exfab/anguy344/gate-worktrees/<cluster>-<sha9>`; a 24-shard
  `--array=0-23` on `short` (8 CPU, 24 GB, 2 h, `-n $SLURM_CPUS_PER_TASK`), files
  round-robin by index; a **foundation lane** (one `short` job, 16 CPU, 32 GB)
  running every torch file in the foundation venv; an `afterany` cleanup that
  removes the worktree. Each shard prints `dirty=0` at start and end.
- This plan's Markdown is the source of truth. Re-render the artifact after any
  edit (render command at the end of this file).

## Phases

| Phase | Document | Depends on | Tasks | Gate |
|---|---|---|---|---|
| P0 | [p0-validation-and-golden.md](p0-validation-and-golden.md) | #281 merged | 2 | per-step guards |
| P1 | [p1-embedder-refactor.md](p1-embedder-refactor.md) | P0 T2 | 2 | golden byte-identical |
| P2 | [p2-large-models-batch-encoder.md](p2-large-models-batch-encoder.md) | P0 T1, P1 | 4 | unit tests; GPU batch-invariance job on 3 GPU models; **deep review** |
| P3 | [p3-store-transaction-and-signal.md](p3-store-transaction-and-signal.md) | P1 | 2 | store-transaction tests, independent reader |
| P4 | [p4-local-stage4.md](p4-local-stage4.md) | P2, P3 | 4 | local end-to-end, crash injection; **deep review P3–P4** |
| P5 | [p5-slurm-controller.md](p5-slurm-controller.md) | P4 | 4 | fake-scheduler tests; live `exfab` smoke; **deep review** |
| P6 | [p6-measure-republish.md](p6-measure-republish.md) | P5 | 2 | measure-over-staged end-to-end; **deep review** |
| P7 | [p7-docs-and-regression.md](p7-docs-and-regression.md) | P6 | 2 | simplify pass, full sharded regression |

22 tasks in all.

## Execution

**File overlap.** These files are touched by more than one phase, so the main
line runs **sequentially in one worktree**:

| File | Phases |
|---|---|
| `abc_/_measure_embedding.py` | P1 P2 |
| `measure/_measure_dino_embedding.py` | P1 P2 |
| `detect/nn/_helper/_dino_support.py` | P2 |
| `detect/nn/_helper/_batch_encoder.py` (new) | P2 P4 |
| `sdk_/_measurement_tables.py` | P3 P6 |
| `_cli/_cli_output_manager.py` | P6 |
| `_cli/_cli_checkpoint_handler.py` | P4 P6 |
| `sdk_/generate_report.py`, `_cli/_dashboard/_manifest_builder.py` | P4 |
| `_cli/_cli_stage4_signal.py` (new) | P3 P4 |
| `_cli/_cli_embed_stage.py` (new) | P4 P5 P6 |
| `_cli/_cli_stage4_worker.py` (new) | P4 P5 P6 |
| `_cli/_cli_staged_resume.py` | P4 P6 |
| `_cli/_cli_staged_workers.py` | P4 |
| `_cli/_cli_staged_strategy.py` | P4 |
| `_cli/_cli_staged_slurm_worker.py` | P4 P5 P6 |
| `_cli/_cli_execution_strategies.py` | P4 P5 P6 |
| `_cli/_cli_process_single.py` | P4 P5 P6 |
| `_cli/_cli_staged_controller.py` | P5 P6 |
| `_cli/_cli_staged_slurm.py` | P4 P5 P6 |
| `_cli/_cli_staged_orchestration.py` | P5 P6 |
| `_cli/_cli_preflight.py` | P5 |
| `phenotypicCLI.py` | P4 P5 P6 |
| `_cli/_stages.py`, `_cli_update_state.py` | P4 |
| `_cli/CLAUDE.md`, `docs/source/**` | P7 |
| `tests/_tiny_run.py`, `tests/conftest.py` | every phase |

**Side lanes**, zero file overlap with the main line:
- **Lane A:** P0 T1, the logic-validation scripts (docs only). Runs alongside P0 T2.
- **Lane B:** P2 T4, the GPU batch-invariance job. Its script lives beside this
  plan; its jobs read a detached worktree, so P3 proceeds while it queues. **P4
  does not start until its result is in** (stop rule 3).

**Clusters.** One agent per cluster. "Opus" is the frontier tier, "Sonnet" the
mid tier. A reviewer is never weaker than its implementer.

| # | Tasks | Shape | Model | Gate after |
|---|---|---|---|---|
| C1 | P0 T1, logic-validation scripts (Lane A) | Leaf | Sonnet | light: the scripts exit 0 and fail on a planted wrong constant |
| C2 | P0 T2, inline golden capture | Leaf | Sonnet | light: golden green on the #281 merge commit, torch lane |
| C3 | P1 T1, T2, embedder refactor | Keystone | Opus | light: golden byte-identical, `tests/unit/measure`, `tests/unit/abc_` |
| C4 | P2 T1, large models, `compute_dtype`, `compute` block, hub verification | Keystone | Opus | light |
| C5 | P2 T2, T3, determinism, batch size, `BatchEncoder` | Keystone | Opus | light |
| C6 | P2 T4, batch-invariance job (Lane B) | Leaf | Sonnet writes; orchestrator submits | **deep: P2** (implementation-test-reviewer) after the measurement lands |
| C7 | P3 T1, `replace_run_embeddings` | Keystone | Opus | light, plus independent reader |
| C8 | P3 T2, Stage-4 signal | Seam | Opus | light |
| C9 | P4 T1, resolver, R1–R3, recording | Seam | Opus | light, plus `test_cli_preflight_ordering.py` |
| C10 | P4 T2, deferral, the done-predicate, the completion predicate | Seam, consistency-critical | Opus | light |
| C11 | P4 T3, local Stage-4 worker | Keystone | Opus | light |
| C12 | P4 T4, resume, retry, crash injection | Seam | Opus | **deep: P3–P4** |
| C13 | P5 T1, controller v2, rounds, worklist | Keystone | Opus | light |
| C14 | P5 T2, scripts, worker `--stage 4`, no-detector plan, U5 pin | Seam | Opus | light |
| C15 | P5 T3, preflight codes and docs rows | Leaf | Sonnet | light, plus `tests/unit/test_docs_preflight_codes.py` |
| C16 | P5 T4, live SLURM smoke | orchestrator, **user approves** | — | **deep: P5** |
| C17 | P6 T1, T2, measure carry, republish, measure-mode finalizer | Seam | Opus | **deep: P6** |
| C18 | P7 T1, docs | Leaf | Sonnet | light, docs build (Slurm, `-D nbsphinx_execute=never`) |
| C19 | P7 T2, simplify then full regression | code-simplifier (Opus), then orchestrator | — | full sharded regression, once |

**Rules for every cluster.**
- The agent runs its own tests and lint, and commits its own work with explicit
  paths. It never commits under `src/phenotypic/_gui/`.
- It reports each commit SHA and the failing-test names it saw before each fix.
- Torch tests run in the foundation venv (`TPT`), and the report says which lane
  each command ran in.
- The orchestrator re-reads the disk at each gate, and never runs verification
  while an agent is mid-edit.
- Reviewers write their report to
  `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/<what-it-reviewed>.md`.
- An open design question found at any gate stops the run until the user
  answers.

**Stop points that need the user.**
1. Any design question the spec does not settle, before the task that consumes
   it. O1–O9 are all decided (below); none is open.
2. **The live SLURM smoke (P5 T4)** spends real GPU time; it runs only on the
   user's go-ahead.
3. **A batch-invariance measurement that refutes the hypothesis** (P2 T4): any
   stored float16 bit that changes with neighbours, position, partition or
   process at one GPU model and physical batch, or any determinism warning. The
   spec's §5 fallback (an m = 1-per-crop mode) is then a user decision, and P4
   does not start.

## Spec coverage

| Spec item | Task |
|---|---|
| §1 goals 1–3 (speed, larger models, consistent embeddings); goal 3 as restated (S1) | P2 T1–T4, P4 T3, P5 T1–T2; S1 in P2 T4's "What it measures" |
| §2 measured facts (device is provenance) | Global Constraints; P2 T1 (`compute` excluded from grouping) |
| U1 GPU Stage 4 after the store is final | P4 T2 (deferral), P4 T3 (worker) |
| U2 `--embed-stage` default `auto` | P4 T1 |
| U3 no switching mid-run (R3) | P4 T1 |
| U4 measure mode uses Stage 4 | P6 T1, T2 |
| U5 (amended) revision resolved once at first submission; a continuation reuses it; Stage-4 runs only | P5 T2 (`resolve_embed_revisions`, `recorded_embed_revisions`) |
| U6 mixed GPU models allowed; device provenance per entry | P2 T1 (`compute` block); P5 T2 (no device constraint across shards) |
| U7 a stall ends unfinished: no record, no journal row, exit 5 | P5 T1 (`test_two_unchanged_rounds_finalize_and_record_nothing`, `test_a_stall_reads_5_to_the_waiter`) |
| U8 a measure-mode Stage-4 failure keeps the old embeddings, journalled `MEASURE_FAILURE_STAGE`, exit 3 | P6 T1 (`carry_embeddings`, `test_a_stage4_failure_keeps_the_old_embeddings_and_a_valid_record`) |
| U9 a CPU pin under `auto` resolves inline with a note; refused with `--embed-stage gpu` (supersedes H9) | P4 T1 (resolver `cpu_pinned`, R2) |
| U10 SLURM measure on the staged controller, with a measure-mode finalizer | P6 T2 |
| U11 DINOv2 only for now: `giant` is the one new size; DINOv3 H+ and 7B deferred | P2 T1 (sizes, validator, tokens, hub check), P2 T4 (DINOv2 rows only), P0 T1 note |
| H1 crop geometry on CPU via `_crop` / `_square_colony_crop` | P1 T2, P4 T3 (loader calls `prepare_regions`) |
| H5 process mode inline on CPU | P4 T1 (R1), P4 T2 (`test_process_mode_ignores_embed_stage`) |
| H6 reuse `--gpu-slurm`, `--gpu-shards` | P5 T2 |
| H7 `compute_dtype` field in the descriptor | P2 T1 |
| H9 superseded by U9 | P4 T1 |
| H10 backbone-major, one model resident, shared loads | P4 T3 (`plan_stage4`) |
| §4.1 where Stage 4 runs; recorded config, outside the work id | P4 T1, P5 T2, P6 T1 |
| §4.2 deferral, signal, `replace_run_embeddings`, record once | P3 T1, P3 T2, P4 T2, P4 T3 |
| §4.3 controller v2, `_classify_stage4`, rounds, worklist, array over shards, `CUBLAS_WORKSPACE_CONFIG`, `StageTag` and per-run terminal stage | P5 T1, P5 T2, P4 T2 (`StageTag`) |
| §4.4 plan, loop, DataLoader, `BatchEncoder`, batch size, determinism, embedder refactor, `entry_descriptor_for` | P1 T1–T2, P2 T2–T3, P4 T3 |
| §4.5 (as amended by U11) `giant`, size/version validator, hub verification, licence gate twice, `compute` block | P2 T1, P4 T3 (licence at shard start), P5 T3 (licence at preflight is the existing `PF-LICENSE`) |
| §4.6 done/pending, `ResumeStage` `"stage4"`, gated shortcuts, interrupted states, Stage-1 re-run deletes signals, `--retry-failures`, measure carry and republish, manifest 4 → 5 | P4 T2, P4 T3 (`stage4_status`), P4 T4, P6 T1 |
| §4.7 R1–R3, `PF-EMBED-SHARD-AMORTIZE`, `PF-EMBED-RESOURCES` | P4 T1, P5 T3 |
| §4.8 failure modes (OOM, host OOM, no CUDA, corrupt store, grid outside, non-finite, root changed, walltime, fence, cold load) | P2 T3 (OOM halving), P3 T1 (root changed), P4 T1 (no CUDA at startup), P4 T3 (CUDA, corrupt, non-finite, root change non-terminal, fence on both seams), P4 T4 (walltime by real-crash injection), P5 T1 (rounds), P5 T3 (cold-load warning) |
| §5 golden; batch invariance on hardware; transaction tests; local e2e with crash injection; controller fake-scheduler tests; live smoke; measure over staged; full regression | P0 T2 + P1 T2; P2 T4; P3 T1; P4 T4; P5 T1; P5 T4; P6 T2; P7 T2 |
| §6 phases | this README's phase table |
| §7 left to the builder: pool sizes, `per_sample`, `accelerate`, internal names | P0 T1 (`per_sample`, memory bounds), P2 T3 (`max_in_flight_bytes`), P4 T3 (one commit thread), P2 T1 Step 5 (`accelerate`: avoided, see below) |

**Decisions this plan takes where the spec leaves the builder a choice** (§7):
- `device_map={"":"cuda"}` needs `accelerate`. The plan loads with
  `from_pretrained(dtype=compute_dtype)` then `.to("cuda")` instead, so the
  `foundation` extra gains no dependency. Giant in float32 is about 4.4 GB of
  host RAM during the load (U11; the deferred 7B would be 13.4 GB in bf16).
  P2 T1 Step 5 records the peak.
- The commit pool has **one** thread (the spec's §4.4 said two; plan review
  Cuts: one is enough to overlap the GPU, and it makes per-image ordering
  trivial). The DataLoader has `resolve_local_worker_count(config.n_jobs, n) − 1`
  workers locally (the allocation's on SLURM), `spawn`, `prefetch_factor=2`.
- A signal carries `root_sha256`, the root its payloads were computed against;
  a commit checks it against the live root and uses it as
  `expected_root_sha256`.
- `masked_dense` tiles are normalised by the HF processor in the loader, exactly
  as inline does, so the two paths share bits on the CPU (plan review I1).
- Stage 4 reads an unpinned revision from `BackboneKey.revision` and never
  rewrites the embedder's `revision` field, so `producer.params_sha256` is the
  same as an inline run's.

## Decisions on the open questions

Revision 1 raised O1–O9. All are decided; nothing in the plan is marked pending.

| # | Question | Decision | Where |
|---|---|---|---|
| O1 | Where the new sizes live | Accepted: a separate `EmbeddingDinoSize`; the detectors' `DinoSize` is unchanged | P2 T1 |
| O2 | Tokens for the new sizes | Accepted for the one new size under U11: `dinov2-vitg14`; the existing tokens are pinned by a test | P2 T1 |
| O3 | A non-DINO embedder under Stage 4 | Accepted: refused under R2 | P4 T1 |
| O4 | `compute` beside `model.device` | Accepted with a condition: one producer writes both, and a test asserts they agree | P2 T1 |
| O5 | `--embed-stage gpu` with no embedder | Accepted: resolves inline with a note | P4 T1 |
| O6 | The amortisation warning | Accepted with conditions: a GPU-only constant (439 / 297), worded as efficiency | P0 T1, P5 T3 |
| O7 | SLURM measure on the controller | Became **U10** (user, 2026-11-19), with a measure-mode finalizer | P6 T2 |
| O8 | Zero-progress leftovers | Became **U7** (user): no record, no journal row, exit 5; revision 1's journalling is removed | P5 T1 |
| U11 | DINOv3 H+ and 7B (user, 2026-11-20) | **DINOv2 only for now**: the cluster's HF token has expired and the DINOv3 licence is not accepted, so their weights and hub configs cannot be fetched or verified. `giant` is the one new size; existing DINOv3 sizes are untouched | P2 T1, P2 T4, Follow-ups |
| O9 | U5's scope | Accepted: Stage-4 runs only | P5 T2 |

## Follow-ups (not in this plan)

- **The GUI run console** has no `--embed-stage` control, so it launches a
  no-detector run that resolves inline. The control is the GUI follow-up of
  spec §6, with its `FEATURES.md` and `WORKFLOWS.md` rows.
- **DINOv3 H+ and 7B:** needs an accepted licence and a valid HF token;
  re-add the sizes, hub verification and P2 T4 rows (U11).
- **`promote_store`'s two renames:** a walltime kill between them can lose a
  store. Pre-existing; Stage 4 crosses the window once more per image. P7
  documents it as a known limit; a fix belongs to the store writer, not here.

## Render

From this folder, with the files in phase order:

```bash
uv run --no-project python ~/.claude/tools/render_md_artifact.py \
  README.md p0-validation-and-golden.md p1-embedder-refactor.md \
  p2-large-models-batch-encoder.md p3-store-transaction-and-signal.md \
  p4-local-stage4.md p5-slurm-controller.md p6-measure-republish.md \
  p7-docs-and-regression.md \
  -o ../../artifacts/2026-11-12-gpu-embedding-staging/plan.html \
  --title "GPU Embedding Plan" \
  --eyebrow "Build plan · PhenoTypic #282 · 2026-11-19"
```

## Revision 2

Revised 2026-11-19 against `reports/2026-11-12-gpu-embedding-staging/plan-review.md`
(plan at `1604d2ff9`, code at `26d7f86a6`) and the spec as amended in
`c33b58193`. Each finding, and where it is fixed:

| Finding | Fix | Pinned by |
|---|---|---|
| **C1** resume re-runs Stages 2–3; SLURM Stage-3 `SystemExit`; no-detector continuation | One predicate, `stages_1_to_3_done`, returns early everywhere and publishes nothing; it also gates Stage 2 (`_cli_staged_strategy.py:250-259`); `run_stage3_step` returns before the prereq check (`:536-552`); explicit no-detector continuation, local and SLURM; a core re-run deletes the image's signals. P4 T2, P4 T4, P5 T2 | `test_a_waiting_image_is_neither_redetected_nor_remeasured`, `test_the_slurm_stage3_step_returns_early_for_a_waiting_image`, `test_a_core_rerun_deletes_the_images_signals`, `test_a_no_detector_continuation_skips_the_core`, `test_the_single_pass_role_skips_a_finished_image` |
| **C2** local completion counts Stage-3 markers | Under deferral the predicate is record-based (`_unpublished_images` in place of `_stage3_less_images`, `phenotypicCLI.py:1535,4329`); every image Stage 4 is handed ends published or journalled, or the stage raises. P4 T2, P4 T3 | `test_a_stage4_failure_exits_3_and_is_not_staged_complete` |
| **C3** measure-mode failure atomicity | U8: `replace_image_tables(carry_embeddings=True)`; a republish-mode failure republishes the record and journals `MEASURE_FAILURE_STAGE`. P6 T1 | `test_a_stage4_failure_keeps_the_old_embeddings_and_a_valid_record`, `test_carry_keeps_the_runs_embedding_files_and_block` |
| **C4** no progress with several backbones under a walltime | Signals reused per key; the loader prepares only missing keys; the progress digest counts valid signal keys. (The alternative, documenting the walltime, was not taken.) P3 T2, P4 T3, P5 T1 | `test_a_second_round_embeds_only_the_keys_it_lacks`, `test_written_signals_count_as_progress` |
| **C5** stale signals loop; terminal second `StoreRootChangedError` | Valid = signal root equals live root; stale signals deleted and re-embedded; a root change is never terminal. P3 T2, P4 T3 | `test_signals_are_classified_per_key_against_the_live_root`, `test_a_signal_from_an_old_root_is_deleted_and_reembedded`, `test_a_root_change_at_commit_leaves_the_image_pending` |
| **I1** tile normalisation differs | Tiles go through the HF processor in `prepare_regions`; real-backbone golden gains `crop_mean` and `masked_dense`; P2 T3 and T4 gain `masked_dense`. P0 T2, P1 T2, P2 T3, P2 T4 | `test_dense_tiles_are_the_processors_pixels`, `test_the_batch_encoder_matches_the_inline_encoder[masked_dense]` |
| **I2** BatchEncoder holds images and memory | Out-of-order drain, a byte-bounded flush (revision 3); memory modelled in `stage4_memory.py`. P0 T1, P2 T3 | `test_a_finished_image_is_drained_before_a_partial_batch`, `test_crop_pipelines_keep_full_batches`, `test_dense_grids_flush_on_the_byte_bound` |
| **I3** DataLoader hazards | Per-backbone loader, `spawn`, not persistent, identity `collate_fn`. P4 T3 | `test_loader_items_arrive_as_ndarrays` |
| **I4** `CUBLAS_WORKSPACE_CONFIG` only in the SLURM prelude | `prepare_stage4_environment` sets it right after resolution. P4 T1 | `test_the_cublas_variable_is_set_before_any_strategy` |
| **I5** flat order, non-crashing crashes, half a fence | One commit thread and order compared per image; crashes are `os._exit(137)` in a subprocess; the fence test runs on both seams. P4 T3, P4 T4 | `test_commit_then_record_then_signals_per_image`, `test_a_real_crash_resumes_from_disk`, `test_a_fenced_epoch_commits_nothing` |
| **I6** wrong does-not-fit case | The unsupported set is computed; the refusal now uses giant fp32 on an explicit 4 GiB `gpu_total_bytes`, since no DINOv2 row is unsupported (U11). P0 T1, P2 T2 | `physical_batch_size.py` checks 1–3, `test_weights_that_do_not_fit_are_refused` |
| **I7** orphan cleanup unreachable | `sweep_orphan_stage4_signals` over the whole inventory, locally and in the controller. P4 T3, P5 T1 | `test_orphan_signals_of_a_done_image_are_swept`, `test_orphans_are_swept_before_classifying` |
| **I8** missed event callers; test file name | All six callers listed; `test_cli_update_state_stage.py`. P4 T2 | `test_stage3_completion_is_not_completion_when_the_run_embeds_on_gpu` |
| **I9** SLURM measure finalizer not wired | `RUN_MODE: "measure"`; no forward run proof; `measure_failed` from the journal under the epoch. P6 T2 | `test_the_measure_finalizer_publishes_no_run_proof_and_counts_measure_failures` |
| **I10** API errors | Forward runs use `journal_table_run(store)`; measure uses `owned_table_run_id(block, pipeline_sha256=...)`. P4 T3, P6 T1 | the P4 T3 and P6 T1 suites |
| **I11** stale recomputation cannot fire | The commit reads the digests from the live store; a mismatch fails the image (revision 3). P3 T1, P4 T3 | `test_a_payload_bound_to_another_image_fails_the_image` |
| **I12** zero-region `encode()` loads a model | It returns `(0, 0)` without `_dim()`; `payload_from` reshapes. P1 T1, P1 T2 | `test_zero_regions_call_neither_the_encoder_nor_dim`, `test_zero_regions_trigger_no_second_load` |
| **I13** non-determinism; tests that cannot fail | Golden embedders pin `device="cpu"`; `stage4_cpu` hides CUDA; the one-value float16 test is gone; registers use `crop_mean`; a leaky stub proves the isolation check can fail. P0 T2, P2 T3, P4 | `test_the_registers_case_differs_from_a_registerless_run`, `test_the_isolation_check_catches_a_leaky_model` |
| **I14** stub without parameters; delegate missing `dtype` | Parametric stub; the measurer's `load_dino_backbone` delegate gains `dtype`. P0 T2, P2 T1 | `test_a_bfloat16_load_reaches_the_weights` |
| **I15** `SLURM_CPUS_PER_TASK` `KeyError` | `resolve_local_worker_count`. P4 T3, P5 T2 | `test_an_unset_cpu_count_does_not_raise` |
| **I16** no-CUDA detected after Stages 1–3 | Local device check at startup. P4 T1 | `test_no_cuda_stops_before_stage1` |
| Minor: P2 T4 additions | m vs m/2, separate-process repeat, warnings as errors, `masked_dense` row. P2 T4 | the report's table |
| Minor: OOM once then succeed | P2 T3 | `test_oom_once_halves_and_succeeds` |
| Minor: amortisation constant | GPU-only: 439 base, 297 large. P0 T1, P5 T3 | `shard_amortization.py` check 1 |
| Minor: loader memory | Layers and uint16 counted. P0 T1, P5 T3 | `test_a_uint16_plate_needs_more` |
| Minor: republish byte-identity | P6's review checks equality apart from P2's added entry fields. P6 | deep review P6 |
| Minor: R3 remedy | Names `--gpu-slurm` on SLURM. P4 T1 | `test_r3_names_the_recorded_flag_and_the_remedy` |
| Minor: `tracked_state.md` rows | Signals, worklist, counters. P7 T1 | docs build |
| Minor: promote's two renames | Documented as a known limit, not fixed (pre-existing, the store writer's). P7 T1, Follow-ups | — |
| Minor: GUI run console | A GUI follow-up; no `_gui/` change here. Follow-ups | — |
| O1–O9 | See "Decisions on the open questions" | — |
| Cut: `worklist_partition.py` | Dropped; the rule is a unit test. P0 T1, P5 T1 | `test_the_partition_rule` |
| Cut: `_journal_stage4_no_progress` | Dropped under U7. P5 T1 | `test_two_unchanged_rounds_finalize_and_record_nothing` |
| Cut: one republish carrier | The manifest flag; no config key, no `--republish`. P5 T1–T2, P6 T1 | `test_stage4_script_is_one_unchunked_gpu_array`, `test_a_slurm_measure_with_gpu_slurm_uses_the_controller` |
| Cut: one-thread commit pool | P4 T3 | `test_commit_then_record_then_signals_per_image` |
| Spec U5 (amended), U7, U8, U9, U10, S1 | See "Spec coverage" | — |

Not adopted: none. Where the review offered alternatives (C4 "or document the
walltime", I1 "or record the drift", I16 "refuse or warn"), the plan takes the
first, and says so in the row.

## Revision 3

Revised 2026-11-19 on top of `671665959`, against the re-review of revision 2.
Each item, and where it is fixed:

| Item | Fix | Pinned by |
|---|---|---|
| **1** local no-detector finalization calls `staged_detector_slot` | `reconcile_stage3_publications` (`phenotypicCLI.py:4264-4271`) is guarded on a `GpuDetector`; its comment (`:4265-4266`) corrected. P4 T2 | `test_a_no_detector_local_run_finalizes`; `test_local_stage4_matches_inline_deliverables[single-pass]` (P4 T4) |
| **2a** the loader ignores `--njobs` | `resolve_local_worker_count(config.n_jobs, n) − 1` locally; `--njobs 1` gives 0 workers, in-process. P4 T3 | `test_njobs_1_runs_the_loader_in_process` |
| **2b** fault injection by monkeypatch is invisible to spawned workers | An on-disk trigger: `FaultyDinoEmbedding` and `$PHENOTYPIC_TEST_FAULT_DIR`. P4 (fixtures), P4 T3, P4 T4, P6 T1 | `test_a_loader_worker_error_fails_only_its_image[0/2]`, `test_a_stage4_failure_exits_3_and_is_not_staged_complete`, `test_retry_failures_reruns_only_stage4`, `test_a_stage4_failure_keeps_the_old_embeddings_and_a_valid_record` |
| **2c** a worker loads its own processor | Processors are loaded in the main process and pickled into the dataset. P4 T3 | `test_the_dataset_pickles_with_its_processor` |
| **3** U8 with a differing measure pipeline | `replace_image_tables(carry_embeddings=[(run_id, key), …])`; the measure core carries matching key and `params_sha256` from its own run, else from the record's certified run (`certified_table_run`, `_cli_recompile_recovery.py:339`); an unmatched key has nothing to keep, and a Stage-4 failure journals `MEASURE_FAILURE_STAGE`, exit 3. P6 T1 | `test_a_differing_measure_pipeline_carries_matching_entries`, `test_carry_from_another_run_into_a_new_folder`, `test_carry_refuses_an_undeclared_pair` |
| **4** in-flight bound by image count | `max_in_flight_bytes` (`MAX_IN_FLIGHT_OUTPUT_BYTES`, P0 T1 `stage4_memory.py` checks 3–4). P2 T3 | `test_crop_pipelines_keep_full_batches`, `test_dense_grids_flush_on_the_byte_bound` |
| **5** `stage4_cpu` relies on `CUDA_VISIBLE_DEVICES` | It also patches `torch.cuda.is_available`; so does the subprocess runner. P4 (fixtures) | every `stage4_cpu` test on a GPU node |
| **6** no-CUDA refusal, `--dry-run` cost, CUBLAS values | Exit `1` via `click.ClickException`, one line, before Stage 1; the torch import is only for an explicit local `--embed-stage gpu`, and `--dry-run` pays it (P4 T1, tutorial in P7 T1); `check_cublas_workspace` accepts `:4096:8` and `:16:8` (P2 T2) | `test_no_cuda_stops_before_stage1`, `test_cublas_workspace_values[:16:8]` |
| Minor: progress digest reads footers | One `stat` per key (existence and `st_mtime_ns`). P5 T1 | `test_the_progress_digest_reads_no_parquet` |
| Minor: stale at commit | A per-image `stage4` failure; nothing committed or published. P4 T3 | `test_a_payload_bound_to_another_image_fails_the_image` |
| Minor: orphans of a finished run | The sweep runs before the "All images already processed!" exit (`phenotypicCLI.py:3889-3897`). P4 T4 | `test_a_finished_rerun_sweeps_orphans_before_it_exits` |
| Minor: a local root change leaves an image pending | `run_local_stage4` exits `1` with "still need Stage 4". P4 T3 | `test_a_local_run_left_pending_exits_1` |
| Minor: U5 reuse for local continuations | Revisions recorded in `processing_state.json` too; local continuations reuse them. P5 T2 | `test_a_local_continuation_reuses_the_recorded_revision` |
| Minor: a deferring core leaves an old `stages.stage3` | It drops it (and the image's signals) first. P4 T2 | `test_a_core_rerun_drops_the_old_stage3_first` |
| Minor: `stage="stage3"` tag | Only under deferral. P4 T2 | `test_inline_events_stay_untagged` |
| Minor: the dense-tile test compares `"np"` with itself | It compares with the inline `"pt"` path. P1 T2 | `test_dense_tiles_are_the_processors_pixels` |
| #281 defect: `_commit_hash` unset on transformers ≥ 5.18 | Every revision comes from `resolved_revision` (gridname's fallback, in flight on #281); a guard forbids `_commit_hash` outside `_dino_support.py`. P2 T1, P5 T2 | `test_the_revision_comes_from_resolved_revision`, `test_no_282_module_reads_commit_hash_directly`, `test_u5_records_a_revision_when_commit_hash_is_absent` |

The two revision tests depend on gridname's #281 fix; until it merges they
fail, which is the behaviour they pin. Revision 2's table above is kept as
history; where revision 3 renamed a test, the table names the new one.

### U11 (2026-11-20)

The user chose DINOv2 only for now (spec `52674eac2`: U11, §1 goal 2, §4.5).
Applied as docs-only plan edits:
- **P2 T1:** `EmbeddingDinoSize = Literal["small", "base", "large", "giant"]`
  (the merged tree's `DinoSize` is `Literal["small", "base", "large"]`,
  unchanged); `giant` with version 3 is refused; one token, `dinov2-vitg14`;
  hub verification covers `facebook/dinov2-giant` only; no
  `_DINOV3_SIZE_TO_REPO` or `Dinov3CheckpointManager` change.
- **P2 T2:** no DINOv2 combination is unsupported on the GPU table (the
  committed script's unsupported set is all `dinov3-7b`), so the refusal test
  uses an explicit 4 GiB `gpu_total_bytes`, with a 16 GiB control that keeps it
  able to fail.
- **P2 T4:** DINOv2 rows only (base `crop_cls`, base `masked_dense`, large fp32,
  giant bf16); the float16-range check covers ViT-g.
- **P0:** the scripts stay as committed; a note marks their DINOv3 rows
  informational, and the memory bounds sized for 7B (`ENCODER_OUTPUT_BYTES` at
  4096-d, the 1.54 GB grid, `MAX_IN_FLIGHT_OUTPUT_BYTES`) as conservative,
  since giant's grid is about 0.75 GB.
- **P5 T3:** the licence preflight test uses DINOv3 base, not 7B.
- **P7:** float32 is recommended for every size, giant included; bf16 is documented as a memory-bound opt-in (H7, amended 2026-11-20 after P2 T4).
- **README:** stop point 4 removed; U11 in the decisions table and spec
  coverage; O2 narrowed to `dinov2-vitg14`; a DINOv3 follow-up.
