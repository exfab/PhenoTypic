# GPU Embedding Stage 4 Implementation Plan

**Status:** draft for review · 2026-11-19
**Spec:** `docs/superpowers/specs/2026-11-12-gpu-embedding-staging/design.md` (approved 2026-11-19)
**Parent:** #281, `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`
**Base:** `feat/gpu-embedding-staging` (PR #282), after the orchestrator merges #281
(`worktree-dvc-embeddings-refactor` at `d3c54f723` or later) into it. Every
`file:line` below is read from the #281 worktree at `d3c54f723`, not from the
#282 branch as it stands today.

---

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every `MeasureDinoEmbedding` of a staged run on GPUs, in a Stage 4
after the store is final. One resident model per GPU shard, batched across
images, with vectors that do not depend on sharding or batching.

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
  fields where they are (open question O4).
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
  | `--mode measure`, SLURM with `--gpu-slurm` | `gpu` |
  | `--mode measure`, otherwise | `inline` |
  | `--mode process` (copy or in place) | `inline`; an explicit `gpu` is refused (R1) |
  | a pipeline with no in-scope embedder | `inline` (open question O5) |

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
  - the Stage-4 script prelude exports `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Physical batch:**
  `m = min(batch_size, pow2floor((0.8·GPU_mem − weights − 1 GiB) / per_sample))`.
  It depends on the GPU model (total memory), the backbone and `crop_px` only,
  never on free memory. On OOM, m halves once for the shard and is recorded in
  `compute.physical_batch`. At m = 1, or on a second OOM, the shard exits
  non-zero. Constants live in P0's logic-validation script.
- **Inside Stage 4, an embedder's device is `cuda`.** A missing or unusable CUDA
  fails the shard, never falls back to CPU. The test-only escape is
  `PHENOTYPIC_STAGE4_DEVICE=cpu`.
- **One writer per artifact.** Stage 3 (and the single-pass core in a Stage-4
  run) writes the store with its measurements and `stages.stage3`, and writes
  **no** record and no embeddings. Stage 4 writes the embeddings (one store
  transaction), then the record, then consumes its signals, in that order.
- **Signals:** `.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet`,
  atomic, with `work_id`, `run_id` and `root_sha256` in the Parquet key-value
  metadata.
- **Process mode** (copy and in place) and AutoConvertRaw-GC's direct
  `process_single_apply_only_core` calls stay inline on CPU, byte for byte (H5).
- No new resource flags: Stage 4 runs on `resolve_stage_slurm_args(gpu_slurm_args, slurm_args)`
  (`_cli_staged_slurm.py:110`) and `--gpu-shards` (H6).
- Heavy imports (torch, transformers, huggingface_hub) stay inside functions.
  `tests/unit/ci/test_startup_imports.py` and `test_deferred_imports.py` stay
  green. `sdk_` never imports `_cli` (`tests/unit/sdk_/test_run_state_layering.py`).
- **No file under `src/phenotypic/_gui/` changes.** The run-console control for
  `--embed-stage` is a GUI follow-up (spec §6).
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
   with one error line and exit `1`, journal no per-image failure, and leave every
   image Stage-4 pending, so a re-run on a GPU node finishes it. Pinned in P4 T3
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
  needs torch), `BatchEncoder` or the Stage-4 worker run in the foundation venv,
  from the worktree root:

  ```bash
  TPT='env QT_QPA_PLATFORM=offscreen PYTHONPATH=src:. /bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/python -m pytest -q --no-header -p no:randomly -o addopts= -m "not slow"'
  ```

  Each task marks its torch tests **[torch]**. A torch test in the shared env
  must skip through `pytest.importorskip("torch")`, never fail.
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
| `sdk_/_measurement_tables.py` | P3 |
| `_cli/_cli_stage4_signal.py` (new) | P3 P4 |
| `_cli/_cli_embed_stage.py` (new) | P4 P5 P6 |
| `_cli/_cli_stage4_worker.py` (new) | P4 P5 P6 |
| `_cli/_cli_staged_resume.py` | P4 P6 |
| `_cli/_cli_staged_workers.py` | P4 |
| `_cli/_cli_staged_strategy.py` | P4 |
| `_cli/_cli_execution_strategies.py` | P4 P5 P6 |
| `_cli/_cli_process_single.py` | P4 P5 P6 |
| `_cli/_cli_staged_controller.py` | P5 P6 |
| `_cli/_cli_staged_slurm.py`, `_cli_staged_slurm_worker.py` | P5 P6 |
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
| C10 | P4 T2, deferral and gated shortcuts | Seam, consistency-critical | Opus | light |
| C11 | P4 T3, local Stage-4 worker | Keystone | Opus | light |
| C12 | P4 T4, resume, retry, crash injection | Seam | Opus | **deep: P3–P4** |
| C13 | P5 T1, controller v2, rounds, worklist | Keystone | Opus | light |
| C14 | P5 T2, scripts, worker `--stage 4`, no-detector plan, U5 pin | Seam | Opus | light |
| C15 | P5 T3, preflight codes and docs rows | Leaf | Sonnet | light, plus `tests/unit/test_docs_preflight_codes.py` |
| C16 | P5 T4, live SLURM smoke | orchestrator, **user approves** | — | **deep: P5** |
| C17 | P6 T1, T2, measure republish | Seam | Opus | **deep: P6** |
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
1. Any design question the spec does not settle, including O1–O9 below, before
   the task that consumes it.
2. **The live SLURM smoke (P5 T4)** spends real GPU time; it runs only on the
   user's go-ahead.
3. **A batch-invariance measurement that refutes the hypothesis** (P2 T4): any
   stored float16 bit that changes with neighbours, position or partition on a
   fixed physical shape. The spec's §5 fallback (an m = 1-per-crop mode) is then a
   user decision, and P4 does not start.
4. **DINOv3 weights.** The 7B and H+ rows of P2 T4 need the gated licence
   accepted on the Hugging Face account and a token on the cluster. If either is
   missing, the orchestrator asks rather than skipping the rows silently.

## Spec coverage

| Spec item | Task |
|---|---|
| §1 goals 1–3 (speed, larger models, consistent embeddings) | P2 T1–T4, P4 T3, P5 T1–T2 |
| §2 measured facts (device is provenance) | Global Constraints; P2 T1 (`compute` excluded from grouping) |
| U1 GPU Stage 4 after the store is final | P4 T2 (deferral), P4 T3 (worker) |
| U2 `--embed-stage` default `auto` | P4 T1 |
| U3 no switching mid-run (R3) | P4 T1 |
| U4 measure mode uses Stage 4 | P6 T1, T2 |
| U5 unpinned revision resolved once at submission | P5 T2 |
| U6 mixed GPU models allowed; device provenance per entry | P2 T1 (`compute` block); P5 T2 (no device constraint across shards) |
| H1 crop geometry on CPU via `_crop` / `_square_colony_crop` | P1 T2, P4 T3 (loader calls `prepare_regions`) |
| H5 process mode inline on CPU | P4 T1 (R1), P4 T2 (`test_process_mode_bytes_unchanged`) |
| H6 reuse `--gpu-slurm`, `--gpu-shards` | P5 T2 |
| H7 `compute_dtype` field in the descriptor | P2 T1 |
| H9 `device="cpu"` embedder refused when Stage 4 is on (R2) | P4 T1 |
| H10 backbone-major, one model resident, shared loads | P4 T3 (`plan_stage4`) |
| §4.1 where Stage 4 runs; recorded config, outside the work id | P4 T1, P5 T2, P6 T1 |
| §4.2 deferral, signal, `replace_run_embeddings`, record once | P3 T1, P3 T2, P4 T2, P4 T3 |
| §4.3 controller v2, `_classify_stage4`, rounds, worklist, array over shards, `CUBLAS_WORKSPACE_CONFIG`, `StageTag` and per-run terminal stage | P5 T1, P5 T2, P4 T2 (`StageTag`) |
| §4.4 plan, loop, DataLoader, `BatchEncoder`, batch size, determinism, embedder refactor, `entry_descriptor_for` | P1 T1–T2, P2 T2–T3, P4 T3 |
| §4.5 `giant`/`huge_plus`/`7b`, size/version validator, hub verification, licence gate twice, `compute` block | P2 T1, P4 T3 (licence at shard start), P5 T3 (licence at preflight is the existing `PF-LICENSE`) |
| §4.6 done/pending, `ResumeStage` `"stage4"`, gated shortcuts, interrupted states, Stage-1 re-run deletes signals, `--retry-failures`, measure republish, manifest 4 → 5 | P4 T2, P4 T4, P6 T1 |
| §4.7 R1–R3, `PF-EMBED-SHARD-AMORTIZE`, `PF-EMBED-RESOURCES` | P4 T1, P5 T3 |
| §4.8 failure modes (OOM, host OOM, no CUDA, corrupt store, grid outside, non-finite, root changed, walltime, fence, cold load) | P2 T3 (OOM halving), P3 T1 (root changed), P4 T3 (CUDA, corrupt, grid, non-finite, fence), P4 T4 (walltime by crash injection), P5 T1 (rounds), P5 T3 (cold-load warning) |
| §5 golden; batch invariance on hardware; transaction tests; local e2e with crash injection; controller fake-scheduler tests; live smoke; measure over staged; full regression | P0 T2 + P1 T2; P2 T4; P3 T1; P4 T4; P5 T1; P5 T4; P6 T2; P7 T2 |
| §6 phases | this README's phase table |
| §7 left to the builder: pool sizes, `per_sample`, `accelerate`, internal names | P0 T1 (`per_sample`), P2 T3 (pool sizes), P2 T1 Step 5 (`accelerate`: avoided, see below) |

**Decisions this plan takes where the spec leaves the builder a choice** (§7):
- `device_map={"":"cuda"}` needs `accelerate`. The plan loads with
  `from_pretrained(dtype=compute_dtype)` then `.to("cuda")` instead, so the
  `foundation` extra gains no dependency. 7B in bf16 is 13.4 GB of host RAM
  during the load, inside a 48 GB `--mem`. P2 T1 Step 5 records the peak.
- The commit pool has 2 threads; the DataLoader has `cpus_per_task − 1` workers
  (local: `len(os.sched_getaffinity(0)) − 1`), `prefetch_factor=2`.
- A signal carries `root_sha256`, the root its payloads were computed against,
  so a commit in a later round uses the right `expected_root_sha256`.
- Stage 4 reads an unpinned revision from `BackboneKey.revision` and never
  rewrites the embedder's `revision` field, so `producer.params_sha256` is the
  same as an inline run's.

## Open questions

Each blocks only the task named; the plan is written with the recommended
default and marks it **(pending Ox)**.

- **O1 (P2 T1). Where the new sizes live.** `DinoSize` (`sdk_/typing_.py:163`)
  is shared by `DinoSam2Detector`, `FssDinoDetector` and `Insid3Detector`.
  Extending it, as spec §4.5 says literally, lets those detectors accept and
  tune-enumerate `giant`/`huge_plus`/`7b`. **Recommended:** a separate
  `EmbeddingDinoSize` Literal for the embedder only.
- **O2 (P2 T1). Model tokens for the new sizes.** They become folder names and
  keys. **Recommended:** the hub's own spelling, `dinov2-vitg14`,
  `dinov3-vith16plus`, `dinov3-vit7b16`.
- **O3 (P4 T1). A non-DINO `MeasureEmbedding` in a Stage-4 run.** Stage 4 has
  backbone code only for DINO. **Recommended:** refuse it under R2's line, since
  running it inline would mix CPU into a GPU run.
- **O4 (P2 T1). `compute` beside #281's `model.device`/`model.device_name`.**
  **Recommended:** keep both; `compute.device` and `compute.device_name` repeat
  them, so #281 entries keep their shape.
- **O5 (P4 T1). `--embed-stage gpu` with no in-scope embedder.**
  **Recommended:** resolve `inline` and print one note line; not a refusal.
- **O6 (P5 T3). What "too few images per shard to pay off the cold load" means.**
  **Recommended:** warn when the cold load exceeds the shard's estimated warm GPU
  time (about 200 images per shard for DINOv2-base at the measured 3.8 s/image
  and 750 s cold). The alternative, break-even against inline CPU, is about 12.
- **O7 (P6 T2). SLURM measure mode moves engines.** With Stage 4 on, a SLURM
  `--mode measure` runs through the staged controller (phases
  `["stage3","stage4"]`, a measure role) instead of `AutonomousSLURMStrategy`.
  The spec implies this (U4 and §4.6 need controller rounds) but does not say it.
- **O8 (P5 T1). The spec's "exit 3" on zero progress.** `phenotypic-wait` exits
  `3` only when every accepted image is verified or failed. For two unchanged
  Stage-4 rounds to end at `3`, the controller must journal each remaining image
  as a terminal `stage4` failure before the finalizer; otherwise the run reads
  `5`. **Recommended:** journal them, so `--retry-failures` retries only their
  Stage 4.
- **O9 (P5 T2). U5's scope.** **Recommended:** resolve at submission only for
  Stage-4 runs, as the spec places it; inline SLURM runs keep per-process
  resolution and `PF-EMBED-REVISION-UNPINNED`.

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
  --eyebrow "Implementation plan · PhenoTypic #282 · 2026-11-19"
```
