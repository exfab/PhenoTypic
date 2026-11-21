# Design: GPU embedding stage (Stage 4)

**Status:** draft for approval · 2026-11-19 · supersedes the 2026-11-12 scoping stub
**Parent:** `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md` (#281).
This change is stacked on it as #282.
**Option:** C, a GPU Stage 4 (user choice, 2026-11-19).
**Blueprint:** a read-only architecture pass over #281's code (2026-11-19). File
anchors below are from that pass. Lines marked *inferred* were reasoned, not read,
and are verified during the build.

## 1. Intent

#281 ships `MeasureDinoEmbedding` on **CPU**, run inline by every per-image
writer. That is fine for AutoConvertRaw-GC: two grid embedders add about 7% to
its roughly 612 s per image. It is not fine for full PhenoTypic runs, where
crops and wells cost about 64 s per image on four cores. Over 33,000 images that
is about 590 CPU-node-hours, against about 16 hours on one GPU.

This change runs embedding on GPUs, in its own stage, with three goals the user
set:

1. **Speed.** One resident model per GPU worker processes many images, batched
   across images. The cold model load is paid once per shard, not once per
   process.
2. **Larger models.** DINOv2-large and -giant, in reduced precision where
   needed. (DINOv3 H+ and 7B are deferred, U11.)
3. **Consistent embeddings.** A staged run embeds every image on GPUs, never
   mixing in CPU. Mixed GPU models on SLURM are fine. Within one GPU model
   and physical batch size, vectors must not depend on how images were
   sharded or batched (restated 2026-11-19, plan review S1: m is chosen per
   GPU model and OOM halving changes it, so an absolute guarantee cannot
   hold; across GPU models the measured difference is about 1e-7).

## 2. Measured facts (2026-11-19)

The measurements used real 4012 × 6016 corrected plates (66 colonies, 96
wells), from code at `d1bc6febe`. Logs are in
`/bigdata/exfab/anguy344/gate-results/gpu-baseline/`.

**Warm time per image:**

| | CPU, 4 cores, base | GPU (RTX 6000 Ada), base | GPU, large |
|---|---|---|---|
| Object crops (66) | 26.1 s | 0.72 s | 1.05 s |
| Wells grid (96 cells) | 37.9 s | 0.99 s | 1.48 s |
| Agar cell (1) | 5.5 s | 2.1 s | 2.1 s |

- **The agar cell's time is a CPU resize** of a 3224 × 4836 crop, on either
  device.
- **Cold first call:** 650–750 s (imports plus weights from GPFS). The large
  model on a warm node took 33 s.
- **Peak RSS:** about 6 GB.

**Device differences** (DINOv2-base, 198 crops, 288 cells, stored float16
vectors; cosine distance):

| Comparison | Median | Max |
|---|---|---|
| CPU vs GPU | 2e-6 | 1.4e-5 |
| GPU vs GPU (Blackwell vs Ada) | about 0 | 1.2e-7 (last float16 bit) |
| CPU vs CPU, two nodes | bit-identical | |
| Same colony 6 h later | 0.125 | min 0.025 |

The device is therefore provenance, not descriptor identity (parent E5, D3).

## 3. Decisions

### Made by the user (2026-11-19)

| # | Decision |
|---|---|
| U1 | Option C: a GPU Stage 4 after the store is final. |
| U2 | **Opt-in is `auto`.** A new `--embed-stage {auto,gpu,inline}` defaults to `auto`. On SLURM, `auto` resolves to `gpu` when the run already has a GPU profile (a `GpuDetector`, or `--gpu-slurm` given), otherwise `inline`. A local run resolves `inline` unless `--embed-stage gpu`. |
| U3 | **No switching mid-run.** A continuation whose resolved `embed_stage` differs from the recorded one is refused, naming the recorded flag, and `--mode measure --embed-stage gpu` to re-embed. |
| U4 | **`--mode measure` uses Stage 4 in v1.** It is the documented remedy for an unreadable embedding (parent E5), and re-embedding a GPU run must stay on GPU. |
| U5 | **An unpinned `revision` is resolved once, at a run's first submission,** and recorded in job metadata, so every shard and round loads the same commit; a continuation reuses the recorded commit rather than re-resolving (2026-11-19). It applies to Stage-4 runs only. Offline, it falls back to per-process resolution with the existing warning. |
| U7 | **A stalled Stage-4 round ends unfinished** (2026-11-19, plan review O8): after two unchanged retryable sets the finalizer runs, the leftover images keep no record and no journal row, and the run exits 5 ("re-run the same command"); a plain re-run retries them. GPU OOM and a changed root are node or concurrency state, never terminal per-image failures. |
| U8 | **A measure-mode Stage-4 failure keeps the old embeddings** (2026-11-19, plan review C3): the measure core carries the image's existing embeddings block forward (still valid, since measure never writes the objmap); Stage 4 replaces it. A Stage-4 failure leaves the previous embeddings and a valid record, and is journalled as `MEASURE_FAILURE_STAGE`, so the run exits 3. |
| U9 | **An explicit `device="cpu"` pin wins** (2026-11-19): under `auto`, a run whose in-scope embedder is pinned to CPU resolves `inline`, with a one-line note; `--embed-stage gpu` with such a pin is refused. |
| U10 | **SLURM `--mode measure` moves onto the staged controller** (2026-11-19, plan review O7), with a measure step, Stage 4 rounds, and a measure-mode finalizer (`measure_failed`, no forward run proof). |
| U11 | **DINOv2 only for now** (2026-11-20). The new size this PR adds is DINOv2 `giant` (ViT-g/14). DINOv3 H+ and 7B are deferred to a follow-up, because their gated weights and hub configs need a licence and a valid token, which the cluster does not have; nothing about them could be verified. Existing DINOv3 sizes are untouched. The batch-invariance job (§5) runs DINOv2 models only. |
| U6 | **Mixed GPU models are allowed.** `device` and `device_name` are provenance, recorded per entry. |

### Defaults taken (the user did not object)

| # | Decision |
|---|---|
| H1 | **Crop geometry and the skimage resize stay on the CPU,** in DataLoader workers, through the same `_crop` / `_square_colony_crop` functions the inline path uses. The GPU runs only the backbone. CPU and GPU vectors then differ only by backbone arithmetic. |
| H5 | **Process mode, copy and in place, stays inline on CPU in v1.** That covers AutoConvertRaw-GC's direct `process_single_apply_only_core` calls. Process stores are write-once and byte-reproducible, and Stage 4 would add a second, device-dependent transaction. A GPU in-place backfill is a v2 candidate. |
| H6 | **`--gpu-slurm` and `--gpu-shards` are reused.** No new resource flags in v1. |
| H7 | **`compute_dtype: Literal["float32","bfloat16","float16"] = "float32"`** is a new field. It is part of the descriptor, because bf16 moves vectors far more than 1e-6. Storage stays float16 (parent D1). The docs recommend float32 for every DINOv2 size, giant included; it fits on every cluster GPU (2026-11-20). bf16 is an opt-in for memory-bound models only, documented with P2 T4's measurement: within a bf16 run, mixed GPU models differ by up to 3.2e-4 in 1 − cos and an OOM batch halving by up to 4.3e-4, against ≤ 2.3e-7 and ≤ 6e-9 in float32. No warning or refusal; documentation only (user, 2026-11-20). |
| H9 | Superseded by U9: an explicit CPU pin makes the run resolve inline; refused only with `--embed-stage gpu`. |
| H10 | **Backbone-major.** One model is resident at a time. Embedders sharing a backbone share one load. |

## 4. Design

### 4.1 Where Stage 4 runs

| Mode | Embedding |
|---|---|
| Full, `GpuDetector`, SLURM | Stage 4; Stage 3 defers |
| Full, `GpuDetector`, local | inline in Stage 3, unless `--embed-stage gpu` |
| Full, no `GpuDetector`, SLURM with `--gpu-slurm` | Stage 4, through the staged controller with phases `["stage3", "stage4"]`; the "stage3" role runs the single-pass core with `defer_embeddings=True` |
| Full, no `GpuDetector`, SLURM without `--gpu-slurm` | inline (`AutonomousSLURMStrategy`), as today |
| Full, local | inline, unless `--embed-stage gpu` |
| `--mode measure` | Stage 4 under the same `auto` rule, where only `--gpu-slurm` counts as a GPU profile; republish mode (§4.6) |
| `--mode process`, copy and in place | always inline on CPU; `--embed-stage gpu` refused |
| `--layer objmap`, recompile, migrate | no embedder runs |

The resolved value goes into `processing_state.json` config and into
`job_metadata.json` (`EMBED_STAGE`). It is a config value, not a new tracked
state, and it stays out of the work-id digest.

### 4.2 Work unit and one writer per artifact

**Stage 3 defers two things:** `embed()` and the per-image record. This applies
to the single-pass core too, in a Stage-4 run. It still writes the store with
its measurements, and writes `stages.stage3`.

Stage 4 reads that store read-only, computes the payloads, and writes one
**Stage-4 signal** per embedder key:
`.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet`.
Each signal is atomic and fenced, with the `work_id` and `run_id` in its key-value
metadata. Then Stage 4 commits:

1. **The store.** `replace_run_embeddings(store, run_id, payloads,
   expected_root_sha256=…)`, a new function on `_rewrite_store_tables`.
   - `clear_table_run=f"{run_id}/embeddings"`.
   - Measurements, metadata, other runs and `upstream` are carried, as hard
     links.
   - It rewrites only that run's `embeddings` block, with `stale` recomputed
     against the store's own objmap and array digests (D7).
   - The root is written last, and re-consolidated if it was consolidated.
2. **The record,** published for the first and only time in this pass.
3. **The signals,** consumed.

Deferring the record, rather than republishing it, keeps "one writer per
artifact, per pass" true (`_cli/CLAUDE.md`). Each Parquet has one writer, the
record is written once, and the root is never certified before its last write.
An image whose Stage 4 fails keeps **no** record, so the run reads unfinished,
never verified with a silent hole. The spec of the "One writer" section, and
parent D5's writer list, gain the Stage-4 rewrite.

### 4.3 Controller

- **Config version goes 1 → 2,** with a `phases` list: `["stage1","stage2",
  "stage3","stage4"]`, or `["stage3","stage4"]` without a detector.
- **After the last Stage-3 chunk,** the controller classifies Stage 4 and
  either submits a GPU shard round or the finalizer.
- **`_classify_stage4`:**
  - done when the record is valid;
  - terminal on a recorded failure, missing `stages.stage3`, a store that
    doesn't match its work id, or a changed reference digest;
  - retryable otherwise.
- **Rounds** use their own counters: two unchanged retryable sets in a row go to
  the finalizer; the leftover images stay unrecorded and the run exits 5 (U7). The ledger role is
  `stage4` and the token `stage4-round-<n>`, so `live_slurm_job_ids` sees them.
- **A per-round worklist** (`stage4_worklist.json`) partitions only the
  retryable set across shards. After a walltime, the leftover work spreads over
  every GPU.
- **Stage 4 is an array over shards and is never chunked.** It has no trigger
  entries and no sidecar jobs. Its resources are the Stage-2 GPU profile. The
  script prelude sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Progress:** `StageTag` gains `"stage4"`, and the terminal stage for event
  counting becomes per-run.

### 4.4 Worker

- **Plan.** The `(meas_key, embedder)` pairs are grouped by `BackboneKey(version,
  size, revision, compute_dtype)` in sorted order. Inside Stage 4, an embedder's
  device is forced to `cuda`. A missing CUDA fails the shard, never falling back
  to CPU, except for a test-only `PHENOTYPIC_STAGE4_DEVICE=cpu`.
- **Loop.**
  1. Fence check, then filter: an image whose store already holds every key
     fresh, with matching params, goes to publish-only.
  2. `configure_determinism()`.
  3. For each backbone: load once (`dtype=compute_dtype`, then `.to("cuda")`;
     `device_map` would need `accelerate`, which the foundation extra does not
     carry), stream images through a `BatchEncoder`, and write signals as each
     image's key completes. Then free the model.
  4. A one-thread commit pool commits images whose signals are complete, under
     the commit guard, overlapping GPU work. (One thread, not two, per plan
     review I5: per-image call order then holds by construction; commits are
     I/O-bound and short next to GPU work.)
- **DataLoader.** One item is one image, with workers `cpus_per_task - 1`
  (from the CLI's allocation-aware worker count), one loader per backbone,
  `spawn` context, not persistent, order-preserving, identity `collate_fn`
  (plan review I3). Workers read the store read-only, compute the
  binds digests and cut and resize every crop through the embedder's own
  functions. For `masked_dense`, they cut patch-aligned tiles.
- **BatchEncoder.** It keeps one queue per input geometry. Every forward pass
  of a queue has exactly that queue's m inputs; the last is zero-padded.
  Inputs are normalised as `embed_crops` does, cast to `compute_dtype`, run
  under `inference_mode`, and pooled in float32. Vectors are routed back by
  `(image, key, row)`, and each drained image reports every pass size its rows
  ran at. After any failed forward pass, the encoder re-queues that pass's
  rows and refuses all further use, so the shard fails rather than continue
  with images a failed pass held (2026-11-20, P1/P2 review I6).
- **Batch size.**
  `m = min(batch_size, pow2floor((0.8·GPU_mem − weights − 1 GiB) / per_sample))`.
  It is a function of the GPU model, the backbone and the input geometry only,
  never of free memory. m is **per queue geometry** within one backbone
  (`crop_px`, or a `masked_dense` tile's h × w): embedders sharing a model load
  may differ in `crop_px`, and a 448-px crop costs about four times a 224-px
  one (P1/P2 review I2). One function computes it from the loaded model
  (`physical_batch_for`, review I3); where embedders share a geometry, the
  smallest `batch_size` caps it. On OOM, that queue's m halves once for the
  shard, and every pass size an image's rows ran at is recorded in its
  `compute` block (§4.5). At m = 1, or on a second OOM in that queue, the shard
  exits non-zero, and the next round retries. The formula's constants live in
  a logic-validation script.
- **Determinism.**
  - cuDNN deterministic, no benchmark.
  - TF32 off for both matmul and cuDNN (DINO's patch embedding is a conv).
  - `set_float32_matmul_precision("highest")`.
  - `use_deterministic_algorithms(True, warn_only=True)`.
- **Batch invariance is a hypothesis until measured** (§5). At a fixed physical
  shape, a row's vector should be bit-identical whatever its batch neighbours,
  position or shard.
- **Embedder refactor.** `MeasureEmbedding` splits into three steps:
  - `prepare_regions(image)` (CPU), which produces keys, inputs, binds and
    input;
  - `encode(prepared, encoder)`;
  - `payload_from(prepared, vectors, descriptor)`.

  The inline `payload()` becomes prepare, inline encode, then `payload_from`,
  so CPU bytes are unchanged; a golden pins that. `entry_descriptor_for(model,
  compute)` avoids a second model load.

### 4.5 Larger models

- The embedding size type gains `giant`: DINOv2 only, ViT-g/14, 1536-d.
  (`huge_plus`, ViT-H+/16, and `7b`, ViT-7B/16, both DINOv3, are deferred:
  U11.)
- A validator refuses impossible size/version pairs.
- Repository ids and dimensions are verified against the hub before use.
- DINOv3 stays licence-gated (`require_license_acceptance`), refused at
  preflight and again at shard start, before any load.
- A per-entry provenance block, `compute: {device, device_name, capability,
  torch, cuda, physical_batch, stage}`, is excluded from descriptor grouping.
  `compute_dtype` is not excluded. `physical_batch` is the one pass size the
  entry's rows ran at. When they ran at several (an OOM halving part-way
  through the image, or an inline image's shorter last pass),
  `physical_batch` is null and the block gains `physical_batches`, every size,
  largest first, e.g. `[64, 32]`; an entry that ran no pass records
  `physical_batch: null` (P1/P2 review I1, M10).

### 4.6 Continuation and measure mode

- **Done.** "Stage 4 done" means the record is valid. "Stage 4 pending" means
  `stages.stage3` is present, the store matches the work id, a measurement table
  exists, and there is no record. `ResumeStage` gains `"stage4"`.
- **Recovery shortcuts.** Those that publish today (local
  `_terminal_output_exists`, SLURM `run_stage3_step`, and Stage 3's own publish)
  are gated on `defers_embeddings`. Otherwise they would certify an image with no
  embeddings.
- **Interrupted states resume content-defined:**
  - signals complete but no commit: commit only;
  - store promoted but no record: publish only;
  - record published but signals remaining: orphan cleanup.
- **A Stage-1 re-run deletes the image's Stage-4 signals.** `--restart` clears
  them with `progress/`.
- **`--retry-failures`** re-runs only the GPU embedding of a Stage-4 failure.
- **Measure mode (U4, U8, U10).** The measure core carries the existing
  embeddings forward and defers `embed` and `_republish_table_marker`. Stage 4
  runs in *republish* mode, flagged in the
  manifest (version 4 → 5), so it rewrites an already-certified store's
  embeddings and then republishes its record. That is a separate invocation, so
  "one writer per pass" holds.

### 4.7 Refusals and preflight

These are each one usage-error line in the read-only half, outside
`--skip-validation`, above `--restart`, `--overwrite`, `mint_run_identity` and
`--dry-run`:

- **R1:** `--embed-stage gpu` with `--mode process`.
- **R2 (U9, O3):** `--embed-stage gpu` with an in-scope embedder pinned to `device="cpu"`, or with an in-scope embedder that has no Stage-4 encoder (any `MeasureEmbedding` but `MeasureDinoEmbedding`). Under `auto`, either resolves the run inline instead, with a one-line note naming the embedder (O3 amended 2026-11-20).
- **R3 (U3):** a continuation whose `embed_stage` differs from the recorded one.

New preflight warnings, each with `HINTS` and a docs row:
- `PF-EMBED-SHARD-AMORTIZE`: too few images per shard to pay off the cold load;
- `PF-EMBED-RESOURCES`: `--mem` too low for the loader.

### 4.8 Failure modes

| Failure | Behaviour |
|---|---|
| GPU OOM | Halve m once; at m = 1 the shard exits. Node state, never a terminal per-image record; the next round retries |
| Host OOM (loader) | The shard dies; the next round retries |
| No usable CUDA (e.g. a P100 without `gpu_latest`) | Checked at shard start against the torch arch list; fails fast |
| Corrupt store at load | Per-image scientific failure (`stage4`); the shard continues |
| `GridCellOutsideImageError`, non-finite float16 | Per-image failure, as inline |
| Root changed before commit | `StoreRootChangedError`, the part removed; a plain retry next round |
| Walltime mid-batch | Uncommitted images untouched (root-last); written signals survive; the next round reclassifies |
| Epoch fenced | Nothing is published after the fence |
| Cold load | Paid once per shard per round; a preflight warning when shards are too small; pre-staged weights and `HF_HUB_OFFLINE=1` recommended |

## 5. Verification

- **Byte-identity of the inline path.** A golden of today's inline CPU bytes
  (objects and grid, stub backbone, and a real small backbone where cached)
  stays identical through the refactor.
- **Batch invariance, measured on hardware.** On `exfab` and on `short_gpu
  --constraint=gpu_latest` (Ada, A100, Blackwell), a crop's vector stays
  bit-identical when its batch neighbours, position or partition change. The job
  also reports cross-GPU distance and float16 range for ViT-g. The spec's
  "vectors don't depend on sharding" rests on this measurement. If it fails, the
  fallback is an m = 1-per-crop mode for runs that need byte stability.
- **`replace_run_embeddings`:**
  - root-last;
  - measurement and metadata inodes carried;
  - other runs and `upstream` byte-equal;
  - `stale` recomputed;
  - root-change retry;
  - re-consolidation;
  - checked by an independent reader (zarr-python, pyarrow).
- **Local end-to-end with a stub** (`PHENOTYPIC_STAGE4_DEVICE=cpu`), with crash
  injection after the signal, after the promote and after the record. The
  deliverable must be byte-equal to an inline run's **when both run the same
  pass shape** (the inline embedder's `batch_size` at least its region count,
  so each side runs one pass). Otherwise they agree only within the cosine
  bound: inline runs its last pass unpadded while Stage 4 pads to `m`, and
  CPU GEMM bits depend on row count (measured by C5, 2026-11-20: max 4.77e-7
  in float32, thread-count dependent).
- **Controller unit tests with a fake scheduler:**
  - rounds and zero-progress;
  - no Stage-2/Stage-4 counter crosstalk;
  - ledger roles;
  - sidecar absence.
- **Live SLURM smoke on `exfab`.** A short `--time` forces a walltime kill, and
  recovery happens by round. Also a no-detector run with `--gpu-slurm`.
- **Measure mode over a staged run** re-embeds on GPU, republishes, and exits 0.
- **The end-of-change full sharded regression,** once.

## 6. Build phases

| Phase | Content | Gate |
|---|---|---|
| P0 | Logic-validation script for m, amortisation and partition; inline CPU golden | per-step guards |
| P1 | Embedder refactor (prepare / encode / payload) | golden byte-identical |
| P2 | Large models, `compute_dtype`, `BatchEncoder`, determinism, batch size | unit tests, plus the GPU batch-invariance job across three GPU models |
| P3 | `replace_run_embeddings`, Stage-4 signal | store-transaction tests, independent reader |
| P4 | Local Stage 4: resolver, refusals, deferral, gated recovery, resume | local end-to-end and crash injection |
| P5 | SLURM: controller v2, rounds, worklist, no-detector plan, preflight | fake-scheduler tests, live `exfab` smoke |
| P6 | Measure mode, republish Stage 4 | measure-over-staged end-to-end |
| P7 | Docs (`_cli/CLAUDE.md`, `tracked_state.md`, the batch-processing tutorial, the embeddings how-to), then the full regression | full sharded regression |

A run-console control for `--embed-stage` is a GUI follow-up. It needs the
FEATURES.md and WORKFLOWS.md ledgers.

## 7. Left to the builder

- Exact pool sizes, and the `per_sample` constant (validated by the P0 script
  and the P2 measurement).
- Whether `accelerate` is needed for `device_map` loading, which is a dependency
  question for the `foundation` extra.
- Internal names, beyond those other modules consume.
