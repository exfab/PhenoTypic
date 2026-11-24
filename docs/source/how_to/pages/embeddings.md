# Embed colonies on GPUs

`MeasureDinoEmbedding` turns each colony crop, or each cell of a grid, into a
DINO feature vector. On a CPU that is slow: about 64 s per plate for 66 colony
crops and a 96-cell grid on four cores, against under 2 s on one RTX 6000 Ada
(spec §2; see **Measured numbers** at the end). This page shows how to run every
`MeasureDinoEmbedding` of a CLI run on GPUs, in a separate **Stage 4** after
each image's store is final, and what that changes about the results.

Stage 4 is a CLI feature. `embedder.measure(image)` in a notebook, or
`pipeline.measure(image)`, is unchanged and runs the backbone where its
`device` says.

## Where embeddings run: `--embed-stage`

```bash
python -m phenotypic --mode full --pipeline pipeline.json \
    --input /plates/ --output /out/ --embed-stage gpu
```

`--embed-stage` takes three values; the default is `auto`.

- **`gpu`**: the per-image core measures everything else, writes the store and
  defers the embeddings. Stage 4 then reads each finished store read-only,
  embeds it on a GPU, rewrites the store's embedding tables, and only then
  publishes the image's per-image record.
- **`inline`**: the embedders run beside the other measurers, as before this
  option existed.
- **`auto`**: resolved once, before anything is written, by the rules below.

| Run | `auto` resolves to |
|---|---|
| `--mode full` on SLURM with a `GpuDetector` in the pipeline, or with `--gpu-slurm` given | `gpu` |
| `--mode full` on SLURM with neither | `inline` |
| any local run | `inline` |
| `--mode measure` on SLURM with `--gpu-slurm` given | `gpu` |
| `--mode measure` otherwise | `inline` |
| `--mode process` (copy or in place) | `inline` |
| a pipeline with an embedder pinned to `device="cpu"` | `inline`, with a note naming it |
| a pipeline with an embedder that is not a `MeasureDinoEmbedding` | `inline`, with a note naming it |
| a pipeline with no embedder | `inline` |

A local run uses Stage 4 only when you ask for it with `--embed-stage gpu`. In
`--mode measure` a `GpuDetector` does not count as a GPU profile, because measure
mode runs no detector; only `--gpu-slurm` does. Only `MeasureDinoEmbedding` has a
Stage-4 encoder, so any other `MeasureEmbedding` in the pipeline keeps the whole
run inline. The resolved value is recorded in `.phenotypic/processing_state.json`
(`embed_stage`) and is configuration, not part of an image's work id, so changing
it never invalidates finished images.

### What is refused

Each refusal is one usage-error line (exit `2`) before anything under `--output`
changes, and `--skip-validation` does not skip it.

- **`--embed-stage gpu` with `--mode process`.** Process mode always embeds
  inline on the CPU.
- **`--embed-stage gpu` with an embedder Stage 4 cannot run**: one pinned to
  `device="cpu"`, or one that is not a `MeasureDinoEmbedding`. The message names
  the `meas` key. Drop the flag (or pass `--embed-stage inline`), or set
  `device="auto"` or `"cuda"`.
- **A continuation that would switch stage.** Under `auto`, a continuation keeps
  the stage its run started with and prints a note saying so, whatever `auto`
  would resolve to today. An explicit `--embed-stage` that differs from the
  recorded one is refused, so a run never mixes CPU and GPU embeddings. The
  message names the recorded flag; to move a finished run to the GPU, re-embed it
  with `--mode measure --embed-stage gpu` (below).
- **A SLURM continuation of a `gpu` run with no GPU profile.** If the run
  recorded `gpu` and this invocation has neither a `GpuDetector` nor
  `--gpu-slurm`, Stage 4 would be submitted on the `--slurm` CPU profile. The CLI
  refuses and asks for the run's `--gpu-slurm` options.

### A local `--embed-stage gpu`

A local Stage 4 runs in the CLI process. Before Stage 1 writes anything, the CLI
checks that this machine has a usable CUDA device: torch is installed,
`torch.cuda.is_available()`, and the torch build has kernels for the device. If
not, it prints one line and exits `1`. That check imports torch, so a `--dry-run`
with `--embed-stage gpu` pays the import too. Stage 4 never falls back to the CPU.

The DataLoader that cuts and resizes the crops uses one worker per CPU after the
first: `--njobs N` gives `N - 1` workers, and `--njobs 1` prepares the crops in
the main process.

## On SLURM

```bash
python -m phenotypic --mode full --pipeline pipeline.json \
    --input /plates/ --output /out/ \
    --slurm slurm_partition=intel --slurm mem_gb=16 --slurm time=60 \
    --gpu-slurm slurm_partition=exfab --gpu-slurm slurm_account=exfab \
    --gpu-slurm mem_gb=24 --gpu-slurm time=120 \
    --gpu-shards 2
```

- **Resources.** Stage 4 uses the same GPU profile as a detector's Stage 2:
  `--gpu-slurm` options inherit from and override `--slurm`, and one GPU per task
  is added. There are no new resource flags.
- **Shards.** `--gpu-shards N` runs Stage 4 as an array of up to N whole-GPU
  tasks. Each loads each backbone once and embeds its share of the images. Only
  the shards that have work are submitted.
- **Few shards for a small run.** Each shard pays one model load before its first
  image. With cached weights it took 39.7–297.9 s to the first commit in the P5
  live smoke (dinov2-small, RTX 6000 Ada and short_gpu nodes), and 87–303 s in
  the P6 measure check, against 2.7–4.6 s per image after that
  (`live-smoke.md`). A run of a few hundred images gains little from many shards.
- **Loader workers.** A shard uses one loader worker per allocated CPU after the
  first, so `--gpu-slurm slurm_cpus_per_task=` sets them. The preflight warns
  (`PF-EMBED-RESOURCES`) when the GPU profile's memory is below what those
  workers need for your largest plate.
- **Without a `GpuDetector`.** A pipeline with no GPU detector embeds on GPUs on
  SLURM once you give `--gpu-slurm`. Its per-image core runs as CPU array tasks
  on the `--slurm` profile, then Stage 4 runs on the GPU profile.
- **Weights.** Pre-stage the weights in the Hugging Face cache on shared storage
  and run with `HF_HUB_OFFLINE=1`, so no shard downloads them.
- **Pin `revision=`.** An unpinned `MeasureDinoEmbedding(revision=None)` is
  resolved to one hub commit when the run is first submitted, recorded in the
  run's state, and loaded by every shard and every continuation. Offline, or on
  a hub error, it cannot be resolved; the CLI warns, and each shard then
  resolves on its own. With pre-staged weights and `HF_HUB_OFFLINE=1`, pinning
  `revision` to a commit is the only safe pairing (see `PF-EMBED-REVISION-UNPINNED`
  in the [preflight table](../../tutorials/pages/cli_batch_processing.md)).
- **`CUBLAS_WORKSPACE_CONFIG`.** The Stage-4 script exports `:4096:8`, which
  deterministic cuBLAS needs. Locally the CLI sets `:4096:8` when the variable is
  unset, keeps a `:16:8` you set, and refuses any other value (exit `1`).

`phenotypic-wait` turns the submission's exit `4` into the run's own exit code,
and returns only after every job the submission queued has ended, including the
last recovery controller.

## Model sizes

| `size` | Backbone (DINOv2) | Vector | Weights per process, float32 |
|---|---|---|---|
| `"small"` | ViT-S/14, `dinov2-vits14` | 384 | about 90 MB |
| `"base"` (default) | ViT-B/14, `dinov2-vitb14` | 768 | about 350 MB |
| `"large"` | ViT-L/14, `dinov2-vitl14` | 1024 | about 1.2 GB |
| `"giant"` | ViT-g/14, `dinov2-vitg14` | 1536 | about 4.4 GB |

The memory column is the `MeasureDinoEmbedding` docstring's; `bfloat16` halves
it. `giant` is DINOv2 only: `MeasureDinoEmbedding(size="giant", version=3)` is
refused when it is constructed, because DINOv3 has no ViT-g. DINOv3 `small`,
`base` and `large` are unchanged, and still need
`PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3`, which Stage 4 checks before it loads
anything. DINOv3's ViT-H+ and 7B are not available yet.

## `compute_dtype`

`compute_dtype` is the type the backbone runs in: `"float32"` (the default),
`"bfloat16"` or `"float16"`. Vectors are pooled in float32 and stored as
float16 whatever it is.

**Use `float32` for every size, `giant` included.** That giant fits in float32 is
the result of Stage 4's batch-size rule, not a measurement: the rule finds no
DINOv2 size and crop size that does not fit in float32 on any GPU in its table
(`physical_batch_size.py`, the change's logic-validation script), but giant in
float32 was not run on a GPU for this change. The batch-invariance measurement
(`batch-invariance.md`) ran giant in `bfloat16` only. There, large in float32
moved by at most 2.27e-7 in cosine distance across three GPU models, while giant
in `bfloat16` moved by:

- up to 3.16e-4 in cosine distance between two GPU models (median 3.1e-5 to
  3.3e-5), against at most 2.27e-7 in float32;
- up to 4.29e-4 when its batch is halved after a GPU out-of-memory error,
  against at most 6.27e-9 in float32.

So use `bfloat16` only for a model that does not fit in float32. `compute_dtype`
is part of the table's descriptor: tables computed in different types are never
grouped as one model. `"float16"` is accepted but was not measured, and ViT-g is
known to overflow float16 activations, which fails every image.

## What each entry records

Every embedding entry in a store records where it was computed, in a `compute`
block beside `model.device` and `model.device_name`:

| Field | Holds |
|---|---|
| `device` | the device that ran the backbone: `"cuda"` (or `"cuda:N"`), or `"cpu"` |
| `device_name` | the GPU's name, e.g. `"NVIDIA RTX 6000 Ada Generation"` |
| `capability` | the GPU's compute capability, e.g. `"8.9"`; `null` on the CPU |
| `torch`, `cuda` | the torch and CUDA versions |
| `physical_batch` | the number of inputs per forward pass |
| `physical_batches` | present only when the image's rows ran at more than one pass size (an out-of-memory halving part-way through), every size, largest first; `physical_batch` is then `null` |
| `stage` | `"stage4"`, or `"inline"` |

The block is provenance. Tables are grouped into one descriptor without it, so
a run whose shards ran on different GPU models publishes one descriptor.
`compute_dtype` is not in this block; it is part of the descriptor.

## Repeatability

**Within one GPU model and one physical batch size, a colony's stored vector does
not depend on how the run was sharded or batched.** P2's batch-invariance job
(`batch-invariance.md`) ran 486 crops and cells of three corrected plates on an
RTX 6000 Ada, an A100 and an RTX PRO 6000 Blackwell, for DINOv2 base
(`crop_cls` and `masked_dense`) and large in float32, and giant in `bfloat16`.
On each GPU, every stored
float16 vector was bit-identical to the reference when its batch neighbours, its
position in the batch, its padding or its shard changed, and in a repeat in a
separate process (on another node, for the Blackwell). Every run used a physical
batch of 64.

What can change the bits:

- **A different GPU model.** In float32, at most 2.27e-7 in cosine distance
  (large, Ada vs A100) and at most 6.76e-9 for base. You may mix GPU models in
  one run: the device is provenance, recorded per entry.
- **An out-of-memory halving.** After a GPU out-of-memory error, that input
  shape's batch is halved once for the shard, and the entry records every size
  it ran at. In float32 the halved batch gave identical bits on Ada and A100, and
  at most 6.27e-9 on Blackwell.
- **The CPU.** A CPU and a GPU embedding of one DINOv2-base crop differ by a
  median 2e-6 and at most 1.4e-5 in cosine distance (spec §2, 198 crops and 288
  cells). For scale, the same colony imaged 6 h later differs by a median 0.125.
  A Stage-4 run never mixes in the CPU.

**A re-measure reproduces the vectors.** In the P6 live check
(`live-smoke.md`), a same-pipeline `--mode measure --embed-stage gpu` over 400
images re-embedded every image on the same GPU model as the forward run, and all
800 embedding files were byte-identical to the forward run's.

## When a run is interrupted

Run the same command again; there is no `--resume` flag.

- **Walltime.** A shard killed by its walltime leaves the images it had not
  committed untouched, and keeps every per-image signal it had written. The
  controller then works out what is left and submits another round over every
  shard, so the leftover work spreads over all the GPUs. In the P5 live smoke
  (`live-smoke.md`, attempt 2: 400 plates, two shards on `exfab`,
  `--gpu-slurm time=8`), round 1's two shards were killed at 8:14 and 8:19 after
  committing 335 images; round 2 committed the other 65 (33 and 32) in about
  2 min; no image was embedded twice, all 400 records verified, and
  `phenotypic-wait` exited `0`.
- **A stall.** When two rounds in a row change nothing, the run ends unfinished:
  the leftover images get no record and no failure row, and `phenotypic-wait`
  exits `5`. The finalizer's log repeats why. A shard that cannot commit its
  first image within its walltime causes this; raise `--gpu-slurm time=`, then
  run the same command again.
- **A killed CPU task.** A Stage-1 or Stage-3 task that dies without a result
  (an `OUT_OF_MEMORY` kill, a node failure) is resubmitted, only those tasks.
  After two retry rounds that change nothing, those images stay unfinished and
  the run exits `5`; raise `mem_gb` in `--slurm` and run the same command again.
- **GPU out of memory that halving cannot absorb, a missing CUDA, a failed
  forward pass.** These are node state, never an image's failure. Nothing is recorded as failed; on SLURM the
  next round retries the shard's images, and locally the run stops with exit `1`
  and a plain re-run finishes it.

## When an image fails

An image whose Stage 4 fails (a store that does not decode, a grid cell
outside the image, a non-finite vector) is recorded as a terminal failure at stage
`stage4`, has no record, and makes the run exit `3`. Its store keeps its
measurements. A plain re-run skips it, as it skips every terminal failure. In
a local run, `--retry-failures` retries it and only its Stage 4 runs again: the
failure kept the image's Stage-3 results, and a test pins this. On SLURM the
code takes the same path (the image is planned as "Stage 4 only", and the
controller retries a failure recorded by an earlier submission), but no test
yet runs it end to end.

## Re-embed a finished run: `--mode measure`

`--mode measure --embed-stage gpu` re-measures and re-embeds an existing run's
stores on GPUs. Use it to move a run that embedded inline onto GPUs, or to
replace an embedding a reader cannot use.

```bash
phenotypic-wait --output /out/ -- python -m phenotypic --mode measure \
    --pipeline pipeline.json --output /out/ --embed-stage gpu \
    --slurm slurm_partition=intel --gpu-slurm slurm_partition=exfab \
    --gpu-slurm slurm_account=exfab
```

- **The old embeddings are kept until the new ones are written.** The measure
  step carries each image's readable embeddings forward, and Stage 4 replaces
  them. If Stage 4 fails for an image, the image keeps the embeddings the
  measure step carried and a record that verifies, the failure is recorded, and
  the run exits `3`.
- **Exit `0` means every image was re-measured and re-embedded.** If any image
  the forward run published is left neither re-embedded nor recorded as failed
  (a Stage-4 stall, a measure step that kept dying), the pass ends unfinished:
  `phenotypic-wait` exits `5` and the finalizer's log names each such image.
  Locally the same case is exit `1`, with the images named.
- **A killed measure step is re-run, never certified.** On SLURM, each measure step writes
  a small marker under `.phenotypic/progress/measure_done/` once it has rewritten
  the image's table. An image without one for this pass is resubmitted, and Stage
  4 never takes it.
- **The weights do not move.** An unpinned `revision` reuses the commit the run
  recorded. Pin `revision=` in the measure pipeline to move to another commit.
  A forward run that embedded inline recorded none, so the measure resolves one,
  once, at submission.
- **No preview.** `--mode measure` refuses `--dry-run`, so the first measure
  invocation is the real one.

## Exit codes

| Code | With Stage 4 |
|---|---|
| `0` | every image published (a measure pass: every image re-measured and re-embedded) |
| `1` | locally: no usable CUDA (before Stage 1), Stage 4 stopped on node state, images left pending, or a local measure pass that left images neither re-embedded nor failed |
| `2` | a refused option, including the refusals above |
| `3` | finished, and some images failed terminally, Stage-4 failures included |
| `4` | submitted to SLURM; `phenotypic-wait` turns it into the run's own code |
| `5` | `phenotypic-wait`: the run ended unfinished (a Stage-4 stall, CPU tasks that kept dying, an unfinished measure pass); re-run the same command |
| `6` | `phenotypic-wait` gave up after `--timeout-seconds` |

## Known limitations

- **An image whose crops are huge is slow, and no stop rule catches it.** When
  the detected objects sprawl across the plate, each padded crop is thousands of
  pixels on a side, and cutting and resizing them on the CPU takes minutes per
  image: in the P5 smoke, up to about 35 min for one image (`live-smoke.md`, F3).
  Every round still commits a few images, so the stall rule never fires, and such
  a run can go on for days. The remedy is in the pipeline: refine the objects so
  the crops are colony-sized.
- **An image that crashes its shard every round ends the run.** An image that,
  for example, exhausts the host's memory kills its shard each round. Once it is
  the only image left, its rounds change nothing, so the run ends with exit `5`,
  and a plain re-run repeats that. Raise the shard's memory in `--gpu-slurm`, or
  leave the image out with `--image-manifest`.
- **One local invocation per output folder.** Two local invocations over one
  `--output` at the same time are unsupported, in `--mode full` and in
  `--mode measure`, as before Stage 4 existed. Each store write is checked
  against the store it read, so the second writer leaves that image unfinished
  (exit `1`) rather than corrupt it, but both may publish one image's record.
- **The GUI run console has no `--embed-stage` control yet.**

## Measured numbers

Every number on this page comes from one of these, in
`docs/superpowers/reports/2026-11-12-gpu-embedding-staging/` unless noted:

- `batch-invariance.md`: invariance, OOM halving and cross-GPU distances on
  Ada, A100 and Blackwell (torch 2.11, CUDA 13.0).
- `live-smoke.md`: the P5 live SLURM smoke (400 Maresca plates, dinov2-small,
  two shards) and its P6 section, the SLURM `--mode measure` check.
- `docs/superpowers/specs/2026-11-12-gpu-embedding-staging/design.md` §2: the
  CPU and GPU time per plate and the CPU-vs-GPU distances.
