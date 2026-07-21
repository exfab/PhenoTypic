# GPU-Accelerated Colony Detection

Set up and use deep-learning-based colony detectors (SAM2, micro-sam, SAM3,
DinoSam2) with GPU acceleration.

```{warning}
**Behaviour change (this release).** The DINO-backed detectors
(`FssDinoDetector`, `Insid3Detector`) previously fed every tile to the ViT
backbone at a fixed 224x224 regardless of `tile_px` -- the checkpoint's
*classification* preset, not the tile you asked for. They now feed the
backbone the tile at its native geometry, so native pixels per patch equal the
backbone's `patch_size` (14 for DINOv2, 16 for DINOv3) instead of collapsing to
whatever 224 implied. Pipelines deserialized from JSON keep their pinned
`tile_px` (`to_json()` writes every field), but **will produce different,
higher-resolution masks** and cost substantially more GPU time -- re-serialize
to adopt the new `tile_px` defaults described below. Separately,
`Sam2Detector.crop_n_layers` moves from `0` to `1` (~5.2x cost) for
**newly-constructed** detectors only; existing serialized pipelines are
unaffected.
```

## Installation

The GPU detectors have different packaging constraints:

| Detector            | Package(s) needed          | Available via              | CUDA-capable?          |
|---------------------|----------------------------|----------------------------|------------------------|
| `Sam2Detector`      | `torch`, `torchvision`, `sam2` | **PyPI** (ships in `phenotypic[torch]`) | Yes — Linux + CUDA |
| `MicroSamDetector`  | `micro_sam` (+ `torch`)    | **conda-forge only**, not on PyPI | CPU by default; user-managed CUDA possible |
| `Sam3Detector`      | `transformers`, `huggingface_hub` (+ `torch`) | **PyPI** (ships in `phenotypic[foundation]`); weights **gated** | Yes — Linux + CUDA |
| `DinoSam2Detector`  | `transformers`, `sam2` (+ `torch`) | **PyPI** (ships in `phenotypic[foundation]`); DINOv2 weights **ungated** | Yes — Linux + CUDA |

### Per-model license posture

| Model         | Code license          | Weights license            | Gated? |
|---------------|-----------------------|----------------------------|--------|
| SAM2          | Apache-2.0            | Apache-2.0                 | No     |
| SAM3          | Apache-2.0 (`transformers`) | **SAM License** (commercial-OK) | **Yes** — accept on Hugging Face |
| DINOv2        | Apache-2.0            | Apache-2.0                 | No     |
| DINOv3 (opt-in) | Apache-2.0          | **DINOv3 License** (custom Meta) | **Yes** — accept on Hugging Face |
| INSID3 method (`Insid3Detector`) | Apache-2.0 (clean-room, no code vendored) | DINOv3-native backbone (gated) | **Yes** (via DINOv3) |
| FSSDINO method (`FssDinoDetector`) | paper **CC BY-NC-SA** (clean-room, no code vendored) | DINOv2 default (ungated) / DINOv3 opt-in | No (DINOv2) / Yes (DINOv3) |

PhenoTypic **does not redistribute model weights** — each weight is downloaded
by you from the upstream source under that model's license, which you accept
(see `NOTICE` and `licenses/`). The two semantic detectors
(`Insid3Detector`, `FssDinoDetector`) carry no vendored upstream code: INSID3 is
clean-room-reimplemented from its Apache-2.0 method (attributed), and FSSDINO is
clean-room-reimplemented from the paper only (the reference repo is
all-rights-reserved). When `dino_version=3` is selected, PhenoTypic displays
"Built with DINOv3" per the DINOv3 License.

PhenoTypic itself is distributed via PyPI and managed with `uv`. `micro_sam`
is not published on PyPI, so it is **not** included in any `phenotypic`
extra. Users who need `MicroSamDetector` must install `micro_sam`
themselves; the tutorial recipe below uses `pixi` for that.

> **Installing the detector packages and downloading their weights is covered in
> the {ref}`Deep Learning Detectors`
> section of the Getting Started tutorial.** In short: `uv add "phenotypic[torch]"`
> for SAM2, `"phenotypic[foundation]"` for the SAM3/DINO detectors, the `[gpu]`
> umbrella for all of them, and conda-forge `micro_sam` for `MicroSamDetector`.
> In a source checkout, use `uv sync --extra foundation` to install the same
> foundation-model dependencies. Gated SAM3/DINOv3 weights need a one-time
> Hugging Face license handshake (`hf auth login` after accepting the model
> license on Hugging Face).

`MicroSamDetector` is importable from `phenotypic.detect.nn` even when
`micro_sam` is missing; the `ImportError` is deferred to the first
`apply()` call and points back at the installation instructions.

## Downloading Model Checkpoints

Every detector downloads its checkpoints automatically on first use, and the
`phenotypic.detect.nn` CLI (`download` / `list` / `clear`) pre-fetches and
inspects them — essential before submitting SLURM jobs, since compute nodes
often lack internet access. The full command set, cache-location environment
variables (`TORCH_HOME`, `MICROSAM_CACHEDIR`, `HF_HOME`,
`PHENOTYPIC_ACCEPT_MODEL_LICENSE`, …), and the login-node pre-staging workflow
for gated foundation weights are documented in the
{ref}`Deep Learning Detectors`
tutorial section. The SLURM-specific staging is expanded under
[SLURM Deployment](#slurm-deployment) below.

## Using Sam2Detector

`Sam2Detector` wraps Meta's SAM2 automatic mask generator. It lays a grid of
prompt points over the RGB image, predicts masks at each point, filters by
quality, and assembles a labelled object map.

```python
from phenotypic.detect.nn import Sam2Detector

# Basic usage with default parameters
detector = Sam2Detector()

# Tuned for dense plates with small colonies
detector = Sam2Detector(
    model_size="small",
    points_per_side=48,
    pred_iou_thresh=0.6,
    min_mask_region_area=200,
)

# Apply to an image (downloads checkpoint on first use)
result = detector.apply(image)
print(result.num_objects)
```

### Parameter tuning for colony detection

- **`points_per_side`** (default 32): Controls the density of the prompt grid.
  Use 16 for large, well-separated colonies. Increase to 48--64 for dense
  plates with many small colonies. Higher values increase inference time
  quadratically.
- **`points_per_batch`** (default 8): Controls how many existing prompt-grid
  points are decoded at once. Lower this first when a run exhausts GPU or unified
  memory. It changes throughput, not crop resolution, prompt positions, or prompt
  density.
- **`input_scaling`** (default `"image_max"`): Converts non-uint8 model input
  in bounded row chunks. The default retains historical per-image contrast
  normalization and performed better on the labeled uint16 accuracy fixture.
  Use `"dtype_range"` when absolute sensor-scale intensity must be preserved.
- **`pred_iou_thresh`** (default 0.7): Minimum predicted IoU for keeping a
  mask. Raise to 0.85--0.95 for conservative detection (fewer false
  positives); lower to 0.5 to catch faint or ambiguous colonies.
- **`stability_score_thresh`** (default 0.92): Filters masks by boundary
  stability. Higher values keep only masks with crisp edges.
- **`min_mask_region_area`** (default 100): Minimum mask area in pixels.
  Increase to suppress agar texture, dust, and other small artefacts that
  SAM2 segments as objects. Typical range: 50--500 depending on image
  resolution.
- **`model_size`** (default `"tiny"`): `"tiny"` is fastest and sufficient
  for most colony plates. Use `"large"` for maximum mask quality on
  publication figures.
- **`crop_n_layers`** (default 1): Number of additional crop-pyramid layers.
  SAM2's encoder resizes the whole image to a fixed 1024x1024 square, so small
  colonies on a multi-megapixel plate can be lost to downsampling; each added
  layer re-tiles the image into nearer-native-resolution crops and merges them
  by NMS that prefers masks from smaller crops. `0` keeps a single full-image
  pass; the default `1` costs 5 encoder passes instead of 1 (~5.2x wall-clock)
  but recovered far more ground-truth colonies in measurement (see the
  behaviour-change notice above).
- **`box_nms_thresh`** (default 0.7): Box-IoU cutoff for non-maximum
  suppression between the dense point grid's redundant proposals *within* one
  crop -- distinct from `crop_nms_thresh`, which deduplicates the same colony
  seen in two different overlapping crops.

## Using MicroSamDetector

`MicroSamDetector` uses SAM models finetuned on large-scale microscopy
datasets. It is particularly effective for brightfield and darkfield
microscopy images of agar plates.

```python
from phenotypic.detect.nn import MicroSamDetector

# Default: ViT-Base light microscopy model
detector = MicroSamDetector()

# Use the larger model for higher accuracy
detector = MicroSamDetector(model_type="vit_l_lm")

result = detector.apply(image)
```

### Model selection

Light microscopy models (recommended for agar plate imaging):

- `"vit_t_lm"` -- ViT-Tiny, fastest, good for rapid screening
- `"vit_b_lm"` -- ViT-Base (default), best speed/accuracy trade-off
- `"vit_l_lm"` -- ViT-Large, highest accuracy, most VRAM

Electron microscopy models (for organelle segmentation):

- `"vit_b_em_organelles"` -- ViT-Base
- `"vit_l_em_organelles"` -- ViT-Large

Base SAM checkpoints (without microscopy finetuning):

- `"vit_t"`, `"vit_b"`, `"vit_l"`, `"vit_h"`

## Using Sam3Detector

`Sam3Detector` wraps Meta's **text-prompted** SAM3 foundation model. Unlike
SAM2's dense point grid, SAM3 segments everything matching a short text
`prompt` (default `"colony"`) in one true batched forward pass, then assembles
the predicted instance masks into a labelled `objmap` (`output_kind="instance"`).

```python
from phenotypic.detect.nn import Sam3Detector

# Override the prompt per run — SAM3 has no prompt-free "segment everything" mode.
det = Sam3Detector(prompt="yeast colony", score_thresh=0.5)
```

SAM3 weights are **gated** (SAM License). Accept the gate and authenticate
once (see {ref}`Deep Learning Detectors`)
before the first `apply()`.

**Dense plates.** SAM3 caps at 200 instances per forward. `facebook/sam3` is a
gated repository whose processor config we cannot read (requests return 403),
so the claim that it runs at 1008 px internally is carried as an
**assumption**, not a verified fact. `Sam3Detector` tiles large plates into
fixed `tile_px` crops with `tile_overlap`, infers each tile, and merges the
per-tile instances by **centroid-in-core assignment**
(`~phenotypic.detect.nn._tiling.assign_by_centroid_core`): each instance is
kept by the one tile whose core -- the Voronoi cell of the tile centres --
contains its centroid, so a colony straddling a seam is claimed by exactly one
tile and the fragment a neighbouring tile saw is never kept as its own
colony. There is no IoU-NMS deduplication step here; duplicates cannot occur
by construction. Images that fit one tile run un-tiled.

Key parameters:

- `prompt` — free text describing the target (`"colony"`, `"bacterial colony"`).
- `score_thresh` / `mask_threshold` — instance-confidence and mask-probability
  cutoffs.
- `min_mask_region_area` — drop masks smaller than this (default 100).
- `tile_px` / `tile_overlap` — dense-plate tiling controls (default 1008 / 0.15,
  the 1008 figure being the unverified assumption noted above). `tile_px` is a
  compute/context knob, not a fidelity knob: attention cost is quadratic in
  tokens per tile, while the total tokens across the plate are roughly fixed,
  so a larger `tile_px` costs more without segmenting any more finely.
- `tile_merge_iou` — **deprecated and ignored.** The cross-tile merge is
  centroid-in-core (above), which needs no IoU threshold. The field is
  retained only so pipelines serialized before this change still deserialize.

## Using DinoSam2Detector

`DinoSam2Detector` is a **training-free** instance detector that composes two
ungated foundation models: SAM2's automatic mask generator produces
class-agnostic proposals, and a **DINOv2** backbone (Apache-2.0, ungated by
default) supplies dense patch features. Each proposal is scored by cosine
similarity of its pooled DINO feature to a foreground prototype; background-like
proposals are dropped, near-duplicates merged by IoU, and survivors painted into
a labelled `objmap`.

```python
from phenotypic.detect.nn import DinoSam2Detector

# Recommended config is fully ungated (SAM2 Apache + DINOv2 Apache).
det = DinoSam2Detector(dino_size="base", similarity_thresh=0.5)
```

`dino_version` selects the backbone generation: `2` = DINOv2 (default, ungated),
`3` = DINOv3 (gated opt-in). Selecting `dino_version=3` routes the snapshot pull
through the DINOv3-License acceptance gate (pre-stage it with the
`download --model-type dinov3 --accept-license` command from the
{ref}`Deep Learning Detectors`
tutorial section). The default DINOv3 checkpoint id is
`dinov3-vitb16-pretrain-lvd1689m`.

Key parameters:

- `dino_version` / `dino_size` — backbone generation (2/3) and size.
- `sam2_model_size` — SAM2 variant for the proposal generator.
- `points_per_batch` — SAM2 proposal-decoder batch size (default 8). Lowering it
  reduces peak memory without reducing proposal density or crop resolution.
- `similarity_thresh` — minimum cosine-to-prototype score to keep a proposal.
- `merge_iou_thresh` — IoU above which two survivors are merged.
- `tile_px` / `tile_overlap` — geometry of the tiles the DINO backbone scores
  proposals against (defaults 518 / 0.15). Each SAM2 proposal is pooled from
  the tile whose core contains its centroid. Scoring against one whole-plate
  feature grid instead would leave every colony smaller than a single patch, so
  each proposal's pooled feature would collapse to the same zero vector and the
  re-ranking would do no work.
- `crop_n_layers` / `crop_nms_thresh` / `crop_overlap_ratio` /
  `crop_n_points_downscale_factor` — SAM2 crop-pyramid controls for the
  *proposal* half, forwarded to the same mask generator `Sam2Detector` uses and
  documented under {ref}`Using Sam2Detector`. `crop_n_layers` defaults to `1`
  here too.

## Semantic few-shot detectors (`Insid3Detector`, `FssDinoDetector`)

`Insid3Detector` (one-shot, in-context) and `FssDinoDetector` (few-shot) are
**semantic** detectors: they emit a binary `objmask` (`output_kind="semantic"`),
not their own instance labels. The mask auto-labels into the shared `objmap`
backend exactly like a threshold detector, so the repo's downstream watershed
(`SeparateObjects`) turns it into instances — pair them with a separation step
in your pipeline, just as you would `OtsuDetector`. Both run on a frozen DINO
backbone and write only `objmask`; `objmap[:] > 0` then equals `objmask[:]`.

Both ship a **curated colony exemplar** (a reference colony RGB + its mask,
rendered once from `load_synth_yeast_plate()`) as the **default** reference /
support set, so they work out of the box. Supply your own annotated exemplar to
transfer to a new colony appearance.

### `Insid3Detector` — one-shot in-context (DINOv3-native, gated)

A faithful clean-room reimplementation of INSID3 (Apache-2.0). Given a single
**reference image + reference mask**, it pools an in-context prototype and
cosine-matches every query patch — but first removes DINOv3's **positional bias**
(estimated by SVD and projected out, INSID3's defining step) so patches match on
appearance, not position. It is DINOv3-native (gated, `dino_version=3` default);
a `dino_version=2` opt-in runs gate-free (the debias is then a near-no-op).

Because `dino_version=3` is the default and DINOv3 weights are gated, the
constructor now emits a `UserWarning` at construction time (rather than
deferring the failure to first `apply()`) pointing at the Hugging Face access
request and `PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3`. Contrast
`FssDinoDetector` below, which defaults to the ungated DINOv2 backbone and so
needs no access request. (Both detectors additionally warn at model-load time
if `tile_px` is not a multiple of the loaded backbone's `patch_size` — which
happens when you flip `dino_version` and leave `tile_px` at the other
backbone's default.)

```{warning}
`Insid3Detector` is **weak at its default `similarity_thresh=0.5`**. On the
bundled synthetic plate it recovers only 8 of 96 colonies, and its own
functional test lowers the floor to `similarity_thresh=0.0` to obtain a
non-empty mask at all. This is a threshold-calibration problem that predates
the resolution work described in the behaviour-change notice above, and it is
independent of `tile_px`: the detection *rate* is the same on a small plate and
on a plate twelve times its area. Tune `similarity_thresh` down for your own
plates before trusting its output.
```

```python
from phenotypic.detect.nn import Insid3Detector

# Default: bundled exemplar + gated DINOv3 (accept the DINOv3 License first).
det = Insid3Detector(similarity_thresh=0.5)

# Override the in-context reference with your own annotated colony pair:
det = Insid3Detector(
    reference_image="ref_plate.tiff",
    reference_mask="ref_plate_mask.png",
    similarity_thresh=0.6,
)

# Gate-free DINOv2 variant (no Hugging Face token needed):
det = Insid3Detector(dino_version=2, dino_size="small")
```

Key parameters:

- `reference_image` / `reference_mask` — the in-context exemplar (defaults to
  the bundled colony exemplar).
- `dino_version` / `dino_size` — backbone (3 = DINOv3 default/gated, 2 = DINOv2).
- `similarity_thresh` — cosine cutoff binarising the match map.
- `svd_components` — INSID3's positional-debias strength (leading SVD directions
  removed; default 4 ≈ DINOv3's register-token count; `0` disables the debias).
- `tile_px` / `tile_overlap` — large-plate tiling (defaults 512 / 0.15). `512 =
  16 * 32` is an exact multiple of DINOv3's patch size (16). Under the native
  processor geometry, native pixels per patch equal the backbone's
  `patch_size` regardless of `tile_px`, so this is a compute/context choice,
  not a fidelity one: 512 and 1024 both resolve to 16.0 native px/patch, but
  1024 costs 2.6x more (attention is quadratic in tokens per tile, while the
  total tokens across the plate are roughly fixed). Raising it buys no extra
  detail. Note `config.image_size` reports 224 for DINOv3 — that is a
  classification preset, not a native-resolution signal, and is not used to
  pick this default.

### `FssDinoDetector` — few-shot (DINOv2 default, ungated)

A clean-room reimplementation **from the paper only** of FSSDINO
(arXiv:2602.07550, CC BY-NC-SA; the reference repo is all-rights-reserved and is
not vendored). From a **support set** it builds `n_clusters` class-specific
prototypes (k-means) plus a Gram matrix (channel co-occurrence), scores each
query patch by cosine to the prototypes and a Gram-refined energy, combines the
maps (mean ⊙ max) and assigns foreground vs background by `argmax`. It defaults
to **DINOv2** (ungated), so it runs gate-free.

```python
from phenotypic.detect.nn import FssDinoDetector

# Default: bundled one-shot exemplar + ungated DINOv2.
det = FssDinoDetector(n_clusters=5)

# A true few-shot support set:
det = FssDinoDetector(
    support_images=["s1.tiff", "s2.tiff", "s3.tiff"],
    support_masks=["s1_mask.png", "s2_mask.png", "s3_mask.png"],
    n_clusters=5,
)
```

Key parameters:

- `support_images` / `support_masks` — the few-shot support set (defaults to the
  bundled one-shot colony exemplar).
- `n_clusters` — class-specific prototypes per class (paper default 5).
- `feature_layer` — transformer hidden-state layer for the dense features.
  FSSDINO's "Semantic Selection Gap" finding is that intermediate layers often
  beat the last, but cannot be reliably selected unsupervised — so the default
  is `-1` (the last layer, the paper's safe default).
- `dino_version` / `dino_size` — backbone (2 = DINOv2 default/ungated, 3 = DINOv3
  gated opt-in).
- `similarity_thresh` — foreground-score floor on top of the fg-vs-bg argmax.
- `tile_px` / `tile_overlap` — large-plate tiling (defaults 518 / 0.15). `518 =
  14 * 37` is an exact multiple of DINOv2's patch size (14). Native pixels per
  patch equal the backbone's `patch_size` regardless of `tile_px`, so 518 and
  1022 both resolve to 14.0 native px/patch — 1022 just costs 3.3x more
  (attention is quadratic in tokens per tile, while total tokens across the
  plate are roughly fixed). Smaller is cheaper at equal fidelity; raising it
  buys no extra detail.

## Pipeline Integration

GPU detectors work like any other PhenoTypic operation in a pipeline:

```python
import phenotypic as pht
from phenotypic.detect.nn import Sam2Detector
from phenotypic.measure import SizeMeasurer

pipeline = pht.ImagePipeline(
    ops=[Sam2Detector(model_size="tiny", points_per_side=32)],
    measurer=SizeMeasurer(),
    name="sam2_colony_pipeline",
)

# Run the pipeline
results = pipeline.operate([image])
df = pipeline.measure([image])
```

### JSON serialization

Pipelines containing GPU detectors can be saved and loaded just like any
other pipeline. The detector parameters are serialized; the model weights
are not (they are re-downloaded or loaded from cache when needed):

```python
# Save
pipeline.to_json("sam2_pipeline.json")

# Load -- works without torch installed (model loads lazily on apply)
restored = pht.ImagePipeline.from_json("sam2_pipeline.json")
```

Internal state (attributes prefixed with `_`, such as the loaded model) is
excluded from serialization. The model is rebuilt transparently on the next
call to `apply`.

## Local Staged GPU Detection (CLI)

When you run a pipeline through the CLI (`python -m phenotypic`) and it contains
a `GpuDetector`, detection runs as **three internal stages** rather than invoking
the GPU model once per image. The segmentation model is built **once** and every
image is streamed through it — far more efficient than the notebook per-image
path when processing a directory:

1. **Stage 1 — CPU preprocess.** Every prior `ImageOperation` (enhancers,
   corrections) is applied per image and the result is saved to the normal
   per-image HDF (`results/<dataset>/hdf/<stem>.h5`).
2. **Stage 2 — resident-model GPU detect.** The detector's model is built once
   and kept resident while each staged HDF is streamed through
   `preprocess → infer_batch`. The labelled object map is written to a per-image
   `.npy` **sidecar** at `results/<dataset>/objmap/<stem>.npy`; the HDF is opened
   read-only here, so an interrupted run never corrupts it.
3. **Stage 3 — CPU merge + measure.** The sidecar is merged back into the image
   through the object-map accessor, the post-detector refiners and the
   measurement queue run, the HDF is re-saved atomically, and the sidecar is
   **deleted**.

The output folder is identical to a single-pass run — staging is an internal
optimization, not a different output contract.

**Resume is content-defined.** Re-running the same command skips any image whose
work already exists: Stage 1 skips when the HDF exists, Stage 2 skips when the
sidecar *or* the measurement parquet exists (Stage 3 deletes the sidecar, so the
parquet is the durable "done" marker), and Stage 3 skips when the parquet exists.
Progress is **stage-tagged** in the event log, so the run dashboard can show how
far each image has moved through the three stages. If Stage 1 fails for an image
(e.g. an unreadable file), Stages 2 and 3 skip it and record a structured failure
instead of aborting the batch.

```bash
# Forward run: detection stages automatically because the pipeline has a GpuDetector
python -m phenotypic --pipeline sam2_pipeline.json --input /plates/ -o /output/

# Export just the object maps (runs Stages 1-2, then writes one objmap PNG per image)
python -m phenotypic --mode process --layer objmap \
    --pipeline sam2_pipeline.json --input /plates/ -o /output/
```

## SLURM Deployment

When a pipeline contains a `GpuDetector` operation (either `Sam2Detector` or
`MicroSamDetector`), the CLI automatically adapts:

**Local execution:** Forward GPU runs use the staged engine above (the model
loads once and streams every image); see "Local Staged GPU Detection". The
legacy per-image path (measure-only and non-objmap layer exports) still forces
sequential processing (`n_jobs=1`) to avoid multiple workers competing for the
same GPU.

**SLURM execution:** Automatically adds `--gpus-per-node=1` to the SLURM
job if GPU resources were not explicitly requested.

```bash
# GPU resources are auto-requested when the pipeline contains a GpuDetector
python -m phenotypic --pipeline sam2_pipeline.json --input /plates/ -o /output/

# Override with explicit SLURM GPU arguments
python -m phenotypic --pipeline sam2_pipeline.json --input /plates/ -o /output/ \
    --slurm slurm_gpus_per_node=2 \
    --slurm slurm_partition=gpu
```

Pre-cache checkpoints on the login node before submitting (see
"Downloading Model Checkpoints" above).

### SLURM Staged GPU Detection

A forward GPU run on SLURM (`--slurm ...` with a `GpuDetector` pipeline) runs as
the staged engine above, coordinated by an **epoch-fenced dependent controller**
with **per-stage resources**:

- **Stage 1** — a CPU array over images (preprocess → staged HDF), on the
  `--slurm` profile.
- **Stage 2** — a GPU array over **shards** (`--gpu-shards N`): each task is one
  whole GPU running a resident model that streams its shard of HDFs to objmap
  sidecars, on the `--gpu-slurm` profile.
- **Stage 3** — a CPU array over images (merge sidecar → measure), on the
  `--slurm` profile.

Controllers wait with `afterany`, so a handful of per-image failures never block
the next decision. Because Stage 2 writes a `.npy` sidecar with the HDF opened
read-only, there is **no HDF5 write-locking on the GPU nodes**; Stages 1 & 3
write the HDF atomically (temp + rename) on CPU nodes.

```bash
# CPU partition for Stages 1 & 3 (--slurm); GPU partition + 2 concurrent GPUs for Stage 2
python -m phenotypic --pipeline sam2_pipeline.json --input /plates/ -o /output/ \
    --slurm slurm_partition=batch --slurm slurm_time=02:00:00 \
    --gpu-slurm slurm_partition=gpu --gpu-shards 2
```

GPU distribution uses `--gpu-shards` (whole GPUs, across nodes).
`--gpu-workers-per-gpu` is reserved; the current worker uses one model per shard.
`--gpu-slurm` inherits/deltas over `--slurm`, so shared keys
(account, qos, time) carry over and only the GPU partition/account need
restating; one GPU is requested automatically.

**Walltime survival.** Before a controller decides what comes next, it has a
dependent recovery controller in the queue. After a Stage-2 array terminates,
the controller checks atomic sidecars, resume parquets, missing Stage-1 HDFs,
and the current epoch's terminal-failure journal. If retryable images remain,
it submits another Stage-2 array and moves the recovery controller behind that
array master. Workers do not install a walltime signal handler and do not call
`scontrol requeue`.

One unchanged retryable-set round is retried. If a second consecutive round
makes no progress, the remaining images become terminal failures and Stage 3
continues for completed images. All dynamic job IDs are written to the run
ledger, which cancellation uses after atomically deactivating the epoch.

Without `--wait`, the command prints `PROCESSING SUBMITTED` and returns. The
dependent finalizer alone publishes aggregate measurements, QC, analyses,
dashboard, HTML report, README, and the atomic completion marker. With
`--wait`, the CLI monitors that marker; Ctrl+C only detaches monitoring.

**Pre-staging gated weights.** For offline compute nodes, download checkpoints on
the login node first and export `HF_HUB_OFFLINE=1` (and `hf auth login` for gated
Hugging Face models). Gated foundation-model weights are never bundled; accept
their license once via `PHENOTYPIC_ACCEPT_MODEL_LICENSE=<model>` (see the
`require_license_acceptance` hook).

**Custom detectors on SLURM.** Each SLURM stage worker is a fresh process that
deserializes the pipeline with `ImagePipeline.from_json`, which resolves operation
classes from the `phenotypic` namespace. If your pipeline uses a detector defined
**outside** the package, set
`PHENOTYPIC_PRELOAD_MODULES=your.module[,another.module]` — the worker imports
each before deserializing, so a self-registering module can make its class
resolvable on the compute node. `sbatch --export=ALL` (the default) propagates the
variable.

## Device Selection

Both detectors accept a `device` parameter that controls where inference runs.

### Automatic detection (default)

With `device="auto"` (the default), PhenoTypic probes accelerators in priority
order:

1. **CUDA** -- NVIDIA GPUs
2. **MPS** -- Apple Silicon (macOS)
3. **XPU** -- Intel GPUs
4. **HPU** -- Habana Gaudi accelerators

If none is found, a `RuntimeError` is raised.

### Explicit device

```python
# Force a specific device
Sam2Detector(device="cuda")   # NVIDIA GPU
Sam2Detector(device="mps")    # Apple Silicon
Sam2Detector(device="xpu")    # Intel GPU
Sam2Detector(device="cpu")    # CPU (very slow, but always available)
```

When an explicit accelerator is requested but unavailable, a `RuntimeError`
is raised with a descriptive message.

### `resolve_device()` utility

The device resolution logic is available as a standalone function for custom
workflows:

```python
from phenotypic.detect.nn._checkpoint_manager import resolve_device

device = resolve_device("auto")           # raises if no accelerator
device = resolve_device("auto", allow_cpu=True)  # falls back to CPU with warning
```

## Listing and Clearing Models

### List cached checkpoints

```bash
python -m phenotypic.detect.nn list
```

This prints a table showing all cached SAM2 and micro-sam checkpoints with
their file sizes and paths.

### Clear cached checkpoints

```bash
# Clear all cached checkpoints (prompts for confirmation)
python -m phenotypic.detect.nn clear

# Clear only SAM2 checkpoints
python -m phenotypic.detect.nn clear --model-type sam2

# Clear only micro-sam checkpoints
python -m phenotypic.detect.nn clear --model-type microsam
```

## Troubleshooting

### `ImportError: Sam2Detector requires the sam2 package`

PyTorch and the model packages are not installed. Install the `torch` extra:

```bash
uv add "phenotypic[torch]"
```

(Linux/macOS only — `sam2` is not packaged for Windows.)

### `ImportError: MicroSamDetector requires the micro_sam package`

`micro_sam` is conda-only and must be installed separately. See the
{ref}`Deep Learning Detectors`
tutorial section for the conda-forge / `pixi` recipe.

### `RuntimeError: No accelerator available`

No GPU was detected. Options:

- Ensure your GPU drivers and CUDA toolkit are installed correctly.
- On macOS with Apple Silicon, ensure PyTorch >= 2.0 with MPS support.
- Pass `device="cpu"` to force CPU inference (very slow):

```python
Sam2Detector(device="cpu")
```

### `RuntimeError: device='cuda' requested but CUDA is not available`

CUDA was explicitly requested but is not available. Check:

- `nvidia-smi` shows your GPU.
- PyTorch was installed with CUDA support (`torch.cuda.is_available()`
  returns `True`).
- On SLURM, the job was submitted to a GPU partition.

### Out of memory (OOM) errors

SAM models require significant GPU memory. To reduce VRAM or unified-memory usage
without lowering segmentation resolution:

- Lower `points_per_batch` first (for example, 4 instead of the default 8). This
  keeps the same prompt grid, crop pyramid, thresholds, and mask geometry.
- Both `input_scaling="dtype_range"` and `input_scaling="image_max"` use the
  same bounded row-chunk conversion, so this setting is not an OOM control.
  Keep the `"image_max"` default for the validated compatibility policy; use
  `"dtype_range"` only when absolute sensor-scale intensity is required.
- Use a smaller model: `Sam2Detector(model_size="tiny")` instead of `"large"`.
- Use `MicroSamDetector(model_type="vit_t_lm")` for the smallest micro-sam
  model.
- If memory is still insufficient, reduce `points_per_side` (for example, 16
  instead of 32), then reduce `crop_n_layers` or downscale the image. These later
  options reduce prompt density or effective segmentation resolution and can
  affect accuracy.

### Checkpoint not found on SLURM compute nodes

Compute nodes often lack internet access. Pre-download checkpoints on the
login node:

```bash
python -m phenotypic.detect.nn download --model-type sam2 --model-size tiny
python -m phenotypic.detect.nn download --model-type microsam --model-name vit_b_lm
python -m phenotypic.detect.nn list  # verify
```

Ensure `TORCH_HOME` and `MICROSAM_CACHEDIR` (if customised) point to a
shared filesystem accessible from compute nodes.

### `Illegal instruction (core dumped)` on a SLURM compute node

A stage worker exits with code 132 and the SLURM `.err` log shows `Illegal
instruction (core dumped)`. The node's CPU is too old for the installed
numpy/scipy/torch wheels (a pre-AVX node on a heterogeneous partition). This
affects **any** PhenoTypic SLURM run, not just the staged GPU engine. Pin jobs to
modern nodes — use a homogeneous modern partition, or add a SLURM feature
constraint that excludes the old CPUs, e.g.:

```bash
python -m phenotypic --pipeline p.json --input /plates/ -o /out/ \
    --slurm slurm_partition=<modern-partition> \
    --gpu-slurm slurm_partition=<gpu-partition>
```

(Stage 2's GPU work runs on GPU nodes, which are typically consistent; the CPU
Stages 1 & 3 are the ones exposed to a heterogeneous CPU partition.)
