# GPU-Accelerated Colony Detection

Set up and use deep-learning-based colony detectors (SAM2, micro-sam, SAM3,
DinoSam2) with GPU acceleration.

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
themselves; the recipe below uses `pixi` for that.

### Installing `Sam2Detector` (PyPI-only)

On Linux or macOS:

```bash
uv add "phenotypic[torch]"          # torch + torchvision + sam2
# or, inside a uv-managed project:
uv sync --extra torch
```

The `torch` extra is not available on Windows — `sam2` requires CUDA `nvcc`
and has no pre-built Windows wheels. Use WSL2 (Ubuntu) instead.

### Installing the foundation models (`Sam3Detector`, `DinoSam2Detector`)

The foundation detectors collapse to two pure-Python, permissive libraries —
`transformers` (>=5.2.0, the first release shipping `Sam3Model`/`Sam3Processor`)
and `huggingface_hub`. Both ship in the `foundation` extra (which pulls
`torch`); the `gpu` umbrella extra installs every GPU detector at once.

```bash
uv sync --extra foundation           # SAM3 + DINO-based detectors
uv sync --extra gpu                  # umbrella: every GPU detector (dev env)
# as a dependency of a downstream project:
uv add 'phenotypic[foundation]'
```

Installing any extra **never pulls encumbered material** — only the gated
*weights* (SAM3, DINOv3) require a license handshake, and they are fetched at
runtime, never at install time (see *Gated weights* below).

### Enabling `micro_sam` (optional, self-service)

`micro_sam` is only published on conda-forge. Because PhenoTypic does not
own your environment, we recommend managing the combined stack in your own
project with [pixi](https://pixi.sh), which speaks both conda-forge and
PyPI in a single lockfile. Create a `pixi.toml` in *your* project (not in
PhenoTypic):

```toml
[project]
name = "my-phenotyping-project"
channels = ["conda-forge"]
platforms = ["osx-arm64", "linux-64", "win-64"]

[pypi-dependencies]
phenotypic = "*"
# Or, while developing against a local checkout:
# phenotypic = { path = "../PhenoTypic", editable = true }

[dependencies]
micro_sam = "*"
```

Then:

```bash
pixi install
pixi run python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/
```

Because conda's `micro_sam` pulls in CPU-only conda PyTorch, combining it
with `Sam2Detector`'s CUDA wheels in the same environment requires extra
care (the conda torch will typically win). Keep SAM2 and micro-sam work
in separate environments if you need both with GPU acceleration.

`MicroSamDetector` is importable from `phenotypic.detect.nn` even when
`micro_sam` is missing; the `ImportError` is deferred to the first
`apply()` call and points back at these instructions.

### Alternative: pip + conda

If you already manage your environment with conda:

```bash
pip install phenotypic                        # base (or phenotypic[torch] on non-Windows)
conda install -c conda-forge micro_sam        # adds MicroSamDetector support
```

## Downloading Model Checkpoints

Both SAM2 and micro-sam download checkpoints automatically on first use.
However, on SLURM clusters the compute nodes often lack internet access, so
you should pre-download checkpoints on a login node before submitting jobs.

### SAM2 checkpoints

```bash
# Download the default (tiny) SAM2 checkpoint
python -m phenotypic.detect.nn download

# Download a specific size
python -m phenotypic.detect.nn download --model-type sam2 --model-size large

# Download all SAM2 sizes at once
python -m phenotypic.detect.nn download --model-type sam2 --all

# Force re-download even if cached
python -m phenotypic.detect.nn download --model-type sam2 --model-size tiny --force
```

SAM2 checkpoints are stored in the `torch.hub` cache directory
(`~/.cache/torch/hub/checkpoints/` by default). Set the `TORCH_HOME`
environment variable to change this location.

Available SAM2 sizes: `tiny` (~39 MB), `small`, `base_plus`, `large` (~900 MB).

### micro-sam checkpoints

```bash
# Download the default (vit_b_lm) micro-sam model
python -m phenotypic.detect.nn download --model-type microsam

# Download a specific model
python -m phenotypic.detect.nn download --model-type microsam --model-name vit_l_lm

# Download all micro-sam models
python -m phenotypic.detect.nn download --model-type microsam --all
```

micro-sam stores checkpoints via `platformdirs`. Set `MICROSAM_CACHEDIR` to
override the cache location.

### Foundation-model weights (SAM3 + DINOv3 gated, DINOv2 ungated)

SAM3 and the DINO backbones live on the Hugging Face Hub and are downloaded via
`huggingface_hub.snapshot_download` into the HF cache. **SAM3 weights
(~3.45 GB) and DINOv3 weights are gated**; DINOv2 is ungated.

**One-time human handshake for a gated model (per user, per model):**

1. Have a Hugging Face account.
2. Accept the gate on the model page — <https://huggingface.co/facebook/sam3>
   for SAM3, <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
   for DINOv3 (this *is* the license acceptance).
3. Authenticate locally once: `uv run hf auth login` (stores a token), or
   export `HF_TOKEN`.

Then download with the extended `phenotypic.detect.nn` CLI:

```bash
# SAM3 (gated): --accept-license acknowledges the SAM License non-interactively
uv run python -m phenotypic.detect.nn download --model-type sam3 --accept-license

# DINOv3 (gated): --accept-license acknowledges the DINOv3 License
#   (required for Insid3Detector's native backbone + the dino_version=3 opt-in)
uv run python -m phenotypic.detect.nn download --model-type dinov3 --dino-size base --accept-license

# DINOv2 (ungated): no acceptance, no token
uv run python -m phenotypic.detect.nn download --model-type dinov2 --dino-size base
```

If the gate is not accepted or no token is present, the download fails with an
actionable message ("Request access at <url>, then run `uv run hf auth
login`"). The informational acceptance gate is also satisfied non-interactively
by `PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3,dinov3` (a comma list for several
models) — this is the layer the `--accept-license` flag sets for you, on top of
the binding Hugging Face gate.

### Foundation-model environment variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | auth for gated download (alternative to `hf auth login`) |
| `HF_HOME` (or `HF_HUB_CACHE`) | cache location → point at shared HPCC storage |
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | force load-from-cache on offline compute nodes |
| `PHENOTYPIC_ACCEPT_MODEL_LICENSE` | non-interactive acknowledgement for batch/SLURM jobs |

### SLURM pre-caching workflow

On a cluster, download models on the login node first:

```bash
# On the login node (has internet access)
python -m phenotypic.detect.nn download --model-type sam2 --model-size tiny
python -m phenotypic.detect.nn download --model-type microsam --model-name vit_b_lm

# Verify they are cached
python -m phenotypic.detect.nn list

# Now submit SLURM jobs -- compute nodes will use the cached checkpoints
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/
```

**Foundation models on HPCC (offline compute nodes).** Never download inside a
job — accept the gate and pre-stage weights to shared storage on the login node,
then run the compute job offline:

```bash
# On the login node (has internet): cache to shared storage, accept once
export HF_HOME=/bigdata/exfab/<...>/hf_cache
uv run hf auth login
uv run python -m phenotypic.detect.nn download --model-type sam3 --accept-license
uv run python -m phenotypic.detect.nn download --model-type dinov2

# In the SLURM job: same HF_HOME, force offline, carry the accepted acknowledgement
export HF_HOME=/bigdata/exfab/<...>/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/
```

The binding acceptance (HF gate + token) happens **once on the login node**;
the compute job carries only the already-accepted token plus
`PHENOTYPIC_ACCEPT_MODEL_LICENSE` and reads the pre-staged cache. Acceptance
never happens silently inside a batch job.

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
once (see *Foundation-model weights* above) before the first `apply()`.

**Dense plates.** SAM3 caps at 200 instances per forward and runs at 1008 px
internally, so `Sam3Detector` tiles large plates into fixed `tile_px` crops
with `tile_overlap`, infers each tile, offsets the masks back to full
coordinates, and merges cross-tile duplicates by IoU-NMS — all automatically.
Images that fit one tile run un-tiled.

Key parameters:

- `prompt` — free text describing the target (`"colony"`, `"bacterial colony"`).
- `score_thresh` / `mask_threshold` — instance-confidence and mask-probability
  cutoffs.
- `min_mask_region_area` — drop masks smaller than this (default 100).
- `tile_px` / `tile_overlap` — dense-plate tiling controls (defaults 1008 / 0.15).

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
`download --model-type dinov3 --accept-license` command above).

Key parameters:

- `dino_version` / `dino_size` — backbone generation (2/3) and size.
- `sam2_model_size` — SAM2 variant for the proposal generator.
- `similarity_thresh` — minimum cosine-to-prototype score to keep a proposal.
- `merge_iou_thresh` — IoU above which two survivors are merged.

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
- `tile_px` / `tile_overlap` — large-plate tiling (defaults 1024 / 0.15).

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
- `tile_px` / `tile_overlap` — large-plate tiling (defaults 512 / 0.15).

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
the staged engine above, submitted as a **3-link `afterany` dependency chain**
with **per-stage resources**:

- **Stage 1** — a CPU array over images (preprocess → staged HDF), on the
  `--slurm` profile.
- **Stage 2** — a GPU array over **shards** (`--gpu-shards N`): each task is one
  whole GPU running a resident model that streams its shard of HDFs to objmap
  sidecars, on the `--gpu-slurm` profile.
- **Stage 3** — a CPU array over images (merge sidecar → measure), on the
  `--slurm` profile.

`afterany` between stages means a handful of per-image failures never block the
next stage. Because Stage 2 writes a `.npy` sidecar with the HDF opened
read-only, there is **no HDF5 write-locking on the GPU nodes**; Stages 1 & 3
write the HDF atomically (temp + rename) on CPU nodes.

```bash
# CPU partition for Stages 1 & 3 (--slurm); GPU partition + 2 concurrent GPUs for Stage 2
python -m phenotypic --pipeline sam2_pipeline.json --input /plates/ -o /output/ \
    --slurm slurm_partition=batch --slurm slurm_time=02:00:00 \
    --gpu-slurm slurm_partition=gpu --gpu-shards 2
```

The resource nesting is three levels: `--gpu-shards` (whole GPUs, across nodes)
→ `--gpu-workers-per-gpu` (replicas packed per GPU for small models) →
`--gpu-batch-size` (images per forward pass; batchable models, `auto` in
Spec 2). `--gpu-slurm` inherits/deltas over `--slurm`, so shared keys
(account, qos, time) carry over and only the GPU partition/account need
restating; one GPU is requested automatically.

**Walltime survival.** The Stage-2 script carries `#SBATCH --signal=B:TERM@120`,
so SLURM sends `SIGTERM` shortly before the walltime. The shard-worker catches
it and `sbatch`-resubmits its shard (`afterany` on itself); content-defined skip
means the continuation processes only the remaining sidecar-less images, and it
repeats until the shard is complete — so a `TIMEOUT` never loses work and never
needs a manual restart.

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

`micro_sam` is conda-only and must be installed separately. See
[Enabling `micro_sam` (optional, self-service)](#enabling-micro_sam-optional-self-service)
above.

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

SAM models require significant GPU memory. To reduce VRAM usage:

- Use a smaller model: `Sam2Detector(model_size="tiny")` instead of `"large"`.
- Use `MicroSamDetector(model_type="vit_t_lm")` for the smallest micro-sam
  model.
- Reduce `points_per_side` (e.g., 16 instead of 32) to generate fewer
  candidate masks.
- Process smaller images or downscale before detection.

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
