# GPU-Accelerated Colony Detection

Set up and use deep-learning-based colony detectors (SAM2, micro-sam) with
GPU acceleration.

## Installation

The two GPU detectors have different packaging constraints:

| Detector            | Package(s) needed          | Available via              | CUDA-capable?          |
|---------------------|----------------------------|----------------------------|------------------------|
| `Sam2Detector`      | `torch`, `torchvision`, `sam2` | **PyPI** (ships in `phenotypic[torch]`) | Yes — Linux + CUDA |
| `MicroSamDetector`  | `micro_sam` (+ `torch`)    | **conda-forge only**, not on PyPI | CPU by default; user-managed CUDA possible |

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
