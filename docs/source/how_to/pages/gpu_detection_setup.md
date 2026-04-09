# GPU-Accelerated Colony Detection

Set up and use deep-learning-based colony detectors (SAM2, micro-sam) with
GPU acceleration.

## Installation

GPU detectors require PyTorch and model-specific packages. Install the optional
`torch` extra:

```bash
# Using pixi (recommended)
pixi install -e full
```

This installs PyTorch, torchvision, and the `sam2` package.

**micro-sam** has C++ dependencies not available on PyPI and must be installed
separately via conda:

```bash
conda install -c conda-forge micro_sam
```

`MicroSamDetector` handles the missing import gracefully at runtime — you only
need micro-sam installed if you plan to use it.

## Downloading Model Checkpoints

Both SAM2 and micro-sam download checkpoints automatically on first use.
However, on SLURM clusters the compute nodes often lack internet access, so
you should pre-download checkpoints on a login node before submitting jobs.

### SAM2 checkpoints

```bash
# Download the default (tiny) SAM2 checkpoint
python -m phenotypic.nn download

# Download a specific size
python -m phenotypic.nn download --model-type sam2 --model-size large

# Download all SAM2 sizes at once
python -m phenotypic.nn download --model-type sam2 --all

# Force re-download even if cached
python -m phenotypic.nn download --model-type sam2 --model-size tiny --force
```

SAM2 checkpoints are stored in the `torch.hub` cache directory
(`~/.cache/torch/hub/checkpoints/` by default). Set the `TORCH_HOME`
environment variable to change this location.

Available SAM2 sizes: `tiny` (~39 MB), `small`, `base_plus`, `large` (~900 MB).

### micro-sam checkpoints

```bash
# Download the default (vit_b_lm) micro-sam model
python -m phenotypic.nn download --model-type microsam

# Download a specific model
python -m phenotypic.nn download --model-type microsam --model-name vit_l_lm

# Download all micro-sam models
python -m phenotypic.nn download --model-type microsam --all
```

micro-sam stores checkpoints via `platformdirs`. Set `MICROSAM_CACHEDIR` to
override the cache location.

### SLURM pre-caching workflow

On a cluster, download models on the login node first:

```bash
# On the login node (has internet access)
python -m phenotypic.nn download --model-type sam2 --model-size tiny
python -m phenotypic.nn download --model-type microsam --model-name vit_b_lm

# Verify they are cached
python -m phenotypic.nn list

# Now submit SLURM jobs -- compute nodes will use the cached checkpoints
python -m phenotypic pipeline.json /plates/ /output/
```

## Using Sam2Detector

`Sam2Detector` wraps Meta's SAM2 automatic mask generator. It lays a grid of
prompt points over the RGB image, predicts masks at each point, filters by
quality, and assembles a labelled object map.

```python
from phenotypic.nn import Sam2Detector

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
from phenotypic.nn import MicroSamDetector

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
from phenotypic.nn import Sam2Detector
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

## SLURM Deployment

When a pipeline contains a `GpuDetector` operation (either `Sam2Detector` or
`MicroSamDetector`), the CLI automatically adapts:

**Local execution:** Forces sequential processing (`n_jobs=1`) to avoid
multiple workers competing for the same GPU.

**SLURM execution:** Automatically adds `--gpus-per-node=1` to the SLURM
job if GPU resources were not explicitly requested.

```bash
# GPU resources are auto-requested when the pipeline contains a GpuDetector
python -m phenotypic sam2_pipeline.json /plates/ /output/

# Override with explicit SLURM GPU arguments
python -m phenotypic sam2_pipeline.json /plates/ /output/ \
    --slurm-args slurm_gpus_per_node=2 \
    --slurm-args slurm_partition=gpu
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
from phenotypic.nn._checkpoint_manager import resolve_device

device = resolve_device("auto")           # raises if no accelerator
device = resolve_device("auto", allow_cpu=True)  # falls back to CPU with warning
```

## Listing and Clearing Models

### List cached checkpoints

```bash
python -m phenotypic.nn list
```

This prints a table showing all cached SAM2 and micro-sam checkpoints with
their file sizes and paths.

### Clear cached checkpoints

```bash
# Clear all cached checkpoints (prompts for confirmation)
python -m phenotypic.nn clear

# Clear only SAM2 checkpoints
python -m phenotypic.nn clear --model-type sam2

# Clear only micro-sam checkpoints
python -m phenotypic.nn clear --model-type microsam
```

## Troubleshooting

### `ImportError: Sam2Detector requires the sam2 package`

PyTorch and the model packages are not installed. Install the optional extra:

```bash
pixi install -e full
```

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
python -m phenotypic.nn download --model-type sam2 --model-size tiny
python -m phenotypic.nn download --model-type microsam --model-name vit_b_lm
python -m phenotypic.nn list  # verify
```

Ensure `TORCH_HOME` and `MICROSAM_CACHEDIR` (if customised) point to a
shared filesystem accessible from compute nodes.
