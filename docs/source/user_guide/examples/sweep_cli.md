# Sweep CLI Tutorial

Batch parameter sweeps from the command line.

The Sweep CLI (`python -m phenotypic.sweep`) processes a flat directory of images through
every pipeline configuration defined in a sweep manifest. It parallelizes pipelines per
image using joblib locally, or distributes images across a SLURM cluster as array tasks.

## Prerequisites

Install PhenoTypic with the default extras:

```bash
pip install phenotypic
```

Or with uv:

```bash
uv pip install phenotypic
```

## Quick Start

```bash
# 1. Generate a sweep manifest (Python)
python -c "
from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize, MeasureMorphology

config = [
    Sweep(GaussianBlur, sigma=(1.0, 1.5, 2.0)),
    Sweep(CLAHE, clip_limit=(1.5, 2.5)),
    Sweep(OtsuDetector),
]
meas = [MeasureSize(), MeasureMorphology()]

generate_sweep_manifest(config, meas=meas, filepath='my_sweep.json')
"

# 2. Run the sweep
python -m phenotypic.sweep my_sweep.json ./plate_images/ -o ./sweep_results/
```

## Workflow Overview

The Sweep CLI follows a three-stage workflow:

1. **Define** sweep configurations in Python and save as a manifest JSON
2. **Execute** the sweep via the CLI on a flat directory of images
3. **Analyze** the aggregated `master_measurements.csv` with pandas or similar tools

```
┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐
│  Python Script   │     │     Sweep CLI       │     │     Analysis         │
│                  │     │                     │     │                      │
│  Sweep configs   │────▶│  my_sweep.json      │────▶│  master_measurements │
│  + measurements  │     │  + ./plate_images/  │     │  .csv                │
│  → manifest.json │     │  → ./sweep_results/ │     │                      │
└──────────────────┘     └────────────────────┘     └──────────────────────┘
```

## Creating a Sweep Manifest

A sweep manifest defines all pipeline configurations to test. Use the `Sweep` class to
declare which parameters to vary for each operation.

### Basic Manifest

```python
from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector

# Sweep sigma across 3 values; OtsuDetector uses defaults
config = [
    Sweep(GaussianBlur, sigma=(1.0, 1.5, 2.0)),
    Sweep(OtsuDetector),
]

manifest = generate_sweep_manifest(config, filepath="sweep.json")
print(f"Total pipelines: {manifest['total_pipelines']}")  # 3
```

### Multi-Parameter Sweeps

When multiple parameters are swept, all combinations (cartesian product) are generated:

```python
from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector

config = [
    Sweep(GaussianBlur, sigma=(1.0, 2.0)),      # 2 values
    Sweep(CLAHE, clip_limit=(1.5, 2.5, 3.5)),   # 3 values
    Sweep(OtsuDetector, ignore_zeros=(True, False)),  # 2 values
]

manifest = generate_sweep_manifest(config, filepath="sweep.json")
print(f"Total pipelines: {manifest['total_pipelines']}")  # 2 x 3 x 2 = 12
```

### Named Configurations

Group different pipeline architectures under named configurations:

```python
from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur, CLAHE, GrayOpening
from phenotypic.detect import OtsuDetector, RankOtsuDetector
from phenotypic.measure import MeasureSize

configs = {
    "blur_otsu": [
        Sweep(GaussianBlur, sigma=(1.0, 2.0)),
        Sweep(OtsuDetector),
    ],
    "clahe_rank": [
        Sweep(CLAHE, clip_limit=(1.5, 2.5)),
        Sweep(GrayOpening, radius=(2, 3)),
        Sweep(RankOtsuDetector),
    ],
}

meas = [MeasureSize()]
manifest = generate_sweep_manifest(configs, meas=meas, filepath="sweep.json")
```

### Fixed Parameters

Use `Fixed` to pin a tuple value (so it isn't interpreted as sweep values):

```python
from phenotypic.sweep import Sweep, Fixed
from phenotypic.enhance import GaussianBlur

# Without Fixed: sigma=(1.0, 2.0) means sweep over 1.0 and 2.0
# With Fixed: the tuple IS the parameter value
config = [
    Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=Fixed((4.0, 4.0))),
]
```

## Running the Sweep CLI

### Basic Usage

```bash
python -m phenotypic.sweep MANIFEST_JSON INPUT_DIR [OPTIONS]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `MANIFEST_JSON` | Path to the sweep manifest JSON file |
| `INPUT_DIR` | Flat directory containing images (no subdirectories with images) |

### Common Options

```bash
# Specify output directory (auto-timestamped if omitted)
python -m phenotypic.sweep sweep.json ./images/ -o ./results/

# Set image type and grid dimensions
python -m phenotypic.sweep sweep.json ./images/ \
    --image-type GridImage --nrows 8 --ncols 12

# Control parallelism (pipelines per image)
python -m phenotypic.sweep sweep.json ./images/ --n-jobs 4

# Preview without processing
python -m phenotypic.sweep sweep.json ./images/ --dry-run

# Skip validation (useful if validation is slow on large images)
python -m phenotypic.sweep sweep.json ./images/ --skip-validation
```

### Saving Additional Layers

By default the CLI saves measurements (CSV) and overlays (PNG) for each pipeline.
Enable additional output layers with flags:

```bash
python -m phenotypic.sweep sweep.json ./images/ \
    --save-rgb \
    --save-gray \
    --save-detect-mat \
    --save-objmask \
    --save-objmap \
    --save-objmap-overlay \
    --save-detect-mat-overlay \
    --save-objmask-overlay \
    --overlay-mode image \
    --overlay-alpha 0.3
```

### Full Options Reference

```bash
python -m phenotypic.sweep --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | Auto-timestamped | Output directory |
| `--image-type` | `GridImage` | `Image` or `GridImage` |
| `--nrows` | 8 | Grid rows (GridImage only) |
| `--ncols` | 12 | Grid columns (GridImage only) |
| `--bit-depth` | Auto | Bit depth (8 or 16) |
| `--detect-mode` | `gray` | Detection channel |
| `--n-jobs` | -1 (all cores) | Parallel pipeline jobs per image |
| `--dry-run` | Off | Preview configuration without processing |
| `--skip-validation` | Off | Skip first-pipeline validation step |
| `--force-local` | Off | Force local execution even with SLURM args |
| `--overlay-mode` | `image` | `image` (full-res) or `figure` (matplotlib) |
| `--overlay-alpha` | 0.3 | Overlay transparency (0.0-1.0) |

## Output Structure

Results are organized by pipeline name:

```
sweep_results/
├── sweep_manifest.json          # Copy of input manifest
├── master_measurements.csv      # All pipelines combined
├── results/
│   ├── Pipeline_0/
│   │   ├── measurements/
│   │   │   ├── plate_001.csv
│   │   │   └── plate_002.csv
│   │   ├── overlays/
│   │   │   ├── plate_001.png
│   │   │   └── plate_002.png
│   │   └── pipeline_measurements.csv   # All images for this pipeline
│   ├── Pipeline_1/
│   │   └── ...
│   └── Pipeline_N/
│       └── ...
└── logs/
```

### Master CSV

The `master_measurements.csv` file combines measurements from all pipelines with a
`Metadata_Pipeline` column to identify which pipeline produced each row:

```python
import pandas as pd

df = pd.read_csv("sweep_results/master_measurements.csv")

# Compare colony counts across pipelines
summary = df.groupby("Metadata_Pipeline").agg(
    n_objects=("Metadata_ObjectID", "count"),
    mean_area=("Size_Area", "mean"),
).reset_index()
print(summary)
```

## SLURM Cluster Execution

For large sweeps on HPC clusters, pass SLURM parameters to distribute images as array
tasks. Each array task processes one image through all pipelines sequentially.

```bash
# Submit as SLURM array job
python -m phenotypic.sweep sweep.json ./images/ -o ./results/ \
    --slurm time=60 \
    --slurm mem=16G \
    --slurm partition=batch \
    --slurm cpus-per-task=4

# Submit and monitor progress
python -m phenotypic.sweep sweep.json ./images/ -o ./results/ \
    --slurm time=60 \
    --slurm mem=16G \
    --wait

# Force local execution even with SLURM args defined
python -m phenotypic.sweep sweep.json ./images/ -o ./results/ \
    --slurm time=60 \
    --force-local
```

### Parallelism Model

| Execution Mode | Image Loop | Pipeline Loop |
|----------------|------------|---------------|
| **Local** | Sequential (one at a time) | Parallel (joblib, `--n-jobs`) |
| **SLURM** | Parallel (one array task per image) | Sequential (within each task) |

## Complete Example

End-to-end workflow for optimizing colony detection on 96-well plate images:

```python
"""Step 1: Generate sweep manifest."""
from phenotypic.sweep import Sweep, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur, CLAHE, GrayOpening
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize, MeasureMorphology

config = [
    Sweep(GaussianBlur, sigma=(0.5, 1.0, 1.5, 2.0)),
    Sweep(CLAHE, clip_limit=(1.5, 2.0, 2.5)),
    Sweep(GrayOpening, radius=(2, 3)),
    Sweep(OtsuDetector),
]

meas = [MeasureSize(), MeasureMorphology()]

manifest = generate_sweep_manifest(
    config,
    meas=meas,
    filepath="colony_sweep.json",
    desc="Optimize preprocessing for yeast colony detection",
)
print(f"Generated {manifest['total_pipelines']} pipeline configurations")
```

```bash
# Step 2: Run the sweep
python -m phenotypic.sweep colony_sweep.json ./raw_plates/ \
    -o ./colony_optimization/ \
    --image-type GridImage \
    --nrows 8 --ncols 12 \
    --n-jobs -1
```

```python
"""Step 3: Analyze results."""
import pandas as pd

df = pd.read_csv("colony_optimization/master_measurements.csv")

# Compare detection quality across pipelines
summary = df.groupby("Metadata_Pipeline").agg(
    total_colonies=("Metadata_ObjectID", "count"),
    mean_area=("Size_Area", "mean"),
    mean_circularity=("Morphology_Circularity", "mean"),
).reset_index()

# Find pipelines with the most detected colonies
best = summary.sort_values("total_colonies", ascending=False).head(5)
print("Top 5 pipeline configurations:")
print(best.to_string(index=False))
```

## Input Requirements

The Sweep CLI expects a **flat directory** of images with no subdirectories containing
image files. This keeps the output structure simple -- one organizational dimension
(pipeline name) in the results.

```
# Good: flat directory
plate_images/
├── plate_001.tif
├── plate_002.tif
└── plate_003.tif

# Bad: nested directories (will be rejected)
plate_images/
├── experiment_1/
│   └── plate_001.tif    # Error: images in subdirectory
└── experiment_2/
    └── plate_002.tif
```

```{tip}
For experiments with multiple datasets or nested directory structures, use the main
PhenoTypic CLI (``python -m phenotypic``) which supports hierarchical input directories.
```

## Related

- {py:class}`phenotypic.sweep.Sweep` -- Declare sweep parameters for an operation
- {py:func}`phenotypic.sweep.generate_sweep_manifest` -- Build the manifest JSON
- {py:func}`phenotypic.sweep.load_sweep_manifest` -- Reload pipelines from a manifest
- {py:class}`phenotypic.ImagePipeline` -- The pipeline class used internally
