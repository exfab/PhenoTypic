# Sweep Module

## Overview

Two-step workflow: **generate a manifest** (cartesian product of pipeline configs),
then **deploy it**.

## Step 1: Generate a Sweep Manifest

```python
from phenotypic.sweep import Sweep, Fixed, Presence, generate_sweep_manifest
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize

manifest = generate_sweep_manifest(
    [
        Presence(GaussianBlur, sigma=(1.0, 1.5, 2.0), truncate=4.0),
        Sweep(OtsuDetector, ignore_zeros=(True, False)),
    ],
    meas=[MeasureSize()],
    filepath="my_sweep.json",
    desc="Blur + Otsu exploration",
)
```

- **Tuples** are swept; scalars/lists are fixed. Use `Fixed(value)` for literal tuples.
- `Presence(Op, ...)` like `Sweep` but adds one combo where op is omitted.
- Dict configs: `{"Config_A": [...], "Config_B": [...]}` for named groups.
- `load_sweep_manifest("my_sweep.json")` reconstructs pipelines from saved manifest.

## Step 2: Deploy

```bash
pixi run python -m phenotypic.sweep my_sweep.json ./images \
    --image-type GridImage --nrows 8 --ncols 12 --n-jobs -1

pixi run python -m phenotypic.sweep my_sweep.json ./images \
    --slurm partition=gpu --slurm time=120 --wait

pixi run python -m phenotypic.sweep my_sweep.json ./images --dry-run
```

### CLI Options

| Flag | Purpose |
|------|---------|
| `-o, --output-dir` | Output directory (auto-timestamped if omitted) |
| `--image-type` | `Image` or `GridImage` (default: `GridImage`) |
| `--nrows / --ncols` | Grid dimensions |
| `--detect-mode` | Detection channel (default: `gray`) |
| `--n-jobs` | Parallel jobs, `-1` = all cores |
| `--slurm KEY=VALUE` | SLURM parameters (repeatable) |
| `--force-local` | Force local even with SLURM args |
| `--dry-run` | Preview without executing |
| `--skip-validation` | Skip first-image validation |

### Output Structure

Each image saved as HDF5 + CSV, organized **image-first**:

```
output_dir/
├── results/
│   └── <image_stem>/
│       └── <pipeline_name>/
│           ├── <image_stem>.h5
│           └── <image_stem>.csv
├── logs/ (+ failures/, slurm/)
├── sweep_manifest.json
└── sweep_progress.html
```

Manifest copied into output directory for reproducibility.
