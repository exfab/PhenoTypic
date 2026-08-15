<div style="background-color: white; display: inline-block; padding: 10px; border-radius: 0px;">
  <img src="./src/phenotypic/_assets/logos/400x150/light_logo_exfab.svg" alt="Phenotypic Logo" style="width: 400px; height: auto;">
</div>

# PhenoTypic: A Python Framework for Bio-Image Analysis

![Development Stage](https://img.shields.io/badge/dev_stage-beta-orange)
[![GUI checks](https://github.com/exfab/PhenoTypic/actions/workflows/gui-checks.yml/badge.svg)](https://github.com/exfab/PhenoTypic/actions/workflows/gui-checks.yml)
[![Docs](https://github.com/exfab/PhenoTypic/actions/workflows/docs.yml/badge.svg)](https://github.com/exfab/PhenoTypic/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/phenotypic)](https://pypi.org/project/phenotypic/)
[![Python](https://img.shields.io/pypi/pyversions/phenotypic)](https://pypi.org/project/phenotypic/)
[![License](https://img.shields.io/github/license/exfab/PhenoTypic)](https://github.com/exfab/PhenoTypic/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/exfab/PhenoTypic)](https://github.com/exfab/PhenoTypic/releases)
[![Pull requests](https://img.shields.io/github/issues-pr/exfab/PhenoTypic)](https://github.com/exfab/PhenoTypic/pulls)

A modular image processing framework developed at the NSF Ex-FAB BioFoundry, focused on
arrayed colony phenotyping on solid media.

---

### Links:

[![docs](https://img.shields.io/badge/Documentation-purple?style=for-the-badge)](https://exfab.github.io/PhenoTypic/)

[![exfab](https://img.shields.io/badge/ExFAB_NSF_BioFoundry-blue?style=for-the-badge)](https://exfab.engineering.ucsb.edu/)

## Overview

PhenoTypic provides a modular toolkit designed to simplify and accelerate the
development of reusable bio-image analysis
pipelines. PhenoTypic provides bio-image analysis tools built-in, but has a streamlined
development method
to integrate new tools.

# Installation

## uv (recommended)

**See more** on
<u>[installing uv](https://docs.astral.sh/uv/getting-started/installation/)</u>

**Regular Install** (recommended when deploying on a cluster)

```bash
uv add phenotypic
```

**Interactive / GUI Install** (Plotly dashboards, Jupyter, Dash hub)

```bash
uv add phenotypic --extra gui
```

**napari Desktop Viewer Install** (for `image.*.napari()`, point picker, sweep viewer)

```bash
uv add phenotypic --extra napari
```

## Pip

**Regular Install**

```bash
pip install phenotypic
```

**Interactive / GUI Install**

```bash
pip install "phenotypic[gui]"
```

**napari Desktop Viewer Install**

```bash
pip install "phenotypic[napari]"
```

Note: may not always be the latest version. Install from repo when latest update is
needed

## Manual Installation (For latest updates)

```
git clone https://github.com/exfab/PhenoTypic.git
cd PhenoTypic
uv sync
```

## Dev Installation

For extending PhenoTypic.

```
git clone https://github.com/exfab/PhenoTypic.git
cd PhenoTypic
uv sync --group dev --all-extras
```

## CPU compatibility (polars build)

PhenoTypic installs the **`polars-lts-cpu`** build by default so it runs on
older CPUs (e.g. pre-AVX2 HPCC nodes) without crashing. On AVX2-capable machines
you can swap in the faster stock `polars` build for quicker measurement
compilation — see
[Choosing the polars CPU build](https://exfab.github.io/PhenoTypic/experimental/how_to/pages/polars_cpu_build.html).

## GPU-Accelerated Detection (SAM2, micro-sam)

PhenoTypic ships optional deep-learning detectors backed by Meta's
[Segment Anything Model 2](https://github.com/facebookresearch/sam2) and
[micro-sam](https://github.com/computational-cell-analytics/micro-sam).

See [GPU Detection Setup](https://exfab.github.io/PhenoTypic/experimental/how_to/pages/gpu_detection_setup.html)
for model downloads and SLURM deployment instructions.

## Optional third-party tools

### libvips (faster GUI deep-zoom preparation)

On macOS and Windows, the `[gui]` extra installs the official
[`pyvips[binary]`](https://pypi.org/project/pyvips/) distribution, including a
self-contained native libvips. No Homebrew install or loader-path configuration is
needed for the normal GUI installation.

Linux and HPC installations keep the smaller `pyvips` binding and may provide native
[libvips](https://www.libvips.org/install.html) through the operating system or an
environment module:

```bash
# Debian or Ubuntu
sudo apt install libvips-dev --no-install-recommends
```

Verify that Python can load the native library from the same shell that will launch
the GUI:

```bash
uv run python -c "import pyvips; print('pyvips', pyvips.__version__, 'libvips', '.'.join(str(pyvips.version(i)) for i in range(3)))"
```

The bundled build contains the common image loaders needed by the GUI but omits optional
facilities such as PDF and OpenSlide. Users who intentionally prefer a fuller Homebrew
libvips can install it with `brew install vips`. If `vips --version` then works but
Python cannot find `libvips.42.dylib`, expose Homebrew's library directory before
launching PhenoTypic:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix vips)/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
```

If native libvips or its loader is unavailable on any platform, PhenoTypic automatically
retains the Pillow fallback. Browse still works, but preparing large pyramids can be
slower and use more memory.

### ExifTool (RAW metadata)

To extract metadata from raw images, PhenoTypic uses the `PyExifTool` module. This
requires the external [ExifTool](https://exiftool.org/install.html) application. If
it is unavailable, some RAW metadata may not be imported. See the
[PyExifTool dependency documentation](https://pypi.org/project/PyExifTool/#pyexiftool-dependencies).

# Run the CLI

Process a directory of plate images through a saved pipeline:

```bash
uv run python -m phenotypic --mode full --pipeline pipeline.json --input ./images --output ./out
```

Use `--mode process --layer {rgb|gray|detect_mat|objmap}` for an apply-only export run that
writes a single image layer per input (mirroring the input tree) and skips the
measurement/analysis suite — handy for previewing detection or enhanced layers.

# Launch the GUI

The unified GUI hub bundles the pipeline builder, results viewer, and run console
under one URL. Two equivalent entry points:

```bash
# Console script (preferred)
uv run phenotypic-gui --root ./images --port 8050

# Module entry (works in environments without the console script on PATH)
uv run python -m phenotypic.gui --root ./images --port 8050
```

`--root` freezes the sandbox the GUI's file browser is allowed to see (defaults to
the current working directory). `--host 127.0.0.1` (the default) keeps the server
loopback-only — pair with SSH port forwarding for remote workstations:

```bash
ssh -L 8050:localhost:8050 user@cluster
```

Open `http://localhost:8050/` in your browser. The
[GUI hub guide](docs/source/how_to/pages/gui_hub.md) walks through the file
browser, builder, run console, and results viewer.

For Open OnDemand-style proxies, pass only the browser-visible path prefix:

```bash
uv run phenotypic-gui --root /rhome/ejaco020 --host 0.0.0.0 --port 30099 --url-prefix /node/hz01/30099/
```

Then open the full proxy URL, for example
`https://ondemand.hpcc.ucr.edu/node/hz01/30099/`.

Note: `phenotypic gui` (no hyphen, as a subcommand) is **not supported**. Use
`phenotypic-gui` or `python -m phenotypic.gui`.

# Hyperparameter Tuning

Search an `ImagePipeline`'s parameters to maximize a scorer with the tuning engine:

```bash
uv run python -m phenotypic.tune run spec.json -i ./plates -o ./out
```

Grid and random search work out of the box; the Optuna samplers
(`tpe`/`cmaes`/`gp`/`nsga2`) need the `tune` extra. See the
[tuning how-to](docs/source/how_to/pages/tuning.md) for an end-to-end walkthrough.

# Module Overview

| Module                  | Description                                                                                                                                                                                                              |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `phenotypic.analysis`   | Tools for downstream analysis of the data from phenotypic in various ways such as growth modeling or statistical filtering                                                                                               |
| `phenotypic.correction` | Different methods to improve the data quality of an image such as rotation to improve grid finding                                                                                                                       |
| `phenotypic.data`       | Sample images to experiment your workflow with                                                                                                                                                                           |
| `phenotypic.detect`     | A suite of operations to automatically detect objects in your images                                                                                                                                                     |
| `phenotypic.enhance`    | Preprocessing tools that alter a copy of your image and can improve the results of the detection algorithms                                                                                                              |
| `phenotypic.grid`       | Modules that rely on grid and object information to function                                                                                                                                                             |
| `phenotypic.measure`    | The various measurements PhenoTypic is capable of extracting from objects                                                                                                                                                |
| `phenotypic.detect.nn`  | GPU-accelerated detectors (SAM2, micro-sam) with checkpoint management — see [setup guide](https://exfab.github.io/PhenoTypic/how_to/pages/gpu_detection_setup.html)                                                     |
| `phenotypic.refine`     | Different tools to edit the detected objects such as morphology, relabeling, joining, or removing                                                                                                                        |
| `phenotypic.prefab`     | Various premade image processing pipelines that are in use at ExFAB                                                                                                                                                      |
| `phenotypic.tune`       | Hyperparameter-tuning engine: grid/random search plus Optuna samplers (behind the `tune` extra), pluggable scorers, robust held-out evaluation, distributed search over HPCC SLURM/Postgres, and a `/tune/` GUI co-pilot |

# Sponsors

<div style="background-color: white; display: inline-block; padding: 10px; border-radius: 5px;">
  <img src="./src/phenotypic/_assets/logos/ExFabLogo.svg" alt="Phenotypic Logo" style="width: 400px; height: auto;">
</div>
