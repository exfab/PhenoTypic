<div style="background-color: white; display: inline-block; padding: 10px; border-radius: 0px;">
  <img src="./docs/source/_static/assets/400x150/gradient_logo_exfab.svg" alt="Phenotypic Logo" style="width: 400px; height: auto;">
</div>

# PhenoTypic: A Python Framework for Bio-Image Analysis

![Development Status](https://img.shields.io/badge/dev_status-beta-orange)

A modular image processing framework developed at the NSF Ex-FAB BioFoundry, focused on
arrayed colony phenotyping on solid media.

---

### Links:

[![docs](https://img.shields.io/badge/Documentation-purple?style=for-the-badge)](https://exfab.github.io/PhenoTypic/)

[![exfab](https://img.shields.io/badge/ExFAB_NSF_BioFoundry-blue?style=for-the-badge)](https://exfab.engineering.ucsb.edu/)

## Overview

PhenoTypic provides a modular toolkit designed to simplify and accelerate the development of reusable bio-image analysis
pipelines. PhenoTypic provides bio-image analysis tools built-in, but has a streamlined development method
to integrate new tools.

# Installation

## uv (recommended)

To download the base package (recommended if running on a cluster)

```bash
uv add phenotypic
```

To download the base package plus prototyping environment (recommended for pipeline development)

```bash
uv add phenotypic --groups jupyter
```

## Pip

```
pip install phenotypic
```

Note: may not always be the latest version. Install from repo when latest update is needed

## Manual Installation (For latest updates)

```  
git clone https://github.com/exfab/PhenoTypic.git
cd PhenoTypic
uv pip install -e .
```  

## Dev Installation

```  
git clone https://github.com/exfab/PhenoTypic.git
cd PhenoTypic
uv sync --group dev
```  

## Optional Installation

To extract metadata from raw images, PhenoTypic uses the `PyExifTool` module. This requires an external software called
ExifTool. You can install ExifTool here: https://exiftool.org/install.html. If you don't use it, some metadata from raw
files may not be able to be imported. Read more here: https://pypi.org/project/PyExifTool/#pyexiftool-dependencies

# Module Overview

| Module                  | Description                                                                                                                |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `phenotypic.analysis`   | Tools for downstream analysis of the data from phenotypic in various ways such as growth modeling or statistical filtering |
| `phenotypic.correction` | Different methods to improve the data quality of an image such as rotation to improve grid finding                         |
| `phenotypic.data`       | Sample images to experiment your workflow with                                                                             |
| `phenotypic.detect`     | A suite of operations to automatically detect objects in your images                                                       |
| `phenotypic.enhance`    | Preprocessing tools that alter a copy of your image and can improve the results of the detection algorithms                |
| `phenotypic.grid`       | Modules that rely on grid and object information to function                                                               |
| `phenotypic.measure`    | The various measurements PhenoTypic is capable of extracting from objects                                                  |
| `phenotypic.refine`     | Different tools to edit the detected objects such as morphology, relabeling, joining, or removing                          |
| `phenotypic.prefab`     | Various premade image processing pipelines that are in use at ExFAB                                                        |

# Sponsors

<div style="background-color: white; display: inline-block; padding: 10px; border-radius: 5px;">
  <img src="./docs/source/_static/assets/ExFabLogo.svg" alt="Phenotypic Logo" style="width: 400px; height: auto;">
</div>
