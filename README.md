 <div style="background-color: white; display: inline-block; padding: 10px; border-radius: 5px;">
  <img src="./docs/source/_static/assets/200x150/light_logo_sponsor.png" alt="Phenotypic Logo" style="width: 200px; height: auto;">
</div>

# PhenoTypic: A Python Framework for Bio-Image Analysis

![Development Status](https://img.shields.io/badge/status-beta-orange)

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

```
uv add phenotypic
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
| `phenotypic.objedit`    | Different tools to edit the detected objects such as morphology, relabeling, joining, or removing                          |
| `phenotypic.prefab`     | Various premade image processing pipelines that are in use at ExFAB                                                        |

# Sponsors

## ExFAB NSF BioFoundry

## National Science Foundation
