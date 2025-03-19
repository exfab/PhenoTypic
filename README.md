<div style="background-color: white; display: inline-block; padding: 10px; border-radius: 5px;">
  <img src="docs/assets/logo_background_svg/PhenoScopeLogo.svg" alt="PhenoScope Logo" style="width: 200px; height: auto;">
</div>

# PhenoScope: A Python Framework for Bio-Image Analysis
![Development Status](https://img.shields.io/badge/status-Pre--Alpha-red)

A modular image processing framework developed at the NSF Ex-FAB BioFoundry.

---

## Overview
PhenoScope provides a modular toolkit designed to simplify and accelerate the development of bio-image analysis pipelines. 
Its structured architecture allows researchers and developers to seamlessly integrate custom computer vision modules, avoiding 
the need to build entirely new packages from scratch. Additionally, PhenoScope supports incorporating components from 
other existing image analysis tools into a cohesive, unified ecosystem.


## Installation

### Pip
```
pip install phenoscope
```

### Manual Installation
```  
git clone https://github.com/Xander-git/PhenoScope.git
cd PhenoScope
pip install -e .
```  

## Dev Installation
```  
git clone https://github.com/Xander-git/PhenoScope.git
cd PhenoScope
pip install -e ".[dev]"
```  

## Acknowledgements

### CellProfiler
PhenoScope has drawn inspiration and foundational concepts from [CellProfiler](https://github.com/CellProfiler/CellProfiler), 
an open-source software platform for bio-image analysis developed by the Broad Institute Imaging Platform. CellProfiler's modularity, 
pipeline-oriented design, and extensive documentation have significantly influenced the development approach and structure of PhenoScope.

**Reference:**
- [CellProfiler GitHub Repository](https://github.com/CellProfiler/CellProfiler)
- [CellProfiler Official Website](https://cellprofiler.org/)