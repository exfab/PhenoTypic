"""Phenotypic toolkit for plate-based fungal colony imaging and analysis.

This package bundles image I/O, enhancement, detection, refinement, and measurement
utilities tuned for agar plate assays. Use it to load multi-channel captures,
align plates to grids, extract per-colony masks, and quantify morphology and growth.
Core interfaces are provided via `Image`, `GridImage`, `ImagePipeline`, `ImageSet`,
and `imread`, while subpackages cover enhancement, detection, refinement, measurement,
analytics, prefab pipelines, and supporting utilities for agar-plate workflows.
"""

__version__ = "0.10.0"
__author__ = "Alexander Nguyen"
__email__ = "anguy344@ucr.edu"

from .core._grid_image import GridImage
from .core._image import Image
from .core._image_pipeline import ImagePipeline
from .core._image_set import ImageSet
from .core._imread import imread

from . import (abc_,
               analysis,
               correction,
               data,
               detect,
               enhance,
               grid,
               measure,
               refine,
               tools,
               prefab)

__all__ = [
    "Image",  # Class imported from core
    "imread",  # Function imported from core
    "GridImage",  # Class imported from core
    "ImagePipeline",
    "ImageSet",
    "abc_",
    "analysis",
    "data",
    "detect",
    "measure",
    "grid",
    "refine",
    "prefab",
    "correction",
    "enhance",
    "tools",
]
