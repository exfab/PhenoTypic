"""
A library for processing and analyzing images of microbe colonies on solid media agar.

This module provides tools_ and classes for the manipulation, analysis, and
enhancement of images, specifically tailored for biological applications,
including detecting features of colonies, quantifying growth, and refining image
qualities. Classes such as `Image` and `GridImage` enable flexibility in managing
varied image formats, while the `ImagePipeline` class provides a structured
workflow for image processing. Additionally, submodules offer utilities for
analysis, grid alignment, detection of colonies, enhancement of image clarity,
and correction of artifacts in captured images. This module is designed
primarily for researchers working with images acquired from solid media plates
to study microbial growth patterns.

"""

__version__ = "0.13.0"
__author__ = "Alexander Nguyen"
__email__ = "anguy344@ucr.edu"

from ._core._grid_image import GridImage
from ._core._image import Image
from ._core._image_pipeline import ImagePipeline

# commented out until complete
# from ._core._image_set import ImageSet

from . import (
    abc_,
    analysis,
    correction,
    data,
    detect,
    enhance,
    grid,
    measure,
    refine,
    settings_,
    sweep,
    tools_,
    prefab,
)

__all__ = [
    "Image",  # Class imported from _core
    "GridImage",  # Class imported from _core
    "ImagePipeline",
    # "ImageSet",
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
    "settings_",
    "sweep",
]
