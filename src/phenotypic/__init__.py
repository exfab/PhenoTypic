"""
A library for processing and analyzing images of microbe colonies on solid media agar.

This module provides tools and classes for the manipulation, analysis, and
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

__version__ = "0.17.0b2"
__author__ = "Alexander Nguyen"
__email__ = "anguy344@ucr.edu"

# Import first (before the heavy submodule chain below): stamps the import-start
# time as ``_IMPORT_STARTED_AT`` so launchers can report core-load duration, and
# installs a lazy stub for colour-science's eager-but-unused ``colour.plotting``
# submodule. Both happen as import side effects of ``_startup_perf``.
from ._startup_perf import IMPORT_STARTED_AT as _IMPORT_STARTED_AT  # noqa: F401

# Import abc_ first: its __init__ imports the public ``schema`` package
# (the stdlib-only MeasurementInfo base + leaf enums, cached instantly) then
# triggers tools_ init.  When tools_.constants_ later does
# ``from phenotypic.schema import MeasurementInfo`` the module is already in
# sys.modules, breaking the circular import chain.
from . import abc_  # noqa: F401

from ._core._grid_image import GridImage
from ._core._image import Image
from ._core._image_pipeline import ImagePipeline

from . import (
    analysis,
    correction,
    data,
    detect,
    enhance,
    grid,
    measure,
    refine,
    schema,
    settings_,
    tools_,
    tune,
    util,
    prefab,
)

__all__ = [
    "Image",  # Class imported from _core
    "GridImage",  # Class imported from _core
    "ImagePipeline",
    "abc_",
    "analysis",
    "data",
    "detect",
    "measure",
    "grid",
    "refine",
    "schema",
    "prefab",
    "correction",
    "enhance",
    "tools_",
    "util",
    "settings_",
    "tune",
]
