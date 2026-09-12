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

__version__ = "0.19.0"
__author__ = "Alexander Nguyen"
__email__ = "anguy344@ucr.edu"

# Import first: stamps the import-start time as ``_IMPORT_STARTED_AT`` and installs a
# lazy stub for colour-science's eager-but-unused ``colour.plotting`` submodule before
# anything can import colour. Both happen as import side effects of ``_startup_perf``.
from ._startup_perf import IMPORT_STARTED_AT as _IMPORT_STARTED_AT  # noqa: F401

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Public classes, by the private module that defines each one.
_LAZY_CLASSES: dict[str, str] = {
    "Image": "._core._image",
    "GridImage": "._core._grid_image",
    "ImagePipeline": "._core._image_pipeline",
}

#: Public subpackages. They resolve on first access, so ``import phenotypic`` -- and
#: every console script, which imports this package first -- loads none of them.
_LAZY_SUBPACKAGES: frozenset[str] = frozenset(
    {
        "abc_", "analysis", "correction", "data", "detect", "enhance", "grid", "measure",
        # ``plotting`` is not in ``__all__`` but is public in the docs
        # (``phenotypic.plotting.PlotDiagnostics``), so it resolves here too.
        "plotting",
        "prefab", "refine", "schema", "sdk_", "settings", "tune", "util",
    }
)


def __getattr__(name: str) -> _Any:
    """Import a public class or subpackage on first access and cache it on the package.

    Unknown names raise :class:`AttributeError`;
    ``SerializablePipeline._find_class_in_phenotypic`` relies on that to fall through
    to the subpackages when resolving an operation class by name.
    """
    if name in _LAZY_CLASSES:
        value = getattr(_importlib.import_module(_LAZY_CLASSES[name], __name__), name)
    elif name in _LAZY_SUBPACKAGES:
        value = _importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from . import (
        abc_,
        analysis,
        correction,
        data,
        detect,
        enhance,
        grid,
        measure,
        prefab,
        refine,
        schema,
        sdk_,
        settings,
        tune,
        util,
    )
    from ._core._grid_image import GridImage
    from ._core._image import Image
    from ._core._image_pipeline import ImagePipeline

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
    "sdk_",
    "util",
    "settings",
    "tune",
]
