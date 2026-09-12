"""Abstract interfaces for fungal colony image operations.

Defines the base contracts that power the processing pipeline: enhancers, detectors,
refiners, grid operations, and measurement classes. Implement these to add new steps
tailored to agar plate imaging, building on `MeasurementInfo`, `MeasureFeatures`,
`ImageOperation`, `GridOperation`, and the prefab pipeline foundation.
"""

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Re-exports that pull in the image core: ``PrefabPipeline`` imports the pipeline, and
#: ``DetectionMode``/``register_detection_mode`` import ``phenotypic._core``, whose
#: ``__init__`` loads the whole image handler chain. Resolving them on first access keeps
#: every ``phenotypic.abc_.*`` import free of the core. It also breaks the cycle
#: ``analysis.abc_._model_fitter`` -> ``abc_.plotting`` -> this ``__init__`` -> pipeline
#: core -> ``_model_fitter``. Defined before the eager imports below, so an import that
#: re-enters this package mid-initialisation still resolves them.
_LAZY_ATTRS: dict[str, str] = {
    "PrefabPipeline": "phenotypic.abc_._prefab_pipeline",
    "DetectionMode": "phenotypic._core._image_parts.detection_modes",
    "register_detection_mode": "phenotypic._core._image_parts.detection_modes",
}


def __getattr__(name: str) -> _Any:
    """Resolve a core-dependent re-export on first access and cache it on the package."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if _TYPE_CHECKING:
    from phenotypic._core._image_parts.detection_modes import (
        DetectionMode,
        register_detection_mode,
    )

    from ._prefab_pipeline import PrefabPipeline

from phenotypic.schema import MeasurementInfo  # noqa: E402
from phenotypic.sdk_ import FootprintMixin  # noqa: E402
from ._measure_features import MeasureFeatures  # noqa: E402
from ._image_operation import ImageOperation  # noqa: E402
from ._image_enhancer import ImageEnhancer  # noqa: E402
from ._image_denoiser import ImageDenoiser  # noqa: E402
from ._enhance_markers import (  # noqa: E402
    FocusEdge,
    FocusBlob,
    Smoothing,
    BackgroundSubtraction,
    MorphologicalFiltering,
    ContrastAdjustment,
)
from ._image_corrector import ImageCorrector  # noqa: E402
from ._object_detector import ObjectDetector  # noqa: E402
from ._object_refiner import ObjectRefiner  # noqa: E402
from ._threshold_detector import ThresholdDetector  # noqa: E402
from ._gpu_detector import GpuDetector  # noqa: E402
from ._grid_operation import GridOperation  # noqa: E402
from ._grid_corrector import GridCorrector  # noqa: E402
from ._grid_object_refiner import GridObjectRefiner  # noqa: E402
from ._grid_measure import GridMeasureFeatures  # noqa: E402
from ._grid_finder import GridFinder  # noqa: E402
from ._base_operation import BaseOperation  # noqa: E402
from ._grid_object_detector import GridObjectDetector  # noqa: E402
from ._post_measurement import PostMeasurement  # noqa: E402

__all__ = [
    "MeasureFeatures",
    "ImageOperation",
    "ImageEnhancer",
    "ImageDenoiser",
    "FocusEdge",
    "FocusBlob",
    "Smoothing",
    "BackgroundSubtraction",
    "MorphologicalFiltering",
    "ContrastAdjustment",
    "ImageCorrector",
    "ObjectDetector",
    "ObjectRefiner",
    "ThresholdDetector",
    "GpuDetector",
    "GridOperation",
    "GridFinder",
    "GridCorrector",
    "GridObjectRefiner",
    "GridMeasureFeatures",
    "BaseOperation",
    "MeasurementInfo",
    "GridObjectDetector",
    "PrefabPipeline",
    "FootprintMixin",
    "PostMeasurement",
    "DetectionMode",
    "register_detection_mode",
]
