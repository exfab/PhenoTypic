"""Abstract interfaces for fungal colony image operations.

Defines the base contracts that power the processing pipeline: enhancers, detectors,
refiners, grid operations, and measurement classes. Implement these to add new steps
tailored to agar plate imaging, building on `MeasurementInfo`, `MeasureFeatures`,
`ImageOperation`, `GridOperation`, and the prefab pipeline foundation.
"""

from phenotypic._shared_modules._measurement_info import MeasurementInfo
from ._measure_features import MeasureFeatures
from ._image_operation import ImageOperation
from ._image_enhancer import ImageEnhancer
from ._image_corrector import ImageCorrector
from ._object_detector import ObjectDetector
from ._object_refiner import ObjectRefiner
from ._threshold_detector import ThresholdDetector
from ._grid_operation import GridOperation
from ._grid_corrector import GridCorrector
from ._grid_map_modifier import GridRefiner
from ._grid_measure import GridMeasureFeatures
from ._grid_finder import GridFinder
from ._base_operation import BaseOperation
from ._grid_object_detector import GridObjectDetector
from ._prefab_pipeline import PrefabPipeline

__all__ = [
    "MeasureFeatures",
    "ImageOperation",
    "ImageEnhancer",
    "ImageCorrector",
    "ObjectDetector",
    "ObjectRefiner",
    "ThresholdDetector",
    "GridOperation",
    "GridFinder",
    "GridCorrector",
    "GridRefiner",
    "GridMeasureFeatures",
    'BaseOperation',
    "MeasurementInfo",
    "GridObjectDetector",
    "PrefabPipeline",
]
