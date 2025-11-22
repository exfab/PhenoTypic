"""
This module provides tools for modifying objects in image processing
tasks, including removing objects with low circularity, small objects, objects
at the borders, and reducing the number of objects in a location such as a grid
"""

from ._circularity_modifier import LowCircularityRemover
from ._small_object_modifier import SmallObjectRemover
from ._border_object_modifier import BorderObjectRemover
from ._center_deviation_reducer import CenterDeviationReducer
from ._mask_fill import MaskFill
from ._mask_opener import MaskOpener
from ._white_tophat_modifier import WhiteTophatModifier
from ._border_object_modifier import BorderObjectRemover
from ._grid_oversized_object_remover import GridOversizedObjectRemover
from ._min_residual_error_reducer import MinResidualErrorReducer
from ._residual_outlier_remover import ResidualOutlierRemover

__all__ = [
    "LowCircularityRemover",
    "SmallObjectRemover",
    "BorderObjectRemover",
    "CenterDeviationReducer",
    "MaskFill",
    "MaskOpener",
    "WhiteTophatModifier",
    "BorderObjectRemover",
    "GridOversizedObjectRemover",
    "MinResidualErrorReducer",
    "ResidualOutlierRemover",
]
