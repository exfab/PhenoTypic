"""
This module provides tools for modifying objects in image processing
tasks, including removing objects with low circularity, small objects, objects
at the borders, and reducing the number of objects in a location such as a grid
"""

from ._circularity_modifier import LowCircularityRemover
from ._small_object_modifier import SmallObjectRemover
from ._border_object_modifier import BorderObjectRemover
from ._center_deviation_reducer import CenterDeviationReducer

__all__ = [
    "LowCircularityRemover",
    "SmallObjectRemover",
    "BorderObjectRemover",
    "CenterDeviationReducer",
]