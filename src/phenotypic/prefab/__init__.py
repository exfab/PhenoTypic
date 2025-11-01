"""
This module contains premade, tested prefab that we use in ExFAB workflows
"""

from ._adv_watershed_pipeline import AdvWatershedPipeline
from ._adv_otsu_pipeline import AdvOtsuPipeline

__all__ = [
    "AdvWatershedPipeline",
    "AdvOtsuPipeline",
]
