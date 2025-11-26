"""
The detection module contains classes for detecting objects in images.

"""

from ._canny_detector import CannyDetector
from ._gitter_detection import GitterDetector
from ._isodata_detector import IsodataDetector
from ._li_detector import LiDetector
from ._mean_detector import MeanDetector
from ._minimum_detector import MinimumDetector
from ._otsu_detector import OtsuDetector
from ._triangle_detector import TriangleDetector
from ._watershed_detector import WatershedDetector
from ._yen_detector import YenDetector

__all__ = [
    "CannyDetector",
    "GitterDetector",
    "IsodataDetector",
    "LiDetector",
    "MeanDetector",
    "MinimumDetector",
    "OtsuDetector",
    "TriangleDetector",
    "WatershedDetector",
    "YenDetector",
]
