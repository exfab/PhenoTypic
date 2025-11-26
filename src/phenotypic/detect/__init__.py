"""Colony/object detectors for agar plate images.

Implements thresholding- and edge-based approaches to turn enhanced grayscale images
into binary colony masks, with options suited to faint growth, uneven agar, or dense plates.
Includes global histogram methods (Otsu, Li, Yen, Isodata, Triangle, Mean, Minimum),
edge-aware variants (Canny), grid-aware detection (Gitter), and watershed-based
segmentation for clustered colonies.
"""

from ._canny_detector import CannyDetector
from ._gitter_detector import GitterDetector
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
