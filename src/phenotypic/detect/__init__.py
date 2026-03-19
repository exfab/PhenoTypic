"""Colony/object detectors for agar plate images.

Implements thresholding- and edge-based approaches to turn detection matrix images
into binary colony masks, with options suited to faint growth, uneven agar, or dense plates.
Includes global histogram methods (Otsu, Li, Yen, Isodata, Triangle, Mean, Minimum, Manual),
edge-aware variants (Canny), grid-aware detection (Gitter), and watershed-based
segmentation for clustered colonies.
"""

from ._canny_detector import CannyDetector
from ._chan_vese_detector import ChanVeseDetector
from ._hysteresis_detector import HysteresisDetector
from ._mad_hysteresis_detector import MadHysteresisDetector
from ._round_peaks_detector import RoundPeaksDetector
from ._isodata_detector import IsodataDetector
from ._li_detector import LiDetector
from ._manual_detector import ManualDetector
from ._manual_grid_detector import ManualGridDetector
from ._mean_detector import MeanDetector
from ._minimum_detector import MinimumDetector
from ._otsu_detector import OtsuDetector
from ._triangle_detector import TriangleDetector
from ._watershed_detector import WatershedDetector
from ._yen_detector import YenDetector
from ._rank_otsu import RankOtsuDetector
from ._secondary_otsu import SecondaryOtsuDetector
from ._sine_peak_detector import SinePeakDetector
from ._composite_detector import CompositeDetector
from ._filamentous_fungi_detector import FilamentousFungiDetector
from ._inoculum_detector import InoculumDetector

__all__ = [
    "CannyDetector",
    "ChanVeseDetector",
    "CompositeDetector",
    "FilamentousFungiDetector",
    "HysteresisDetector",
    "InoculumDetector",
    "MadHysteresisDetector",
    "IsodataDetector",
    "LiDetector",
    "ManualDetector",
    "ManualGridDetector",
    "MeanDetector",
    "MinimumDetector",
    "OtsuDetector",
    "RankOtsuDetector",
    "RoundPeaksDetector",
    "SecondaryOtsuDetector",
    "SinePeakDetector",
    "TriangleDetector",
    "WatershedDetector",
    "YenDetector",
]
