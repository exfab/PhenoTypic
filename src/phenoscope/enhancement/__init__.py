"""
The enhancement module is designed to enhance the image's enhancement matrix (which is a copy of the image matrix)
in order to improve detection & segmentation results.
"""

from ._clahe import CLAHE
from ._gaussian_preprocessor import GaussianPreprocessor
from ._median_preprocessor import MedianPreprocessor
from ._rank_median_preprocessor import RankMedianPreprocessor
from ._rolling_ball_preprocessor import RollingBallPreprocessor
from ._white_tophat_preprocessor import WhiteTophatPreprocessor
from ._laplace_preprocessor import LaplacePreprocessor
from ._contrast_streching import ContrastStretching

__all__ = [
    "CLAHE",
    "GaussianPreprocessor",
    "MedianPreprocessor",
    "RankMedianPreprocessor",
    "RollingBallPreprocessor",
    "WhiteTophatPreprocessor",
    "LaplacePreprocessor",
]