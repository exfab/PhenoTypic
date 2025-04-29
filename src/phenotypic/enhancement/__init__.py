"""
The enhancement module is designed to enhance the image's enhancement matrix (which is a copy of the image matrix)
in order to improve detection & segmentation results.
"""

from ._clahe import CLAHE
from ._gaussian_preprocessor import GaussianSmoother
from ._median_preprocessor import MedianEnhancer
from ._rank_median_preprocessor import RankMedianEnhancer
from ._rolling_ball_preprocessor import RollingBallEnhancer
from ._white_tophat_preprocessor import WhiteTophatEnhancer
from ._laplace_preprocessor import LaplaceEnhancer
from ._contrast_streching import ContrastStretching

__all__ = [
    "CLAHE",
    "GaussianSmoother",
    "MedianEnhancer",
    "RankMedianEnhancer",
    "RollingBallEnhancer",
    "WhiteTophatEnhancer",
    "LaplaceEnhancer",
    "ContrastStretching"
]