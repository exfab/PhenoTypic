"""
The enhancement module provides a collection of image enhancement operations designed to improve
detection and segmentation results by modifying the image's enhancement matrix.

Available enhancers:
    - CLAHE: Contrast Limited Adaptive Histogram Equalization for local contrast enhancement
    - GaussianSmoother: Applies Gaussian blur to reduce noise while preserving edges
    - MedianEnhancer: Uses median filtering for noise reduction
    - RankMedianEnhancer: Applies rank-based median filtering for enhanced noise removal
    - RollingBallEnhancer: Implements rolling ball algorithm for background subtraction
    - WhiteTophatEnhancer: Performs white tophat transformation for feature extraction
    - LaplaceEnhancer: Applies Laplacian operator for edge detection
    - ContrastStretching: Enhances image contrast through intensity stretching

Each enhancer operates on a copy of the original image matrix to preserve the source data
while allowing for multiple enhancement operations to be applied sequentially.
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