"""
The enhance module provides a collection of image enhance operations designed to improve
detection and segmentation results by modifying the image's enhance gray.

Available enhancers:
    - CLAHE: Contrast Limited Adaptive Histogram Equalization for local contrast enhance
    - GaussianBlur: Applies Gaussian blur to reduce noise while preserving edges
    - MedianEnhancer: Uses median filtering for noise reduction
    - RankMedianEnhancer: Applies rank-based median filtering for enhanced noise removal
    - RollingBallEnhancer: Implements rolling ball algorithm for background subtraction
    - GaussianRemoveBG: Uses Gaussian blur for background estimation and subtraction
    - WhiteTophatEnhancer: Performs white tophat transformation for feature extraction
    - LaplaceEnhancer: Applies Laplacian operator for edge detection
    - ContrastStretching: Enhances image contrast through intensity stretching

Each enhancer operates on a copy of the original image gray to preserve the source data
while allowing for multiple enhance operations to be applied sequentially.
"""

from ._clahe import CLAHE
from ._gaussian_preprocessor import GaussianBlur
from ._median_enhancer import MedianEnhancer
from ._rank_median_preprocessor import RankMedianEnhancer
from ._rolling_ball_remove_bg import RollingBallRemoveBG
from ._gaussian_remove_bg import GaussianRemoveBG
from ._white_tophat_preprocessor import WhiteTophatEnhancer
from ._laplace_preprocessor import LaplaceEnhancer
from ._contrast_streching import ContrastStretching
from ._sobel_filter import SobelFilter

__all__ = [
    "CLAHE",
    "GaussianBlur",
    "MedianEnhancer",
    "RankMedianEnhancer",
    "RollingBallRemoveBG",
    "GaussianRemoveBG",
    "WhiteTophatEnhancer",
    "LaplaceEnhancer",
    "ContrastStretching",
    "SobelFilter",
]
