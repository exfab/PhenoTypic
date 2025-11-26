"""
The enhance module provides a collection of image enhance operations designed to improve
detection and segmentation results by modifying the image's enhance gray.

Available enhancers:
    - CLAHE: Contrast Limited Adaptive Histogram Equalization for local contrast enhance
    - GaussianBlur: Applies Gaussian blur to reduce noise while preserving edges
    - MedianFilter: Uses median filtering for noise reduction
    - RankMedianEnhancer: Applies rank-based median filtering for enhanced noise removal
    - RollingBallEnhancer: Implements rolling ball algorithm for background subtraction
    - GaussianSubtract: Uses Gaussian blur for background estimation and subtraction
    - WhiteTophatEnhancer: Performs white tophat transformation for feature extraction
    - LaplaceEnhancer: Applies Laplacian operator for edge detection
    - ContrastStretching: Enhances image contrast through intensity stretching
    - BM3DDenoiser: Block-matching and 3D filtering for advanced noise removal

Each enhancer operates on a copy of the original image gray to preserve the source data
while allowing for multiple enhance operations to be applied sequentially.
"""

from ._clahe import CLAHE
from ._gaussian_blur import GaussianBlur
from ._median_filter import MedianFilter
from ._rank_median_enhancer import RankMedianEnhancer
from ._rolling_ball_remove_bg import RollingBallRemoveBG
from ._gaussian_subtract import GaussianSubtract
from ._white_tophat_enhancer import WhiteTophatEnhancer
from ._laplace_enhancer import LaplaceEnhancer
from ._contrast_streching import ContrastStretching
from ._sobel_filter import SobelFilter
from ._bm3d_denoiser import BM3DDenoiser

__all__ = [
    "CLAHE",
    "GaussianBlur",
    "MedianFilter",
    "RankMedianEnhancer",
    "RollingBallRemoveBG",
    "GaussianSubtract",
    "WhiteTophatEnhancer",
    "LaplaceEnhancer",
    "ContrastStretching",
    "SobelFilter",
    "BM3DDenoiser",
]
