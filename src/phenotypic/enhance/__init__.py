"""Image enhancers to boost fungal colonies on agar backgrounds.

Preprocessing steps that denoise, normalize, and emphasize colony structure before
detection. The module covers local contrast equalization (CLAHE), Gaussian/median/rank
denoising, rolling-ball and Gaussian background subtraction, tophat and Laplacian edge
accentuation, Sobel gradients, contrast stretching, unsharp masking, bilateral denoising,
and BM3D denoising for clean plates. All operate on copies of the grayscale view to keep
raw data intact.
"""

from ._bilateral_denoise import BilateralDenoise
from ._bm3d_denoiser import BM3DDenoiser
from ._clahe import CLAHE
from ._contrast_streching import ContrastStretching
from ._gaussian_blur import GaussianBlur
from ._gaussian_subtract import GaussianSubtract
from ._laplace_enhancer import LaplaceEnhancer
from ._median_filter import MedianFilter
from ._rank_median_enhancer import RankMedianEnhancer
from ._rolling_ball_remove_bg import RollingBallRemoveBG
from ._sobel_filter import SobelFilter
from ._unsharp_mask import UnsharpMask
from ._white_tophat_enhancer import WhiteTophatEnhancer

__all__ = [
    "BilateralDenoise",
    "BM3DDenoiser",
    "CLAHE",
    "ContrastStretching",
    "GaussianBlur",
    "GaussianSubtract",
    "LaplaceEnhancer",
    "MedianFilter",
    "RankMedianEnhancer",
    "RollingBallRemoveBG",
    "SobelFilter",
    "UnsharpMask",
    "WhiteTophatEnhancer",
]
