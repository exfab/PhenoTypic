"""Image enhancers to boost fungal colonies on agar backgrounds.

Preprocessing steps that denoise, normalize, and emphasize colony structure before
detection. The module covers local contrast equalization (CLAHE), Gaussian/median/rank
denoising, rolling-ball and Gaussian background subtraction, tophat and Laplacian edge
accentuation, Sobel gradients, contrast stretching, and BM3D denoising for clean plates.
All operate on copies of the grayscale view to keep raw data intact.
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
