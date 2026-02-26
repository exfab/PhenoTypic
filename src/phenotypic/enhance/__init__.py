"""Image enhancers to boost fungal colonies on agar backgrounds.

Preprocessing steps that denoise, normalize, and emphasize colony structure before
detection. The module covers local contrast equalization (CLAHE), Gaussian/median/rank
denoising, rolling-ball and Gaussian background subtraction, tophat and Laplacian edge
accentuation, Sobel gradients, contrast stretching, unsharp masking, bilateral denoising,
BM3D denoising, Hessian-based ridge detection (Frangi vesselness, Sato tubeness, Meijering
neuriteness, Hessian filter) for filamentous structure detection, morphological operations
(opening, closing, erosion, dilation, gradient, black tophat) for noise removal and boundary
enhancement, and more for clean plates. All operate on copies of the grayscale view to keep
raw data intact.
"""

from ._anscombe_forward import AnscombeForward
from ._anscombe_inverse import AnscombeInverse
from ._bilateral_denoise import BilateralDenoise
from ._bayesshrink_enhancer import BayesShrinkEnhancer
from ._bm3d_denoiser import BM3DDenoiser
from ._clahe import CLAHE
from ._coherence_enhancing_diffusion import CoherenceEnhancingDiffusion
from ._contrast_streching import ContrastStretching
from ._frangi_vesselness import FrangiVesselness
from ._gaussian_blur import GaussianBlur
from ._hessian_filter import HessianFilter
from ._gaussian_subtract import GaussianSubtract
from ._image_inverter import ImageInverter
from ._laplace_enhancer import LaplaceEnhancer
from ._median_filter import MedianFilter
from ._phase_congruency import PhaseCongruencyEnhancer
from ._meijering_ridge_filter import MeijeringRidgeFilter
from ._non_local_means import NonLocalMeansDenoiser
from ._opening_subtract_bg import OpeningSubtractBg
from ._rank_median_enhancer import RankMedianEnhancer
from ._rolling_ball_remove_bg import RollingBallRemoveBG
from ._sato_ridge_filter import SatoRidgeFilter
from ._sobel_filter import SobelFilter
from ._unsharp_mask import UnsharpMask
from ._visushrink_enhancer import VisuShrinkEnhancer
from ._white_tophat_subtract import WhiteTophatSubtract
from ._white_tophat_enhance import WhiteTophatEnhance
from ._gray_opening import GrayOpening
from ._set_detect_mode import SetDetectMode

__all__ = [
    "AnscombeForward",
    "AnscombeInverse",
    "BayesShrinkEnhancer",
    "BilateralDenoise",
    "BM3DDenoiser",
    "CLAHE",
    "CoherenceEnhancingDiffusion",
    "ContrastStretching",
    "FrangiVesselness",
    "GaussianBlur",
    "GaussianSubtract",
    "GrayOpening",
    "HessianFilter",
    "ImageInverter",
    "LaplaceEnhancer",
    "MedianFilter",
    "MeijeringRidgeFilter",
    "NonLocalMeansDenoiser",
    "OpeningSubtractBg",
    "PhaseCongruencyEnhancer",
    "RankMedianEnhancer",
    "RollingBallRemoveBG",
    "SatoRidgeFilter",
    "SobelFilter",
    "UnsharpMask",
    "VisuShrinkEnhancer",
    "WhiteTophatEnhance",
    "WhiteTophatSubtract",
    "SetDetectMode",
]
