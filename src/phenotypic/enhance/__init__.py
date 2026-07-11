"""Image enhancers to boost fungal colonies on agar backgrounds.

Preprocessing steps that denoise, normalize, and emphasize colony structure before
detection. The module covers local contrast equalization (EnhanceLocalContrast), Gaussian/median/rank
denoising, rolling-ball and Gaussian background subtraction, tophat and Laplacian edge
accentuation, Sobel gradients, contrast stretching, unsharp masking, bilateral denoising,
BM3D denoising, Hessian-based ridge detection (Frangi vesselness, Sato tubeness, Meijering
neuriteness, Hessian filter) for filamentous structure detection, morphological operations
(opening, closing, erosion, dilation, gradient, black tophat) for noise removal and boundary
enhancement, and more for clean plates. All operate on copies of the grayscale view to keep
raw data intact.
"""

from ._local_edge_denoise import LocalEdgeDenoise
from ._bayesshrink_enhancer import BayesShrinkEnhancer
from ._enhance_block_match import EnhanceBlockMatch
from ._enhance_local_contrast import EnhanceLocalContrast
from ._structure_smoothing import StructureSmoothing
from ._contrast_streching import ContrastStretching
from ._focus_edge_frangi import FocusEdgeFrangi
from ._gaussian_blur import GaussianBlur
from ._focus_edge_hessian import FocusEdgeHessian
from ._subtract_gaussian import SubtractGaussian
from ._image_inverter import ImageInverter
from ._focus_edge_laplace import FocusEdgeLaplace
from ._median_filter import MedianFilter
from ._focus_edge_phase import FocusEdgePhase
from ._focus_edge_monogenic_phase import FocusEdgeMonogenicPhase
from ._focus_edge_color_phase import FocusEdgeColorPhase
from ._focus_edge_meijering import FocusEdgeMeijering
from ._non_local_means import NonLocalMeansDenoiser
from ._subtract_opening import SubtractOpening
from ._rank_median_enhancer import RankMedianEnhancer
from ._subtract_rolling_ball import SubtractRollingBall
from ._focus_edge_sato import FocusEdgeSato
from ._focus_edge_sobel import FocusEdgeSobel
from ._sharpen_edge_gauss import SharpenEdgeGauss
from ._visushrink_enhancer import VisuShrinkEnhancer
from ._subtract_white_tophat import SubtractWhiteTophat
from ._white_tophat_enhance import WhiteTophatEnhance
from ._gray_opening import GrayOpening
from ._flatten_illumination import FlattenIllumination
from ._focus_blob_log import FocusBlobLoG
from ._set_detect_mode import SetDetectMode
from ._composite_enhance import CompositeEnhance

__all__ = [
    "BayesShrinkEnhancer",
    "LocalEdgeDenoise",
    "EnhanceBlockMatch",
    "EnhanceLocalContrast",
    "StructureSmoothing",
    "ContrastStretching",
    "FocusEdgeFrangi",
    "GaussianBlur",
    "SubtractGaussian",
    "GrayOpening",
    "FocusEdgeHessian",
    "FlattenIllumination",
    "ImageInverter",
    "FocusEdgeLaplace",
    "MedianFilter",
    "FocusEdgeMeijering",
    "FocusBlobLoG",
    "NonLocalMeansDenoiser",
    "SubtractOpening",
    "FocusEdgePhase",
    "FocusEdgeMonogenicPhase",
    "FocusEdgeColorPhase",
    "RankMedianEnhancer",
    "SubtractRollingBall",
    "FocusEdgeSato",
    "FocusEdgeSobel",
    "SharpenEdgeGauss",
    "VisuShrinkEnhancer",
    "WhiteTophatEnhance",
    "SubtractWhiteTophat",
    "SetDetectMode",
    "CompositeEnhance",
]
