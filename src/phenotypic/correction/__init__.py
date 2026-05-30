"""Image/grid correction for agar plate captures.

Offers operations that realign grids or correct field-of-view drift so detected colonies
stay anchored to their intended wells or pins. The grid aligner adjusts spacing and
offsets using reference points or heuristics suited to arrayed plate layouts. Also includes
wavelet denoising correctors for full-image noise removal across all components (RGB, gray,
detect_mat).
"""

from ._bayesshrink_corrector import BayesShrinkCorrector
from ._color_correction import CaptureMetadata, ColorCheckerProfile, ColorCorrector
from ._color_denoise import ColorDenoise
from ._stable_denoise import StableDenoise
from ._grid_aligner import GridAligner
from ._visushrink_corrector import VisuShrinkCorrector
from ._image_cropper import ImageCropper
from ._image_padder import ImagePadder

__all__ = [
    "CaptureMetadata",
    "ColorCheckerProfile",
    "ColorCorrector",
    "ColorDenoise",
    "StableDenoise",
    "GridAligner",
    "ImageCropper",
    "ImagePadder",
    "BayesShrinkCorrector",
    "VisuShrinkCorrector",
]
