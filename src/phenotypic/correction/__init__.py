"""Image/grid correction for agar plate captures.

Offers operations that realign grids or correct field-of-view drift so detected colonies
stay anchored to their intended wells or pins. The grid aligner adjusts spacing and
offsets using reference points or heuristics suited to arrayed plate layouts. Also includes
wavelet denoising correctors for full-image noise removal across all components (RGB, gray,
detect_mat).
"""

import warnings

from ._bayesshrink_corrector import BayesShrinkCorrector
from ._color_correction import (
    CaptureMetadata,
    ColorCheckerProfile,
    ColorCorrector,
)
from ._color_denoise import ColorDenoise
from ._denoise_block_match import DenoiseBlockMatch
from ._grid_aligner import GridAligner
from ._visushrink_corrector import VisuShrinkCorrector
from ._image_cropper import CropImage
from ._image_padder import PadImage

__all__ = [
    "CaptureMetadata",
    "ColorCheckerProfile",
    "ColorCorrector",
    "ColorDenoise",
    "DenoiseBlockMatch",
    "GridAligner",
    "CropImage",
    "PadImage",
    "BayesShrinkCorrector",
    "VisuShrinkCorrector",
]

_LEGACY_OPERATION_NAMES = {
    "ImageCropper": CropImage,
    "ImagePadder": PadImage,
}


def __getattr__(name: str):
    """Resolve deprecated correction operation names without GUI duplicates."""
    operation = _LEGACY_OPERATION_NAMES.get(name)
    if operation is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{name} is deprecated; use {operation.__name__} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return operation
