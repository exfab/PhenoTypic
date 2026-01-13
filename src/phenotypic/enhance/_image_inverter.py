from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np

from ..abc_ import ImageEnhancer


class ImageInverter(ImageEnhancer):
    """
    Invert the enhanced grayscale image (negate pixel intensities).

    Reverses the brightness scale so dark regions become bright and vice versa.
    This can be useful for correcting inverted input images or for exploring
    alternative visual representations before detection.

    Use cases (agar plates):
    - Correct for inverted scan output from some imaging systems.
    - Invert low-contrast plates where colony boundaries are defined by dark
      edges on a bright background (converted to bright edges on dark background).
    - Preprocessing step before detectors that expect colonies as bright regions
      when source images have colonies as dark regions.

    Tuning and effects:
    - This operation has no parameters; it performs a simple brightness reversal.
    - For uint8 images: values are inverted as `255 - pixel`.
    - For floating-point images: values are inverted as `1.0 - pixel` (assumes
      normalized [0, 1] range) or as `max_value - pixel` if range differs.

    Caveats:
    - Inversion can amplify noise in low-signal regions.
    - May not improve detection if the fundamental issue is poor contrast rather
      than inverted polarity.
    - Verify that inversion is actually needed before applying; use visual inspection
      or test detection performance on a sample image.

    Attributes:
        None. ImageInverter has no configurable parameters.
    """

    def __init__(self):
        """Initialize the ImageInverter with no parameters."""
        pass

    @staticmethod
    def _operate(image: Image) -> Image:
        enh = image.enh_gray[:]

        # Invert based on data type
        if enh.dtype == np.uint8:
            # For 8-bit integer: 255 - value
            inverted = 255 - enh
        else:
            # For floating-point: find max value and invert
            max_val = enh.max()
            inverted = max_val - enh

        image.enh_gray[:] = inverted
        return image
