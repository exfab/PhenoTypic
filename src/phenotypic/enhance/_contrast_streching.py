from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import rescale_intensity

from ..abc_ import ImageEnhancer


class ContrastStretching(ImageEnhancer):
    """Stretch the intensity range of detect_mat to fill the full dynamic range.

    Rescales pixel values based on lower and upper percentiles, compressing
    outliers (specular highlights, deep shadows) while expanding the range
    where colony intensities reside. Simpler and faster than EnhanceLocalContrast, with no
    local tile artifacts.

    Args:
        lower_percentile: Dark clipping point. Pixels below this percentile
            are mapped to the minimum. Typical range: 1--5. Default: 2.
        upper_percentile: Bright clipping point. Pixels above this percentile
            are mapped to the maximum. Typical range: 95--99. Default: 98.

    Returns:
        Image: Input image with ``detect_mat`` rescaled to the full dynamic
        range. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Plates with narrow histograms (under-exposed or low-contrast).
        - Normalizing exposure across different imaging sessions.
        - Quick preprocessing before global thresholding (Otsu, Triangle).

    Consider Also:
        - :class:`EnhanceLocalContrast` when illumination varies spatially across the plate.
        - :class:`FlattenIllumination` when the primary issue is a brightness
          gradient rather than narrow dynamic range.

    See Also:
        :doc:`/how_to/notebooks/enhance_low_contrast` for a comparison of
        contrast enhancement methods.
        :doc:`/explanation/what_enhancement_does` for how enhancement fits
        into the pipeline model.
    """

    lower_percentile: int = 2
    upper_percentile: int = 98

    def _operate(self, image: Image) -> Image:
        p_lower, p_upper = np.percentile(
                image.detect_mat[:], (self.lower_percentile, self.upper_percentile)
        )
        image.detect_mat[:] = rescale_intensity(
                image=image.detect_mat[:],
                in_range=(p_lower, p_upper),
                out_range=(0, 1),
        )
        return image
