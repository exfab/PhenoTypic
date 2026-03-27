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
    where colony intensities reside. Simpler and faster than CLAHE, with no
    local tile artifacts.

    Best For:
        - Plates with narrow histograms (under-exposed or low-contrast).
        - Normalizing exposure across different imaging sessions.
        - Quick preprocessing before global thresholding (Otsu, Triangle).

    Consider Also:
        - :class:`CLAHE` when illumination varies spatially across the plate.
        - :class:`HomomorphicFilter` when the primary issue is a brightness
          gradient rather than narrow dynamic range.

    Caveats:
        Heavy percentile clipping can reduce the apparent intensity of
        biologically bright colonies and bias downstream measurements.
    - Contrast stretching is global; it will not fix spatially varying illumination
      on its own (consider `SubtractGaussian` or `SubtractRollingBall`).

    Parameters:
        lower_percentile (int): Lower percentile used to define the input range
            for rescaling. Pixels below this are mapped to the minimum.
        upper_percentile (int): Upper percentile used to define the input range
            for rescaling. Pixels above this are mapped to the maximum.
    """

    def __init__(self, lower_percentile: int = 2, upper_percentile: int = 98):
        """
        Parameters:
            lower_percentile (int): Dark clipping point. Increase to suppress
                deep shadows/edge artifacts; too high may remove meaningful dark
                background structure. Typical range: 1–5.
            upper_percentile (int): Bright clipping point. Decrease to suppress
                glare/highlights; too low may flatten bright colonies. Typical
                range: 95–99.
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

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
