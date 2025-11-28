from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.exposure import rescale_intensity

from ..abc_ import ImageEnhancer


class ContrastStretching(ImageEnhancer):
    """
    Contrast stretching for plate images.

    Rescales intensities based on chosen lower/upper percentiles so the bulk of
    pixel values expands to the full display range. For agar plate colony images,
    this often improves separability between colonies and agar by compressing
    extreme outliers (e.g., specular highlights, deep shadows) while expanding the
    dynamic range where colonies reside.

    Use cases (agar plates):
    - Normalize exposure across scans or camera shots before thresholding.
    - Recover contrast on low-contrast plates without amplifying noise as much as
      aggressive histogram equalization might.
    - Prepare images for global methods (e.g., Otsu) or for visualization.

    Tuning and effects:
    - lower_percentile: Increasing this value clips more of the darkest pixels
      (e.g., shadows/edge artifacts), which brightens the image and can reveal
      translucent colonies. Too high can erase true dark background structure.
    - upper_percentile: Decreasing this value clips more highlights (e.g., glare,
      dust reflections), preventing them from dominating contrast. Too low flattens
      bright colonies or pigmented regions.

    Caveats:
    - If outliers are biological signals (very bright colonies), heavy clipping can
      reduce their apparent intensity and bias measurements.
    - Contrast stretching is global; it will not fix spatially varying illumination
      on its own (consider `GaussianSubtract` or `RollingBallRemoveBG`).

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
        p_lower, p_upper = np.percentile(image.enh_gray[:], (self.lower_percentile, self.upper_percentile))
        image.enh_gray[:] = rescale_intensity(image=image.enh_gray[:], in_range=(p_lower, p_upper))
        return image
