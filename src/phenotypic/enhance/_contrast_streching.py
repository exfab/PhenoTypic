from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import rescale_intensity

from ..abc_ import ContrastAdjustment


class ContrastStretching(ContrastAdjustment):
    """Stretch the intensity range of ``detect_mat`` to fill the full dynamic range.

    Rescales pixel values by clipping at lower and upper percentiles, then
    linearly remapping the retained range to [0, 1]. Outliers such as specular
    highlights and deep shadows are clamped, expanding the range where colony
    intensities reside. Simpler and faster than :class:`EnhanceLocalContrast`,
    with no local tile artefacts.

    For how contrast adjustment fits into the pipeline, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates with narrow intensity histograms from under-exposure or low
          scanner gain.
        - Normalizing exposure variation across imaging sessions or plate batches.
        - Quick preprocessing before global thresholding (Otsu, Triangle).
        - Images with bright specular highlights or very dark border regions
          that compress the useful intensity range.

    Consider Also:
        - :class:`EnhanceLocalContrast` when illumination varies spatially
          across the plate and per-tile equalization is needed.
        - :class:`FlattenIllumination` when the primary issue is a large-scale
          brightness gradient rather than a narrow dynamic range.

    Args:
        lower_percentile: Dark clipping point. Pixels below this percentile
            are mapped to 0. Typical range: 1--5. Default: 2.
        upper_percentile: Bright clipping point. Pixels above this percentile
            are mapped to 1. Typical range: 95--99. Default: 98.

    Returns:
        Image: Input image with ``detect_mat`` rescaled to the full dynamic
        range. ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/how_to/notebooks/enhance_low_contrast` for a comparison of
        contrast enhancement methods on real plate images.
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
