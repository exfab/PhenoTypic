from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import skimage.filters

from phenotypic.abc_ import FocusEdge


class FocusEdgeSobel(FocusEdge):
    """Highlight colony edges in detect_mat using the Sobel gradient operator.

    Computes the gradient magnitude to emphasize intensity transitions at
    colony boundaries. The output is an edge-strength map, not a corrected
    image — useful as a preprocessing step before watershed seeds or
    contour-based detectors.

    Returns:
        Image: Input image with ``detect_mat`` set to gradient magnitude.
        ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Pre-filtering before watershed or contour-based detection.
        - Separating touching colonies when combined with marker-based
          segmentation.
        - Visualizing colony boundary sharpness for quality assessment.

    Consider Also:
        - :class:`SharpenEdgeGauss` when you want to sharpen edges without
          converting to a pure edge map.
        - :class:`FocusEdgeLaplace` for second-derivative edge detection
          that responds to ridges and valleys.

    See Also:
        :doc:`/explanation/what_enhancement_does` for how edge enhancement
        fits into the pipeline model.
    """

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = skimage.filters.sobel(
                image=image.detect_mat[:]
        )
        return image
