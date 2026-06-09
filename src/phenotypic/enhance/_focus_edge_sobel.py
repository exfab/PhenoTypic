from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import skimage.filters

from phenotypic.abc_ import FocusEdge


class FocusEdgeSobel(FocusEdge):
    """Highlight colony edges in ``detect_mat`` using the Sobel gradient operator.

    Computes the gradient magnitude across both image axes to emphasize
    intensity transitions at colony boundaries. The output is an edge-strength
    map suitable as a preprocessing step before watershed seeding or
    contour-based detectors, not a corrected grayscale image.

    For how edge enhancement fits into the pipeline, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Pre-filtering before watershed or contour-based colony detection.
        - Separating touching colonies when combined with marker-based
          segmentation.
        - Visualizing colony boundary sharpness for image quality assessment.

    Consider Also:
        - :class:`SharpenEdgeGauss` when sharpening colony edges while
          retaining the original intensity profile rather than producing a
          pure gradient map.
        - :class:`FocusEdgeLaplace` for second-derivative edge detection that
          responds to ridges, valleys, and ring-like swarming fronts.

    Returns:
        Image: Input image with ``detect_mat`` set to the Sobel gradient
        magnitude. ``rgb`` and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a visual
        walkthrough of edge enhancement on plate images.
        :doc:`/explanation/what_enhancement_does` for how edge-response maps
        fit into the pipeline model.
    """

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = skimage.filters.sobel(
                image=image.detect_mat[:]
        )
        return image
