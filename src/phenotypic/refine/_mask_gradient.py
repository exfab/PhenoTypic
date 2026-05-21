from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField

import numpy as np
from skimage.morphology import dilation, erosion


class MaskGradient(ObjectRefiner, FootprintMixin):
    """Extract object boundaries via morphological gradient (dilation minus erosion).

    Computes the difference between dilation and erosion of the binary mask,
    producing a thin outline of each object's boundary pixels. Interior and
    exterior pixels are removed, leaving only the colony perimeter for
    edge-focused analysis or visualization.

    Args:
        shape: Structuring element for gradient computation. ``"auto"``
            selects a disk scaled to image size, ``"disk"``, ``"square"``,
            or ``"diamond"`` use a named shape at the given width, a NumPy
            array provides a custom element, and ``None`` uses the library
            default. Default: None.
        width: Footprint width in pixels when using named shapes or
            auto-scaling. Larger values produce thicker boundaries.
            Typical range: 1--5. Default: 1.

    Returns:
        Image: Input image with ``objmask`` replaced by the gradient
        boundary mask.

    Raises:
        AttributeError: If an invalid ``shape`` type is provided.

    Best For:
        - Extracting colony perimeters for boundary roughness or circularity
          measurements.
        - Creating boundary masks for edge-specific color or texture analysis.
        - Visualizing colony contours as QC overlays on raw images.
        - Detecting spreading or filamentous edges extending from colony cores.

    Consider Also:
        - :class:`Skeletonize` when you need medial-axis topology rather
          than boundary outlines.
        - :class:`MaskEroder` for uniform inward shrinking without
          extracting boundaries.
        - :class:`Thinning` for iterative boundary peeling that preserves
          connectivity.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for boundary
        extraction workflows.
        :doc:`/explanation/refinement_strategies` for a comparison of
        morphological refinement methods.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: int = 1

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    "disk", width=max(1, round(np.min(image.shape) * 0.002))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(self.shape, width=self.width)
        elif not self.shape:
            footprint = None
        else:
            raise AttributeError("Invalid shape type")

        # Compute morphological gradient: dilated - eroded
        mask = image.objmask[:]
        dilated_mask = dilation(mask, footprint=footprint)
        eroded_mask = erosion(mask, footprint=footprint)
        gradient_mask = dilated_mask & ~eroded_mask  # Boundary pixels

        image.objmask[:] = gradient_mask
        return image
