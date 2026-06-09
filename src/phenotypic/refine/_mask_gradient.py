from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField, TuneSpec

import numpy as np
from skimage.morphology import dilation, erosion


class MaskGradient(ObjectRefiner, FootprintMixin):
    """Extract colony boundary outlines via morphological gradient.

    Computes the morphological gradient (dilation minus erosion) of the binary
    object mask, retaining only the pixels that lie on object boundaries. The
    result is a thin perimeter outline of each colony with interior and exterior
    pixels set to background. Boundary thickness scales with ``width``.

    For an overview of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Extracting colony perimeters for boundary roughness or circularity
          measurements that require edge-only pixel sets.
        - Creating boundary masks for edge-specific color or texture sampling
          to characterize colony rim pigmentation.
        - Generating QC overlays that visualize detected colony contours on
          top of the raw RGB image.
        - Isolating spreading or filamentous edges extending from colony cores
          as a precursor to morphological analysis.

    Consider Also:
        - :class:`Skeletonize` when medial-axis topology or branch structure
          is needed rather than a boundary outline.
        - :class:`Thinning` for iterative boundary peeling that preserves
          connectivity and produces a thinner skeleton-like result.
        - :class:`MaskErosion` for uniform inward shrinking of the full mask
          without reducing the result to boundary pixels only.

    Args:
        shape: Structuring element shape for gradient computation. ``"auto"``
            scales a disk to the image size; ``"disk"``, ``"square"``, and
            ``"diamond"`` use named shapes at the given ``width``; a NumPy
            array provides a custom element; ``None`` uses the skimage library
            default. Default: ``None``.
        width: Footprint width in pixels when using a named shape or auto
            scaling. Larger values produce thicker boundary outlines. Typical
            range: 1--5. Default: 1.

    Returns:
        Image: Input image with ``objmask`` replaced by the gradient boundary
        mask; interior and exterior pixels are set to background.

    Raises:
        AttributeError: If an unrecognised ``shape`` value is provided.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for boundary
        extraction workflows on real plate images.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: Annotated[int, TuneSpec(1, 5)] = 1

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
