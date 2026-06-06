from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField

import numpy as np
from skimage.morphology import erosion


class MaskErosion(ObjectRefiner, FootprintMixin):
    """Shrink colony masks inward by morphological erosion to remove boundary noise.

    Removes outer boundary pixels from every detected object, eliminating thin
    whiskers, isolated specks, and uncertain soft-edge pixels produced by
    over-sensitive thresholding. The core colony structure is preserved but
    masks are permanently reduced in area; pair with :class:`MaskDilation` or
    use :class:`MaskOpening` if area must be recovered.

    For an overview of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Removing thin protrusions or whisker artefacts extending from colony
          edges after edge-sensitive detection.
        - Eliminating isolated noise specks that survived previous cleanup
          steps and registered as very small detections.
        - Excluding uncertain boundary pixels before high-precision shape or
          area measurements to tighten the colony footprint.

    Consider Also:
        - :class:`MaskDilation` for the opposite effect — expanding masks
          outward to recover halos or bridge gaps.
        - :class:`MaskOpening` for erosion followed by dilation that removes
          thin protrusions without permanently shrinking the colony body.
        - :class:`SmallObjectRemover` for discarding small objects entirely by
          area threshold rather than shrinking all objects uniformly.

    Args:
        shape: Structuring element shape for the erosion footprint. ``"auto"``
            scales a disk to the image size; ``"disk"``, ``"square"``, and
            ``"diamond"`` use named shapes at the given ``width``; a NumPy
            array provides a custom element; ``None`` uses the skimage library
            default. Default: ``None``.
        width: Footprint width in pixels when using a named shape. Larger values
            erode more deeply inward. Typical range: 1--7. Default: 3.
        n_iter: Number of erosion iterations applied sequentially. Each
            additional iteration removes one further layer of boundary pixels.
            Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` eroded inward.

    See Also:
        :doc:`/explanation/refinement_strategies` for the recommended
        morphological refinement sequence.
    """

    shape: Literal["auto", "square", "diamond", "disk"] | NdArrayField | None = None
    width: int = 3
    n_iter: int = 1

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    "disk", width=max(2, round(np.min(image.shape) * 0.003))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(self.shape, width=self.width)
        elif not self.shape:
            footprint = None
        else:
            raise AttributeError("Invalid shape type")

        for _ in range(self.n_iter):
            image.objmask[:] = erosion(image.objmask[:], footprint=footprint)
        return image
