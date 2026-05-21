from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField

import numpy as np
from skimage.morphology import erosion


class MaskEroder(ObjectRefiner, FootprintMixin):
    """Shrink colony masks inward to remove thin protrusions and noise pixels.

    Removes outer boundary pixels from all objects, eliminating thin
    whiskers, isolated specks, and uncertain boundary pixels from soft
    edges. Leaves the core colony structure intact.

    Args:
        shape: Structuring element. ``'auto'``, ``'disk'``, ``'square'``,
            ``'diamond'``, or custom ndarray. Default: ``None``.
        width: Footprint width in pixels. Default: 3.
        n_iter: Number of erosion iterations. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` eroded.

    Best For:
        - Removing thin protrusions or whiskers from colony edges.
        - Eliminating noise specks that survived previous cleanup.
        - Excluding uncertain boundary pixels for higher-precision measurements.

    Consider Also:
        - :class:`MaskDilator` for the opposite effect — expanding masks
          outward.
        - :class:`MaskOpener` for erosion-then-dilation that removes thin
          features without permanently shrinking colonies.
        - :class:`SmallObjectRemover` for removing small objects by area
          rather than shrinking all objects.

    See Also:
        :doc:`/explanation/refinement_strategies` for the recommended
        refinement sequence.
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
