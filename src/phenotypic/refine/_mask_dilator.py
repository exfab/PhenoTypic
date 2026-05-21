from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import NdArrayField

import numpy as np
from skimage.morphology import dilation


class MaskDilator(ObjectRefiner, FootprintMixin):
    """Expand colony masks outward using morphological dilation.

    Adds pixels around object boundaries, bridging small gaps between
    nearby fragments and recovering faint halos excluded by strict
    thresholding. Dilation inflates area; follow with erosion (closing)
    if area accuracy is critical.

    Args:
        shape: Structuring element. ``'auto'``, ``'disk'``, ``'square'``,
            ``'diamond'``, or custom ndarray. Default: ``None``.
        width: Footprint width in pixels. Default: 3.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` dilated.

    Best For:
        - Bridging thin gaps between fragments of the same colony.
        - Recovering faint colony halos near detection boundaries.
        - Preprocessing before merge-based refinement operations.

    Consider Also:
        - :class:`MaskCloser` for dilation-then-erosion that bridges gaps
          without inflating colony size.
        - :class:`MaskEroder` for the opposite effect — shrinking masks
          to remove thin protrusions.

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
                    shape="diamond", width=max(2, round(np.min(image.shape) * 0.003))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(shape=self.shape,
                                             width=self.width)
        elif not self.shape:
            footprint = None
        else:
            raise AttributeError("Invalid shape type")

        for _ in range(self.n_iter):
            image.objmask[:] = dilation(image.objmask[:], footprint=footprint)
        return image
