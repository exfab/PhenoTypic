from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import dilation


class MaskDilator(ObjectRefiner, FootprintMixin):
    """Expand colony masks outward using morphological dilation.

    Adds pixels around object boundaries, bridging small gaps between
    nearby fragments and recovering faint halos excluded by strict
    thresholding. Dilation inflates area; follow with erosion (closing)
    if area accuracy is critical.

    Best For:
        - Bridging thin gaps between fragments of the same colony.
        - Recovering faint colony halos near detection boundaries.
        - Preprocessing before merge-based refinement operations.

    Consider Also:
        - :class:`MaskCloser` for dilation-then-erosion that bridges gaps
          without inflating colony size.
        - :class:`MaskEroder` for the opposite effect — shrinking masks
          to remove thin protrusions.

    Args:
        shape: Structuring element. ``'auto'``, ``'disk'``, ``'square'``,
            ``'diamond'``, or custom ndarray. Default: ``None``.
        width: Footprint width in pixels. Default: 3.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` dilated.

    See Also:
        :doc:`/explanation/refinement_strategies` for the recommended
        refinement sequence.
    """

    def __init__(
            self,
            shape: Literal["auto", "square", "diamond", "disk"] |
                       np.ndarray[int] | None = None,
            width: int = 3,
            n_iter: int = 1,
    ):
        """Initialize the dilator.

        Args:
            shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for dilation. Use:

                - "auto" to select a disk shape scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom shape,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified width,
                - or ``None`` to use the library default.

                Larger widths expand objects more and bridge wider gaps, but
                risk merging distinct colonies and inflating size measurements
                beyond recovery.
            width (int): Footprint width in pixels when using named shapes
                or auto-scaling. Default: 3 pixels (moderate expansion).
            n_iter (int): Number of times to apply dilation. Repeated dilation
                with a small element produces smoother results than a single
                pass with a larger element. Default: 1.
        """
        super().__init__()
        self.shape = shape
        self.width = width
        self.n_iter = n_iter

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
