from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import erosion


class MaskEroder(ObjectRefiner, FootprintMixin):
    """Shrink colony masks inward to remove thin protrusions and noise pixels.

    Removes outer boundary pixels from all objects, eliminating thin
    whiskers, isolated specks, and uncertain boundary pixels from soft
    edges. Leaves the core colony structure intact.

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

    Args:
        shape: Structuring element. ``'auto'``, ``'disk'``, ``'square'``,
            ``'diamond'``, or custom ndarray. Default: ``None``.
        width: Footprint width in pixels. Default: 3.
        n_iter: Number of erosion iterations. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` eroded.

    See Also:
        :doc:`/explanation/refinement_strategies` for the recommended
        refinement sequence.
    """

    def __init__(
            self,
            shape: Literal[
                           "auto", "square", "diamond", "disk"] | np.ndarray | None = None,
            width: int = 3,
            n_iter: int = 1,
    ):
        """Initialize the eroder.

        Args:
            shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for erosion. Use:

                - "auto" to select a disk shape scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom shape,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified width,
                - or ``None`` to use the library default.

                Larger widths erode more aggressively, removing larger features
                but risking elimination of small colonies and over-shrinkage
                of area measurements.
            width (int): Footprint width in pixels when using named shapes
                or auto-scaling. Default: 3 pixels (moderate erosion).
            n_iter (int): Number of times to apply erosion. Repeated erosion
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
