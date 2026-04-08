from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import closing


class MaskCloser(ObjectRefiner, FootprintMixin):
    """Bridge small gaps and fill holes in colony masks using morphological closing.

    Applies binary closing (dilation then erosion) to reconnect nearby
    fragments of the same colony separated by thin background channels.
    Preserves overall colony shape and size while reducing fragmentation.

    Best For:
        - Colonies fragmented by uneven pigmentation or shadow effects.
        - Small internal holes from condensation or glare.
        - Reconnecting nearby fragments before measurement.

    Consider Also:
        - :class:`MaskFill` for filling enclosed holes without bridging
          separate objects.
        - :class:`MaskOpener` for the opposite effect — breaking thin
          connections between distinct colonies.
        - :class:`NearestNeighborMerger` for merging distant fragments
          based on proximity.

    Args:
        shape: Structuring element. ``'auto'``, ``'disk'``, ``'square'``,
            ``'diamond'``, or custom ndarray. Default: ``None``.
        width: Footprint width in pixels. Larger values bridge wider gaps
            but risk merging distinct colonies. Typical range: 3--9.
            Default: 5.
        n_iter: Number of closing iterations. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` morphologically
        closed.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for fragment
        merging strategies.
    """

    def __init__(
            self,
            shape: Literal[
                           "auto", "square", "diamond", "disk"] | np.ndarray | None = None,
            width: int = 5,
            n_iter: int = 1,
    ):
        """Initialize the closer.

        Args:
            shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for closing. Use:
                - "auto" to select a disk shape scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom shape,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified width,
                - or ``None`` to use the library default.

                Larger widths fill wider gaps and smoother colony boundaries,
                but risk merging adjacent colonies and losing edge sharpness.
            width (int): Footprint width in pixels when using named shapes
                or auto-scaling. Default: 5 pixels (moderate gap-filling).
            n_iter (int): Number of times to apply closing. Repeated closing
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
                    "disk", width=max(3, round(np.min(image.shape) * 0.005))
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
            image.objmask[:] = closing(image.objmask[:], footprint=footprint)
        return image
