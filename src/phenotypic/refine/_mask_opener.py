from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import opening
from phenotypic.tools_.typing_ import FootprintShape


class MaskOpener(ObjectRefiner, FootprintMixin):
    """Smooth mask boundaries and break thin bridges between touching colonies.

    Applies binary opening (erosion then dilation) to remove small isolated
    pixels and narrow connections. Colonies linked by faint film or agar
    artifacts become separated without significantly shrinking well-formed
    colony masks.

    Args:
        shape: Structuring element. ``'auto'`` selects based on detected
            objects. ``'diamond'``, ``'square'``, ``'disk'``, or a custom
            ndarray. Default: ``None``.
        width: Size of the structuring element in pixels. Larger values
            smooth more aggressively. Typical range: 3--9. Default: 5.
        n_iter: Number of opening iterations. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` morphologically
        opened.

    Best For:
        - Splitting colonies connected by 1--2 pixel bridges after thresholding.
        - Removing tiny noise specks from the detection mask.
        - Smoothing jagged mask edges before measurement.

    Consider Also:
        - :class:`MaskCloser` for the opposite effect — filling small gaps
          and bridging fragments.
        - :class:`SmallObjectRemover` for removing small objects by area
          rather than morphology.
        - :class:`SeparateObjects` for splitting merged colonies using
          watershed-based separation.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
        :doc:`/explanation/refinement_strategies` for the recommended
        refinement sequence.
    """

    def __init__(
            self,
            shape: Literal["auto"] | FootprintShape | np.ndarray | None = None,
            width: int = 5,
            n_iter: int = 1,
    ):
        """Initialize the opener.

        Args:
            shape (Literal["auto"] | np.ndarray | int | None): Structuring
                element for opening. Use:

                - "auto" to select a diamond shape scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom shape,
                - an ``int`` width to build a diamond shape of that size,
                - or ``None`` to use the library default.

                Larger widths disconnect wider bridges and suppress more
                speckles, but erode edges and can remove small colonies.
            n_iter (int): Number of times to apply opening. Repeated opening
                with a small element produces smoother results than a single
                pass with a larger element. Default: 1.
        """
        super().__init__()
        self.shape: Literal["auto"] | FootprintShape | np.ndarray | None = shape
        self.width = width
        self.n_iter = n_iter

    def _operate(self, image: Image) -> Image:
        if self.shape == "auto":
            footprint = FootprintMixin._make_footprint(
                    "diamond", width=max(3, round(np.min(image.shape) * 0.005))
            )
        elif isinstance(self.shape, np.ndarray):
            footprint = self.shape
        elif self.shape in self._footprint_shapes:
            footprint = FootprintMixin._make_footprint(self.shape,
                                                       width=int(self.width))
        elif not self.shape:
            footprint = self.shape
        else:
            raise AttributeError("Invalid shape type")

        for _ in range(self.n_iter):
            image.objmask[:] = opening(image.objmask[:], footprint=footprint)
        return image
