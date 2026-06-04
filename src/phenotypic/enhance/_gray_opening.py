from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

from skimage import morphology

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ImageEnhancer
from phenotypic.tools_.mixin import FootprintMixin
from phenotypic.tools_.typing_ import TuneSpec


class GrayOpening(ImageEnhancer, FootprintMixin):
    """Remove small bright artifacts from ``detect_mat`` via morphological opening.

    Applies erosion followed by dilation with a structuring element, removing
    bright features smaller than the element while preserving the shape of
    larger structures. Effectively suppresses dust particles, small noise
    speckles, and tiny satellite colonies.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        shape: Structuring element geometry. ``'square'`` (default) preserves
            edges; ``'diamond'`` is more rounded at diagonals; ``'disk'``
            provides uniform circular operations.
        width: Diameter of the structuring element in pixels. Larger values
            remove larger features. Typical range: 3--15. Default: 5.
        n_iter: Number of times to apply the opening. Repeated opening with
            a small element produces smoother results than a single pass
            with a larger element. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` morphologically opened.
        ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Removing dust particles and small bright noise from plate scans.
        - Suppressing tiny satellite colonies that interfere with detection
          of larger colonies.
        - Smoothing the detection surface before background subtraction.

    Consider Also:
        - :class:`WhiteTophatEnhance` when you want to isolate (not remove)
          small bright structures.
        - :class:`SubtractWhiteTophat` for subtracting small bright artifacts
          while retaining the background.
        - :class:`LocalEdgeDenoise` for noise reduction that preserves edges
          without morphological assumptions.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
    """

    shape: Literal["square", "diamond", "disk"] = "square"
    # TODO: review bound (unverified vs literature)
    width: Annotated[int, TuneSpec(3, 15, step=2)] = 5
    n_iter: Annotated[int, TuneSpec(1, 3)] = 1

    def _operate(self, image: Image) -> Image:
        footprint = self._make_footprint(shape=self.shape, width=self.width)
        for _ in range(self.n_iter):
            image.detect_mat[:] = morphology.opening(
                    image=image.detect_mat[:],
                    footprint=footprint,
            )
        return image
