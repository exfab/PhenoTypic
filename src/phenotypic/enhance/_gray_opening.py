from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from skimage import morphology

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ImageEnhancer
from phenotypic.tools_.mixin import FootprintMixin


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
        - :class:`BilateralDenoise` for noise reduction that preserves edges
          without morphological assumptions.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement pipelines on plate images.
    """

    def __init__(self, shape: Literal["square", "diamond", "disk"] = "square",
                 width: int = 5, n_iter: int = 1):
        """
        A kernel configuration class for image processing tasks, particularly suited for applications
        such as analyzing and processing images of microbe colonies on solid media agar. This class
        enables the definition of a kernel shape and size, which significantly impacts the morphological
        operations applied to the image (e.g., filtering, dilation, erosion). Adjusting these parameters
        can enhance or hinder the detection and analysis of colony boundaries, shapes, and distribution.

        Attributes:
            shape (Literal["square", "diamond", "disk"]): The geometric shape of the kernel. This attribute
                governs the pattern and extent of neighboring pixels involved in the processing operation.
                Choosing "square" results in a uniform rectangular influence, which may be suitable for
                isotropic features but could introduce angular artifacts in circular features like microbe
                colonies. The "diamond" shape provides a more angular neighborhood pattern that helps
                preserve diagonal structures. On the other hand, "disk" introduces a circular pattern
                that can align well with colony boundaries and reduce distortions in rounded features.

            width (int): The size (diameter) of the kernel in pixels. A larger width increases the
                area of influence during image processing, which can smooth out smaller features like
                noise but potentially merge closely spaced microbe colonies into larger regions. Smaller
                values offer finer detail and greater distinction between colonies but may leave noise
                unprocessed or small artifacts unchanged.

            n_iter (int): Number of times to apply the opening operation.
                Repeated opening with a small element produces smoother results
                than a single pass with a larger element. Default: 1.
        """
        self.shape = shape
        self.width = width
        self.n_iter = n_iter

    def _operate(self, image: Image) -> Image:
        footprint = self._make_footprint(shape=self.shape, width=self.width)
        for _ in range(self.n_iter):
            image.detect_mat[:] = morphology.opening(
                    image=image.detect_mat[:],
                    footprint=footprint,
            )
        return image
