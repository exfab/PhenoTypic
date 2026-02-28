from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import erosion


class MaskEroder(ObjectRefiner, FootprintMixin):
    """Morphologically erode binary masks to remove thin protrusions and noise.

    Intuition:
        Binary erosion shrinks all object regions by removing outer pixels,
        effectively eliminating thin protrusions, small isolated specks, and
        faint boundary pixels. On agar plates, this removes dust, sensor noise,
        condensation artifacts, and uneven staining artifacts while preserving
        the _core structure of well-formed colonies.

    Why this is useful for agar plates:
        Colony boundaries detected via thresholding often include noise pixels,
        thin speckles from uneven illumination, and uncertain boundary pixels
        from soft edges. Erosion strips away these artifacts, leaving a more
        robust _core colony shape. This is useful for reducing false-positive
        signal and improving measurement precision.

    Use cases:
        - Remove thin protrusions or whiskers extending from colony edges.
        - Eliminate isolated noise specks and dust artifacts before merging.
        - Shrink masks to exclude uncertain boundary pixels, improving precision.
        - Prepare masks for subsequent operations (e.g., erosion-dilation to
          refine morphology without changing overall size).

    Caveats:
        - Too large a shape eliminates small colonies entirely or severely
          shrinks area measurements, reducing sensitivity for small colonies.
        - Aggressive erosion can disconnect weakly-stained colonies or separate
          merged colonies at the cost of breaking apart the same colony.
        - Erosion reduces per-colony area measurements, which may bias phenotypic
          analysis if not compensated downstream.

    Attributes:
        shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for erosion. A larger shape removes
            more boundary pixels and thin features but risks shrinking colonies
            too aggressively.
        width (int): Footprint width in pixels. Larger values erode more deeply
            but risk eliminating small colonies entirely.

    Examples:
        Remove thin protrusions and noise from colony masks:

        >>> from phenotypic.refine import MaskEroder
        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread("colony_plate.jpg")  # doctest: +SKIP
        >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
        >>> # Erode with auto-scaled shape to remove specks
        >>> refiner = MaskEroder(shape='auto')  # doctest: +SKIP
        >>> eroded = refiner.apply(detected)  # doctest: +SKIP
        >>> # Or use a small fixed disk shape (width 1) for gentle erosion
        >>> refiner = MaskEroder(shape='disk', width=1)  # doctest: +SKIP
        >>> eroded = refiner.apply(detected, inplace=True)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``shape`` type is provided (checked
            during operation).
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
