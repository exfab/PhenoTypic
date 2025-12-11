from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner

import numpy as np
from skimage.morphology import binary_erosion


class MaskEroder(ObjectRefiner):
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
        robust _core colony footprint. This is useful for reducing false-positive
        signal and improving measurement precision.

    Use cases:
        - Remove thin protrusions or whiskers extending from colony edges.
        - Eliminate isolated noise specks and dust artifacts before merging.
        - Shrink masks to exclude uncertain boundary pixels, improving precision.
        - Prepare masks for subsequent operations (e.g., erosion-dilation to
          refine morphology without changing overall size).

    Caveats:
        - Too large a footprint eliminates small colonies entirely or severely
          shrinks area measurements, reducing sensitivity for small colonies.
        - Aggressive erosion can disconnect weakly-stained colonies or separate
          merged colonies at the cost of breaking apart the same colony.
        - Erosion reduces per-colony area measurements, which may bias phenotypic
          analysis if not compensated downstream.

    Attributes:
        footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for erosion. A larger footprint removes
            more boundary pixels and thin features but risks shrinking colonies
            too aggressively.
        radius (int): Footprint radius in pixels. Larger values erode more deeply
            but risk eliminating small colonies entirely.

    Examples:
        .. dropdown:: Remove thin protrusions and noise from colony masks

            >>> from phenotypic.refine import MaskEroder
            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> image = Image.from_image_path("colony_plate.jpg")  # doctest: +SKIP
            >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
            >>> # Erode with auto-scaled footprint to remove specks
            >>> refiner = MaskEroder(footprint='auto')  # doctest: +SKIP
            >>> eroded = refiner.apply(detected)  # doctest: +SKIP
            >>> # Or use a small fixed disk footprint (radius 1) for gentle erosion
            >>> refiner = MaskEroder(footprint='disk', radius=1)  # doctest: +SKIP
            >>> eroded = refiner.apply(detected, inplace=True)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``footprint`` type is provided (checked
            during operation).
    """

    def __init__(
            self,
            footprint: Literal[
                           "auto", "square", "diamond", "disk"] | np.ndarray | None = None,
            radius: int = 3
    ):
        """Initialize the eroder.

        Args:
            footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for erosion. Use:
                - "auto" to select a disk footprint scaled to image size
                  (larger plates → slightly larger radius),
                - a NumPy array to pass a custom footprint,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified radius,
                - or ``None`` to use the library default.

                Larger radii erode more aggressively, removing larger features
                but risking elimination of small colonies and over-shrinkage
                of area measurements.
            radius (int): Footprint radius in pixels when using named shapes
                or auto-scaling. Default: 3 pixels (moderate erosion).
        """
        super().__init__()
        self.footprint = footprint
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        if self.footprint == "auto":
            footprint = self._make_footprint(
                    "disk", radius=max(2, round(np.min(image.shape)*0.003))
            )
        elif isinstance(self.footprint, np.ndarray):
            footprint = self.footprint
        elif self.footprint in self._footprint_shapes:
            footprint = self._make_footprint(self.footprint, radius=self.radius)
        elif not self.footprint:
            footprint = None
        else:
            raise AttributeError("Invalid footprint type")

        image.objmask[:] = binary_erosion(image.objmask[:], footprint=footprint)
        return image
