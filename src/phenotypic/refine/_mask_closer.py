from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_ import FootprintMixin

import numpy as np
from skimage.morphology import closing


class MaskCloser(ObjectRefiner, FootprintMixin):
    """Morphologically close binary masks to fill small holes and gaps.

    Intuition:
        Binary closing (dilation followed by erosion) fills small holes and
        thin gaps between nearby objects. On agar plates, this bridges fragments
        of the same colony that are separated by thin channels of background
        (from uneven staining, condensation, or shadow effects) while preserving
        the overall shape and size of larger colonies.

    Why this is useful for agar plates:
        Colonies may fragment due to uneven pigmentation, staining patterns, or
        internal voids from gas pockets. A gentle closing step reconnects nearby
        fragments, improving count accuracy and morphological measurements
        (area, perimeter) by representing each colony as a single contiguous object.

    Use cases:
        - Reconnect nearby fragments of the same colony separated by thin channels.
        - Fill small internal holes from uneven staining or shadow effects.
        - Smooth minor gaps in colony boundaries after thresholding.

    Caveats:
        - Too large a shape may merge adjacent but distinct colonies into
          a single object, inflating size measurements and reducing count accuracy.
        - Closing can obscure true biological gaps in filamentous or spreading
          growth phenotypes.
        - Large radii produce blunter colony edges and may remove thin appendages.

    Attributes:
        shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for closing. A larger or denser shape fills
            wider gaps but risks merging adjacent colonies.
        width (int): Footprint width in pixels. Larger values fill bigger gaps
            but risk over-connecting separate objects.

    Examples:
        .. dropdown:: Fill small gaps in colony masks

            >>> from phenotypic.refine import MaskCloser
            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> image = Image.imread("colony_plate.jpg")  # doctest: +SKIP
            >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
            >>> # Fill gaps from uneven staining with auto-scaled shape
            >>> refiner = MaskCloser(shape='auto')  # doctest: +SKIP
            >>> refined = refiner.apply(detected)  # doctest: +SKIP
            >>> # Or use a fixed disk shape with width 3
            >>> refiner = MaskCloser(shape='disk', width=3)  # doctest: +SKIP
            >>> refined = refiner.apply(detected, inplace=False)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``shape`` type is provided (checked
            during operation).
    """

    def __init__(
            self,
            shape: Literal[
                           "auto", "square", "diamond", "disk"] | np.ndarray | None = None,
            width: int = 5
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
        """
        super().__init__()
        self.shape = shape
        self.width = width

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

        image.objmask[:] = closing(image.objmask[:], footprint=footprint)
        return image
