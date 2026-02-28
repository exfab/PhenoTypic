from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

import numpy as np
from skimage.morphology import dilation


class MaskDilator(ObjectRefiner, FootprintMixin):
    """Morphologically dilate binary masks to expand colonies and bridge gaps.

    Intuition:
        Binary dilation expands all object regions by adding pixels around the
        boundaries. On agar plates, this bridges small gaps between nearby
        fragments of the same colony (from uneven staining or shadow effects)
        and expands masks to include faint halos or uncertain boundary pixels
        that belong to the colony but were excluded by strict thresholding.

    Why this is useful for agar plates:
        Fragmented colony detections are common from uneven illumination,
        pigmentation heterogeneity, or internal voids. Dilation reconnects
        nearby fragments and recovers faint outer regions, improving count
        accuracy and colony area estimates. It's particularly useful as a
        preprocessing step for merging operations or boundary recovery.

    Use cases:
        - Merge fragmented colony detections separated by thin gaps.
        - Bridge shadow-induced gaps in detection masks.
        - Expand masks to recover faint colony halos near detection boundaries.
        - Prepare masks for merge-based refinement operations.

    Caveats:
        - Too large a shape merges distinct adjacent colonies into a single
          object, reducing count accuracy and biasing spatial analysis.
        - Dilation inflates area measurements; if area accuracy is critical,
          follow dilation with erosion (closing) or record original measurements.
        - Large radii can create bridges between originally separate colonies
          that were only separated by a thin gap, introducing false merges.

    Attributes:
        shape (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for dilation. A larger shape expands
            objects more aggressively but risks merging adjacent colonies.
        width (int): Footprint width in pixels. Larger values bridge bigger gaps
            but risk over-connecting separate objects.

    Examples:
        Merge fragmented colonies via dilation:

        >>> from phenotypic.refine import MaskDilator
        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread("colony_plate.jpg")  # doctest: +SKIP
        >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
        >>> # Dilate with auto-scaled shape to bridge nearby fragments
        >>> refiner = MaskDilator(shape='auto')  # doctest: +SKIP
        >>> dilated = refiner.apply(detected)  # doctest: +SKIP
        >>> # Or use a fixed disk shape with width 2 for moderate expansion
        >>> refiner = MaskDilator(shape='disk', width=2)  # doctest: +SKIP
        >>> dilated = refiner.apply(detected, inplace=False)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``shape`` type is provided (checked
            during operation).
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
