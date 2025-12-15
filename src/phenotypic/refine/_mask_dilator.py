from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ObjectRefiner

import numpy as np
from skimage.morphology import binary_dilation


class MaskDilator(ObjectRefiner):
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
        - Too large a footprint merges distinct adjacent colonies into a single
          object, reducing count accuracy and biasing spatial analysis.
        - Dilation inflates area measurements; if area accuracy is critical,
          follow dilation with erosion (closing) or record original measurements.
        - Large radii can create bridges between originally separate colonies
          that were only separated by a thin gap, introducing false merges.

    Attributes:
        footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
            Structuring element used for dilation. A larger footprint expands
            objects more aggressively but risks merging adjacent colonies.
        width (int): Footprint width in pixels. Larger values bridge bigger gaps
            but risk over-connecting separate objects.

    Examples:
        .. dropdown:: Merge fragmented colonies via dilation

            >>> from phenotypic.refine import MaskDilator
            >>> from phenotypic import Image
            >>> from phenotypic.detect import OtsuDetector
            >>> image = Image.imread("colony_plate.jpg")  # doctest: +SKIP
            >>> detected = OtsuDetector().apply(image)  # doctest: +SKIP
            >>> # Dilate with auto-scaled footprint to bridge nearby fragments
            >>> refiner = MaskDilator(footprint='auto')  # doctest: +SKIP
            >>> dilated = refiner.apply(detected)  # doctest: +SKIP
            >>> # Or use a fixed disk footprint with width 2 for moderate expansion
            >>> refiner = MaskDilator(footprint='disk', width=2)  # doctest: +SKIP
            >>> dilated = refiner.apply(detected, inplace=False)  # doctest: +SKIP

    Raises:
        AttributeError: If an invalid ``footprint`` type is provided (checked
            during operation).
    """

    def __init__(
            self,
            footprint: Literal["auto", "square", "diamond", "disk"] |
                       np.ndarray[int] | None = None,
            width: int = 3
    ):
        """Initialize the dilator.

        Args:
            footprint (Literal["auto", "square", "diamond", "disk"] | np.ndarray | None):
                Structuring element for dilation. Use:
                - "auto" to select a disk footprint scaled to image size
                  (larger plates → slightly larger width),
                - a NumPy array to pass a custom footprint,
                - one of the named shapes ("disk", "square", "diamond") with
                  a specified width,
                - or ``None`` to use the library default.

                Larger widths expand objects more and bridge wider gaps, but
                risk merging distinct colonies and inflating size measurements
                beyond recovery.
            width (int): Footprint width in pixels when using named shapes
                or auto-scaling. Default: 3 pixels (moderate expansion).
        """
        super().__init__()
        self.footprint = footprint
        self.width = width

    def _operate(self, image: Image) -> Image:
        if self.footprint == "auto":
            footprint = self._make_footprint(
                    shape="diamond", width=max(2, round(np.min(image.shape)*0.003))
            )
        elif isinstance(self.footprint, np.ndarray):
            footprint = self.footprint
        elif self.footprint in self._footprint_shapes:
            footprint = self._make_footprint(shape=self.footprint,
                                             width=self.width)
        elif not self.footprint:
            footprint = None
        else:
            raise AttributeError("Invalid footprint type")

        image.objmask[:] = binary_dilation(image.objmask[:],
                                           footprint=footprint)
        return image
