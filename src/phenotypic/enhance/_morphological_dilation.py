from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from scipy.ndimage import binary_dilation
import numpy as np
from ..abc_ import ImageEnhancer


class MorphologicalDilation(ImageEnhancer):
    """
    Morphological dilation for expanding bright regions and merging nearby colonies.

    Performs binary dilation to expand foreground objects (colonies), filling gaps
    and connecting nearby regions. This is useful for bridging fragmented detections
    and improving colony contiguity on agar plates.

    Use cases (agar plates):
    - Merge fragmented colony detections caused by uneven staining or light surface
      texture within a single colony.
    - Bridge narrow gaps between nearly touching colonies to treat them as single
      units.
    - Fill small internal holes in colonies, improving measurement accuracy.
    - Enhance faint colony edges by expanding bright regions slightly.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic dilation that
      expands colonies equally in all directions. 'square' can prefer orthogonal
      expansion. 'diamond' reduces diagonal expansion.
    - radius: Dilation radius in pixels. Determines how much colonies expand.
      Larger radius merges more distant colonies but risks creating false mergers
      of distinct objects.

    Caveats:
    - Dilation expands all bright regions and can merge adjacent colonies,
      reducing the count of detected objects.
    - Test carefully to ensure that nearby but distinct colonies are not merged
      unintentionally.
    - Dilation makes colonies larger, which affects downstream size and area
      measurements.
    - Dilation works on binary (thresholded) data; the quality of the binary image
      is critical.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int): Dilation radius in pixels. Larger values expand colonies more.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import MorphologicalDilation
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Dilate to merge fragmented colonies
        >>> dilater = MorphologicalDilation(shape='disk', radius=3)
        >>> dilated = dilater.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], dilated.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], dilated.rgb[:])
        >>> assert np.array_equal(image.gray[:], dilated.gray[:])
    """

    def __init__(
        self,
        shape: Literal["disk", "square", "diamond"] = "disk",
        radius: int = 3,
    ):
        """
        Parameters:
            shape (str): Footprint shape. 'disk' (default) for isotropic dilation.
                'square' for grid-aligned patterns. 'diamond' for specialized cases
                with reduced diagonal dilation.
            radius (int): Dilation radius in pixels. Determines how much colonies
                expand. Larger values merge more distant objects, but can create
                false mergers. Must be at least 1. Recommended: 2-5 for typical
                colony plates. Balance between merging fragments and preserving
                distinct colonies.
        """
        # Validate shape
        if shape not in ["disk", "square", "diamond"]:
            raise ValueError(
                f"shape must be one of 'disk', 'square', 'diamond'; got '{shape}'"
            )
        self.shape = shape

        # Validate radius
        if not isinstance(radius, int) or radius < 1:
            raise ValueError(f"radius must be a positive integer; got {radius}")
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        """Apply morphological dilation to enh_gray."""
        # Get footprint
        footprint = self._make_footprint(self.shape, self.radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation (binary dilation needs thresholding)
        threshold = np.mean(enh)
        binary = enh > threshold
        result = binary_dilation(binary, structure=footprint)

        # Convert back to grayscale and write
        image.enh_gray[:] = (result * 255).astype(enh.dtype)

        return image
