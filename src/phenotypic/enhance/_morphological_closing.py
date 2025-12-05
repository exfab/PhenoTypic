from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from scipy.ndimage import binary_closing
import numpy as np
from ..abc_ import ImageEnhancer


class MorphologicalClosing(ImageEnhancer):
    """
    Morphological closing for filling small holes within colonies.

    Performs binary closing (dilation followed by erosion) to fill small background
    holes within foreground objects (colonies). This is useful for smoothing colony
    boundaries and merging slightly fragmented detections on agar plates.

    Use cases (agar plates):
    - Fill small holes or voids within colony regions caused by uneven illumination,
      surface texture, or water droplets.
    - Bridge narrow gaps between slightly separated colony fragments, improving
      colony contiguity for measurement.
    - Smooth colony boundaries to reduce boundary artifacts from thresholding and
      improve shape-based measurements.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic smoothing and
      better preserves rounded colony shapes. 'square' aligns with pixel grid.
      'diamond' is specialized for cases where diagonal connectivity should be
      reduced.
    - radius: Determines maximum size of holes filled. Choose smaller than the
      minimum expected gap width between distinct colonies to avoid merging
      separate colonies. Larger radius fills more holes but risks merging nearby
      colonies into one.

    Caveats:
    - Large radius values can merge adjacent colonies, reducing their individual
      detection; test on reference plates carefully.
    - Closing assumes that holes are darker than colony interiors; inverted or
      low-contrast images may behave unexpectedly.
    - This operation works on binary (thresholded) data, so the quality of the
      binary image significantly affects the result.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int): Footprint radius in pixels. Determines the maximum size of
            holes filled by the closing operation.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import MorphologicalClosing
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Fill small holes within colonies and smooth boundaries
        >>> closer = MorphologicalClosing(shape='disk', radius=3)
        >>> smoothed = closer.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], smoothed.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], smoothed.rgb[:])
        >>> assert np.array_equal(image.gray[:], smoothed.gray[:])
    """

    def __init__(
        self,
        shape: Literal["disk", "square", "diamond"] = "disk",
        radius: int = 3,
    ):
        """
        Parameters:
            shape (str): Footprint shape. 'disk' (default) for isotropic processing
                that preserves rounded colony shapes. 'square' for grid-aligned
                patterns. 'diamond' for specialized cases with reduced diagonal
                connectivity.
            radius (int): Footprint radius in pixels. Determines the maximum size
                of holes filled within colonies. Must be at least 1. Recommended:
                2-5 for typical colony plates; balance between filling internal
                voids and avoiding unwanted colony merging.
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
        """Apply morphological closing to enh_gray."""
        # Get footprint
        footprint = self._make_footprint(self.shape, self.radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation (binary closing needs thresholding)
        threshold = np.mean(enh)
        binary = enh > threshold
        result = binary_closing(binary, structure=footprint)

        # Convert back to grayscale and write
        image.enh_gray[:] = (result * 255).astype(enh.dtype)

        return image
