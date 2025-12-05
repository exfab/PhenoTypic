from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from scipy.ndimage import binary_opening
import numpy as np
from ..abc_ import ImageEnhancer


class MorphologicalOpening(ImageEnhancer):
    """
    Morphological opening for removing small noise regions.

    Performs binary opening (erosion followed by dilation) to remove small
    foreground objects (noise, speckles, minor artifacts) while preserving
    colony structure. This is useful after thresholding to clean up salt-and-pepper
    noise on agar plates before downstream analysis.

    Use cases (agar plates):
    - Remove small noise regions and speckles that confuse thresholding.
    - Clean up minor surface artifacts (dust, scratches) that create false small
      detections.
    - Reduce false positives in automated colony detection by eliminating noise
      regions smaller than the structuring element.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic smoothing and
      better preserves rounded colony shapes. 'square' aligns with pixel grid and
      can suppress grid-aligned artifacts. 'diamond' is specialized for certain
      diagonal-sensitive cases.
    - radius: Determines maximum size of noise regions removed. Choose smaller than
      the minimum colony diameter to avoid destroying small but real colonies.
      Larger radius removes more noise but risks removing fine colony details.

    Caveats:
    - Large radius values can remove small but valid colonies; test carefully.
    - Opening assumes that noise is darker than the background after thresholding;
      inverted images may require adjustment.
    - This operation works on binary (thresholded) data, so the quality of the
      binary image (via thresholding) affects the result.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int): Footprint radius in pixels. Determines the size of structures
            removed by the opening operation.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import MorphologicalOpening
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Remove small noise regions (salt-and-pepper speckles)
        >>> opener = MorphologicalOpening(shape='disk', radius=3)
        >>> cleaned = opener.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], cleaned.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], cleaned.rgb[:])
        >>> assert np.array_equal(image.gray[:], cleaned.gray[:])
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
                artifacts. 'diamond' for specialized cases with reduced diagonal
                connectivity.
            radius (int): Footprint radius in pixels. Determines the size threshold
                below which foreground objects are removed. Must be at least 1.
                Recommended: 1-5 for typical colony plates; larger values remove
                more noise but risk losing small colonies.
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
        """Apply morphological opening to enh_gray."""
        # Get footprint
        footprint = self._make_footprint(self.shape, self.radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation (binary opening needs thresholding)
        threshold = np.mean(enh)
        binary = enh > threshold
        result = binary_opening(binary, structure=footprint)

        # Convert back to grayscale and write
        image.enh_gray[:] = (result * 255).astype(enh.dtype)

        return image
