from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from scipy.ndimage import binary_erosion
import numpy as np
from ..abc_ import ImageEnhancer


class MorphologicalErosion(ImageEnhancer):
    """
    Morphological erosion for shrinking bright regions and removing weak colonies.

    Performs binary erosion to shrink foreground objects (colonies) and remove small
    weak regions. This is useful for eliminating faint noise and fine boundary
    artifacts, leaving only robust, well-defined colonies.

    Use cases (agar plates):
    - Remove weak or faint colonies that may be noise or low-viability organisms.
    - Eliminate thin boundary artifacts created by thresholding; only robust,
      thick-boundary colonies survive.
    - Reduce false positives by requiring colonies to be sufficiently well-defined
      (resistant to erosion).
    - Separate touching colonies by eroding enough to break weak connections.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic erosion that
      equally shrinks colonies in all directions. 'square' can prefer orthogonal
      shrinkage. 'diamond' reduces diagonal erosion.
    - radius: Erosion radius in pixels. Determines how much colonies are shrunk.
      Larger radius removes more faint features but also shrinks real colonies.
      Choose carefully: too large will eliminate small but viable colonies.

    Caveats:
    - Erosion shrinks all bright regions; small colonies may be eliminated entirely.
    - Eroding too much can distort colony morphology and make size measurements
      unreliable.
    - After erosion, colonies are smaller, which can affect downstream area-based
      measurements.
    - Erosion works on binary (thresholded) data; the quality of the binary image
      is critical.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int): Erosion radius in pixels. Larger values shrink colonies more.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import MorphologicalErosion
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Erode to remove weak colonies and noise
        >>> eroder = MorphologicalErosion(shape='disk', radius=2)
        >>> eroded = eroder.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], eroded.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], eroded.rgb[:])
        >>> assert np.array_equal(image.gray[:], eroded.gray[:])
    """

    def __init__(
        self,
        shape: Literal["disk", "square", "diamond"] = "disk",
        radius: int = 2,
    ):
        """
        Parameters:
            shape (str): Footprint shape. 'disk' (default) for isotropic erosion.
                'square' for grid-aligned patterns. 'diamond' for specialized cases
                with reduced diagonal erosion.
            radius (int): Erosion radius in pixels. Larger values shrink colonies
                more, but risk eliminating small colonies entirely. Must be at
                least 1. Recommended: 1-3 for typical colony plates. Start with
                small radius and increase cautiously.
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
        """Apply morphological erosion to enh_gray."""
        # Get footprint
        footprint = self._make_footprint(self.shape, self.radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation (binary erosion needs thresholding)
        threshold = np.mean(enh)
        binary = enh > threshold
        result = binary_erosion(binary, structure=footprint)

        # Convert back to grayscale and write
        image.enh_gray[:] = (result * 255).astype(enh.dtype)

        return image
