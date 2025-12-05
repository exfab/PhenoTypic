from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from scipy.ndimage import morphological_gradient
import numpy as np
from ..abc_ import ImageEnhancer


class MorphologicalGradient(ImageEnhancer):
    """
    Morphological gradient for edge enhancement and boundary extraction.

    Computes the morphological gradient (difference between dilation and erosion)
    to highlight object boundaries and edges. This is useful for edge-based
    detection, improving colony boundary definition, and reducing background noise
    while preserving edge sharpness on agar plates.

    Use cases (agar plates):
    - Enhance colony boundaries for improved edge-based detection methods.
    - Improve contrast between colonies and background by amplifying gradient
      information at edges.
    - Prepare images for edge detection algorithms (Canny, Sobel) by sharpening
      colony boundaries.
    - Reduce flat background regions while emphasizing colony contours and texture.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic edge
      enhancement. 'square' emphasizes orthogonal edges. 'diamond' emphasizes
      certain diagonal directions.
    - radius: Determines the scale of edge features extracted. Larger radius
      captures broader boundary transitions; smaller radius captures fine detail.
      Typical range: 1-3 pixels.

    Caveats:
    - Gradient emphasizes all edges, including noise artifacts and surface texture.
      Combine with smoothing (GaussianBlur) before gradient for cleaner results.
    - Gradient output is typically lower intensity than the original image; may
      require subsequent contrast adjustment.
    - Large radius values can blur fine colony boundary details.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int): Radius of the structuring element for gradient computation.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import MorphologicalGradient
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Compute morphological gradient for edge enhancement
        >>> grad_enhancer = MorphologicalGradient(shape='disk', radius=2)
        >>> edge_enhanced = grad_enhancer.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], edge_enhanced.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], edge_enhanced.rgb[:])
        >>> assert np.array_equal(image.gray[:], edge_enhanced.gray[:])
    """

    def __init__(
        self,
        shape: Literal["disk", "square", "diamond"] = "disk",
        radius: int = 2,
    ):
        """
        Parameters:
            shape (str): Footprint shape. 'disk' (default) for isotropic edge
                extraction. 'square' for orthogonal edge emphasis. 'diamond' for
                specialized diagonal-sensitive cases.
            radius (int): Structuring element radius in pixels. Controls the scale
                of edge features extracted. Must be at least 1. Recommended: 1-3
                for typical colony boundaries. Larger values extract coarser
                boundaries; smaller values capture finer details.
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
        """Apply morphological gradient to enh_gray."""
        # Get footprint
        footprint = self._make_footprint(self.shape, self.radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation (works on grayscale directly, no thresholding needed)
        result = morphological_gradient(enh.astype(float), structure=footprint)

        # Convert back to original dtype
        image.enh_gray[:] = result.astype(enh.dtype)

        return image
