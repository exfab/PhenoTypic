from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic import Image

from skimage.morphology import black_tophat
import numpy as np
from ..abc_ import ImageEnhancer


class BlackTophatEnhancer(ImageEnhancer):
    """
    Black top-hat transform to enhance small dark structures.

    Computes the black top-hat (closing minus original) to extract small dark
    regions and objects that stand out from lighter backgrounds. This is useful
    for detecting small dark colonies, spots, or features on lighter agar plates.

    Use cases (agar plates):
    - Enhance small dark colonies that may be difficult to detect against light
      agar backgrounds.
    - Extract dark spots, blemishes, or contamination features for downstream
      analysis.
    - Improve visibility of poorly pigmented or light-colored colonies by
      emphasizing local contrast.
    - Prepare low-contrast images for thresholding by boosting dark feature
      visibility.

    Tuning and effects:
    - shape: Footprint geometry. 'disk' (default) provides isotropic processing
      that preserves rounded colony shapes. 'square' aligns with pixel grid.
      'diamond' is specialized for cases where diagonal connections should be
      reduced.
    - radius: Sets the size of the closing operation. Larger radius captures
      larger dark features; smaller radius captures fine dark details. Choose
      based on typical dark feature size to extract.

    Caveats:
    - Works best on images where small dark features contrast with lighter
      backgrounds; may be ineffective on uniformly dark or inverted images.
    - Large radius values can over-smooth and merge nearby dark features.
    - Complements WhiteTophatEnhancer (for bright features); use appropriate
      tophat operation for your image contrast pattern.

    Attributes:
        shape (str): Footprint shape: 'disk', 'square', or 'diamond'.
        radius (int | None): Footprint radius in pixels; if None, a small default
            is derived from the image size.

    Examples:
        >>> from phenotypic import Image
        >>> from phenotypic.enhance import BlackTophatEnhancer
        >>> import numpy as np
        >>>
        >>> # Load an agar plate image
        >>> image = Image.from_image_path('agar_plate.jpg')
        >>>
        >>> # Enhance small dark features
        >>> tophat = BlackTophatEnhancer(shape='disk', radius=3)
        >>> enhanced = tophat.apply(image)
        >>>
        >>> # Verify enhanced grayscale was modified
        >>> assert not np.array_equal(image.enh_gray[:], enhanced.enh_gray[:])
        >>>
        >>> # Original RGB and grayscale remain unchanged
        >>> assert np.array_equal(image.rgb[:], enhanced.rgb[:])
        >>> assert np.array_equal(image.gray[:], enhanced.gray[:])
    """

    def __init__(
        self,
        shape: Literal["disk", "square", "diamond"] = "disk",
        radius: int | None = None,
    ):
        """
        Parameters:
            shape (str): Footprint geometry controlling how dark features are
                extracted. 'disk' (default) provides isotropic behavior; 'square'
                aligns with pixel grid; 'diamond' provides specialized connectivity.
            radius (int | None): Maximum dark-feature size (in pixels) to enhance.
                If None, a small default based on image dimensions is used
                (0.4% of smallest dimension). Smaller values extract finer dark
                details; larger values capture coarser dark structures.
        """
        # Validate shape
        if shape not in ["disk", "square", "diamond"]:
            raise ValueError(
                f"shape must be one of 'disk', 'square', 'diamond'; got '{shape}'"
            )
        self.shape = shape

        # Validate radius if provided
        if radius is not None and (not isinstance(radius, int) or radius < 1):
            raise ValueError(f"radius must be a positive integer or None; got {radius}")
        self.radius = radius

    def _operate(self, image: Image) -> Image:
        """Apply black top-hat transform to enh_gray."""
        # Get footprint radius
        radius = self._get_footprint_radius(image.enh_gray[:])
        footprint = self._get_footprint(radius)

        # Read enhanced grayscale
        enh = image.enh_gray[:]

        # Apply operation
        tophat_results = black_tophat(enh, footprint=footprint)

        # Write back to enhanced grayscale
        image.enh_gray[:] = tophat_results.astype(enh.dtype)

        return image

    def _get_footprint_radius(self, detection_matrix: np.ndarray) -> int:
        """Determine footprint radius, using default if not specified."""
        if self.radius is None:
            return int(np.min(detection_matrix.shape) * 0.004)
        else:
            return self.radius

    def _get_footprint(self, radius: int) -> np.ndarray:
        """Create footprint with specified shape."""
        return self._make_footprint(shape=self.shape, radius=radius)
