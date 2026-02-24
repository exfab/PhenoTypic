from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
if TYPE_CHECKING:
    from phenotypic import Image

from skimage.filters import (
    apply_hysteresis_threshold,
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_triangle,
    threshold_yen,
)
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class HysteresisDetector(ThresholdDetector):
    """Hysteresis threshold detector for robust colony segmentation with dual thresholds.

    HysteresisDetector applies a two-threshold algorithm that seeds strong colony
    regions (high threshold) and expands via connectivity to include weaker regions
    (low threshold). This provides robust segmentation when colonies have variable
    intensity, background noise, or uneven illumination, especially when a single
    global threshold is insufficient.

    Args:
        low: Lower threshold. Either string method name ('otsu', 'triangle', 'li',
            'yen', 'isodata', 'mean', 'minimum') for automatic computation, or float
            for manual value (0-255 for 8-bit). Default 'mean'. Controls sensitivity;
            lower values include more faint colonies but increase false positives.

        high: Upper threshold. Same format as low. Default 'otsu'. Must be >= low
            after computation. Higher values are more conservative, detecting only
            bright colonies.

        ignore_zeros: If True (default), exclude zero pixels from automatic threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at boundaries.

    Attributes:
        low: The low threshold (method name or float value).
        high: The high threshold (method name or float value).
        ignore_zeros: Whether to exclude zero pixels.
        ignore_borders: Whether to remove edge-touching colonies.

    Returns:
        Image: Input image with objmask set to binary mask. True pixels represent
        colonies (including faint pixels connected to strong regions), False = background.

    Raises:
        ValueError: If high < low after computation or invalid method name provided.

    **Use cases**

    - **Variable colony intensity:** Colonies vary in brightness (small vs large, young
      vs mature growth). Hysteresis bridges fragments via connectivity.
    - **Noisy backgrounds:** Agar texture, dust, or scanner noise. Hysteresis rejects
      isolated noise while preserving connected weak regions (faint colonies).
    - **Uneven illumination:** Plates with vignetting or gradient backgrounds. More
      flexible than single-threshold methods.
    - **Touching colonies:** When colonies merge, two thresholds help distinguish
      boundaries based on intensity peaks.

    **Limitations**

    - Two thresholds to tune (more parameters than single-threshold methods). Test
      combinations on representative images.
    - Global method (no spatial adaptation). For severe vignetting, local thresholding
      may perform better.
    - Connectivity assumption: Isolated pixels above high threshold may not connect
      to weak regions. Ensure low is meaningfully lower than high.
    - Threshold order required: high >= low. If computed methods violate this, raises
      ValueError.
    - Fallback behavior: If low == high after computation, automatically performs
      simple threshold segmentation (mask = image >= threshold) instead of hysteresis.

    **Parameter effects on colony detection**

    - **low (float or str):** Controls expansion sensitivity. Method strings ('mean',
      'minimum') are conservative; numeric values control absolute cutoff. Lower
      values → more detected colonies, more noise.
    - **high (float or str):** Seeds strong regions. Higher values → fewer/larger
      colonies, lower false positives.
    - **ignore_zeros/ignore_borders:** Remove preprocessing artifacts (black borders,
      edge-touching partial colonies).

    Examples:
        Basic detection with automatic thresholds::

            from phenotypic import Image
            from phenotypic.detect import HysteresisDetector

            plate = Image.imread("agar_plate.jpg")
            detector = HysteresisDetector(low='mean', high='otsu')
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels")

        Pipeline with preprocessing::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import HysteresisDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                HysteresisDetector(low='mean', high='otsu', ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(
        self,
        low: Union[str, float] = "mean",
        high: Union[str, float] = "otsu",
        ignore_zeros: bool = True,
        ignore_borders: bool = True,
    ):
        self.low = low
        self.high = high
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Apply hysteresis thresholding to detect colonies.

        Computes low and high thresholds (automatically from method names or from
        manual float values), validates that high >= low, then applies
        apply_hysteresis_threshold() to identify regions that exceed the high
        threshold OR exceed the low threshold while connected to regions above
        the high threshold. If low == high, performs simple threshold segmentation
        instead.

        Args:
            image: The input image object. Must have ``detect_mat`` attribute
                (detection matrix for processing). Uses ``bit_depth`` to
                determine nbins for automatic threshold computation.

        Returns:
            Image: The input image with ``objmask`` attribute set to the binary
            mask (True = colony or connected weak region, False = background).

        Raises:
            ValueError: If high < low after computation, or if threshold_spec
                contains an invalid method name.
        """
        enh_matrix = image.detect_mat[:]

        # Prepare data for threshold computation (exclude zeros if requested)
        if self.ignore_zeros:
            thresh_data = enh_matrix[enh_matrix != 0]
        else:
            thresh_data = enh_matrix

        # Compute low threshold
        low_val = self._compute_threshold(self.low, thresh_data, image.bit_depth)

        # Compute high threshold
        high_val = self._compute_threshold(
            self.high, thresh_data, image.bit_depth
        )

        # Validate threshold order
        if high_val < low_val:
            raise ValueError(
                f"High threshold ({high_val:.2f}) must be >= low threshold "
                f"({low_val:.2f})"
            )

        # Apply thresholding (fallback to simple threshold if low == high)
        if high_val == low_val:
            # Simple threshold segmentation when thresholds are identical
            mask = enh_matrix >= low_val
        else:
            # Hysteresis thresholding with dual thresholds
            mask = apply_hysteresis_threshold(enh_matrix, low_val, high_val)
            # Ensure mask is boolean (apply_hysteresis_threshold returns int64)
            mask = mask.astype(bool)

        # Optionally clear borders
        mask = clear_border(mask) if self.ignore_borders else mask

        # Set objmask
        image.objmask = mask
        return image

    @staticmethod
    def _compute_threshold(
        threshold_spec: Union[str, float],
        data: np.ndarray,
        bit_depth: int,
    ) -> float:
        """Compute threshold value from specification.

        Args:
            threshold_spec: Either method name (str) or manual value (float).
                If string, must be one of: 'otsu', 'isodata', 'li', 'mean',
                'minimum', 'triangle', 'yen'.
            data: Image data (usually with zeros excluded) for automatic
                threshold computation.
            bit_depth: Image bit depth (8 or 16) to determine nbins.

        Returns:
            Computed threshold value as float.

        Raises:
            ValueError: If threshold_spec is a string but not a valid method name.
        """
        if isinstance(threshold_spec, (int, float)):
            # Manual threshold value
            return float(threshold_spec)

        # Map method names to scikit-image functions
        method_map = {
            "otsu": threshold_otsu,
            "isodata": threshold_isodata,
            "li": threshold_li,
            "mean": threshold_mean,
            "minimum": threshold_minimum,
            "triangle": threshold_triangle,
            "yen": threshold_yen,
        }

        # Methods that accept nbins parameter
        # Note: li does NOT accept nbins despite being histogram-based
        methods_with_nbins = {"otsu", "isodata", "triangle", "yen"}

        method_name = threshold_spec.lower()
        if method_name not in method_map:
            raise ValueError(
                f"Unknown threshold method '{threshold_spec}'. "
                f"Valid methods: {list(method_map.keys())}"
            )

        threshold_func = method_map[method_name]

        # Compute threshold with or without nbins depending on method
        if method_name in methods_with_nbins:
            nbins = 2 ** int(bit_depth)
            return float(threshold_func(data, nbins=nbins))
        else:
            # Methods like 'mean' and 'minimum' don't accept nbins
            return float(threshold_func(data))


# Set the docstring so that it appears in the sphinx documentation
HysteresisDetector.apply.__doc__ = HysteresisDetector._operate.__doc__
