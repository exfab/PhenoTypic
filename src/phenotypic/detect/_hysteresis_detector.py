from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
if TYPE_CHECKING:
    from phenotypic._core._image import Image

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
    """Detect colonies by dual-threshold hysteresis, bridging bright cores to faint edges.

    Seed strong colony regions that exceed the high threshold and expand each
    seed via pixel connectivity to include neighbouring pixels above the low
    threshold. This two-pass approach captures colonies whose intensity varies
    from bright centres to faint margins -- a common pattern across growth
    stages and under uneven illumination. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Plates where colony brightness varies (e.g., young versus mature
          growth, or centre-to-edge intensity gradients within a colony).
        * Noisy agar backgrounds where isolated noise pixels sit above a
          single threshold but lack connectivity to true colony regions.
        * Moderate vignetting or lighting gradients that cause a single
          global threshold to over- or under-segment parts of the plate.
        * Mixed-species plates where different organisms produce colonies of
          different intensities on the same agar.

    Consider Also:
        * :class:`OtsuDetector` when colony and background peaks are balanced
          and a single threshold suffices.
        * :class:`WatershedDetector` when touching colonies must be split
          into individually labelled regions.
        * :class:`CannyDetector` when colonies are best delineated by edge
          contrast rather than intensity.

    Args:
        low: Lower threshold controlling expansion sensitivity. Accepts a
            method name (``'otsu'``, ``'triangle'``, ``'li'``, ``'yen'``,
            ``'isodata'``, ``'mean'``, ``'minimum'``) for automatic
            computation, or a float for a manual value (0--255 for 8-bit,
            0--65535 for 16-bit). Default ``'mean'``. Lower values include
            more faint colony pixels but increase false positives. Typical
            tuning: start with ``'mean'`` and switch to a numeric value if
            automatic methods are too aggressive or too conservative.

        high: Upper threshold seeding strong colony regions. Same format as
            *low*. Default ``'otsu'``. Must be >= *low* after computation.
            Higher values restrict seeds to the brightest colony pixels,
            producing fewer but higher-confidence detections.

        ignore_zeros: If True (default), exclude zero-intensity pixels from
            automatic threshold computation. Enable for plates with black
            borders or masked regions.

        ignore_borders: If True (default), remove colonies touching image
            edges via ``clear_border()``. Recommended for grid-based colony
            counting.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        where True pixels are colony foreground (including faint pixels
        connected to strong seed regions). ``objmap`` is not modified.

    Raises:
        ValueError: If the computed high threshold is less than the computed
            low threshold, or if an unrecognised threshold method name is
            provided.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(
        self,
        low: Union[str, float] = "mean",
        high: Union[str, float] = "otsu",
        ignore_zeros: bool = False,
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
