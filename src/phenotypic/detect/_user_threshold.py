from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class UserThreshold(ThresholdDetector):
    """Detect colonies by applying a user-specified intensity threshold.

    Apply a fixed intensity cutoff to the plate detection matrix, producing a
    binary colony mask without any automatic threshold computation. This
    gives explicit control over detection sensitivity and is the preferred
    approach when empirical testing has identified an optimal threshold for a
    specific imaging setup. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Args:
        threshold: Intensity cutoff for binary segmentation. Pixels with
            intensity >= *threshold* become colony (True), others become
            background (False). For 8-bit images the valid range is
            0--255; for 16-bit images 0--65535; for float images 0.0--1.0.
            Higher values are more conservative (fewer colonies detected);
            lower values are more sensitive (more colonies, more noise).
            Default 0.5. Start by inspecting the image histogram to find
            the valley between background and colony peaks.

        ignore_zeros: If True (default), exclude zero-intensity pixels from
            processing. Enable for plates with black borders or masked
            regions.

        ignore_borders: If True (default), remove colonies touching image
            edges via ``clear_border()``. Recommended for grid-based colony
            counting.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        (True = colony, False = background). ``objmap`` is not modified.

    Raises:
        ValueError: If *threshold* is negative.

    Best For:
        * Standardised imaging setups where the optimal threshold has been
          determined empirically and remains stable across plates.
        * Overriding automatic methods (Otsu, triangle, etc.) that
          consistently over- or under-segment on a particular plate type.
        * High-contrast plates where colonies are uniformly bright or dark
          relative to background and a single cutoff cleanly separates
          foreground from background.
        * Reproducibility-critical workflows where a fixed numeric threshold
          eliminates variability introduced by automatic selection.

    Consider Also:
        * :class:`OtsuDetector` when an automatic, parameter-free threshold
          is preferred and the histogram is bimodal.
        * :class:`HysteresisDetector` when colony intensity varies across
          the plate and a single threshold cannot capture all colonies.
        * :class:`TriangleDetector` when colonies are sparse and the
          histogram is skewed toward background.

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
            threshold: float = 0.5,
            ignore_zeros: bool = False,
            ignore_borders: bool = True
    ):
        self.threshold = threshold
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Apply manual binary thresholding to the detection matrix image.

        This function modifies the input image by applying a user-specified threshold
        to its enhanced matrix (``detect_mat``). Pixels with intensity >= threshold
        become foreground (True in the binary mask), pixels < threshold become
        background (False). The resulting binary mask is stored in the image's
        ``objmask`` attribute.

        Args:
            image: The input image object. Must have an ``detect_mat`` attribute
                (detection matrix for processing). Optionally uses
                ``bit_depth`` to validate threshold range.

        Returns:
            Image: The input image with its ``objmask`` attribute updated to the
                computed binary mask.

        Raises:
            ValueError: If threshold is negative or exceeds the image's intensity
                range (inferred from bit depth if available).
        """
        enh_matrix = image.detect_mat[:]

        # Validate threshold range
        if self.threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {self.threshold}")

        # Apply threshold
        mask = enh_matrix >= self.threshold

        # Optionally clear borders
        mask = clear_border(mask) if self.ignore_borders else mask

        # Set objmask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
UserThreshold.apply.__doc__ = UserThreshold._operate.__doc__
