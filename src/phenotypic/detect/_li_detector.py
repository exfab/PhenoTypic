from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_li
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class LiDetector(ThresholdDetector):
    """Detect colonies by minimising cross-entropy between the original and thresholded image.

    Iteratively refine a threshold that minimises the information loss
    (cross-entropy) between the original intensity distribution and the
    binarised result. Performs well on low-contrast or noisy plates where
    the histogram is not clearly bimodal and Otsu's variance-based
    assumption does not hold. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Args:
        ignore_zeros: Exclude zero-intensity pixels from threshold
            computation. Enable for plates with black borders or masked
            regions; disable only when zero is a meaningful intensity value.
            Default: True.

        ignore_borders: Remove colonies touching image edges via
            ``clear_border()``. Recommended for grid-based colony counting
            to eliminate partial colonies at plate boundaries. Default: True.

    Returns:
        Image: Input image with ``objmask`` set to binary mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If threshold computation fails (e.g., degenerate
            histogram with insufficient intensity variation).

    Best For:
        * Low-contrast plates where colony and background intensities
          overlap significantly.
        * Noisy or textured agar backgrounds that create histogram
          irregularities breaking bimodal assumptions.
        * Images where the intensity distribution is unimodal or only
          weakly bimodal.

    Consider Also:
        * :class:`OtsuDetector` when the histogram is clearly bimodal and
          a fast single-pass threshold suffices.
        * :class:`YenDetector` for a correlation-based alternative that
          handles skewed histograms.
        * :class:`HysteresisDetector` when colony brightness varies across
          the plate and a single threshold under-segments faint regions.

    References:
        [1] C. H. Li and C. K. Lee, "Minimum cross entropy thresholding,"
        *Pattern Recognit.*, vol. 26, no. 4, pp. 617--625, 1993.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(self, ignore_zeros: bool = False, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using Li's threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`detect_mat`). The binarization threshold is
        automatically determined using Li's iterative Minimum Cross Entropy method.
        The resulting binary mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `detect_mat`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.detect_mat[:]
        mask = image.detect_mat[:] >= threshold_li(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
LiDetector.apply.__doc__ = LiDetector._operate.__doc__
