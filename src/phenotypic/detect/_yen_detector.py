from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class YenDetector(ThresholdDetector):
    """Detect colonies by maximising the correlation between the original and binarised image.

    Compute a threshold that maximises the correlation coefficient between
    the original intensity image and its binarised version. Handles skewed
    histograms better than Otsu in some scenarios, offering a middle ground
    between variance-based (Otsu) and entropy-based (Li) criteria. For a
    full comparison see :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * High-contrast plates with clear intensity separation between
          colonies and agar.
        * Images with skewed histograms where one class is larger than
          the other.
        * Exploratory analysis when unsure whether a variance-based or
          entropy-based criterion fits the data better.

    Consider Also:
        * :class:`OtsuDetector` for a faster variance-based threshold when
          the histogram is balanced and bimodal.
        * :class:`LiDetector` when the histogram is low-contrast or
          unimodal and entropy-based separation is more appropriate.
        * :class:`TriangleDetector` when colonies are very sparse and the
          histogram is strongly background-dominated.

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

    References:
        [1] J. C. Yen, F. J. Chang, and S. Chang, "A new criterion for
        automatic multilevel thresholding," *IEEE Trans. Image Process.*,
        vol. 4, no. 3, pp. 370--378, 1995.

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
        """Binarizes the given image gray using the Yen threshold method.

        This function modifies the arr image by applying a binary mask to
        its detection matrix (`detect_mat`). The binarization threshold is
        automatically determined using Yen's method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `detect_mat`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.detect_mat[:]
        nbins = 2**image.bit_depth
        mask = image.detect_mat[:] >= threshold_yen(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
YenDetector.apply.__doc__ = YenDetector._operate.__doc__
