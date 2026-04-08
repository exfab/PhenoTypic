from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_minimum
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class MinimumDetector(ThresholdDetector):
    """Detect colonies by finding the valley between two histogram peaks.

    Locate the intensity minimum (valley) between the two dominant peaks of
    the image histogram and threshold at that point. This works well when
    colonies and background form two clearly separated intensity populations,
    as the valley provides a natural separation boundary. For a full
    comparison see :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * High-contrast plates where colony and background intensities form
          two distinct, well-separated histogram peaks.
        * Standardised imaging setups producing consistently bimodal
          histograms across plates.
        * Images where the intensity gap between colonies and agar is wide
          and the valley is unambiguous.

    Consider Also:
        * :class:`OtsuDetector` when the histogram is bimodal but peaks
          are broad or partially overlapping.
        * :class:`LiDetector` when the histogram is unimodal or weakly
          bimodal and a cross-entropy criterion is more appropriate.
        * :class:`HysteresisDetector` when colony brightness varies and a
          single valley-based threshold under-segments faint regions.

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
        ValueError: If the histogram has no clear bimodal distribution and
            no valley can be found.

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
        """Binarizes the given image matrix using the Minimum threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`detect_mat`). The binarization threshold is
        automatically determined using Minimum method. The resulting binary
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
        mask = image.detect_mat[:] >= threshold_minimum(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MinimumDetector.apply.__doc__ = MinimumDetector._operate.__doc__
