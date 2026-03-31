from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_mean
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class MeanDetector(ThresholdDetector):
    """Detect colonies by thresholding at the mean image intensity.

    Use the arithmetic mean of all pixel intensities as the threshold,
    classifying pixels above the mean as colony foreground. This
    parameter-free baseline is fast and deterministic, making it useful for
    quick sanity checks or as a fallback when histogram-adaptive methods
    produce unexpected results. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Quick baseline detection requiring no parameter tuning.
        * Sanity-checking preprocessing steps before applying a more
          specialised detector.
        * Plates where colony and background areas are roughly equal in
          size.
        * Debugging pipelines where a simple, predictable threshold is
          needed.

    Consider Also:
        * :class:`OtsuDetector` for a statistically optimal threshold that
          adapts to the histogram shape.
        * :class:`TriangleDetector` when colonies are sparse and the
          histogram is strongly skewed toward background.
        * :class:`IsodataDetector` for an iterative refinement that
          converges beyond the simple mean.

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
        ValueError: If threshold computation fails (e.g., all pixels share
            the same intensity value).

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the Mean threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`detect_mat`). The binarization threshold is
        automatically determined using Mean method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `detect_mat`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.detect_mat[:]
        mask = image.detect_mat[:] >= threshold_mean(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MeanDetector.apply.__doc__ = MeanDetector._operate.__doc__
