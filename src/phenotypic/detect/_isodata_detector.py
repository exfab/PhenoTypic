from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_isodata
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class IsodataDetector(ThresholdDetector):
    """Detect colonies by iterative ISODATA clustering of the intensity histogram.

    Iteratively partition pixels into foreground and background classes by
    computing class means, then refine the threshold until the estimate
    converges. The resulting binary mask separates colony pixels from agar
    background and works best when both pixel populations have similar variance
    and roughly balanced counts. For a full comparison of detection strategies
    see :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Plates where colony and background pixel counts are roughly balanced.
        - Medium-contrast images where iterative refinement improves on a
          single-pass threshold.
        - Standardised imaging setups producing consistently symmetric
          histograms across plates.

    Consider Also:
        - :class:`OtsuDetector` for a faster single-pass threshold when the
          histogram is clearly bimodal.
        - :class:`LiDetector` when the histogram is skewed or noise dominates
          one side of the distribution.
        - :class:`HysteresisDetector` when colony brightness varies across the
          plate and a single threshold under-segments faint regions.

    Args:
        ignore_zeros: Exclude zero-intensity pixels from the histogram before
            computing the threshold. Enable for plates with black borders or
            masked regions. Default: False.
        ignore_borders: Remove colonies touching image edges via
            ``clear_border()``. Recommended for grid-based colony counting
            to eliminate partial colonies at plate boundaries. Default: True.

    Returns:
        Image: Input image with ``objmask`` set to the binary colony mask and
        ``objmap`` set to labeled connected components.

    Raises:
        ValueError: If threshold computation fails due to a degenerate
            histogram with insufficient intensity variation.

    References:
        [1] T. W. Ridler and S. Calvard, "Picture thresholding using an
        iterative selection method," *IEEE Trans. Syst., Man, Cybern.*,
        vol. 8, no. 8, pp. 630--632, 1978.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies` for a step-by-step
        tutorial on basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm` for guidance on
        selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared` for an in-depth
        comparison of all thresholding strategies.
    """

    ignore_zeros: bool = False
    ignore_borders: bool = True

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the ISODATA threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`detect_mat`). The binarization threshold is
        automatically determined using ISODATA method. The resulting binary
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
        mask = image.detect_mat[:] >= threshold_isodata(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
IsodataDetector.apply.__doc__ = IsodataDetector._operate.__doc__
