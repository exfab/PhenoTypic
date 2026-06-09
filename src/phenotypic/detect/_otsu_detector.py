from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class OtsuDetector(ThresholdDetector):
    """Detect colonies by minimising intra-class variance of the intensity histogram.

    Automatically compute a single global threshold that separates colony
    foreground from agar background by maximising the between-class intensity
    variance. The resulting binary mask cleanly segments plates whose histograms
    are bimodal. For a comparison of all available detection strategies see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Plates imaged under standardised lighting where the intensity
          histogram shows two well-separated peaks.
        - High-throughput screens with uniform agar colour and colony density.
        - Clean plates with minimal dust, scratches, or condensation.
        - Quick baseline detection requiring no parameter tuning.

    Consider Also:
        - :class:`TriangleDetector` when colonies occupy a small fraction of
          the plate and the histogram is strongly skewed toward background.
        - :class:`HysteresisDetector` when colony intensity varies across the
          plate and a single threshold under-segments faint regions.
        - :class:`WatershedDetector` when touching colonies must be split into
          individually labelled objects.

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
            histogram (e.g., all pixels share the same intensity value).

    References:
        [1] N. Otsu, "A threshold selection method from gray-level
        histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1,
        pp. 62--66, 1979.

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
        """Binarizes the given image matrix using the Otsu threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`detect_mat`). The binarization threshold is
        automatically determined using Otsu's method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `detect_mat`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.detect_mat[:]
        nbins = 2 ** int(image.bit_depth)
        mask = image.detect_mat[:] >= threshold_otsu(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
OtsuDetector.apply.__doc__ = OtsuDetector._operate.__doc__
