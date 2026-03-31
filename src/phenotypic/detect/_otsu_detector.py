from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class OtsuDetector(ThresholdDetector):
    """Detect colonies by global Otsu thresholding on bimodal plate histograms.

    Automatically compute a single intensity threshold that minimises
    within-class variance, separating colony foreground from agar background.
    The resulting binary mask cleanly segments plates whose histograms are
    bimodal (one peak for background, one for colonies). For a comparison of
    all available detection strategies see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Plates imaged under standardised lighting where the intensity
          histogram has two well-separated peaks.
        * Quick baseline detection requiring no parameter tuning.
        * High-throughput screens with uniform agar colour and colony density.
        * Comparing automatic methods -- Otsu is the standard reference
          threshold.
        * Clean plates with minimal dust, scratches, or condensation.

    Consider Also:
        * :class:`TriangleDetector` when colonies occupy a small fraction of
          the plate and the histogram is skewed.
        * :class:`HysteresisDetector` when colony intensity varies (e.g.,
          young versus mature growth) and a single threshold under-segments.
        * :class:`WatershedDetector` when touching colonies must be split
          into individually labelled objects.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from
            threshold computation. Enable for plates with black borders or
            masked regions; disable only when zero is a meaningful intensity
            value. Typical range: True for most workflows.

        ignore_borders: If True (default), remove colonies touching image
            edges via ``clear_border()``. Recommended for grid-based colony
            counting to eliminate partial colonies at plate boundaries.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        produced by Otsu thresholding and ``objmap`` set to labeled
        connected components.

    Raises:
        ValueError: If threshold computation fails (e.g., all pixels share
            the same intensity value, producing a degenerate histogram).

    References:
        [1] N. Otsu, "A threshold selection method from gray-level
        histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1,
        pp. 62--66, 1979.

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
