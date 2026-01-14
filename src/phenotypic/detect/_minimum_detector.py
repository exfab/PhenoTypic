from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.filters import threshold_minimum
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class MinimumDetector(ThresholdDetector):
    """Minimum valley threshold detector for bimodal histogram segmentation.

    MinimumDetector finds the minimum value (histogram valley) between two distinct
    peaks, using this valley as the threshold. This method works well on clearly
    bimodal histograms where background and foreground form two separate peaks with
    a distinct valley between them, common in high-contrast imaging.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from minimum thresholding.

    Raises:
        ValueError: If threshold computation fails (e.g., no clear bimodal distribution).

    **Use cases**

    - **Clearly bimodal histograms:** Distinct foreground and background peaks with
      clear valley between them.
    - **High-contrast imaging:** Strong separation between colonies and background
      intensities.
    - **Well-defined regions:** When image has only two intensity clusters.

    **Limitations**

    - Requires clear bimodality. Fails on unimodal or multimodal histograms or when
      background/foreground intensities overlap.
    - Sensitive to noise near peak/valley. Spurious peaks from noise can create false
      valleys.
    - No multi-peak support. Can't handle images with more than two distinct intensity
      groups.
    - May fail on low-contrast images. When peaks are shallow or valleys are gradual,
      method produces poor results.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic minimum detection on bimodal histogram::

            from phenotypic import Image
            from phenotypic.detect import MinimumDetector

            plate = Image.imread("high_contrast_plate.jpg")
            detector = MinimumDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via minimum")

        Pipeline with minimum thresholding::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import MinimumDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                MinimumDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the Minimum threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`enh_gray`). The binarization threshold is
        automatically determined using Minimum method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `enh_gray`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.enh_gray[:]
        nbins = 2**image.bit_depth
        mask = image.enh_gray[:] >= threshold_minimum(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MinimumDetector.apply.__doc__ = MinimumDetector._operate.__doc__
