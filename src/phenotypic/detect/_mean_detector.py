from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.filters import threshold_mean
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class MeanDetector(ThresholdDetector):
    """Mean intensity threshold detector for conservative colony segmentation.

    MeanDetector applies the simplest thresholding approach: using the mean
    (average) intensity of the image as the threshold. Pixels above mean are
    foreground (colonies), below mean are background. This extremely simple,
    fast, parameter-free method serves as a baseline or fallback when more
    sophisticated methods fail.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from mean thresholding.

    Raises:
        ValueError: If threshold computation fails.

    **Use cases**

    - **Quick baseline detection:** Simple, fast method for initial assessment.
    - **Parameter-free baseline:** No tuning needed; useful when method choice uncertain.
    - **Fallback method:** When sophisticated methods fail, mean provides baseline result.
    - **Debugging:** Test if image preprocessing is working at basic level.

    **Limitations**

    - Too simplistic for most real images. Assumes equal foreground/background areas,
      rarely true for agar plates.
    - Sensitive to outliers. Noise or bright artifacts skew mean intensity significantly.
    - Unbalanced histograms. Fails if colonies occupy much smaller/larger fraction
      than background.
    - No adaptation to image content. Ignores histogram shape, distribution, or
      statistical properties.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic mean detection as baseline::

            from phenotypic import Image
            from phenotypic.detect import MeanDetector

            plate = Image.imread("plate.jpg")
            detector = MeanDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via mean")

        Pipeline with mean as fallback::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import MeanDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                MeanDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the Mean threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`enh_gray`). The binarization threshold is
        automatically determined using Mean method. The resulting binary
        mask is stored in the image's `objmask` attribute.

        Args:
            image (Image): The arr image object. It must have an `enh_gray`
                attribute, which is used as the basis for creating the binary mask.

        Returns:
            Image: The arr image object with its `objmask` attribute updated
                to the computed binary mask other_image.
        """
        enh_matrix = image.enh_gray[:]
        mask = image.enh_gray[:] >= threshold_mean(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
MeanDetector.apply.__doc__ = MeanDetector._operate.__doc__
