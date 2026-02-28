from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class YenDetector(ThresholdDetector):
    """Yen's correlation coefficient threshold detector for colony segmentation.

    YenDetector applies Yen's thresholding method, which maximizes the correlation
    coefficient between the original and binarized images. This method performs well
    on high-contrast images and handles skewed histograms better than Otsu in some
    scenarios, offering a middle ground between Otsu and Li methods.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from Yen thresholding.

    Raises:
        ValueError: If threshold computation fails.

    **Use cases**

    - **High-contrast imagery:** Excels on images with clear separation and distinct
      intensity peaks between colonies and background.
    - **Skewed histograms:** Better than Otsu when foreground/background peaks unequal.
    - **Hybrid approach:** Falls between Otsu (variance minimization) and Li (entropy).
      Good when uncertain which method fits image distribution.

    **Limitations**

    - Less commonly used. Limited validation compared to Otsu/Li; behavior on edge
      cases less well-characterized.
    - May over-segment on low-contrast images. Correlation maximization assumes
      detectable signal.
    - Slower than Otsu. More computation than fast variance-minimization method.
    - Limited literature on failure modes. Harder to understand when/why it fails
      compared to Otsu or Li.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic Yen detection::

            from phenotypic import Image
            from phenotypic.detect import YenDetector

            plate = Image.imread("high_contrast_plate.jpg")
            detector = YenDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via Yen")

        Pipeline with Yen::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import YenDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                YenDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
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
