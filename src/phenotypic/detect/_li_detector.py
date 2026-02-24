from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.filters import threshold_li
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class LiDetector(ThresholdDetector):
    """Li's minimum cross-entropy threshold detector for colony segmentation.

    LiDetector applies Li's iterative minimum cross-entropy thresholding method,
    which minimizes the information loss between original and thresholded images.
    This method assumes Gaussian distribution of intensities and performs well on
    low-contrast or noisy images where Otsu's bimodal assumption may not hold.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from Li thresholding.

    Raises:
        ValueError: If threshold computation fails.

    **Use cases**

    - **Low-contrast imaging:** Works better than Otsu on images with low signal-to-noise
      or subtle intensity differences between colonies and background.
    - **Non-bimodal histograms:** Handles images where colony intensity distribution
      doesn't fit Otsu's bimodal assumption.
    - **Noisy backgrounds:** More robust to textured agar or scanner artifacts that
      create histogram irregularities.

    **Limitations**

    - Gaussian assumption. Assumes intensities follow Gaussian distribution; violates
      on highly skewed or multimodal distributions.
    - Slower than Otsu. Iterative refinement takes more computation time.
    - May under-segment variable intensity colonies. Less effective than Otsu when
      images are truly bimodal.
    - Parameter-free but less intuitive. Difficult to understand or debug threshold
      selection compared to Otsu's clear variance minimization objective.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic Li detection::

            from phenotypic import Image
            from phenotypic.detect import LiDetector

            plate = Image.imread("low_contrast_plate.jpg")
            detector = LiDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via Li")

        Pipeline with Li for noisy imaging::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import LiDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=2.0),
                LiDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
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
