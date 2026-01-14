from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image
from skimage.filters import threshold_isodata
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class IsodataDetector(ThresholdDetector):
    """ISODATA iterative clustering threshold detector for colony segmentation.

    IsodataDetector applies ISODATA (Iterative Self-Organizing Data Analysis
    Technique), which iteratively partitions pixels into classes (foreground/background)
    by computing class means and refining threshold. This method converges toward
    an optimal threshold balancing foreground and background statistics.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from ISODATA thresholding.

    Raises:
        ValueError: If threshold computation fails.

    **Use cases**

    - **Balanced class distributions:** Works well when foreground/background pixels
      roughly balanced, with similar variance within each class.
    - **Iterative refinement:** Converges toward optimal separation through multiple
      iterations, robust to initialization.
    - **Medium-contrast images:** Intermediate difficulty; not as easy as Otsu but
      simpler than adaptive methods.

    **Limitations**

    - Computationally expensive. Iterative refinement takes more time than Otsu.
    - Unequal class sizes. Performs poorly when foreground occupies much smaller or
      larger fraction than background.
    - Parameter-free but less intuitive. Harder to understand or debug compared to
      variance-minimization objectives.
    - Convergence issues. Iteration may converge slowly or to suboptimal thresholds
      on some distributions.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic ISODATA detection::

            from phenotypic import Image
            from phenotypic.detect import IsodataDetector

            plate = Image.imread("plate.jpg")
            detector = IsodataDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via ISODATA")

        Pipeline with ISODATA::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import IsodataDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                IsodataDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(self, ignore_zeros: bool = True, ignore_borders: bool = True):
        self.ignore_zeros = ignore_zeros
        self.ignore_borders = ignore_borders

    def _operate(self, image: Image) -> Image:
        """Binarizes the given image matrix using the ISODATA threshold method.

        This function modifies the arr image by applying a binary mask to
        its enhanced matrix (`enh_gray`). The binarization threshold is
        automatically determined using ISODATA method. The resulting binary
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
        mask = image.enh_gray[:] >= threshold_isodata(
            enh_matrix[enh_matrix != 0] if self.ignore_zeros else enh_matrix,
            nbins=nbins,
        )
        mask = clear_border(mask) if self.ignore_borders else mask
        image.objmask = mask
        return image


# Set the docstring so that it appears in the sphinx documentation
IsodataDetector.apply.__doc__ = IsodataDetector._operate.__doc__
