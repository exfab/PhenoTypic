from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border

from ..abc_ import ThresholdDetector


class OtsuDetector(ThresholdDetector):
    """Global Otsu threshold detector for balanced colony segmentation.

    OtsuDetector applies Otsu's method, which automatically computes a global
    threshold that minimizes within-class variance. This assumes a bimodal histogram
    (two intensity peaks) separating background from foreground. Simple, robust,
    and parameter-free, making it ideal for standardized imaging conditions and
    baseline detection.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks. Prevents
            threshold from skewing toward zero.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from Otsu thresholding.

    Raises:
        ValueError: If threshold computation fails (e.g., all pixels same value).

    **Use cases**

    - **Balanced intensity distributions:** Colony and background peaks roughly
      equal in height. Otsu optimally separates two-peak histograms.
    - **Standardized imaging:** Fixed lighting, camera, and agar type produce
      reproducible histograms where Otsu works reliably.
    - **Baseline comparison:** Testing automatic methods; Otsu is standard baseline.
    - **Simple, parameter-free detection:** When minimal tuning is desired.

    **Limitations**

    - Bimodal histogram assumption. Fails on unimodal or multimodal distributions.
      Poorly lit colonies or complex backgrounds violate assumption.
    - Variable colony intensity. Otsu sensitive to brightness variations (young vs
      mature colonies). May over/under-segment depending on growth stage distribution.
    - Uneven illumination. Global threshold doesn't adapt to vignetting or spatial
      gradients. Use local methods (RankOtsuDetector) for uneven lighting.
    - No spatial awareness. Treats all image regions identically. Unsuitable for
      plates with varying background intensity.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for images with black borders. Disable only if zero
      is meaningful intensity.
    - **ignore_borders:** Recommended for grid-based analysis. Disable if edges
      contain valid data.

    Examples:
        Basic Otsu detection::

            from phenotypic import Image
            from phenotypic.detect import OtsuDetector

            plate = Image.imread("plate.jpg")
            detector = OtsuDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via Otsu")

        Pipeline with Otsu for batch processing::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import OtsuDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                OtsuDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
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
