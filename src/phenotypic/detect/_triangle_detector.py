from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import threshold_triangle

from ..abc_ import ThresholdDetector


class TriangleDetector(ThresholdDetector):
    """Triangle threshold detector for background-dominant colony segmentation.

    TriangleDetector applies the triangle (or iso-data) thresholding method,
    which finds the threshold at the base of a histogram triangle formed by
    the minimum value, maximum value, and peak of the histogram. This method
    works well on skewed histograms where background dominates and is effective
    for foreground regions comprising a small fraction of the image.

    Args:
        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

        ignore_borders: If True (default), remove colonies touching image edges via
            clear_border(). Eliminates partial colonies at plate boundaries.

    Attributes:
        ignore_zeros, ignore_borders

    Returns:
        Image: Input image with objmask set to binary mask from triangle thresholding.

    Raises:
        ValueError: If threshold computation fails.

    **Use cases**

    - **Background-dominant images:** Colonies occupy small fraction; background
      dominates histogram. Triangle method biased toward finding background valley.
    - **Skewed histograms:** Better than Otsu when foreground peak small relative
      to background tail.
    - **Sparse detection:** Sparse colonies on large background where global methods
      struggle.

    **Limitations**

    - Assumes skewed histogram. Poor on balanced histograms where colonies and
      background peaks similar.
    - Background assumption. Fails if colonies comprise majority of image or
      background is sparse.
    - Less robust than Otsu. Triangle method less widely-tested; behavior on edge
      cases less predictable.
    - Computation based on histogram extrema. Outliers (noise at extremes) can
      influence threshold.

    **Parameter effects on colony detection**

    - **ignore_zeros:** Enable for black borders. Disable only if zero is meaningful.
    - **ignore_borders:** Recommended for grid analysis.

    Examples:
        Basic triangle detection for sparse imaging::

            from phenotypic import Image
            from phenotypic.detect import TriangleDetector

            plate = Image.imread("sparse_colonies_plate.jpg")
            detector = TriangleDetector()
            detected = detector.apply(plate)
            mask = detected.objmask[:]
            print(f"Detected {mask.sum()} colony pixels via triangle")

        Pipeline with triangle thresholding::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import TriangleDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                TriangleDetector(ignore_zeros=True, ignore_borders=True)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def _operate(self, image: Image) -> Image:
        """
        Applies a thresholding operation on the detection matrix of an image using
        the triangle method.

        Thresholding is performed by comparing each element in the detection matrix
        to the computed triangular threshold, setting the corresponding other_image in
        the output mask (`omask`) to True if the condition is satisfied.

        Args:
            image (Image): The arr image object containing a detection matrix
                (`detect_mat`) which will be processed to generate an output mask.

        Returns:
            Image: The modified image object with an updated output mask (`omask`).
        """
        nbins = 2**image.bit_depth
        image.objmask[:] = image.detect_mat[:] >= threshold_triangle(
            image.detect_mat[:], nbins=nbins
        )
        return image


# Set the docstring so that it appears in the sphinx documentation
TriangleDetector.apply.__doc__ = TriangleDetector._operate.__doc__
