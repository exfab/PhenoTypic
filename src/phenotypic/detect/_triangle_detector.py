from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import threshold_triangle

from ..abc_ import ThresholdDetector


class TriangleDetector(ThresholdDetector):
    """Detects triangles in an image using a thresholding method.

    This class inherits from ThresholdDetector and is specifically designed to
    detect triangles through a thresholding algorithm applied to the image's
    enhance gray. The threshold is calculated using the triangle algorithm,
    and the result modifies the image's object mask.

    Methods:
        apply: Applies triangle thresholding to the enhance gray of the
            image and updates the object mask accordingly.
    """

    def _operate(self, image: Image) -> Image:
        """
        Applies a thresholding operation on the enhanced gray of an image using
        the triangle method.

        Thresholding is performed by comparing each element in the enhanced gray
        to the computed triangular threshold, setting the corresponding other_image in
        the output mask (`omask`) to True if the condition is satisfied.

        Args:
            image (Image): The arr image object containing an enhanced gray
                (`enh_gray`) which will be processed to generate an output mask.

        Returns:
            Image: The modified image object with an updated output mask (`omask`).
        """
        nbins = 2 ** image.bit_depth
        image.objmask[:] = image.enh_gray[:] >= threshold_triangle(image.enh_gray[:], nbins=nbins)
        return image


# Set the docstring so that it appears in the sphinx documentation
TriangleDetector.apply.__doc__ = TriangleDetector._operate.__doc__
