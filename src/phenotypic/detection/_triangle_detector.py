from skimage.filters import threshold_triangle

from ..abstract import ThresholdDetector
from .. import Image


class TriangleDetector(ThresholdDetector):
    """Detects triangles in an image using a thresholding method.

    This class inherits from ThresholdDetector and is specifically designed to
    detect triangles through a thresholding algorithm applied to the image's
    enhancement matrix. The threshold is calculated using the triangle algorithm,
    and the result modifies the image's object mask.

    Methods:
        apply: Applies triangle thresholding to the enhancement matrix of the
            image and updates the object mask accordingly.
    """
    def _operate(self, image: Image) -> Image:
        """
        Applies a thresholding operation on the enhanced matrix of an image using
        the triangle method.

        Thresholding is performed by comparing each element in the enhanced matrix
        to the computed triangular threshold, setting the corresponding value in
        the output mask (`omask`) to True if the condition is satisfied.

        Args:
            image (Image): The input image object containing an enhanced matrix
                (`enh_matrix`) which will be processed to generate an output mask.

        Returns:
            Image: The modified image object with an updated output mask (`omask`).
        """
        image.objmask[:] = image.enh_matrix[:] >= threshold_triangle(image.enh_matrix[:])
        return image

# Set the docstring so that it appears in the sphinx documentation
TriangleDetector.apply.__doc__ = TriangleDetector._operate.__doc__
