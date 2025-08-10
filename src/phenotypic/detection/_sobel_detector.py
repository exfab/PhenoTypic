from skimage.filters import sobel

from ..abstract import ObjectDetector
from .. import Image


class SobelDetector(ObjectDetector):
    """
    A specialized object detection class using Sobel filtering on image enhancement matrices for edge detection.

    This class focuses on processing an image's enhancement matrix using a Sobel filter
    to detect objects. It updates the object's mask property with the Sobel-filtered
    result while leaving other image properties unchanged. It inherits from ObjectDetector,
    providing a specific implementation of object detection suitable for edge enhancement
    and analysis.

    Attributes:
        mask (type): Represents the object mask computed after applying the Sobel
            filter. Its type and structure depend on the parent ObjectDetector class
            implementation.
    """

    def _operate(self, image: Image) -> Image:
        """
        Performs an operation on the input image by applying a Sobel filter to its enhancement
        matrix and storing the result in the object's mask. The processed image is then returned.

        Args:
            image (Image): The input image object which contains an enhancement matrix and
                an object mask to be updated.

        Returns:
            Image: The modified image after applying the Sobel filter on the enhancement
            matrix.
        """
        enh_matrix = image.enh_matrix[:]
        image.objmask[:] = sobel(image=enh_matrix)
        return image
