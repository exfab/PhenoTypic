import numpy as np
from ._image_operation import ImageOperation
from ..util.constants import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG
from .. import Image


# <<Interface>>
class ObjectDetector(ImageOperation):
    """ObjectDetectors are for detecting objects in an image. They change the image object mask and map."""
    def __init__(self):
        pass

    def apply(self, image: Image, inplace: bool = False) -> Image:
        imcopy = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        # Post Operation Checks
        if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(imcopy.enh_matrix[:], output.enh_matrix[:]): raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
