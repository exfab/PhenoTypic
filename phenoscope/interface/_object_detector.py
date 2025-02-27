import numpy as np
from ._image_operation import ImageOperation
from ..util.constants import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG
from .. import Image


# <<Interface>>
class ObjectDetector(ImageOperation):
    def __init__(self):
        pass

    def detect(self, image: Image, inplace: bool = False) -> Image:
        imcopy = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        # Post Operation Checks
        if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(imcopy.det_matrix[:], output.det_matrix[:]): raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
