import numpy as np
from ._image_operation import ImageOperation
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG
from .. import Image


class ImagePreprocessor(ImageOperation):
    def __init__(self):
        pass

    def preprocess(self, image: Image, inplace: bool = False) -> Image:

        # Make a copy for post checking
        input_image: Image = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if not np.array_equal(input_image.array, output.array): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
