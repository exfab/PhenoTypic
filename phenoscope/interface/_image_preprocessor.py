import numpy as np
from ._image_operation import ImageOperation
from ..util.constants import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, C_ImageOperation, C_ImageFormats
from .. import Image


class ImagePreprocessor(ImageOperation):
    def __init__(self):
        pass

    def preprocess(self, image: Image, inplace: bool = False) -> Image:

        # Make a copy for post checking
        imcopy = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if image.schema not in C_ImageFormats.MATRIX_FORMATS:
            if not np.array_equal(output.array[:], imcopy.array[:]):
                raise C_ImageOperation.ComponentChangeError(component='array', operation = self.__class__.__name__)

        if not np.array_equal(output.matrix[:], imcopy.matrix[:]):
            raise C_ImageOperation.ComponentChangeError(component='matrix', operation = self.__class__.__name__)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
