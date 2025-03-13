import numpy as np
from ._image_operation import ImageOperation
from ..util.constants import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, C_ImageOperation, C_ImageFormats
from .. import Image


class ImagePreprocessor(ImageOperation):
    def __init__(self):
        pass

    def apply(self, image: Image, inplace: bool = False) -> Image:
        try:
            # Make a copy for post checking
            imcopy = image.copy()

            if inplace:
                output = self._operate(image)
            else:
                output = self._operate(image.copy())

            if image.schema not in C_ImageFormats.MATRIX_FORMATS:
                if not np.array_equal(output.array[:], imcopy.array[:]):
                    raise C_ImageOperation.IntegrityError(component='array', operation=self.__class__.__name__)

            if not np.array_equal(output.matrix[:], imcopy.matrix[:]):
                raise C_ImageOperation.IntegrityError(component='matrix', operation=self.__class__.__name__)

            return output
        except Exception as e:
            raise C_ImageOperation.OperationError(operation=self.__class__.__name__,
                                                  image_name=image.name,
                                                  err_type=type(e),
                                                  message=str(e)
                                                  )

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
