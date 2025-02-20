import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.constants import INTERFACE_ERROR_MSG, C_ImageOperation, C_ImageFormats


# <<Interface>>
class MapModifier(ImageOperation):
    def modify(self, image: Image, inplace: bool = False) -> Image:
        imcopy = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if output.schema in C_ImageFormats.MATRIX_FORMATS:
            if not np.array_equal(imcopy.array[:], output.array[:]): raise C_ImageOperation.ComponentChangeError(
                component='array', operation=self.__class__.__name__
            )

        if not np.array_equal(imcopy.matrix[:], output.matrix[:]): raise C_ImageOperation.ComponentChangeError(
            component='matrix', operation=self.__class__.__name__
        )

        if not np.array_equal(imcopy.det_matrix[:], output.det_matrix[:]): raise C_ImageOperation.ComponentChangeError(
            component='det_matrix', operation=self.__class__.__name__
        )

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
