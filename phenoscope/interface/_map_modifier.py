import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, MISSING_MAP_ERROR_MSG


# <<Interface>>
class MapModifier(ImageOperation):
    def modify(self, image: Image, inplace: bool = False) -> Image:
        if image.object_map is None: raise ValueError(MISSING_MAP_ERROR_MSG)
        arr = image.matrix
        enhanced_arr = image.enhanced_matrix

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if not np.array_equal(arr, output.matrix): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(enhanced_arr, output.enhanced_matrix): raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
