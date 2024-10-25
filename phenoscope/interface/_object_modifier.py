import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, MISSING_MAP_ERROR_MSG


# <<Interface>>
class ObjectModifier(ImageOperation):
    def modify(self, image: Image, inplace: bool = False) -> Image:
        if image.object_map is None: raise ValueError(MISSING_MAP_ERROR_MSG)
        input_image = image.copy()

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if not np.array_equal(input_image.array, output.array): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(input_image.enhanced_array, output.enhanced_array): raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
