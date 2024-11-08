import numpy as np

from ._image_operation import ImageOperation

from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, MISSING_MASK_ERROR_MSG


# <<Interface>>
class MorphologyMorpher(ImageOperation):
    def morph(self, image: Image, inplace: bool = False) -> Image:
        if image.object_mask is None: raise ValueError(MISSING_MASK_ERROR_MSG)
        input_arr = image.array
        input_enhanced_arr = image.enhanced_array

        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        if not np.array_equal(input_arr, output.array): raise AttributeError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(input_enhanced_arr, output.enhanced_array): raise AttributeError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
