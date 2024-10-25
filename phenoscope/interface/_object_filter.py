import numpy as np

from ._image_operation import ImageOperation
from .. import Image
from ..util.error_message import INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, OUTPUT_NOT_IMAGE_MSG


# <<Interface>>
class ObjectFilter(ImageOperation):
    def __init__(self):
        pass

    def filter(self, image: Image, inplace=False) -> Image:
        # Input Validation
        if image.object_map is None: raise ValueError("Image has no object map")
        input_img = np.copy(image.array)
        input_enhanced = np.copy(image.enhanced_array)

        # Operation
        if inplace:
            output = self._operate(image)
        else:
            output = self._operate(image.copy())

        # Integrity Check
        if not np.array_equal(image.array, input_img): raise ValueError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(image.enhanced_array, input_enhanced): raise ValueError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)

        if isinstance(output, Image) is False: raise RuntimeError(OUTPUT_NOT_IMAGE_MSG)
        return output

    def _operate(self, image: Image) -> Image:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
