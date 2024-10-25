import numpy as np
import pandas as pd

from ._image_operation import ImageOperation
from ..util.error_message import (INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, MASK_CHANGE_ERROR_MSG,
                                  MAP_CHANGE_ERROR_MSG)
from .. import Image


# <<Interface>>
class FeatureExtractor(ImageOperation):
    def extract(self, image: Image, inplace:bool = False) -> pd.DataFrame:
        input_image: Image = image.copy()

        if inplace:
            measurement = self._operate(image)
        else:
            measurement = self._operate(image.copy())

        if not np.array_equal(input_image.array, image.array): raise ValueError(ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(input_image.enhanced_array, image.enhanced_array): raise ValueError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)
        if not np.array_equal(input_image.object_mask, image.object_mask): raise ValueError(MASK_CHANGE_ERROR_MSG)
        if not np.array_equal(input_image.object_map, image.object_map): raise ValueError(MAP_CHANGE_ERROR_MSG)

        return measurement

    def _operate(self, image: Image) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
