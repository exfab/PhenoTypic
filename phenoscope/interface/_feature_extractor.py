import numpy as np
import pandas as pd

from ._image_operation import ImageOperation
from ..util.constants import (INTERFACE_ERROR_MSG, ARRAY_CHANGE_ERROR_MSG, ENHANCED_ARRAY_CHANGE_ERROR_MSG, MASK_CHANGE_ERROR_MSG,
                              MAP_CHANGE_ERROR_MSG)
from .. import Image


# <<Interface>>
class FeatureExtractor(ImageOperation):
    """
    A FeatureExtractor is an interface object intended to essentially perform calculations on the values within detected objects of
    the image array. The __init__ constructor & _operate method is meant to be the only parts overloaded in inherited classes. This is so
    that the main extract method call can contain all the necessary type validation and output validation checks to streamline development.
    """

    def extract(self, image: Image) -> pd.DataFrame:
        imcopy: Image = image.copy()

        measurement = self._operate(image)

        # TODO: Fix checks
        # if not np.array_equal(imcopy.matrix[:], image.matrix[:]): raise ValueError(ARRAY_CHANGE_ERROR_MSG)
        # if not np.array_equal(imcopy.det_matrix[:], image.det_matrix[:]): raise ValueError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)
        # if not np.array_equal(imcopy.obj_mask[:], image.obj_mask[:]): raise ValueError(MASK_CHANGE_ERROR_MSG)
        # if not np.array_equal(imcopy.obj_map[:], image.obj_map[:]): raise ValueError(MAP_CHANGE_ERROR_MSG)

        return measurement

    def _operate(self, image: Image) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
