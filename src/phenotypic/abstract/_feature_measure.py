from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd

from ._docstring_metaclass import MeasureDocstringMeta
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import OperationFailedError, InterfaceError


# <<Interface>>
class FeatureMeasure:
    """
    A FeatureExtractor is an abstract object intended to calculate measurements on the values within detected objects of
    the image array. The __init__ constructor & _operate method are meant to be the only parts overloaded in inherited classes. This is so
    that the main measure method call can contain all the necessary type validation and output validation checks to streamline development.
    """

    def measure(self, image: Image) -> pd.DataFrame:
        try:
            imcopy: Image = image.copy()

            measurement = self._operate(image)

            # TODO: Fix checks
            # if not np.array_equal(imcopy.matrix[:], image.matrix[:]): raise ValueError(ARRAY_CHANGE_ERROR_MSG)
            # if not np.array_equal(imcopy.enh_matrix[:], image.enh_matrix[:]): raise ValueError(ENHANCED_ARRAY_CHANGE_ERROR_MSG)
            # if not np.array_equal(imcopy.omask[:], image.omask[:]): raise ValueError(MASK_CHANGE_ERROR_MSG)
            # if not np.array_equal(imcopy.omap[:], image.omap[:]): raise ValueError(MAP_CHANGE_ERROR_MSG)

            return measurement
        except Exception as e:
            raise OperationFailedError(operation=self.__class__.__name__,
                                       image_name=image.name,
                                       err_type=type(e),
                                       message=str(e)
                                       )

    def _operate(self, image: Image) -> pd.DataFrame:
        raise InterfaceError
