from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd

from ._docstring_metaclass import MeasureDocstringMeta
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import OperationFailedError, InterfaceError, OperationIntegrityError


# <<Interface>>
class FeatureMeasure:
    """
    A FeatureExtractor is an abstract object intended to calculate measurements on the values within detected objects of
    the image array. The __init__ constructor & _operate method is meant to be the only parts overloaded in inherited classes. This is so
    that the main measure method call can contain all the necessary type validation and output validation checks to streamline development.
    """

    def measure(self, image: Image) -> pd.DataFrame:
        try:

            return self._operate(image.copy())
        except Exception as e:
            raise OperationFailedError(operation=self.__class__.__name__,
                                       image_name=image.name,
                                       err_type=type(e),
                                       message=str(e)
                                       )

    @staticmethod
    def _operate(image: Image) -> pd.DataFrame:
        raise InterfaceError
