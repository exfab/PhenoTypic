from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd

from phenotypic.abc_ import MeasureFeatures


class MeasureTemplate(MeasureFeatures):

    def __init__(self, a, b):
        self.a, self.b = a, b

    @staticmethod  # for MeasureFeatures objects, _operate should be static methods with the parameter names matching the class attribute names needed for the calculation
    def _operate(self, image: Image, a, b) -> pd.DataFrame:
        """
        for MeasureFeatures objects, _operate should be static methods with the parameter names
        matching the class attribute names needed for the calculation.

        In this example, self.a and self.b are the exact same as the parameters a and b.
        """
        raise NotImplementedError
