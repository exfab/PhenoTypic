from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage

import pandas as pd


from phenotypic.abstract import MeasureFeatures
from phenotypic.abstract import GridOperation
from phenotypic.util.exceptions_ import GridImageInputError, OutputValueError
from phenotypic.util.funcs_ import validate_measure_integrity

class GridMeasureFeatures(MeasureFeatures):
    def __init__(self, nrows: int=8, ncols: int=12):
        self.nrows = nrows
        self.ncols = ncols

    @validate_measure_integrity()
    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenotypic import GridImage
        if not isinstance(image, GridImage): raise GridImageInputError()
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame): raise OutputValueError("pandas.DataFrame")
        return output