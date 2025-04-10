from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenotypic import GridImage

import pandas as pd


from phenotypic.abstract import FeatureMeasure
from phenotypic.abstract import GridOperation
from phenotypic.util.exceptions_ import GridImageInputError, OutputValueError


class GridFeatureMeasure(FeatureMeasure, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenotypic import GridImage
        if not isinstance(image, GridImage): raise GridImageInputError()
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame): raise OutputValueError("pandas.DataFrame")
        return output