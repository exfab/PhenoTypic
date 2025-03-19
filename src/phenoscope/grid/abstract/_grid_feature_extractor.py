from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenoscope import GridImage

import pandas as pd


from phenoscope.abstract import FeatureExtractor
from phenoscope.grid.abstract import GridOperation
from phenoscope.util.exceptions_ import GridImageInputError, OutputValueError


class GridFeatureExtractor(FeatureExtractor, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenoscope import GridImage
        if not isinstance(image, GridImage): raise GridImageInputError()
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame): raise OutputValueError("pandas.DataFrame")
        return output