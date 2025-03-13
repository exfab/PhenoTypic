from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING: from phenoscope import GridImage

import pandas as pd


from phenoscope.abstract import FeatureExtractor
from phenoscope.grid.abstract import GridOperation
from phenoscope.util.constants import GRID_SERIES_INPUT_IMAGE_ERROR_MSG, OUTPUT_NOT_TABLE_MSG


class GridFeatureExtractor(FeatureExtractor, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def measure(self, image: GridImage) -> pd.DataFrame:
        from phenoscope import GridImage
        if not isinstance(image, GridImage): raise ValueError(GRID_SERIES_INPUT_IMAGE_ERROR_MSG)
        output = super().measure(image)
        if not isinstance(output, pd.DataFrame): raise ValueError(OUTPUT_NOT_TABLE_MSG)
        return output