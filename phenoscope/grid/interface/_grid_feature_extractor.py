from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenoscope.grid import GriddedImage

from phenoscope.interface import FeatureExtractor
from phenoscope.grid.interface import GridOperation
from phenoscope.util.constants import GRID_SERIES_INPUT_IMAGE_ERROR_MSG, OUTPUT_NOT_TABLE_MSG


class GridFeatureExtractor(FeatureExtractor, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def extract(self, image: GriddedImage) -> pd.DataFrame:
        from phenoscope.grid import GriddedImage
        if not isinstance(image, GriddedImage): raise ValueError(GRID_SERIES_INPUT_IMAGE_ERROR_MSG)
        output = super().extract(image)
        if not isinstance(output, pd.DataFrame): raise ValueError(OUTPUT_NOT_TABLE_MSG)
        return output