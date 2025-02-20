from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenoscope.grid import GriddedImage

from phenoscope.interface import ImageTransformer
from phenoscope.grid.interface import GridOperation
from phenoscope.util.constants import GRID_SERIES_INPUT_IMAGE_ERROR_MSG, OUTPUT_NOT_GRIDDED_IMAGE_MSG


class GridTransformer(ImageTransformer, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def morph(self, image: GriddedImage, inplace=False) -> GriddedImage:
        from phenoscope.grid import GriddedImage

        if not isinstance(image, GriddedImage): raise ValueError(GRID_SERIES_INPUT_IMAGE_ERROR_MSG)
        output = super().morph(image, inplace=inplace)
        if not isinstance(output, GriddedImage): raise ValueError(OUTPUT_NOT_GRIDDED_IMAGE_MSG)
        return output

