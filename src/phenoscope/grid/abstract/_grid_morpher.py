from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenoscope import GridImage

from phenoscope.abstract import ImageTransformer
from phenoscope.grid.abstract import GridOperation
from phenoscope.util.exceptions_ import GridImageInputError, OutputValueError


class GridTransformer(ImageTransformer, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenoscope import GridImage

        if not isinstance(image, GridImage): raise GridImageInputError
        output = super().apply(image, inplace=inplace)
        if not isinstance(output, GridImage): raise OutputValueError("GridImage")
        return output

