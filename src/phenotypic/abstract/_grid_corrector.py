from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenotypic import GridImage

from phenotypic.abstract import ImageCorrector
from phenotypic.abstract import GridOperation
from phenotypic.util.exceptions_ import GridImageInputError, OutputValueError


class GridCorrector(ImageCorrector, GridOperation):
    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage): raise GridImageInputError
        output = super().apply(image, inplace=inplace)
        if not isinstance(output, GridImage): raise OutputValueError("GridImage")
        return output

