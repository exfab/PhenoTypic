from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenotypic import GridImage

from phenotypic.abstract import ImageCorrector
from phenotypic.abstract import GridOperation
from phenotypic.util.exceptions_ import GridImageInputError, OutputValueError


class GridCorrector(ImageCorrector, GridOperation):
    def __init__(self, nrows: int = 8, ncols: int = 12):
        self.nrows = nrows
        self.ncols = ncols

    def apply(self, image: GridImage, inplace=False) -> GridImage:
        from phenotypic import GridImage

        if not isinstance(image, GridImage): raise GridImageInputError
        output = super().apply(image, inplace=inplace)
        if not isinstance(output, GridImage): raise OutputValueError("GridImage")
        return output
