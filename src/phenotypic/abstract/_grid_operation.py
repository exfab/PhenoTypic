from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage
from phenotypic.abstract import ImageOperation


class GridOperation(ImageOperation):
    def __init__(self, nrows: int = 8, ncols: int = 12):
        self.nrows = nrows
        self.ncols = ncols

    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        return super().apply(image=image, inplace=inplace)
