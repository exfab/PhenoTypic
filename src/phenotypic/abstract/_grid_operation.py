from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage
from phenotypic.abstract import ImageOperation


class GridOperation(ImageOperation):
    def __init__(self, n_rows: int = 8, n_cols: int = 12):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        return super().apply(image=image, inplace=inplace)
