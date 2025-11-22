from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage
from phenotypic.abc_ import ImageOperation
from abc import ABC


class GridOperation(ImageOperation, ABC):

    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        return super().apply(image=image, inplace=inplace)
