from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import GridImage

from phenoscope.abstract import MapModifier
from phenoscope.grid.abstract import GridOperation
from phenoscope.util.exceptions_ import GridImageInputError, InterfaceError


class GridMapModifier(MapModifier, GridOperation):
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        from phenoscope import GridImage
        if not isinstance(image, GridImage): raise GridImageInputError()
        output = super().apply(image=image, inplace=inplace)
        return output

    def _operate(self, image: GridImage) -> GridImage:
        raise InterfaceError()
