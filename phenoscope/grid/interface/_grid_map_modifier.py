from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope.grid import GriddedImage

from phenoscope.interface import MapModifier
from phenoscope.grid.interface import GridOperation
from phenoscope.util.error_message import GRID_SERIES_INPUT_IMAGE_ERROR_MSG, INTERFACE_ERROR_MSG


class GridMapModifier(MapModifier, GridOperation):
    def modify(self, image: GriddedImage, inplace: bool = False) -> GriddedImage:
        from phenoscope.grid import GriddedImage
        if not isinstance(image, GriddedImage): raise ValueError(GRID_SERIES_INPUT_IMAGE_ERROR_MSG)
        output = super().modify(image=image, inplace=inplace)
        return output

    def _operate(self, image: GriddedImage, inplace: bool = False) -> GriddedImage:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
