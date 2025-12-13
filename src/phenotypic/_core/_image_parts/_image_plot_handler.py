from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np

from .accessors import PlotAccessor
from ._image_objects_handler import ImageObjectsHandler


class ImagePlotHandler(ImageObjectsHandler):
    def __init__(
            self,
            arr: np.ndarray | Image | None = None,
            name: str | None = None,
            bit_depth: int | None = None,
    ):
        super().__init__(arr=arr, name=name, bit_depth=bit_depth)
        self._accessors.plot = PlotAccessor(self)

    @property
    def plot(self) -> PlotAccessor:
        return self._accessors.plot
