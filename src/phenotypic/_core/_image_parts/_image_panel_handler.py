from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .accessors import PanelAccessor
from ._image_plot_handler import ImagePlotHandler

if TYPE_CHECKING:
    from phenotypic import Image


class ImagePanelHandler(ImagePlotHandler):
    def __init__(
            self,
            arr: np.ndarray | Image | None = None,
            name: str | None = None,
            bit_depth: int | None = None,
    ):
        super().__init__(arr=arr, name=name, bit_depth=bit_depth)
        self._accessors.panel = PanelAccessor(self)

    @property
    def panel(self) -> PanelAccessor:
        return self._accessors.panel
