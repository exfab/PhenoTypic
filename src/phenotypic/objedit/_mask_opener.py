from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from phenotypic.ABC_ import MapModifier

import numpy as np
from skimage.morphology import binary_opening, diamond


class MaskOpener(MapModifier):
    """
    Perform morphological binary opening operation on an image's object mask.

    MaskOpener class is a specialized MapModifier that applies a morphological
    binary opening operation to an image's object mask. The operation utilizes
    a structuring element, referred to as the footprint, which can be customized
    or automatically determined based on the input image's shape. This class
    is designed to enhance object detection by refining and smoothing binary
    masks in an image.

    Attributes:
        footprint (np.ndarray | int | None): The structuring element for the
            binary opening operation. Can be a custom numpy array, an integer
            specifying the radius for a diamond-shaped footprint, or None to
            auto-generate the footprint based on the size of the input image.
    """

    def __init__(self, footprint: Literal["auto"] | np.ndarray | int | None = None):
        super().__init__()
        self.footprint: Literal["auto"] | np.ndarray | int | None = footprint

    def _operate(self, image: Image) -> Image:
        if self.footprint == 'auto':
            footprint = diamond(radius=max(3, round(np.min(image.shape)*0.005)))
        elif isinstance(self.footprint, np.ndarray):
            footprint = self.footprint
        elif isinstance(self.footprint, (int, float)):
            footprint = diamond(radius=int(self.footprint))
        elif not self.footprint:
            footprint = self.footprint
        else:
            raise AttributeError('Invalid footprint type')

        image.objmask[:] = binary_opening(image.objmask[:], footprint=footprint)
        return image
