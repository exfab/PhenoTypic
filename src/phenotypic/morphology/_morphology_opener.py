from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from ..abstract import MapModifier

import numpy as np
from skimage.morphology import binary_opening, diamond


class MaskOpener(MapModifier):
    """Performs morphological opening on an image's object mask.

    This class extends `MapModifier` and applies a morphological opening operation
    to an image object's binary mask. Morphological opening is a common image
    processing technique used to remove noise or small artifacts from a binary
    mask. The operation is performed using a structuring element (footprint),
    which can either be provided or dynamically generated based on the size of
    the input image.

    Attributes:
        footprint (np.ndarray | None): A numpy array representing the structuring
            element (footprint) used for the morphological opening operation. If
            None, a default footprint is calculated based on the size of the input
            image.
    """

    def __init__(self, footprint: np.ndarray | None = None):
        super().__init__()
        self.footprint: np.ndarray = footprint

    def _operate(self, image: Image) -> Image:
        footprint = self.footprint or diamond(radius=max(3, round(np.min(image.shape)*0.005)))
        image.objmask[:] = binary_opening(image.objmask[:], footprint=footprint)
        return image
