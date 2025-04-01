from ..abstract import MapModifier
from .. import Image

import numpy as np
from skimage.morphology import binary_opening


class MorphologyOpener(MapModifier):
    def __init__(self, footprint: np.ndarray = None):
        self.__footprint: np.ndarray = footprint

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = binary_opening(image.objmask[:], footprint=self.__footprint)
        return image
