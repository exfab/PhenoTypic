from ..interface import MapModifier
from .. import Image

import numpy as np
from skimage.morphology import binary_opening


class MorphologyOpener(MapModifier):
    def __init__(self, footprint: np.ndarray = None):
        self.__footprint: np.ndarray = footprint

    def _operate(self, image: Image) -> Image:
        image.obj_mask[:] = binary_opening(image.obj_mask[:], footprint=self.__footprint)
        return image
