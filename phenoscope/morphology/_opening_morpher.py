from ..interface import MorphologyMorpher
from .. import Image

import numpy as np
from skimage.morphology import binary_opening


class OpeningMorpher(MorphologyMorpher):
    def __init__(self, footprint: np.ndarray = None):
        self.__footprint: np.ndarray = footprint

    def _operate(self, image: Image) -> Image:
        mask = image.object_mask
        image.object_mask = binary_opening(mask, footprint=self.__footprint)
        return image
