from skimage.morphology import remove_small_objects

from ..abstract import MapModifier
from .. import Image


class SmallObjectRemovalModifier(MapModifier):
    """Removes small objects from an image"""
    def __init__(self, min_size=64):
        self.__min_size = min_size

    def _operate(self, image: Image) -> Image:
        image.omap[:] = remove_small_objects(image.omap[:], min_size=self.__min_size)
        return image
