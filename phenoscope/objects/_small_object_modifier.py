from skimage.morphology import remove_small_objects

from ..interface import MapModifier
from .. import Image


class SmallObjectRemovalModifier(MapModifier):
    """Removes small objects from an image"""
    def __init__(self, min_size=64):
        self.__min_size = min_size

    def _operate(self, image: Image) -> Image:
        image.obj_map[:] = remove_small_objects(image.obj_map[:], min_size=self.__min_size)
        return image
