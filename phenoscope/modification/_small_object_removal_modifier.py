from skimage.morphology import remove_small_objects

from ..interface import ObjectModifier
from .. import Image


class SmallObjectRemovalModifier(ObjectModifier):
    def __init__(self, min_size=64):
        self.__min_size = min_size

    def _operate(self, image: Image) -> Image:
        obj_map = image.object_map
        remove_small_objects(image.object_map, min_size=self.__min_size, out=obj_map)
        image.object_map = obj_map
        return image
