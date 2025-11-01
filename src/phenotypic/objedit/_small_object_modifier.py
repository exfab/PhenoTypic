from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.morphology import remove_small_objects

from ..ABC_ import MapModifier


class SmallObjectRemover(MapModifier):
    """Removes small objects from an image"""

    def __init__(self, min_size=64):
        self.__min_size = min_size

    def _operate(self, image: Image) -> Image:
        image.objmap[:] = remove_small_objects(image.objmap[:], min_size=self.__min_size)
        return image
