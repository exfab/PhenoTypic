import numpy as np
from typing import Optional, Union

from phenotypic import Image
from phenotypic.abstract import MapModifier


class BorderObjectRemover(MapModifier):
    """Removes objects at the border of the image within a certain distance.

    """
    def __init__(self, border_size: Optional[Union[int, float]] = None):
        self.__edge_size = border_size

    def _operate(self, image: Image) -> Image:
        if self.__edge_size is None:
            edge_size = int(np.min(image.shape) * 0.05)
        elif type(self.__edge_size) == float:
            edge_size = int(np.min(image.shape) * self.__edge_size)
        elif type(self.__edge_size) == int:
            edge_size = self.__edge_size
        else:
            raise TypeError('Invalid edge size. Should be int, float, or None to use default edge size.')

        obj_map = image.objmap[:]
        edges = [obj_map[:edge_size - 1, :].ravel(),
                 obj_map[-edge_size:, :].ravel(),
                 obj_map[:, :edge_size - 1].ravel(),
                 obj_map[:, -edge_size:].ravel()
                 ]
        edge_labels = np.unique(np.concatenate(edges))
        for label in edge_labels:
            obj_map[obj_map == label] = 0

        image.objmap = obj_map
        return image