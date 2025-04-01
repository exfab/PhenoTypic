import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from .. import Image
from ..abstract import MapModifier
from ..util.funcs import is_binary_mask


class MorphologyFiller(MapModifier):
    def __init__(self, structure: Optional[np.ndarray] = None, origin: int = 0):
        if structure is not None:
            if not is_binary_mask(structure): raise ValueError('input object array must be a binary array')
        self._structure = structure
        self._origin = origin

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = binary_fill_holes(
            input=image.objmask[:],
            structure=self._structure,
            origin=self._origin
        )
        return image
