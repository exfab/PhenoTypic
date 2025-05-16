import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from .. import Image
from ..abstract import MapModifier
from ..util.funcs_ import is_binary_mask


class MaskFill(MapModifier):
    def __init__(self, structure: Optional[np.ndarray] = None, origin: int = 0):
        if structure is not None:
            if not is_binary_mask(structure): raise ValueError('input_image object array must be a binary array')
        self.structure = structure
        self.origin = origin

    @staticmethod
    def _operate(image: Image, structure, origin) -> Image:
        image.objmask[:] = binary_fill_holes(
            input=image.objmask[:],
            structure=structure,
            origin=origin
        )
        return image
