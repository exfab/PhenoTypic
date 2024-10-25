import numpy as np

from .. import Image
from ..interface import NoisePreprocessor

from skimage.filters import median


class MedianFilter(NoisePreprocessor):
    def __init__(self, mode='nearest', cval: float = 0.0):
        if mode in ['nearest', 'reflect', 'constant', 'mirror', 'wrap']:
            self._mode = mode
            self._cval = cval
        else:
            raise ValueError('mode must be one of "nearest","reflect","constant","mirror","wrap"')

    def _operate(self, image: Image) -> Image:
        image.enhanced_array = median(image=image.enhanced_array, behavior='ndimage', mode=self._mode, cval=self._cval)
        return image
