from skimage.filters import laplace
from typing import Optional
import numpy as np

from ..abstract import ImagePreprocessor
from .. import Image


class LaplacePreprocessor(ImagePreprocessor):
    def __init__(self, ksize: Optional[int] = 3, mask: Optional[np.ndarray] = None):
        self._ksize: Optional[np.ndarray] = ksize
        self._mask:Optional[np.ndarray] = mask

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = laplace(
                image=image.enh_matrix[:],
                ksize=self._ksize,
                mask=self._mask,
        )
        return image
