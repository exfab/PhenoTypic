import numpy as np
from skimage.exposure import equalize_adapthist

from .. import Image
from ..interface import ContrastPreprocessor


class CLAHE(ContrastPreprocessor):
    def __init__(self, kernel_size: int = None):
        self.__kernel_size: int = kernel_size

    def _operate(self, image: Image) -> Image:
        if self.__kernel_size is None:
            image.enhanced_array = equalize_adapthist(
                    image=image.enhanced_array,
                    kernel_size=int(min(image.array.shape[:1]) * (1.0 / 15.0))
            )
            return image
        else:
            image.enhanced_array = equalize_adapthist(
                    image=image.enhanced_array,
                    kernel_size=self.__kernel_size
            )
            return image
