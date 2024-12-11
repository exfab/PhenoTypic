from skimage.exposure import equalize_adapthist

from .. import Image
from ..interface import ImagePreprocessor


class CLAHE(ImagePreprocessor):
    def __init__(self, kernel_size: int = None):
        self.__kernel_size: int = kernel_size

    def _operate(self, image: Image) -> Image:
        if self.__kernel_size is None:
            image.enhanced_matrix = equalize_adapthist(
                    image=image.enhanced_matrix,
                    kernel_size=int(min(image.matrix.shape[:1]) * (1.0 / 15.0))
            )
            return image
        else:
            image.enhanced_matrix = equalize_adapthist(
                    image=image.enhanced_matrix,
                    kernel_size=self.__kernel_size
            )
            return image
