from ..interface import ImageTransformer
from .. import Image

from skimage.color import rgb2gray


class RGB2Grayscale(ImageTransformer):
    def _operate(self, image: Image) -> Image:
        if image.matrix.ndim!=3 and image.matrix.shape[2]!=3: raise ValueError('Image must be RGB to be converted to grayscale')
        image.matrix = rgb2gray(image.matrix)
        return image
