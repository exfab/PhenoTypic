from ..interface import ImageTransformer
from .. import Image

from skimage.color import rgb2gray


class RGB2Grayscale(ImageTransformer):
    def _operate(self, image: Image) -> Image:
        if image.array.ndim!=3 and image.array.shape[2]!=3: raise ValueError('Image must be RGB to be converted to grayscale')
        image.array = rgb2gray(image.array)
        return image
