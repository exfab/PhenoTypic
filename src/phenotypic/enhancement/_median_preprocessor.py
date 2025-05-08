from .. import Image
from ..abstract import ImageEnhancer

from skimage.filters import median


class MedianEnhancer(ImageEnhancer):
    """
    The MedianEnhancer class applies a median filter operation to an image's enhanced matrix using
    specified boundary conditions and a constant fill value if required.

    This preprocessor enhances image data by replacing each pixel value with the median of
    the neighboring pixels, where boundary behavior is determined by the `mode` parameter and
    `cval` as a constant fill value when applicable.

    Attributes:
        mode (str): Defines the behavior at image boundaries, where valid options are
            'nearest', 'reflect', 'constant', 'mirror', and 'wrap'.
        cval (float): The constant value filled in cells beyond the image boundary when
            'constant' mode is set.
    """
    def __init__(self, mode='nearest', cval: float = 0.0):
        if mode in ['nearest', 'reflect', 'constant', 'mirror', 'wrap']:
            self.mode = mode
            self.cval = cval
        else:
            raise ValueError('mode must be one of "nearest","reflect","constant","mirror","wrap"')

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = median(image=image.enh_matrix[:], behavior='ndimage', mode=self.mode, cval=self.cval)
        return image
