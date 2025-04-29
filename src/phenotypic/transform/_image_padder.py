import numpy as np

from .. import Image
from ..abstract import ImageCorrector
from ..util.constants_ import IMAGE_FORMATS


class ImagePadder(ImageCorrector):
    def __init__(self, pad_size: int = 50, mode='edge', **kwargs):
        if type(pad_size) == int:
            self.__pad_size: int = pad_size
        else:
            raise ValueError('pad_size must be an integer.')

        if mode in ['edge', 'maximum', 'linear_ramp', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap', 'empty']:
            self.__mode: str = mode
        else:
            raise ValueError('Input mode not supported.')

        self.__kwargs = kwargs

    def _operate(self, image: Image) -> Image:
        if image.imformat in IMAGE_FORMATS.MATRIX_FORMATS:
            image.set_image(np.pad(array=image.matrix[:],
                                   pad_width=((self.__pad_size, self.__pad_size), (self.__pad_size, self.__pad_size)),
                                   mode=self.__mode, **self.__kwargs
                                   ), imformat=image.imformat
                            )
        else:
            image.set_image(np.pad(
                array=image.array[:],
                pad_width=((self.__pad_size, self.__pad_size), (self.__pad_size, self.__pad_size), (0, 0)),
                mode=self.__mode, **self.__kwargs
            ), imformat=image.imformat
            )

        return image
