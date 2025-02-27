from typing import Optional, Union, Type

import numpy as np
from skimage.color import rgb2hsv

from ._image_components import HSVHandler
from ._image_handler import ImageHandler, Image
from ..util.constants import C_ImageFormats, C_ImageHandler


class ImageHsv(ImageHandler):
    """Adds HSV support for the color measurement module"""

    def __init__(self, input_image: Optional[Union[np.ndarray, Type[Image]]] = None, input_schema=None):
        super().__init__(input_image=input_image, input_schema=input_schema)
        self.__hsv_handler = HSVHandler(self)

    @property
    def _hsv(self):
        """Returns the hsv array of the current image. This can become computationally expensive so implementation may be changed in future."""
        if self.schema in C_ImageFormats.MATRIX_FORMATS:
            raise AttributeError('Grayscale images cannot be directly converted to hsv. Convert to RGB first')
        else:
            match self.schema:
                case C_ImageFormats.RGB:
                    return rgb2hsv(self.array[:])
                case C_ImageFormats.BGR:
                    return rgb2hsv(self._convert_bgr_to_rgb(self.array[:]))
                case _:
                    raise ValueError(f'Unsupported schema {self.schema} for HSV conversion')

    @property
    def hsv(self):
        return self.__hsv_handler

    @hsv.setter
    def hsv(self, value):
        raise C_ImageHandler.IllegalAssignmentError('hsv')
