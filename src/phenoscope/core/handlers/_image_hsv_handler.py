from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

from typing import Optional, Union
import numpy as np
from skimage.color import rgb2hsv
from os import PathLike

from ..accessors import HsvAccessor
from ._image_handler import ImageHandler
from phenoscope.util.constants_ import IMAGE_FORMATS
from phenoscope.util.exceptions_ import IllegalAssignmentError


class ImageHsvHandler(ImageHandler):
    """Adds HSV format support for the color measurement module."""

    def __init__(self, input_image: Optional[Union[np.ndarray, Image, PathLike]] = None, input_schema:str=None):
        super().__init__(input_image=input_image, input_schema=input_schema)
        self.__hsv_handler = HsvAccessor(self)

    @property
    def _hsv(self)->np.ndarray:
        """Returns the hsv array dynamically of the current image.

        This can become computationally expensive so implementation may be changed in the future.

        Returns:
            np.ndarray: The hsv array of the current image.
        """
        if self.schema in IMAGE_FORMATS.MATRIX_FORMATS:
            raise AttributeError('Grayscale images cannot be directly converted to hsv. Convert to RGB first')
        else:
            match self.schema:
                case IMAGE_FORMATS.RGB:
                    return rgb2hsv(self.array[:])
                case IMAGE_FORMATS.BGR:
                    return rgb2hsv(self._convert_bgr_to_rgb(self.array[:]))
                case _:
                    raise ValueError(f'Unsupported schema {self.schema} for HSV conversion')

    @property
    def hsv(self)->HsvAccessor:
        """Returns the HSV accessor.

        This property returns an instance of the HsvAccessor associated with the
        current object, allowing access to HSV (hue, saturation, value) related
        functionalities controlled by this handler.

        Returns:
            HsvAccessor: The instance of the HSV accessor.
        """
        return self.__hsv_handler

    @hsv.setter
    def hsv(self, value):
        raise IllegalAssignmentError('hsv')
