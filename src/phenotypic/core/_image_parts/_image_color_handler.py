from __future__ import annotations
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.color import rgb2hsv
from os import PathLike

from phenotypic.core._image_parts.accessors import HsvAccessor
from ._image_objects_handler import ImageObjectsHandler
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import IllegalAssignmentError


class ImageColorSpace(ImageObjectsHandler):
    """

    """

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | None = None,
                 imformat: str | None = None,
                 name: str | None = None):
        super().__init__(input_image=input_image, imformat=imformat, name=name)
        self._accessors.hsv = HsvAccessor(self)

    @property
    def _hsv(self) -> np.ndarray:
        """Returns the hsv array dynamically of the current image.

        This can become computationally expensive, so implementation may be changed in the future.

        Returns:
            np.ndarray: The hsv array of the current image.
        """
        if self.imformat.is_matrix():
            raise AttributeError('Grayscale images cannot be directly converted to hsv. Convert to RGB first')
        else:
            match self.imformat:
                case IMAGE_FORMATS.RGB:
                    return rgb2hsv(self.array[:])
                case _:
                    raise ValueError(f'Unsupported imformat {self.imformat} for HSV conversion')

    @property
    def hsv(self) -> HsvAccessor:
        """Returns the HSV accessor.

        This property returns an instance of the HsvAccessor associated with the
        current object, allowing access to HSV (hue, saturation, other_image) related
        functionalities controlled by this handler.

        Returns:
            HsvAccessor: The instance of the HSV accessor.
        """
        return self._accessors.hsv

    @hsv.setter
    def hsv(self, value):
        raise IllegalAssignmentError('hsv')
