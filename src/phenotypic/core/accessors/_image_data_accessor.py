import numpy as np

from ._image_accessor import ImageAccessor
from phenotypic.util.exceptions_ import InterfaceError


class ImageDataAccessor(ImageAccessor):
    """
    The base for classes that specifically provide access to data within parent image.

    The ImageAccessor class serves as a base class for interacting with a parent image
    object. It requires an instance of the parent image for initialization to
    enable seamless operations on the image's properties and data.

    Attributes:
        _parent_image (Image): The parent image object that this accessor interacts
            with.
    """

    def shape(self) -> tuple[int, ...]:
        raise InterfaceError

    def isempty(self):
        return True if self.shape[0] == 0 else False
