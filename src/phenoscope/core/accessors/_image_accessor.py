from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

class ImageAccessor:
    """
    A base class that provides access to details and functionalities of a parent image.

    The ImageAccessor class serves as a base class  for interacting with a parent image
    object. It requires an instance of the parent image for initialization to
    enable seamless operations on the image's properties and data.

    Attributes:
        _parent_image (Image): The parent image object that this accessor interacts
            with.
    """
    def __init__(self, parent_image: Image):
        self._parent_image = parent_image