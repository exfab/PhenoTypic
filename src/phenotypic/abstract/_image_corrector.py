from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from phenotypic.abstract import ImageOperation

class ImageCorrector(ImageOperation):
    """
    The ImageCorrector class is for operations that alter the original image data to
    account for error inducing variations, such as color or illumination.
    """

    pass

