from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING: from phenotypic import Image

from ._image_parts._image_io_handler import ImageIOHandler


class Image(ImageIOHandler):
    """A comprehensive class for handling image processing, including manipulation, information sync, metadata management, and format conversion.

    The `Image` class is designed to load, process, and manage image data using different
    representation formats (e.g., arrays and matrices). This class allows for metadata editing,
    schema definition, and subcomponent handling to streamline image processing tasks.

    Note:
        - If the input_image is 2-D, the ImageHandler leaves the array form as empty
        - If the input_image is 3-D, the ImageHandler will automatically set the matrix component to the grayscale representation.
        - Added in v0.5.0, HSV handling support
    """

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | Path | str | None = None,
                 imformat: str | None = None,
                 name: str | None = None,
                 illuminant: str | None = 'D65',
                 color_profile='sRGB',
                 observer='CIE 1931 2 Degree Standard Observer'):
        super().__init__(
            input_image=input_image,
            imformat=imformat,
            name=name,
            illuminant=illuminant,
            color_profile=color_profile,
            observer=observer
        )