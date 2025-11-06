from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING: from phenotypic import Image

from ._image_parts._image_io_handler import ImageIOHandler


class Image(ImageIOHandler):
    """A comprehensive class for handling image processing, including manipulation, information sync, metadata management, and format conversion.

    The `Image` class is designed to load, process, and manage image data using different
    representation formats (e.g., arrays and matrices). This class allows for metadata editing,
    schema definition, and subcomponent handling to streamline image processing tasks.

    Note:
        - If the arr is 2-D, the ImageHandler leaves the rgb form as empty
        - If the arr is 3-D, the ImageHandler will automatically set the gray component to the grayscale representation.
        - Added in v0.5.0, HSV handling support
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None,
                 bit_depth: Literal[8, 16] | None = None,
                 illuminant: str | None = 'D65',
                 color_profile='sRGB',
                 observer='CIE 1931 2 Degree Standard Observer'):
        """
        Initializes an instance of the class with optional attributes for array data,
        name, bit depth, illuminant, color profile, and observer. The class is constructed to handle
        data related to image processing and its various configurations.

        Args:
            arr (np.ndarray | Image | None): An optional array or image object. It
                represents the pixel data or image source.
            name (str | None): An optional name or identifier for the image instance.
            bit_depth (Literal[8, 16] | None): An optional bit depth of the image.
                Either 8 or 16 bits for pixel representation. If None is specified, the bit depth
                will be guessed from the arr dtype. If the arr is a float, the bit_depth will default to 8.
            illuminant (str | None): A string specifying the illuminant standard for
                the image, defaulting to 'D65'.
            color_profile (str): The color profile for the image, defaulting to 'sRGB'.
            observer (str): Observer type in CIE standards, defaulting to 'CIE 1931
                2 Degree Standard Observer'.

        """
        super().__init__(
                arr=arr,
                name=name,
                bit_depth=bit_depth,
                illuminant=illuminant,
                color_profile=color_profile,
                observer=observer
        )
