from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import colour
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
                 gamma_encoding: str | None = 'sRGB',
                 illuminant: str | None = 'D65',
                 observer='CIE 1931 2 Degree Standard Observer'

                 ):
        """
        Initializes the object attributes related to image processing and colorimetry.

        Args:
            arr (np.ndarray | Image | None): The array or image data. Defaults to None.
            name (str | None): The name associated with the object or image. Defaults to None.
            bit_depth (Literal[8, 16] | None): The bit depth for the image (8 or 16). Defaults to None.
            gamma_encoding (str | None): The gamma encoding type, e.g., 'sRGB'. Defaults to 'sRGB'.
            illuminant (str | None): The reference illuminant, e.g., 'D65'. Defaults to 'D65'.
            observer (str): The observer standard, typically 'CIE 1931 2 Degree Standard Observer'.
                Defaults to 'CIE 1931 2 Degree Standard Observer'.
        """
        super().__init__(
                arr=arr,
                name=name,
                bit_depth=bit_depth,
                gamma_encoding=gamma_encoding,
                illuminant=illuminant,
                observer=observer,
        )
