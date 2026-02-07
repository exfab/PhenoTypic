from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.tools_.constants_ import GAMMA_ENCODINGS
from ._image_parts._image_io_handler import ImageIOHandler


class Image(ImageIOHandler):
    """Comprehensive image processing class with integrated data, color, and I/O management.

    The `Image` class is the primary interface for image processing, analysis, and manipulation
    within the PhenoTypic framework. It combines:
    - Data management (image arrays, enhanced versions, object maps)
    - Color space handling (RGB, grayscale, HSV, XYZ, Lab with color corrections)
    - Object detection and analysis (object masks, labels, measurements)
    - File I/O and metadata management (loading, saving, metadata extraction)
    - Image manipulation (rotation, slicing, copying, visualization)

    Image data can be provided as:
    - NumPy arrays (2-D grayscale or 3-D RGB/RGBA)
    - Another Image instance (copies all data)
    - Loaded from file via imread()

    The class automatically manages format conversions and maintains internal consistency
    across multiple data representations. RGB and grayscale forms are kept synchronized,
    and additional representations (detection matrix, object maps) support analysis workflows.

    Notes:
        - 2-D input arrays are treated as grayscale; rgb form remains empty.
        - 3-D input arrays are treated as RGB; grayscale is computed automatically.
        - Color space properties (gamma, illuminant, _observer) are inherited.
        - Object detection and measurements require an ObjectDetector first.
        - HSV color space support added in v0.5.0.

    Examples:
        Create from array:

        >>> import numpy as np
        >>> from phenotypic import Image
        >>> arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> img = Image(arr, name='sample')
        >>> img.show()

        Load from file:

        >>> img = Image.imread('photo.jpg')
        >>> print(img.shape)  # Image dimensions
        >>> img.save2pickle('saved.pkl')
    """

    def __init__(
            self,
            arr: np.ndarray | Image | None = None,
            name: str | None = None,
            bit_depth: Literal[8, 16] | None = None,
            gamma: GAMMA_ENCODINGS = GAMMA_ENCODINGS.SRGB,
            illuminant: str | None = "D65",
    ):
        """Initialize an Image instance with optional image data and color properties.

        Creates a new Image with complete initialization of all data management, color space,
        I/O, and object handling capabilities. The image can be initialized empty or with
        data from a NumPy array or another Image instance.

        Args:
            arr (np.ndarray | Image | None): Optional image data. Can be:
                - A NumPy array of shape (height, width) for grayscale or
                  (height, width, channels) for RGB/RGBA
                - An existing Image instance to copy from
                - None to create an empty image
                Defaults to None.
            name (str | None): Optional human-readable name for the image. If not provided,
                the image UUID will be used as the name. Defaults to None.
            bit_depth (Literal[8, 16] | None): The bit depth of the image data (8 or 16 bits).
                If not specified and arr is provided, bit depth is automatically inferred
                from the array dtype. Defaults to None.
            gamma (GAMMA_ENCODINGS): The gamma encoding used for color correction.
                GAMMA_ENCODINGS.SRGB: applies sRGB gamma correction (standard display gamma)
                GAMMA_ENCODINGS.LINEAR: assumes linear RGB data
                Defaults to GAMMA_ENCODINGS.SRGB.
            illuminant (str | None): The reference illuminant for color calculations.
                'D65': standard daylight illuminant (recommended)
                'D50': standard illumination for imaging
                Defaults to 'D65'.

        Raises:
            ValueError: If gamma is not a GAMMA_ENCODINGS member.
            ValueError: If illuminant is not 'D65' or 'D50'.
            TypeError: If arr is provided but is not a NumPy array or Image instance.

        Examples:
            Create empty image:

            >>> img = Image(name='empty_image')

            Create from grayscale array:

            >>> gray_arr = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
            >>> img = Image(gray_arr, name='grayscale_photo')

            Create from RGB array:

            >>> rgb_arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            >>> from phenotypic.tools_.constants_ import GAMMA_ENCODINGS
            >>> img = Image(rgb_arr, name='color_photo', gamma=GAMMA_ENCODINGS.SRGB)

            Copy another image:

            >>> img1 = Image.imread('original.jpg')
            >>> img2 = Image(img1, name='copy_of_original')
        """
        super().__init__(
                arr=arr,
                name=name,
                bit_depth=bit_depth,
                gamma=gamma,
                illuminant=illuminant,
        )
