from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from .accessors._color_accessor import ColorAccessor
from ._image_objects_handler import ImageObjectsHandler
import colour


class ImageColorSpace(ImageObjectsHandler):
    """Manages color space representation and transformations for image data.

    This class extends ImageObjectsHandler to add comprehensive color space management,
    enabling access to multiple color space representations through a unified ColorAccessor
    interface. It handles:
    - Gamma encoding and color correction (sRGB or linear)
    - Color space transformations (RGB, XYZ, Lab, HSV, etc.)
    - Observer model and illuminant specification for colorimetric calculations
    - Integration with the colour library for accurate color conversions

    The class ensures consistency across different color space representations while
    maintaining the underlying image data integrity. All color transformations are
    computed on-demand through the color accessor to minimize memory overhead.

    Attributes:
        gamma_encoding (str | None): The gamma encoding applied to the image
            ('sRGB' for gamma-corrected, None for linear RGB). Defaults to 'sRGB'.
        observer (str): The CIE standard observer model for color calculations
            (default: 'CIE 1931 2 Degree Standard Observer').
        illuminant (Literal["D65", "D50"]): The reference illuminant defining viewing
            conditions. 'D65' represents standard daylight, 'D50' represents standard
            illumination for imaging. Defaults to 'D65'.
        _accessors.color (ColorAccessor): Unified accessor for color space representations.

    References:
        - Bruce Lindbloom Color Calculator: http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
        - Colour Library Decodings: https://colour.readthedocs.io/en/latest/generated/colour.CCTF_DECODINGS.html
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None,
                 bit_depth: Literal[8, 16] | None = 8,
                 *,
                 gamma_encoding: Literal["sRGB"] | None = 'sRGB',
                 illuminant: Literal["D65", "D50"] = "D65",
                 observer: str = 'CIE 1931 2 Degree Standard Observer',
                 ):
        """Initialize ImageColorSpace with color properties and representations.

        Sets up color space management for the image, including gamma encoding,
        observer model, and illuminant specification. These parameters are critical
        for accurate color space transformations and colorimetric calculations.

        Args:
            arr (np.ndarray | Image | None): Optional initial image data. Can be a NumPy array
                or an existing Image instance. Defaults to None.
            name (str | None): Optional name for the image. Defaults to None.
            bit_depth (Literal[8, 16] | None): The bit depth of the image (8 or 16 bits).
                Defaults to 8.
            gamma_encoding (Literal["sRGB"] | None): The gamma encoding applied to the image.
                'sRGB' applies gamma correction for display, None assumes linear RGB.
                Only 'sRGB' and None are supported. Defaults to 'sRGB'.
            illuminant (Literal["D65", "D50"]): The reference illuminant for color calculations.
                'D65' represents standard daylight, 'D50' represents standard illumination
                for imaging. Defaults to 'D65'.
            observer (str): The CIE standard observer model for color computations.
                Common value is 'CIE 1931 2 Degree Standard Observer'. Defaults to
                'CIE 1931 2 Degree Standard Observer'.

        Raises:
            ValueError: If gamma_encoding is not 'sRGB' or None.
            ValueError: If illuminant is not 'D65' or 'D50'.
        """
        if (gamma_encoding != "sRGB") and (gamma_encoding is not None):
            raise ValueError(f'only sRGB or None for linear is supported for gamma encoding: got {gamma_encoding}')
        if illuminant not in ["D65", "D50"]:
            raise ValueError('illuminant must be "D65" or "D50"')

        self.gamma_encoding = gamma_encoding
        self.observer: str = observer
        self.illuminant: Literal["D50", "D65"] = illuminant
        super().__init__(arr=arr, name=name, bit_depth=bit_depth)

        # Initialize color accessor
        self._accessors.color = ColorAccessor(self)

    @property
    def color(self) -> ColorAccessor:
        """
        Access all color space representations through a unified interface.
        
        This property provides access to the ColorAccessor object, which groups
        all color space transformations and representations including:
        
        - XYZ: CIE XYZ color space
        - XYZ_D65: CIE XYZ under D65 illuminant
        - Lab: CIE L*a*b* perceptually uniform color space
        - xy: CIE xy chromaticity coordinates
        - hsv: HSV (Hue, Saturation, Value) color space
        
        Returns:
            ColorAccessor: Unified accessor for all color space representations.
            
        Examples:
            >>> img = Image.imread('sample.jpg')
            >>> xyz_data = img.color.XYZ[:]
            >>> lab_data = img.color.Lab[:]
            >>> hue = img.color.hsv[..., 0] # hue is the first matrix in the array
        """
        return self._accessors.color
