from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from .accessors._color_accessor import ColorAccessor
from ._image_objects_handler import ImageObjectsHandler
import colour


class ImageColorSpace(ImageObjectsHandler):
    """Represents an image's color space and its associated properties and transformations.

    This class encapsulates various properties of an image's color space, including gamma encoding, observer model,
    illuminant, and color profile. It provides mechanisms for handling image data with specific attributes
    and accessing color transformations through a unified interface. It supports initialization with image
    data as a NumPy array or another compatible image format, along with metadata.

    Attributes:
        color_profile (str): The color profile associated with the image (default: 'sRGB').
        gamma_encoding (colour.CCTF_ENCODINGS | str | None): The gamma encoding applied to the image, as
            either a predefined encoding in `colour.CCTF_ENCODINGS` or a string representation
            (default: 'sRGB').
        observer (str): The observer model employed for the image (default: 'CIE 1931 2 Degree Standard Observer').
        illuminant (Literal["D65", "D50"]): The illuminant type defining the lighting conditions for the image
            (default: "D65").

    References:
        - http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
        - https://colour.readthedocs.io/en/latest/generated/colour.CCTF_DECODINGS.html#colour.CCTF_DECODINGS
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None, bit_depth: Literal[8, 16] | None = 8,
                 *,
                 gamma_encoding: Literal["sRGB"] | None = 'sRGB',
                 illuminant: Literal["D65", "D50"] = "D65",
                 observer: str = 'CIE 1931 2 Degree Standard Observer',
                 ):
        """
        Represents an image with associated metadata and color properties. This class is
        used for handling image data with specific attributes such as bit depth, gamma
        encoding, observer, and illuminant. It provides functionality for working with
        color images, with the ability to set and retrieve associated metadata and color
        attributes.

        Attributes:
            observer: str
                The standard observer used in the color computations, such as 'CIE 1931
                2 Degree Standard Observer'.
            illuminant: Literal["D65", "D50"]
                The illuminant used, typically "D65" for daylight conditions or "D50"
                for standard viewing for imaging.
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
