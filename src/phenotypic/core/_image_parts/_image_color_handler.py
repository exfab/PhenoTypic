from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from os import PathLike

from .accessors._color_accessor import ColorAccessor
from ._image_objects_handler import ImageObjectsHandler
import colour


class ImageColorSpace(ImageObjectsHandler):
    """
    Adds color space handling to the image class

    References:
        - http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
        - https://colour.readthedocs.io/en/latest/generated/colour.CCTF_DECODINGS.html#colour.CCTF_DECODINGS
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None, bit_depth: Literal[8, 16] | None = 8,
                 *,
                 color_profile: str = 'sRGB',
                 gamma_encoding: colour.CCTF_ENCODINGS | str | None = None,
                 observer: str = 'CIE 1931 2 Degree Standard Observer',
                 illuminant: Literal["D65", "D50"] = "D65",
                 ) -> None:
        """
        Initializes an object with attributes for image processing and handling color profiles.

        The constructor accepts an image input along with its format, name, and additional
        optional parameters. These parameters allow customization of the color profile and
        illuminant for color management.

        Args:
            arr: np.ndarray | Image | PathLike | None
                The input image to process, which can be specified as a numpy array,
                a PIL Image object, or a path-like object. Defaults to None.

            name: str | None
                The optional name of the image or the processing instance. Useful for
                tracking or logging purposes. Defaults to None.

            **kwargs: dict
                Additional keyword arguments that allow customization of various settings.

        Attributes:
            color_profile: str
                The color profile associated with the image, specifying how colors are
                represented. Defaults to 'sRGB'.

            observer: str
                The CIE standard observer specification. Defaults to 
                "CIE 1931 2 Degree Standard Observer".

            illuminant: Literal["D50", "D65"]
                The color temperature or reference white point used in image color
                processing. Options are "D50" or "D65". Defaults to "D65".

            color: ColorAccessor
                Provides unified access to all color space representations including
                XYZ, Lab, xy chromaticity, and HSV.

        """
        self.color_profile: str = color_profile

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
            >>> img = Image('sample.jpg')
            >>> xyz_data = img.color.XYZ[:]
            >>> lab_data = img.color.Lab[:]
            >>> hue = img.color.hsv.hue
        """
        return self._accessors.color
