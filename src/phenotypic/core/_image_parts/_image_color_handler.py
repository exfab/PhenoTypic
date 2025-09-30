from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from os import PathLike

from phenotypic.core._image_parts.accessors import HsvAccessor
from .color_space_accessors._xyz_accessor import XyzAccessor
from .color_space_accessors._xyz_d65_accessor import XyzD65Accessor
from .color_space_accessors._cielab_accessor import CieLabAccessor
from .color_space_accessors._chromaticity_xy_accessor import xyChromaticityAccessor

from ._image_objects_handler import ImageObjectsHandler


class ImageColorSpace(ImageObjectsHandler):
    """
    Adds color space handling to the image class

    References:
        - http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
        - https://colour.readthedocs.io/en/latest/generated/colour.CCTF_DECODINGS.html#colour.CCTF_DECODINGS
    """

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | None = None,
                 imformat: str | None = None,
                 name: str | None = None, **kwargs):
        """
        Initializes an object with attributes for image processing and handling color profiles.

        The constructor accepts an image input along with its format, name, and additional
        optional parameters. These parameters allow customization of the color profile and
        illuminant for color management.

        Args:
            input_image: np.ndarray | Image | PathLike | None
                The input image to process, which can be specified as a numpy array,
                a PIL Image object, or a path-like object. Defaults to None.

            imformat: str | None
                The format of the input image, which specifies the structure or encoding
                of the image data. Defaults to None.

            name: str | None
                The optional name of the image or the processing instance. Useful for
                tracking or logging purposes. Defaults to None.

            **kwargs: dict
                Additional keyword arguments that allow customization of various settings.

        Attributes:
            color_profile: str
                The color profile associated with the image, specifying how colors are
                represented. Defaults to 'sRGB'.

            illuminant: Literal["D50", "D65"]
                The color temperature or reference white point used in image color
                processing. Options are "D50" or "D65". Defaults to "D65".

            _known_gamma_decoding: bool
                Tracks whether the gamma decoding for the color profile has been applied.
                A warning is issued if not applied. Defaults to False.

            _accessors.hsb: HsvAccessor
                Provides access to HSV-related processing methods tailored for the image.

        """
        self.color_profile: str = kwargs.get('color_profile', 'sRGB')
        self.observer: str = kwargs.get('observer', "CIE 1931 2 Degree Standard Observer")
        self.illuminant: Literal["D50", "D65"] = kwargs.get('illuminant', "D65")
        super().__init__(input_image=input_image, imformat=imformat, name=name, **kwargs)

        # Device color profiles
        self._accessors.hsv = HsvAccessor(self)
        self._accessors.CieXYZ = XyzAccessor(self)
        self._accessors.CieXYZD65 = XyzD65Accessor(self)
        self._accessors.CieLab = CieLabAccessor(self)
        self._accessors.Ciexy = xyChromaticityAccessor(self)

    @property
    def CieXYZ(self) -> XyzAccessor:
        return self._accessors.CieXYZ

    @property
    def CieXYZ_D65(self) -> XyzD65Accessor:
        return self._accessors.CieXYZD65

    @property
    def Cie_xy(self) -> xyChromaticityAccessor:
        return self._accessors.Ciexy

    @property
    def CieLab(self) -> CieLabAccessor:
        """
        Gets the CieLab color space accessor associated with this instance.

        This property provides access to the CieLabAccessor object,
        allowing operations related to the CieLab color space.

        Returns:
            CieLabAccessor: An accessor for performing operations and
            manipulations in the CieLab color space.
        """
        return self._accessors.CieLab

    @property
    def hsv(self) -> HsvAccessor:
        """Returns the HSV accessor.

        This property returns an instance of the HsvAccessor associated with the
        current object, allowing access to HSV (hue, saturation, other_image) related
        functionalities controlled by this handler.

        Returns:
            HsvAccessor: The instance of the HSV accessor.
        """
        return self._accessors.hsv
