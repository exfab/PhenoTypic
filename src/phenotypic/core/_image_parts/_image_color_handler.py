from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.color import rgb2hsv
from os import PathLike

import colour

from functools import partial

from phenotypic.core._image_parts.accessors import HsbAccessor
from ._image_objects_handler import ImageObjectsHandler
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import IllegalAssignmentError
from phenotypic.util.colourspaces_ import sRGB_D50


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

            _accessors.hsb: HsbAccessor
                Provides access to HSV-related processing methods tailored for the image.

        """
        self.color_profile: str = kwargs.get('color_profile', 'sRGB')

        self.illuminant: Literal["D50", "D65"] = kwargs.get('illuminant', "D65")
        super().__init__(input_image=input_image, imformat=imformat, name=name, **kwargs)

        # Device color profiles

        # This tracks if phenotypic's gamma decoder has been applied. This will trigger a warning if not applied
        self._known_gamma_decoding: bool = False
        self._accessors.hsb = HsbAccessor(self)

    def rgb2xyz(self) -> np.ndarray:
        """
        Converts RGB color values to XYZ color space based on the specified color
        profile and illuminant. This method is dependent on the specified color
        profile (`sRGB`) and reference illuminants (`D50` or `D65`). It invokes
        color transformation based on these parameters and returns an XYZ
        representation of the colors.

        Note that this function relies on the pre-decoding of RGB values using a
        gamma decoder. A warning is issued if the gamma decoding step is not
        acknowledged as done as it may impact the accuracy of the results.

        Raises:
            ValueError: If the specified combination of color profile and illuminant
            is unsupported.

        Returns:
            np.ndarray: A numpy array representing the colors in XYZ color space.
        """
        if not self._known_gamma_decoding:
            warnings.warn('The RGB values have not been decoded using phenotypic\'s gamma decoder, this may lead to inaccurate results.')
        match (self.color_profile, self.illuminant):
            case ("sRGB", "D50"):
                return colour.RGB_to_XYZ(
                    RGB=self.array[:],
                    colourspace=sRGB_D50,
                    illuminant=sRGB_D50.whitepoint,
                )
            case ("sRGB", "D65"):
                return colour.RGB_to_XYZ(
                    RGB=self.array[:],
                    colourspace=colour.RGB_COLOURSPACES["sRGB"],
                    illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"],
                )
            case _:
                raise ValueError(f'Unknown color_profile: {self.color_profile} or illuminant: {self.illuminant}')

    def rgb2xyz_d65(self) -> np.ndarray:
        """
        Converts RGB values to XYZ under the D65 illuminant.

        This method assumes the 2-degree standard observer and handles chromatic
        adaptation as necessary based on the specified color profile and illuminant.
        It uses the Bradford transformation method for chromatic adaptation where
        applicable.

        Returns:
            np.ndarray: The XYZ color representation of the RGB input under the D65
            illuminant.

        Raises:
            ValueError: If an unrecognized color profile or illuminant is provided.
        """
        # We assume 2 degree standard observer for now
        wp = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']

        # Creates a partial function so only the test XYZ whitepoint needs to be supplied
        bradford_cat65 = partial(colour.chromatic_adaptation, XYZ=self.rgb2xyz(), XYZ_wr=colour.xy_to_XYZ(wp['D65']), method='Bradford')

        match (self.color_profile, self.illuminant):
            case ("sRGB", "D65"):
                return self.rgb2xyz()
            case ("sRGB", "D50"):
                return bradford_cat65(XYZ_w=colour.xy_to_XYZ(wp['D50']))
            case _:
                raise ValueError(f'Unknown color_profile: {self.color_profile} or illuminant: {self.illuminant}')

    def xy(self)-> np.ndarray:
        """
        Converts an RGB color to its corresponding xy chromaticity coordinates.

        This function uses the RGB to XYZ color space conversion, followed by the
        transformation from XYZ to xy chromaticity coordinates. The result represents
        the color in terms of its xy chromaticity values, which are commonly used
        for two-dimensional color specifications.

        Returns:
            np.ndarray: A NumPy array containing the xy chromaticity coordinates of
                the input RGB color.
        """
        return colour.XYZ_to_xy(self.rgb2xyz_d65())


    @property
    def _hsb(self) -> np.ndarray:
        """Returns the hsb array dynamically of the current image.

        This can become computationally expensive, so implementation may be changed in the future.

        Returns:
            np.ndarray: The hsb array of the current image.
        """
        if self.imformat.is_matrix():
            raise AttributeError('Grayscale images cannot be directly converted to hsb. Convert to RGB first')
        else:
            match self.imformat:
                case IMAGE_FORMATS.RGB:
                    return rgb2hsv(self.array[:])
                case _:
                    raise ValueError(f'Unsupported imformat {self.imformat} for HSV conversion')

    @property
    def hsb(self) -> HsbAccessor:
        """Returns the HSV accessor.

        This property returns an instance of the HsvAccessor associated with the
        current object, allowing access to HSV (hue, saturation, other_image) related
        functionalities controlled by this handler.

        Returns:
            HsbAccessor: The instance of the HSV accessor.
        """
        return self._accessors.hsb

    @hsb.setter
    def hsb(self, value):
        raise IllegalAssignmentError('hsb')
