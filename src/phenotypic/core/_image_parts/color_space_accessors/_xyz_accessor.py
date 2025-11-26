from __future__ import annotations

import colour
import numpy as np

from phenotypic.tools.colourspaces_ import sRGB_D50
from phenotypic.tools.exceptions_ import IllegalAssignmentError
from phenotypic.tools.funcs_ import normalize_rgb_bitdepth
from ..accessor_abstracts._color_space_accessor import ColorSpaceAccessor
from phenotypic.tools.constants_ import IMAGE_MODE


class XyzAccessor(ColorSpaceAccessor):
    """Provides access to the XYZ color representation of an image.

    Converts image data from different color profiles and illuminants to its
    corresponding XYZ color representation using the
    `colour.RGB_to_XYZ <https://colour.readthedocs.io/en/develop/generated/colour.RGB_to_XYZ.html>`_
    function. The class ensures accurate color management while preventing direct
    modifications to the data.

    Attributes:
        _root_image (Image): The source image object containing all necessary
            information, including color profile, illuminant, and array data.
    """

    _accessor_property_name: str = "color.XYZ"

    @property
    def _subject_arr(self) -> np.ndarray:
        if self._root_image.rgb.isempty():
            raise AttributeError('XYZ conversion is not available for grayscale images')
        norm_rgb = normalize_rgb_bitdepth(self._root_image.rgb[:])
        match (self._root_image.gamma_encoding, self._root_image.illuminant):
            case ('sRGB', "D50"):
                sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[self._root_image.observer]["D50"]
                return colour.RGB_to_XYZ(
                        RGB=norm_rgb,
                        colourspace=sRGB_D50,
                        illuminant=sRGB_D50.whitepoint,
                        cctf_decoding=True,
                )
            case ("sRGB", "D65"):
                return colour.RGB_to_XYZ(
                        RGB=norm_rgb,
                        colourspace=colour.RGB_COLOURSPACES["sRGB"],
                        illuminant=colour.CCS_ILLUMINANTS[self._root_image.observer]["D65"],
                        cctf_decoding=True,
                )
            case (None, "D50"):
                sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[self._root_image.observer]["D50"]
                return colour.RGB_to_XYZ(
                        rgb=norm_rgb,
                        colourspace=colour.RGB_COLOURSPACES["sRGB"],
                        illuminant=sRGB_D50.whitepoint,
                        cctf_decoding=False,
                )
            case (None, "D65"):
                return colour.RGB_to_XYZ(
                        rgb=norm_rgb,
                        colourspace=colour.RGB_COLOURSPACES["sRGB"],
                        illuminant=colour.CCS_ILLUMINANTS[self._root_image.observer]["D65"],
                        cctf_decoding=False,
                )
            case _:
                raise ValueError(
                        f'Unknown color_profile: {self._root_image.gamma_encoding} '
                        f'or illuminant: {self._root_image.illuminant}')

    def __setitem__(self, key, value):
        raise IllegalAssignmentError('XYZ')
