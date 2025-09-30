from __future__ import annotations

import colour
import numpy as np

from phenotypic.util.colourspaces_ import sRGB_D50
from phenotypic.util.exceptions_ import IllegalAssignmentError
from phenotypic.util.funcs_ import normalize_rgb_bitdepth
from ..accessor_abstracts._image_accessor_base import ImageAccessorBase


class XyzAccessor(ImageAccessorBase):
    """Provides access to the XYZ color representation of an image.

    Converts image data from different color profiles and illuminants to its
    corresponding XYZ color representation. The class ensures accurate color
    management while preventing direct modifications to the data.

    Attributes:
        _root_image (Image): The source image object containing all necessary
            information, including color profile, illuminant, and array data.
    """

    @property
    def _subject_arr(self) -> np.ndarray:
        rgb = self._root_image.array[:]
        norm_rgb = normalize_rgb_bitdepth(rgb)
        match (self._root_image.color_profile, self._root_image.illuminant):
            case ("sRGB", "D50"):
                sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[self._root_image.observer]["D50"]
                return colour.RGB_to_XYZ(
                        RGB=norm_rgb,
                        colourspace=sRGB_D50,
                        illuminant=sRGB_D50.whitepoint,
                        cctf_decoding=True,  # Assumes sRGB means non-linear
                )
            case ("sRGB", "D65"):
                return colour.RGB_to_XYZ(
                        RGB=norm_rgb,
                        colourspace=colour.RGB_COLOURSPACES["sRGB"],
                        illuminant=colour.CCS_ILLUMINANTS[self._root_image.observer]["D65"],
                        cctf_decoding=True,  # Assumes sRGB means non-linear
                )
            case _:
                raise ValueError(
                    f'Unknown color_profile: {self._root_image.color_profile} or illuminant: {self._root_image.illuminant}')

    def __getitem__(self, key) -> np.ndarray:
        return self._subject_arr[key].copy()

    def __setitem__(self, key, value):
        raise IllegalAssignmentError('XYZ')
