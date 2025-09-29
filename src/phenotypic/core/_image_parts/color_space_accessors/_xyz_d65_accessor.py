from functools import partial

from phenotypic.util.exceptions_ import IllegalAssignmentError


import colour
import numpy as np

from ..accessor_abstracts._image_accessor_base import ImageAccessorBase

class XyzD65Accessor(ImageAccessorBase):
    """
    Provides functionality for accessing image data converted to the XYZ color space
    under the D65 illuminant.

    This class inherits from ImageAccessorBase and is designed to handle image data
    manipulation in the XYZ color space under specific viewing conditions. It uses
    the chromatic adaptation transformation as needed and supports sRGB-specific
    color profiles and illuminants.

    Attributes:
        _root_image (ImageData): The root image object containing color profile,
            illuminant, and associated data for chromatic adaptation and XYZ
            conversions.
    """
    @property
    def _subject_arr(self) -> np.ndarray:
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
        wp = colour.CCS_ILLUMINANTS[self._root_image.observer]

        # Creates a partial function so only the new XYZ whitepoint needs to be supplied
        bradford_cat65 = partial(colour.chromatic_adaptation, XYZ=self._root_image.CieXYZ[:], XYZ_wr=colour.xy_to_XYZ(wp['D65']), method='Bradford')

        match (self._root_image.color_profile, self._root_image.illuminant):
            case ("sRGB", "D65"):
                return self._root_image.CieXYZ[:]
            case ("sRGB", "D50"):
                return bradford_cat65(XYZ_w=colour.xy_to_XYZ(wp['D50']))
            case _:
                raise ValueError(f'Unknown color_profile: {self._root_image.color_profile} or illuminant: {self._root_image.illuminant}')

    def __getitem__(self, key) -> np.ndarray:
        return self._subject_arr[key].copy()

    def __setitem__(self, key, value):
        raise IllegalAssignmentError('XYZD65')