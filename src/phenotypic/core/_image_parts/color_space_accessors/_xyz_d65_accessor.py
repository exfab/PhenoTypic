from functools import partial

import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class XyzD65Accessor(ColorSpaceAccessor):
    """Provides access to XYZ color space under D65 illuminant.
    
    Converts image data to the XYZ color space under D65 illuminant viewing conditions,
    applying chromatic adaptation transformations as needed. Supports sRGB color profiles
    with both D50 and D65 illuminants.
    
    Attributes:
        _root_image (Image): The root image object containing color profile,
            illuminant, and data for chromatic adaptation and XYZ conversions.
    """

    @property
    def _subject_arr(self) -> np.ndarray:
        """
        Converts RGB values to XYZ under the D65 illuminant.

        This method assumes the 2-degree standard observer and handles chromatic
        adaptation as necessary based on the specified color profile and illuminant.
        It uses the `colour.chromatic_adaptation <https://colour.readthedocs.io/en/develop/generated/colour.chromatic_adaptation.html>`_
        function with the Bradford transformation method for chromatic adaptation where
        applicable.

        Returns:
            np.ndarray: The XYZ color representation of the RGB input under the D65
            illuminant.

        Raises:
            ValueError: If an unrecognized color profile or illuminant is provided.
        """
        wp = colour.CCS_ILLUMINANTS[self._root_image.observer]

        # Creates a partial function so only the new XYZ whitepoint needs to be supplied
        bradford_cat65 = partial(colour.chromatic_adaptation,
                                 XYZ=self._root_image.color.XYZ[:],
                                 XYZ_wr=colour.xy_to_XYZ(wp['D65']),
                                 method='Bradford')

        match self._root_image.illuminant:
            case "D65":
                return self._root_image.color.XYZ[:]
            case "D50":
                return bradford_cat65(XYZ_w=colour.xy_to_XYZ(wp['D50']))
            case _:
                raise ValueError(
                        f'Unknown color_profile: {self._root_image.color_profile} or illuminant: {self._root_image.illuminant}')
