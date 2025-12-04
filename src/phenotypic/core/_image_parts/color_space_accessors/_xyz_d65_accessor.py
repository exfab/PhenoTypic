from functools import partial

import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class XyzD65Accessor(ColorSpaceAccessor):
    """Provides access to XYZ color space under D65 illuminant.

    The XyzD65Accessor transforms image data to the CIE XYZ color space under
    D65 illuminant viewing conditions (standard daylight at 6500 K), assuming a
    2-degree standard observer. It applies chromatic adaptation transformations
    when necessary to convert images from alternative illuminants (such as D50)
    to the D65 reference white point. This accessor supports sRGB color profiles
    with both D50 and D65 illuminants.

    The XYZ color space is a device-independent representation useful for color
    matching, comparison, and conversion to other perceptual color spaces. The
    D65 illuminant is the CIE standard daylight reference and is widely used in
    color management and imaging applications.

    Attributes:
        _root_image (Image): The root image object containing color profile,
            illuminant, and data for chromatic adaptation and XYZ conversions.
        _accessor_property_name (str): The property name on Image for accessing
            this accessor via Image.color.XYZ_D65.

    Examples:
        .. dropdown:: Access XYZ D65 color space with chromatic adaptation

            .. code-block:: python

                from phenotypic import Image
                img = Image('path/to/image.tif')
                # Access XYZ D65 color space
                xyz_d65_arr = img.color.XYZ_D65[:]
                # For images with D65 illuminant, returns XYZ values directly
                # For images with D50 illuminant, applies Bradford chromatic adaptation
    """

    _accessor_property_name: str = "color.XYZ_D65"

    @property
    def _subject_arr(self) -> np.ndarray:
        """Convert RGB values to XYZ under the D65 illuminant.

        Transforms the parent image's RGB data to the CIE XYZ color space under
        the D65 illuminant (standard daylight at 6500 K). If the image data is
        already in the D65 illuminant, returns the XYZ representation directly.
        If the image data is in the D50 illuminant, applies the Bradford chromatic
        adaptation transformation to convert from D50 to D65.

        The method assumes the CIE 1931 2-degree standard observer. The Bradford
        chromatic adaptation method is used as it provides excellent color accuracy
        for illuminant transformations and is widely adopted in color management
        systems (ICC profiles, digital photography, etc.).

        Returns:
            np.ndarray: The CIE XYZ color space representation of the parent image
                under the D65 illuminant. Array shape matches the parent image's
                RGB data (height, width, 3), with values typically in the range
                [0, 100] for Y values normalized to illuminant Y = 100.

        Raises:
            ValueError: If the parent image's illuminant is not one of the
                supported values ('D65' or 'D50'). This error indicates an
                unrecognized color profile or illuminant configuration.

        Notes:
            - For D65 images: Returns the XYZ conversion without modification.
            - For D50 images: Applies Bradford chromatic adaptation using:
              - Source whitepoint (XYZ_w): D50 in the image's observer space
              - Target whitepoint (XYZ_wr): D65 in the image's observer space
            - The Bradford transformation is implemented via the colour library's
              chromatic_adaptation() function with the 'Bradford' method.
            - See: https://colour.readthedocs.io/en/develop/generated/colour.chromatic_adaptation.html

        Examples:
            .. dropdown:: Access the underlying XYZ D65 array

                .. code-block:: python

                    import numpy as np
                    # Access the underlying XYZ D65 array
                    xyz_arr = accessor._subject_arr
                    # Shape is (height, width, 3)
                    # Can be used with colour library for further conversions
                    print(xyz_arr.shape)  # e.g., (480, 640, 3)
                    print(xyz_arr.dtype)  # float64
        """
        wp = colour.CCS_ILLUMINANTS[self._root_image._observer]

        # Creates a partial function so only the new XYZ whitepoint needs to be supplied
        bradford_cat65 = partial(
            colour.chromatic_adaptation,
            XYZ=self._root_image.color.XYZ[:],
            XYZ_wr=colour.xy_to_XYZ(wp["D65"]),
            method="Bradford",
        )

        match self._root_image.illuminant:
            case "D65":
                return self._root_image.color.XYZ[:]
            case "D50":
                return bradford_cat65(XYZ_w=colour.xy_to_XYZ(wp["D50"]))
            case _:
                raise ValueError(
                    f"Unknown color_profile: {self._root_image.color_profile} or illuminant: {self._root_image.illuminant}"
                )
