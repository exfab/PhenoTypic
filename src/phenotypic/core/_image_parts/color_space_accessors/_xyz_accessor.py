from __future__ import annotations

import colour
import numpy as np

from phenotypic.tools.colourspaces_ import sRGB_D50
from phenotypic.tools.exceptions_ import IllegalAssignmentError
from phenotypic.tools.funcs_ import normalize_rgb_bitdepth
from ..accessor_abstracts._color_space_accessor import ColorSpaceAccessor
from phenotypic.tools.constants_ import IMAGE_MODE


class XyzAccessor(ColorSpaceAccessor):
    """Provides read-only access to XYZ color space representation of an image.

    This accessor converts image data from various RGB color profiles and illuminants
    to their corresponding CIE XYZ tristimulus values. XYZ is a device-independent
    color space used as a reference standard in color science for precise color
    representation and transformation between different color spaces.

    The class supports multiple gamma encoding profiles (sRGB or linear/None) and
    illuminants (D50 or D65) with automatic observer-dependent whitepoint selection.
    Conversions are performed on-the-fly via the `colour.RGB_to_XYZ` function from
    the colour-science library, and direct modifications to the data are prevented
    to maintain data integrity.

    Attributes:
        _root_image (Image): The source image object containing the RGB array,
            gamma encoding profile, illuminant specification, observer type, and
            other metadata necessary for accurate color conversion.
        _accessor_property_name (str): The property name "color.XYZ" identifying
            this accessor type for metadata tracking during save/load operations.

    Raises:
        AttributeError: When accessing _subject_arr on grayscale images that lack
            RGB data.
        ValueError: When the image has an unknown gamma encoding profile or
            illuminant combination.
        IllegalAssignmentError: When attempting to modify XYZ data through
            __setitem__.

    Notes:
        XYZ values are computed using the colour-science library's RGB_to_XYZ
        function with the following behavior:
        - For sRGB gamma-encoded images: CCTF (color component transfer function)
          decoding is applied.
        - For linear RGB (gamma_encoding=None): No CCTF decoding is applied.
        - The appropriate standard RGB colorspace and whitepoint are selected based
          on the image's illuminant and observer settings.
    """

    _accessor_property_name: str = "color.XYZ"

    @property
    def _subject_arr(self) -> np.ndarray:
        """Compute and return XYZ tristimulus values from the image's RGB data.

        Converts the parent image's RGB array to CIE XYZ tristimulus values using
        the colour-science library. The conversion accounts for the image's gamma
        encoding profile (sRGB or linear) and illuminant specification (D50 or D65).
        RGB bitdepth is normalized to 0-1 range before conversion.

        The conversion process follows these steps:
        1. Check that the image has RGB data (not grayscale).
        2. Normalize RGB bitdepth to [0, 1] using normalize_rgb_bitdepth().
        3. Select appropriate colorspace and illuminant based on gamma_encoding
           and illuminant attributes.
        4. For D50 illuminant: configure observer-specific D50 whitepoint from
           colour.CCS_ILLUMINANTS.
        5. Call colour.RGB_to_XYZ with appropriate CCTF decoding setting.

        Returns:
            np.ndarray: XYZ tristimulus values with shape (height, width, 3) and
                dtype float64. X, Y, Z values range from 0 to approximately 100
                for typical image data.

        Raises:
            AttributeError: If the parent image is grayscale (RGB data is empty).
            ValueError: If gamma_encoding or illuminant combination is not one of:
                ('sRGB', 'D50'), ('sRGB', 'D65'), (None, 'D50'), (None, 'D65').

        Notes:
            - RGB normalization ensures input values are in [0, 1] range expected
              by colour.RGB_to_XYZ.
            - For sRGB images: cctf_decoding=True applies inverse OECF to convert
              from sRGB's perceptually-encoded values to linear RGB.
            - For linear RGB (gamma_encoding=None): cctf_decoding=False treats
              RGB values as already linear.
            - D50 whitepoint is dynamically set based on self._root_image.observer
              to ensure chromatic adaptation consistency.
            - Results are read-only (non-writeable) due to parent class __getitem__
              implementation.

        Examples:
            .. dropdown:: Basic access to XYZ data

                .. code-block:: python

                    image = Image('photo.jpg')  # sRGB, D65
                    xyz_data = image.color.XYZ[:]
                    print(xyz_data.shape)  # (height, width, 3)
                    print(xyz_data.dtype)  # float64

            .. dropdown:: Access specific region

                .. code-block:: python

                    xyz_region = image.color.XYZ[100:200, 50:150, :]
        """
        if self._root_image.rgb.isempty():
            raise AttributeError("XYZ conversion is not available for grayscale images")
        norm_rgb = normalize_rgb_bitdepth(self._root_image.rgb[:])
        match (self._root_image.gamma_encoding, self._root_image.illuminant):
            case ("sRGB", "D50"):
                sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[
                    self._root_image._observer
                ]["D50"]
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
                    illuminant=colour.CCS_ILLUMINANTS[self._root_image._observer][
                        "D65"
                    ],
                    cctf_decoding=True,
                )
            case (None, "D50"):
                sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[
                    self._root_image._observer
                ]["D50"]
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
                    illuminant=colour.CCS_ILLUMINANTS[self._root_image._observer][
                        "D65"
                    ],
                    cctf_decoding=False,
                )
            case _:
                raise ValueError(
                    f"Unknown color_profile: {self._root_image.gamma_encoding} "
                    f"or illuminant: {self._root_image.illuminant}"
                )

    def __setitem__(self, key, value):
        """Prevent direct modification of XYZ color space data.

        XYZ color space data is read-only to maintain data integrity and prevent
        inconsistencies between the computed XYZ values and the parent image's RGB
        data. All modifications should be made to the underlying RGB data, and
        the XYZ representation will be automatically recomputed when accessed.

        Args:
            key: Index or slice for accessing XYZ data (not used).
            value: Value to assign (not allowed).

        Raises:
            IllegalAssignmentError: Always raised. XYZ data cannot be directly
                modified; modifications to the parent image's RGB data will be
                reflected the next time the XYZ property is accessed.

        Examples:
            .. dropdown:: This operation always fails

                .. code-block:: python

                    image = Image('photo.jpg')
                    image.color.XYZ[0, 0, 0] = 50  # Raises IllegalAssignmentError

            .. dropdown:: To modify color data, work with the RGB accessor instead

                .. code-block:: python

                    image.rgb[0, 0, :] = [255, 128, 64]  # Valid
                    xyz_updated = image.color.XYZ[:]  # Recomputed automatically
        """
        raise IllegalAssignmentError("XYZ")
