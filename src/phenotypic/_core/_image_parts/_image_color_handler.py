from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np

from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS
from .accessors._color_accessor import ColorAccessor
from ._image_plot_handler import ImagePlotHandler


class ImageColorSpace(ImagePlotHandler):
    """Manages color space representation and transformations for image data.

    This class extends ImageObjectsHandler to add comprehensive color space management,
    enabling access to multiple color space representations through a unified ColorAccessor
    interface. It handles:
    - Gamma encoding and color correction (sRGB or linear)
    - Color space transformations (RGB, XYZ, Lab, HSV, etc.)
    - Observer model and illuminant specification for colorimetric calculations
    - Integration with the colour library for accurate color conversions

    The class ensures consistency across different color space representations while
    maintaining the underlying image data integrity. All color transformations are
    computed on-demand through the color accessor to minimize memory overhead.

    Attributes:
        gamma (GAMMA_ENCODINGS): The gamma encoding applied to the image
            (GAMMA_ENCODINGS.SRGB for gamma-corrected, GAMMA_ENCODINGS.LINEAR for linear
            RGB). Defaults to GAMMA_ENCODINGS.SRGB.
        _observer (str): The CIE standard observer model for color calculations
            (default: 'CIE 1931 2 Degree Standard Observer').
        illuminant (Literal["D65", "D50"]): The reference illuminant defining viewing
            conditions. 'D65' represents standard daylight, 'D50' represents standard
            illumination for imaging. Defaults to 'D65'.
        _accessors.color (ColorAccessor): Unified accessor for color space representations.

    References:
        - Bruce Lindbloom Color Calculator: http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
        - Colour Library Decodings: https://colour.readthedocs.io/en/latest/generated/colour.CCTF_DECODINGS.html
    """

    def __init__(
            self,
            arr: np.ndarray | Image | None = None,
            name: str | None = None,
            bit_depth: Literal[8, 16] | None = 8,
            *,
            gamma: GAMMA_ENCODINGS | str | None = GAMMA_ENCODINGS.SRGB,
            illuminant: Literal["D65", "D50"] = "D65",
    ):
        """Initialize ImageColorSpace with color properties and representations.

        Sets up color space management for the image, including gamma encoding,
        observer model, and illuminant specification. These parameters are critical
        for accurate color space transformations and colorimetric calculations.

        Args:
            arr (np.ndarray | Image | None): Optional initial image data. Can be a NumPy array
                or an existing Image instance. Defaults to None.
            name (str | None): Optional name for the image. Defaults to None.
            bit_depth (Literal[8, 16] | None): The bit depth of the image (8 or 16 bits).
                Defaults to 8.
            gamma (GAMMA_ENCODINGS): The gamma encoding applied to the image.
                GAMMA_ENCODINGS.SRGB applies gamma correction for display,
                GAMMA_ENCODINGS.LINEAR assumes linear RGB.
                Defaults to GAMMA_ENCODINGS.SRGB.
            illuminant (Literal["D65", "D50"]): The reference illuminant for color calculations.
                'D65' represents standard daylight, 'D50' represents standard illumination
                for imaging. Defaults to 'D65'.

        Raises:
            ValueError: If gamma is not a GAMMA_ENCODINGS member or a recognized
                string ('sRGB') / None.
            ValueError: If illuminant is not 'D65' or 'D50'.
        """
        if not isinstance(gamma, GAMMA_ENCODINGS):
            _GAMMA_COERCE = {"sRGB": GAMMA_ENCODINGS.SRGB, None: GAMMA_ENCODINGS.LINEAR}
            if gamma not in _GAMMA_COERCE:
                raise ValueError(
                        f"gamma must be a GAMMA_ENCODINGS member, 'sRGB', or None: got {gamma}"
                )
            gamma = _GAMMA_COERCE[gamma]
        if illuminant not in ["D65", "D50"]:
            raise ValueError('illuminant must be "D65" or "D50"')

        self.gamma = gamma
        self.illuminant: Literal["D50", "D65"] = illuminant

        self._observer: str = "CIE 1931 2 Degree Standard Observer"
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
            Access color spaces:

            >>> img = Image.imread('sample.jpg')
            >>> xyz_data = img.color.XYZ[:]
            >>> lab_data = img.color.Lab[:]
            >>> hue = img.color.hsv[..., 0] # hue is the first matrix in the array
        """
        return self._accessors.color
