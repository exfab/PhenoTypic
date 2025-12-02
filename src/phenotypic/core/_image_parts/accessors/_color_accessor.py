from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from ..color_space_accessors._xyz_accessor import XyzAccessor
from ..color_space_accessors._xyz_d65_accessor import XyzD65Accessor
from ..color_space_accessors._cielab_accessor import CieLabAccessor
from ..color_space_accessors._chromaticity_xy_accessor import xyChromaticityAccessor
from ._hsv_accessor import HsvAccessor


class ColorAccessor:
    """Provides unified access to all color space representations of an image.

    This accessor class serves as a facade to multiple specialized color space accessors,
    grouping together various color space transformations and representations. It provides
    convenient access to both device-dependent color spaces (HSV) and CIE standard color
    spaces (XYZ, XYZ-D65, L*a*b*, and xy chromaticity).

    All color space conversions are computed on-demand and returned as read-only numpy
    arrays. Individual accessor objects maintain caching to avoid redundant computations
    when the same color space is accessed multiple times.

    The parent Image object configuration (illuminant, _observer model, gamma encoding) is
    used consistently across all color space transformations to ensure coherent color
    space analysis.

    Attributes:
        _root_image (Image): The parent Image object that this accessor is bound to.
            Used to access raw RGB/grayscale data and color properties (illuminant, _observer).
        _xyz (XyzAccessor): Accessor for CIE XYZ color space representation.
        _xyz_d65 (XyzD65Accessor): Accessor for CIE XYZ under D65 illuminant.
        _cielab (CieLabAccessor): Accessor for perceptually uniform L*a*b* color space.
        _xy (xyChromaticityAccessor): Accessor for CIE xy chromaticity coordinates.
        _hsv (HsvAccessor): Accessor for HSV color space representation.

    Examples:
        .. dropdown:: Access different color spaces from an image

            .. code-block:: python

                from phenotypic import Image

                img = Image.imread('sample.jpg')

                # Access CIE XYZ color space
                xyz_data = img.color.XYZ[:]
                print(xyz_data.shape)  # (height, width, 3)

                # Access perceptually uniform Lab color space
                lab_data = img.color.Lab[:]

                # Access HSV components (hue is first channel)
                hue_channel = img.color.hsv[..., 0]
                saturation_channel = img.color.hsv[..., 1]
                brightness_channel = img.color.hsv[..., 2]

                # Access chromaticity coordinates
                xy_coords = img.color.xy[:]

        .. dropdown:: Use color spaces for analysis

            .. code-block:: python

                import numpy as np

                # Calculate color differences using Lab space
                lab_data = img.color.Lab[:]
                differences = np.sqrt(
                    (lab_data[..., 0] - 50)**2 +
                    (lab_data[..., 1] - 25)**2 +
                    (lab_data[..., 2] - 10)**2
                )

                # Extract hue for color classification
                hue = img.color.hsv[..., 0] * 360  # Convert to degrees
                red_mask = (hue < 30) | (hue > 330)
    """

    def __init__(self, root_image: Image):
        """Initialize the ColorAccessor with a reference to the parent image.

        Creates all subordinate color space accessor objects, each providing specialized
        access to a specific color space representation. All accessors share the same
        parent Image reference to ensure consistent color properties (illuminant, _observer)
        are used across all color space transformations.

        Args:
            root_image (Image): The Image object that this accessor is bound to.
                Used to access RGB/grayscale data and color configuration properties
                (illuminant, _observer, gamma encoding).

        Examples:
            .. dropdown:: Access ColorAccessor through Image.color property

                .. code-block:: python

                    from phenotypic import Image

                    img = Image.imread('photo.jpg')
                    # Direct initialization is not typically needed - use img.color instead
                    accessor = img.color  # Returns ColorAccessor instance
        """
        self._root_image = root_image
        self._xyz = XyzAccessor(root_image)
        self._xyz_d65 = XyzD65Accessor(root_image)
        self._cielab = CieLabAccessor(root_image)
        self._xy = xyChromaticityAccessor(root_image)
        self._hsv = HsvAccessor(root_image)

    @property
    def XYZ(self) -> XyzAccessor:
        """Access the CIE XYZ color space representation.

        Provides access to the CIE XYZ color space representation of the image,
        computed under the parent Image's configured illuminant. XYZ is a device-independent
        color space that forms the basis for many other color space transformations.

        The XYZ color space separates color information into lightness-related luminance (Y)
        and chromaticity values (X, Z). It is particularly useful as an intermediate
        representation for converting between different color spaces.

        Returns:
            XyzAccessor: Accessor providing numpy-like interface to XYZ data.
                Supports array indexing and slicing. Shape is (height, width, 3) where
                the three channels correspond to X, Y, and Z values.

        See Also:
            XYZ_D65: XYZ representation specifically under D65 illuminant conditions.
            xy: Normalized chromaticity coordinates derived from XYZ.

        Examples:
            .. dropdown:: Access and work with XYZ color space

                .. code-block:: python

                    from phenotypic import Image

                    img = Image.imread('photo.jpg')

                    # Get full XYZ array
                    xyz_array = img.color.XYZ[:]
                    print(xyz_array.shape)  # (height, width, 3)

                    # Extract individual channels
                    X = img.color.XYZ[..., 0]
                    Y = img.color.XYZ[..., 1]  # Luminance
                    Z = img.color.XYZ[..., 2]

                    # Slice specific region
                    roi_xyz = img.color.XYZ[100:200, 100:200, :]
        """
        return self._xyz

    @property
    def XYZ_D65(self) -> XyzD65Accessor:
        """Access the CIE XYZ color space under D65 illuminant.

        Provides XYZ representation specifically under D65 (standard daylight) illuminant
        viewing conditions. If the parent Image uses a different illuminant (e.g., D50),
        chromatic adaptation is automatically applied to transform the data to D65 conditions.

        D65 is the CIE standard daylight illuminant with a color temperature of approximately
        6504 K. It is the most commonly used illuminant in color science, photography, and
        display technology. Using D65 as a reference standard enables comparison of color data
        across different imaging systems.

        Returns:
            XyzD65Accessor: Accessor providing numpy-like interface to XYZ D65 data.
                Supports array indexing and slicing. Shape is (height, width, 3) where
                the three channels correspond to X, Y, and Z values under D65 conditions.

        See Also:
            XYZ: XYZ representation under the image's configured illuminant.
            Lab: Perceptually uniform color space (typically uses D65).

        Examples:
            .. dropdown:: Access XYZ color space under D65 illuminant

                .. code-block:: python

                    from phenotypic import Image

                    img = Image.imread('photo.jpg')

                    # Get XYZ D65 representation
                    xyz_d65 = img.color.XYZ_D65[:]

                    # Use D65 for standardized color comparison
                    luminance_d65 = img.color.XYZ_D65[..., 1]

                    # If original illuminant differs from D65, chromatic adaptation is applied
                    # For images originally in D65, this is equivalent to XYZ
        """
        return self._xyz_d65

    @property
    def xy(self) -> xyChromaticityAccessor:
        """Access the CIE xy chromaticity coordinates.

        Provides 2D chromaticity representation derived from the CIE XYZ color space.
        Chromaticity coordinates express color independently of luminance, isolating
        the hue and saturation information. This is particularly useful for color analysis,
        gamut visualization, and studying color without brightness variation.

        The xy coordinates are derived from XYZ using the formulas:
        x = X / (X + Y + Z), y = Y / (X + Y + Z)

        This normalized representation is device-independent and widely used in color
        science for visualizing color spaces on the CIE 1931 chromaticity diagram.

        Returns:
            xyChromaticityAccessor: Accessor providing numpy-like interface to xy data.
                Supports array indexing and slicing. Shape is (height, width, 2) where
                the two channels correspond to x and y chromaticity coordinates
                (both in range [0, 1]).

        See Also:
            XYZ: Full 3D color space representation including luminance.
            Lab: Perceptually uniform color space incorporating both chromaticity and lightness.

        Examples:
            .. dropdown:: Access and visualize xy chromaticity coordinates

                .. code-block:: python

                    from phenotypic import Image
                    import matplotlib.pyplot as plt

                    img = Image.imread('photo.jpg')

                    # Get chromaticity coordinates
                    xy_coords = img.color.xy[:]
                    x = img.color.xy[..., 0]
                    y = img.color.xy[..., 1]

                    # Plot color on CIE 1931 chromaticity diagram
                    plt.scatter(x.flatten(), y.flatten(), c=img.rgb[:].reshape(-1, 3)/255)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.show()

                    # Analyze color composition without luminance effects
                    roi_xy = img.color.xy[100:200, 100:200, :]
        """
        return self._xy

    @property
    def Lab(self) -> CieLabAccessor:
        """Access the CIE L*a*b* color space representation.

        Provides access to the perceptually uniform CIE L*a*b* color space, derived
        from the image's XYZ representation. The Lab color space is designed to approximate
        human visual perception, making it ideal for color analysis, color correction,
        and calculating perceptually meaningful color differences.

        The three channels represent:
        - L* (lightness): Ranges from 0 (black) to 100 (white), representing perceptual brightness
        - a* (green-red opponent): Negative values indicate green, positive indicate red
        - b* (blue-yellow opponent): Negative values indicate blue, positive indicate yellow

        Because Lab is perceptually uniform, Euclidean distances in Lab space correspond
        to perceptual color differences as seen by human observers. This makes it superior
        to RGB for color analysis and comparison tasks.

        Returns:
            CieLabAccessor: Accessor providing numpy-like interface to Lab data.
                Supports array indexing and slicing. Shape is (height, width, 3) where
                the three channels correspond to L*, a*, and b* values.

        See Also:
            XYZ: Device-independent color space basis for Lab calculation.
            XYZ_D65: XYZ under standard D65 illuminant (Lab typically computed from this).

        Examples:
            .. dropdown:: Access Lab color space and calculate color differences

                .. code-block:: python

                    from phenotypic import Image
                    import numpy as np

                    img = Image.imread('photo.jpg')

                    # Access Lab color space
                    lab = img.color.Lab[:]
                    L = img.color.Lab[..., 0]  # Lightness (0-100)
                    a = img.color.Lab[..., 1]  # Green (-) to Red (+)
                    b = img.color.Lab[..., 2]  # Blue (-) to Yellow (+)

                    # Calculate perceptually meaningful color difference
                    reference_lab = np.array([50, 0, 0])  # Mid-gray
                    color_difference = np.sqrt(
                        (lab[..., 0] - reference_lab[0])**2 +
                        (lab[..., 1] - reference_lab[1])**2 +
                        (lab[..., 2] - reference_lab[2])**2
                    )

                    # Find pixels close to a reference color (within ΔE of 5)
                    similar_mask = color_difference < 5

                    # Adjust lightness across entire image
                    brightened = lab.copy()
                    brightened[..., 0] += 10  # Increase L* by 10
        """
        return self._cielab

    @property
    def hsv(self) -> HsvAccessor:
        """Access the HSV (Hue, Saturation, Value) color space representation.

        Provides access to the device-dependent HSV color space, which represents colors
        in a way that is intuitive for human color selection and manipulation. HSV is
        particularly useful for color-based filtering, hue-specific analysis, and applications
        where color properties need to be adjusted independently.

        The three channels represent (all normalized to range [0, 1]):
        - H (hue): Color type, ranges from 0 to 1 (corresponds to 0 to 360 degrees)
        - S (saturation): Color intensity/purity, 0 (grayscale) to 1 (pure color)
        - V (value): Brightness/luminosity, 0 (black) to 1 (brightest)

        HSV is computed from RGB and is device-dependent (unlike CIE color spaces).
        However, it is more intuitive for color-based operations like selecting all red
        pixels or adjusting hue.

        The HsvAccessor includes additional analysis methods such as histograms and
        visualization of HSV components. Note: HSV is only available for RGB images;
        grayscale images do not have HSV representation.

        Returns:
            HsvAccessor: Accessor providing numpy-like interface to HSV data.
                Supports array indexing and slicing. Shape is (height, width, 3) where
                the three channels correspond to H, S, V values (each in range [0, 1]).

        Raises:
            AttributeError: If called on a grayscale image without RGB data.

        See Also:
            XYZ: Device-independent color space alternative.
            Lab: Perceptually uniform color space alternative.

        Examples:
            .. dropdown:: Access HSV components and perform color-based filtering

                .. code-block:: python

                    from phenotypic import Image

                    img = Image.imread('photo.jpg')

                    # Access HSV components
                    hsv = img.color.hsv[:]
                    hue = img.color.hsv[..., 0]  # 0 to 1
                    saturation = img.color.hsv[..., 1]  # 0 to 1
                    brightness = img.color.hsv[..., 2]  # 0 to 1

                    # Convert hue to degrees (0-360)
                    hue_degrees = hue * 360

                    # Extract red pixels (hue near 0 or near 360)
                    red_hue = hue_degrees
                    red_mask = (red_hue < 30) | (red_hue > 330)

                    # Extract highly saturated colors
                    saturated_mask = saturation > 0.5

                    # Find bright colors
                    bright_mask = brightness > 0.7

                    # Visualize HSV components
                    fig, axes = img.color.hsv.show()
        """
        return self._hsv
