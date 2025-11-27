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
    """
    Provides unified access to all color space representations of an image.
    
    This accessor groups together various color space transformations and representations
    including device color spaces (HSV) and CIE standard color spaces (XYZ, Lab, xy).
    All color space conversions are performed lazily and cached for efficiency.
    
    Attributes:
        _root_image: The parent Image object that this accessor is bound to.
        
    Examples:
        >>> from phenotypic import Image
        >>> img = Image.imread('sample.jpg')
        >>> xyz_data = img.color.XYZ[:]
        >>> lab_data = img.color.Lab[:]
        >>> hsv_hue = img.color.hsv[..., 0]
    """

    def __init__(self, root_image: Image):
        """
        Initialize the ColorAccessor with a reference to the parent image.
        
        Args:
            root_image: The Image object that this accessor is bound to.
        """
        self._root_image = root_image
        self._xyz = XyzAccessor(root_image)
        self._xyz_d65 = XyzD65Accessor(root_image)
        self._cielab = CieLabAccessor(root_image)
        self._xy = xyChromaticityAccessor(root_image)
        self._hsv = HsvAccessor(root_image)

    @property
    def XYZ(self) -> XyzAccessor:
        """
        Access the CIE XYZ color space representation.
        
        Converts the image to the XYZ color space under the image's configured
        illuminant. XYZ is a device-independent color space that forms the basis
        for many other color space transformations.
        
        Returns:
            XyzAccessor: Accessor providing numpy-like interface to XYZ data.
        """
        return self._xyz

    @property
    def XYZ_D65(self) -> XyzD65Accessor:
        """
        Access the CIE XYZ color space under D65 illuminant.
        
        Provides XYZ representation specifically under D65 illuminant viewing
        conditions, applying chromatic adaptation if necessary.
        
        Returns:
            XyzD65Accessor: Accessor providing numpy-like interface to XYZ D65 data.
        """
        return self._xyz_d65

    @property
    def xy(self) -> xyChromaticityAccessor:
        """
        Access the CIE xy chromaticity coordinates.
        
        Provides 2D chromaticity representation, expressing color independently
        of luminance. Useful for color analysis and gamut visualization.
        
        Returns:
            xyChromaticityAccessor: Accessor providing numpy-like interface to xy data.
        """
        return self._xy

    @property
    def Lab(self) -> CieLabAccessor:
        """
        Access the CIE L*a*b* color space representation.
        
        Converts to perceptually uniform L*a*b* color space where L* represents
        lightness and a*/b* represent color-opponent dimensions. This space is
        particularly useful for color difference calculations.
        
        Returns:
            CieLabAccessor: Accessor providing numpy-like interface to Lab data.
        """
        return self._cielab

    @property
    def hsv(self) -> HsvAccessor:
        """
        Access the HSV (Hue, Saturation, Value) color space representation.
        
        Provides access to device-dependent HSV color space, which is intuitive
        for color selection and manipulation. Includes methods for extracting
        individual components and object-specific color analysis. All values are in normalized
        range between (0, 1).
        
        Returns:
            HsvAccessor: Accessor providing numpy-like interface to HSV data.
        """
        return self._hsv
