import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class xyChromaticityAccessor(ColorSpaceAccessor):
    """Provides access to CIE xy chromaticity coordinates.

    This accessor converts the parent image's XYZ color space data to
    2D CIE xy chromaticity coordinates. Chromaticity coordinates represent
    color composition independently of luminance (brightness), using only
    the normalized x and y values from the CIE XYZ color space.

    The xy chromaticity coordinates are computed on-the-fly from the
    parent image's XYZ data using the `colour.XYZ_to_xy
    <https://colour.readthedocs.io/en/latest/generated/colour.XYZ_to_xy.html>`_
    function. The resulting data is read-only to prevent accidental
    modification of the computed color transformation.

    Attributes:
        _accessor_property_name (str): The property name on Image that returns
            this accessor. Set to "color.xy" for CIE xy chromaticity access.

    Examples:
        .. dropdown:: Access CIE xy chromaticity coordinates from an image

            .. code-block:: python

                image = Image(...)
                xy_coords = image.color.xy[:]  # Get full xy array
                xy_subset = image.color.xy[100:200, 50:150]  # Get region of interest
                shape = image.color.xy.shape  # (height, width, 2)
    """

    _accessor_property_name: str = "color.xy"

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return CIE xy chromaticity coordinates derived from XYZ data.

        Converts the parent image's XYZ color space data to 2D CIE xy
        chromaticity coordinates. The xy values represent normalized color
        composition (hue and saturation) independent of luminance.

        The computation uses the standard CIE formula:
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        Returns:
            np.ndarray: Array of shape (height, width, 2) containing CIE xy
                chromaticity coordinates. Each pixel has [x, y] values in the
                range [0, 1], where (x, y) are the normalized chromaticity
                coordinates.

        Notes:
            - This is a read-only property. The returned array cannot be
              modified due to ColorSpaceAccessor's design.
            - Values are computed on-the-fly each time this property is accessed.
            - The conversion is performed by the Colour science library's
              `colour.XYZ_to_xy` function.
        """
        return colour.XYZ_to_xy(XYZ=self._root_image.color.XYZ[:])
