import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class xyChromaticityAccessor(ColorSpaceAccessor):
    """Provides access to CIE xy chromaticity coordinates.

    Converts XYZ color space data to 2D chromaticity coordinates
    using the `colour.XYZ_to_xy <https://colour.readthedocs.io/en/develop/generated/colour.XYZ_to_xy.html>`_
    function, representing color independently of luminance.
    """

    _accessor_property_name: str = "color.xy"

    @property
    def _subject_arr(self) -> np.ndarray:
        return colour.XYZ_to_xy(XYZ=self._root_image.color.XYZ[:])
