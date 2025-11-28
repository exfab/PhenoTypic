import colour
import numpy as np

from ..accessor_abstracts import ColorSpaceAccessor


class CieLabAccessor(ColorSpaceAccessor):
    """Provides access to CIE L*a*b* color space representation.

    Converts XYZ color space data to perceptually uniform L*a*b* coordinates
    using the `colour.XYZ_to_Lab <https://colour.readthedocs.io/en/develop/generated/colour.XYZ_to_Lab.html>`_
    function, where L* represents lightness and a*/b* represent color dimensions.
    """

    _accessor_property_name: str = "color.Lab"

    @property
    def _subject_arr(self) -> np.ndarray:
        return colour.XYZ_to_Lab(
            XYZ=self._root_image.color.XYZ[:],
            illuminant=colour.CCS_ILLUMINANTS[self._root_image.observer][self._root_image.illuminant],
        )
