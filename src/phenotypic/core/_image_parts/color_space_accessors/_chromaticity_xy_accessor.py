import colour
import numpy as np

from ..accessor_abstracts._image_accessor_base import ImageAccessorBase

from phenotypic.util.colourspaces_ import sRGB_D50

from phenotypic.util.funcs_ import normalize_rgb_bitdepth


class xyChromaticityAccessor(ImageAccessorBase):

    @property
    def _subject_arr(self) -> np.ndarray:
        return colour.XYZ_to_xy(XYZ=self._root_image.CieXYZ[:])

    def __getitem__(self, key) -> np.ndarray:
        return self._subject_arr[key].copy()
