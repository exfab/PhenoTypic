import colour
import numpy as np

from ..accessor_abstracts._image_accessor_base import ImageAccessorBase

class CieLabAccessor(ImageAccessorBase):
    @property
    def _subject_arr(self) -> np.ndarray:
        return colour.XYZ_to_Lab(
            XYZ=self._root_image.CieXYZ[:],
            illuminant=colour.CCS_ILLUMINANTS[self._root_image.observer][self._root_image.illuminant],
        )

    def __getitem__(self, key) -> np.ndarray:
        return self._subject_arr[key].copy()
