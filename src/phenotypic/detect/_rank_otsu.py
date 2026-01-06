from __future__ import annotations

import numpy as np
import skimage.filters.rank as rank
from skimage.util import img_as_ubyte, img_as_uint
from phenotypic.abc_ import ObjectDetector, FootprintMixin
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image


class RankOtsuDetector(ObjectDetector, FootprintMixin):
    def __init__(
            self,
            shape: Literal["square", "diamond", "disk"] = "square",
            width: int | None = None,
            ignore_zeros: bool = False,
    ):
        self.shape = shape
        self.width = width
        self.ignore_zeros = ignore_zeros

    def _operate(self, image: Image) -> Image:
        enh_gray = img_as_ubyte(image.enh_gray[:])
        if self.ignore_zeros:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[enh_gray.nonzero()] = 1
            mask = mask > 0
        else:
            mask = None

        width = min(image.shape[:2]) // 8 if self.width is None else self.width

        image.objmask[:] = enh_gray >= rank.otsu(
                image=enh_gray,
                footprint=self._make_footprint(
                        shape=self.shape,
                        width=width,
                ),
                mask=mask,
        )
        return image
