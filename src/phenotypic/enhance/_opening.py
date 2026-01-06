from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from PIL import ImageEnhance
from skimage import morphology

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ImageEnhancer, FootprintMixin


class Opening(ImageEnhancer, FootprintMixin):
    def __init__(self, shape: Literal["square", "diamond", "disk"] = "square",
                 width: int = 5):
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = morphology.opening(
                image=image.enh_gray[:],
                footprint=self._make_footprint(
                        shape=self.shape,
                        width=self.width,
                ),
        )
        return image
