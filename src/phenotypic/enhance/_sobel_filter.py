from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.filters import sobel

from phenotypic.ABC_ import ImageEnhancer


class SobelFilter(ImageEnhancer):
    """Applies the Sobel filter to enhance images.

    This class provides functionality for applying the Sobel filter to
    images. This class highlights edges within the image. Consider combining with `GaussianBlur`.

    Attributes:
        None
    """

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = sobel(image=image.enh_matrix[:])
        return image
