from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from skimage.exposure import equalize_adapthist

from phenotypic.abc_ import ImageEnhancer


class CLAHE(ImageEnhancer):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)

    This class is a specific implementation of the ImageEnhancer, designed to
    apply the CLAHE algorithm to enhance the contrast of an image. CLAHE is a
    variant of adaptive histogram equalization that limits the amplification of
    noise by clipping the histogram before equalization. It is useful for improving
    the contrast of images with poor lighting conditions or high dynamic range.

    Attributes:
        kernel_size (int or None): The size of the kernel to use for the histogram
            equalization. If not provided, it will be automatically calculated
            using the dimensions of the image.
        clip_limit (float): The contrast limit for local areas (normalized). A
            smaller value decreases the contrast amplification. Defaults to 0.01.
    """

    def __init__(self, kernel_size: int | None = None, clip_limit: float = 0.01, ):
        self.kernel_size: int = kernel_size
        self.clip_limit: float = clip_limit

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = equalize_adapthist(
                image=image.enh_gray[:],
                kernel_size=self.kernel_size if self.kernel_size else self._auto_kernel_size(image),
                clip_limit=self.clip_limit,
                nbins=2 ** int(image.bit_depth)
        )
        return image

    @staticmethod
    def _auto_kernel_size(image: Image) -> int:
        return int(min(image.gray.shape[:1])*(1.0/15.0))
