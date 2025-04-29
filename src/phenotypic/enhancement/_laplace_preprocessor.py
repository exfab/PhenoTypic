from skimage.filters import laplace
from typing import Optional
import numpy as np

from ..abstract import ImageEnhancer
from .. import Image


class LaplaceEnhancer(ImageEnhancer):
    """
    The LaplaceEnhancer class applies a Laplacian filter to an image's enhanced matrix.

    This class is designed to preprocess images by enhancing their features using
    a Laplacian operation. The filter applies edge detection, which emphasizes
    areas of rapid intensity change. Users can specify the kernel size and an
    optional mask for the operation.

    Parameters:
        kernel_size (Optional[int]): The size of the kernel used for the Laplacian filter.
        mask (Optional[numpy.ndarray]): An optional mask to limit the operation to
            specified regions of the image.
    """
    def __init__(self, kernel_size: Optional[int] = 3, mask: Optional[np.ndarray] = None):
        self.kernel_size: Optional[np.ndarray] = kernel_size
        self.mask:Optional[np.ndarray] = mask

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = laplace(
                image=image.enh_matrix[:],
                ksize=self.kernel_size,
                mask=self.mask,
        )
        return image