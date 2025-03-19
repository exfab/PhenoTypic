from skimage.exposure import equalize_adapthist

from .. import Image
from ..abstract import ImagePreprocessor


class CLAHE(ImagePreprocessor):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    This class is used to preprocess images by applying the CLAHE algorithm, which
    enhances the contrast of an image by adjusting the local contrast within specified
    regions (or kernels). This algorithm is particularly useful for improving the
    visibility of features in low-contrast images or images with varying illumination.

    Parameters:
        kernel_size (int): The size of the kernel used for the local histogram. If not
            provided, an adaptive size based on the image dimensions is used.
    """
    def __init__(self, kernel_size: int = None):
        self.kernel_size: int = kernel_size

    def _operate(self, image: Image) -> Image:
        if self.kernel_size is None:
            image.enh_matrix[:] = equalize_adapthist(
                    image=image.enh_matrix[:],
                    kernel_size=int(min(image.matrix.shape[:1]) * (1.0 / 15.0))
            )
            return image
        else:
            image.enh_matrix[:] = equalize_adapthist(
                    image=image.enh_matrix[:],
                    kernel_size=self.kernel_size
            )
            return image

CLAHE.apply.__doc__ = CLAHE._operate.__doc__