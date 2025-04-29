from skimage.filters import gaussian

from ..abstract import ImageEnhancer
from .. import Image


class GaussianSmoother(ImageEnhancer):
    """
    Applies Gaussian smoothing (blurring) to the enhanced matrix of an image; Helps with salt & pepper noise.

    The GaussianPreprocessor class is used to enhance the pixel quality of an image by applying a
    Gaussian filter. It operates on the enhanced matrix of an image object. It allows customization
    of the Gaussian filter parameters. The class is designed for use in image enhancement pipelines.

    Parameters:
        sigma (float): The standard deviation for Gaussian kernel. Higher values result in more
            blurring. Default is 2.
        mode (str): The mode used to handle pixels outside the image boundaries. Common modes
            include 'reflect', 'constant', 'nearest', etc. Default is 'reflect'.
        truncate (float): Truncate the filter at this many standard deviations. This determines
            the size of the Gaussian kernel. Default is 4.0.
        channel_axis (Optional[int]): The axis in the image that represents color channels. Set
            to None for grayscale images. Default is None.
    """
    def __init__(self, sigma=2, mode='reflect', truncate=4.0, channel_axis=None):
        self.sigma = sigma
        self.mode = mode
        self.truncate = truncate
        self.channel_axis = channel_axis

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = gaussian(
                image=image.enh_matrix[:],
                sigma=self.sigma,
                mode=self.mode,
                truncate=self.truncate,
                channel_axis=self.channel_axis
        )
        return image