from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import gaussian

from ..abstract import ImageEnhancer


class GaussianSmoother(ImageEnhancer):
    """
    Applies Gaussian smoothing (blurring) to the enhanced matrix of an image; Helps with salt & pepper noise.

    The GaussianPreprocessor class is used to enhance the pixel quality of an image by applying a
    Gaussian filter. It operates on the enhanced matrix of an image object. It allows customization
    of the Gaussian filter parameters. The class is designed for use in image enhancement prefab.

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

    def __init__(self, sigma: int = 2, mode: str = 'reflect', truncate: float = 4.0):
        if isinstance(sigma, int):
            self.sigma = sigma
        else:
            raise TypeError('sigma must be an integer')

        if mode in ['reflect', 'constant', 'nearest']:
            self.mode = mode
        else:
            raise ValueError('mode must be one of "reflect", "constant", "nearest"')
        
        self.truncate = truncate

    def _operate(self, image: Image) -> Image:
        image.enh_matrix[:] = gaussian(
                image=image.enh_matrix[:],
                sigma=self.sigma,
                mode=self.mode,
                truncate=self.truncate,
                channel_axis=-1
        )
        return image
