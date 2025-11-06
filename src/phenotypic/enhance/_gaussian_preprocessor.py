from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import gaussian

from ..abc_ import ImageEnhancer


class GaussianBlur(ImageEnhancer):
    """
    Applies a Gaussian blur filter to an image.

    The GaussianBlur class inherits from ImageEnhancer and provides the functionality
    to apply Gaussian blurring to an image. It allows customization of the Gaussian
    kernel using parameters such as sigma, mode, cval, and truncate. This class is
    designed for enhancing images through controlled blurring.

    Attributes:
        sigma (int): Standard deviation for Gaussian kernel. Determines the amount
            of blurring. Must be an integer.
        mode (str): Boundary mode to handle edges. Supported modes are 'reflect',
            'constant', and 'nearest'.
        cval (float): Value to set constant boundaries when mode is 'constant'.
        truncate (float): Radius of the Gaussian kernel in terms of standard
            deviations. Truncates the kernel beyond this value.
    """

    def __init__(self, sigma: int = 2,
                 *,
                 mode: str = 'reflect',
                 cval=0.0,
                 truncate: float = 4.0):
        if isinstance(sigma, int):
            self.sigma = sigma
        else:
            raise TypeError('sigma must be an integer')

        if mode in ['reflect', 'constant', 'nearest']:
            self.mode = mode
        else:
            raise ValueError('mode must be one of "reflect", "constant", "nearest"')

        self.cval = cval

        self.truncate = truncate

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = gaussian(
                image=image.enh_gray[:],
                sigma=self.sigma,
                mode=self.mode,
                truncate=self.truncate,
                cval=self.cval,
                channel_axis=-1
        )
        return image
