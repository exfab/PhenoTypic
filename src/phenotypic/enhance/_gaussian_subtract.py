from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.filters import gaussian

from phenotypic.abc_ import ImageEnhancer


class GaussianSubtract(ImageEnhancer):
    """
    Provides methods for image enhance on the enhanced grayscale gray
    using Gaussian background subtraction.

    The class implements an algorithm that estimates the background using a
    Gaussian blur and subtracts it from the original image. This technique is
    useful for removing slowly varying background illumination while preserving
    sharp features and edges in the foreground.

    Attributes:
        sigma (float): Standard deviation for Gaussian kernel. Larger values
            create a smoother background estimate.
        mode (str): How to handle values outside the image borders. Options
            include 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
        cval (float): Value to fill past edges when mode is 'constant'.
        truncate (float): Truncate the filter at this many standard deviations.
        preserve_range (bool): Whether to keep the original range of values.
    """

    def __init__(self,
                 sigma: float = 50.0,
                 mode: str = 'reflect',
                 cval: float = 0.0,
                 truncate: float = 4.0,
                 preserve_range: bool = True):
        """
        Initializes an instance of the class with parameters to configure Gaussian
        background subtraction behavior.

        Args:
            sigma (float): Standard deviation for Gaussian kernel. Larger values
                result in more aggressive background smoothing. Default is 50.0.
            mode (str): The mode parameter determines how the array borders are
                handled. Default is 'reflect'.
            cval (float): Value to fill past edges of the input if mode is 'constant'.
                Default is 0.0.
            truncate (float): Truncate the filter at this many standard deviations.
                Default is 4.0.
            preserve_range (bool): Whether to keep the original range of values in
                the output. Default is True.
        """
        self.sigma: float = sigma
        self.mode: str = mode
        self.cval: float = cval
        self.truncate: float = truncate
        self.preserve_range: bool = preserve_range

    def _operate(self, image: Image):
        background = gaussian(image=image.enh_gray[:],
                              sigma=self.sigma,
                              mode=self.mode,
                              cval=self.cval,
                              truncate=self.truncate,
                              preserve_range=self.preserve_range)
        image.enh_gray[:] = image.enh_gray[:] - background
        return image
