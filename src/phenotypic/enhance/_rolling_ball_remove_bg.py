from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.restoration import rolling_ball

from phenotypic.abc_ import ImageEnhancer


class RollingBallRemoveBG(ImageEnhancer):
    """
    Provides methods for image enhance on the enhanced grayscale gray
    using a rolling ball subtraction technique.

    The class implements an algorithm that uses a rolling ball approach to subtract
    background or smooth out variations in image brightness. It is configurable with
    parameters such as kernel size, radius for operations, and threading options.
    This is particularly used in preprocessing steps for image analysis where uniform
    lighting is desired.

    Attributes:
        radius (int): The radius for the rolling ball operation defining the extent of
            influence or smoothing.
        kernel (np.ndarray): An optional kernel gray used for computational purposes
            during the enhance operation.
        nansafe (bool): A boolean flag indicating whether computations should handle
            NaN (Not a Number) values safely.
        num_threads (int): The number of threads allocated for computations to enable
            parallel processing, if supported.
    """

    def __init__(self,
                 radius: int = 100,
                 kernel: np.ndarray = None,
                 nansafe: bool = False):
        """
        Initializes an instance of the class with parameters to configure its behavior.

        Args:
            radius (int): The radius defining the extent of an operation or calculation. Default
                is 100.
            kernel (np.ndarray): an alternative way to express the rolling ball operation
            nansafe (bool): A flag to enable or disable operations that handle NaN (Not a Number)
                values gracefully. Default is False.
        """
        self.radius: int = radius
        self.kernel: np.ndarray = kernel
        self.nansafe: bool = nansafe

    def _operate(self, image: Image):
        image.enh_gray[:] = image.enh_gray[:] \
                            - rolling_ball(image=image.enh_gray[:],
                                           radius=self.radius,
                                           kernel=self.kernel,
                                           nansafe=self.nansafe)
        return image
