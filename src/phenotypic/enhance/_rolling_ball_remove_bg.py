from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.restoration import rolling_ball

from phenotypic.abc_ import ImageEnhancer


class RollingBallRemoveBG(ImageEnhancer):
    """
    Rolling-ball background removal (ImageJ-style) for agar plates.

    Models the background as the surface traced by rolling a ball under the
    image intensity landscape, then subtracts it. For colony images, this
    effectively removes slow illumination gradients and agar shading while
    preserving colony structures.

    Use cases (agar plates):
    - Correct uneven backgrounds from scanner vignetting, lid glare, or agar
      thickness variations.
    - Improve segmentation of dark colonies on bright agar by flattening the
      background.

    Tuning and effects:
    - radius: Core scale of the rolling ball. Set larger than typical colony
      diameter so colonies are not smoothed into the background. Too small
      a radius will erode colonies and create halos.
    - kernel: Custom structuring element defining the ball shape. Providing a
      kernel overrides `radius` and allows non-spherical shapes if needed.
    - nansafe: Enable if your images contain masked/NaN regions (e.g., plate
      outside masked to NaN). When False, NaNs may propagate or cause artifacts.

    Caveats:
    - Overly small radius removes real features and can bias size/area metrics.
    - May introduce edge effects near the plate boundary; consider masking the
      plate region or using `nansafe` with an appropriate mask.

    Attributes:
        radius (int): Ball radius (in pixels) controlling the background scale;
            choose > colony diameter.
        kernel (np.ndarray): Optional custom kernel; overrides `radius` when set.
        nansafe (bool): Handle NaNs during computation to respect masked regions.
    """

    def __init__(self,
                 radius: int = 100,
                 kernel: np.ndarray = None,
                 nansafe: bool = False):
        """
        Parameters:
            radius (int): Rolling-ball radius (pixels). Use a value larger than
                colony diameter to avoid removing colony signal. Default 100.
            kernel (np.ndarray): Optional custom ball/footprint; when provided it
                overrides `radius`.
            nansafe (bool): If True, treat NaNs as missing data to avoid artifacts
                when using masked images (e.g., outside the plate).
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
