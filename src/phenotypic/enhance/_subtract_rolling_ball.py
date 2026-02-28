from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.restoration import rolling_ball

from phenotypic.abc_ import ImageEnhancer


class SubtractRollingBall(ImageEnhancer):
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
    - width: Core scale of the rolling ball. Set larger than typical colony
      diameter so colonies are not smoothed into the background. Too small
      a width will erode colonies and create halos.
    - kernel: Custom structuring element defining the ball shape. Providing a
      kernel overrides `width` and allows non-spherical shapes if needed.
    - nansafe: Enable if your images contain masked/NaN regions (e.g., plate
      outside masked to NaN). When False, NaNs may propagate or cause artifacts.

    Caveats:
    - Overly small width removes real features and can bias size/area metrics.
    - May introduce edge effects near the plate boundary; consider masking the
      plate region or using `nansafe` with an appropriate mask.

    Attributes:
        radius (int): Ball width (in pixels) controlling the background scale;
            choose > colony diameter.
        kernel (np.ndarray): Optional custom kernel; overrides `width` when set.
        nansafe (bool): Handle NaNs during computation to respect masked regions.
    """

    def __init__(
            self, radius: int = 100, kernel: np.ndarray = None, nansafe: bool = False
    ):
        """
        Parameters:
            radius (int): Rolling-ball width (pixels). Use a value larger than
                colony diameter to avoid removing colony signal. Default 100.
            kernel (np.ndarray): Optional custom ball/shape; when provided it
                overrides `width`.
            nansafe (bool): If True, treat NaNs as missing data to avoid artifacts
                when using masked images (e.g., outside the plate).
        """
        self.radius: int = radius
        self.kernel: np.ndarray = kernel
        self.nansafe: bool = nansafe

    def _operate(self, image: Image):
        image.detect_mat[:] = image.detect_mat[:] - rolling_ball(
                image=image.detect_mat[:],
                radius=self.radius,
                kernel=self.kernel,
                nansafe=self.nansafe,
        )
        return image
