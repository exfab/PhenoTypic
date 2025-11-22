from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from skimage.filters import gaussian

from phenotypic.abc_ import ImageEnhancer


class GaussianSubtract(ImageEnhancer):
    """
    Background correction by Gaussian subtraction.

    Estimates a smooth background via Gaussian blur and subtracts it from the
    image. For agar plate colony analysis, this removes gradual illumination
    gradients (vignetting, agar thickness, scanner shading) while retaining sharp
    colony features, improving downstream thresholding and edge detection.

    Use cases (agar plates):
    - Correct uneven lighting across plates or across scan beds.
    - Enhance visibility of dark colonies on bright agar by flattening the
      background.
    - Normalize batches captured with varying exposure/illumination profiles.

    Tuning and effects:
    - sigma: Sets the spatial scale of the background. Choose a value larger than
      the typical colony diameter so colonies are not treated as background. Too
      small will subtract colony signal and can invert contrast around edges.
    - mode/cval: Controls border handling; 'reflect' often avoids rim artifacts
      on circular plates. 'constant' may require matching `cval` to background.
    - truncate: Extent of the Gaussian in standard deviations; rarely needs change.
    - preserve_range: Keep original intensity range after filtering; useful when
      subsequent steps assume the same data range/bit depth.

    Caveats:
    - If sigma is too low, colonies can be attenuated or produce halos.
    - Very large sigma can oversmooth and retain large shadows or plate rim effects.
    - Background subtraction may re-center intensities around zero; ensure later
      steps handle negative values or re-normalize if needed.

    Attributes:
        sigma (float): Gaussian std for background scale; use > colony diameter.
        mode (str): Border handling: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
        cval (float): Fill value if `mode='constant'`.
        truncate (float): Gaussian support in standard deviations.
        preserve_range (bool): Preserve input value range during filtering.
    """

    def __init__(self,
                 sigma: float = 50.0,
                 mode: str = 'reflect',
                 cval: float = 0.0,
                 truncate: float = 4.0,
                 preserve_range: bool = True):
        """
        Parameters:
            sigma (float): Background scale. Set larger than colony diameter so
                colonies are preserved while slow illumination is removed.
            mode (str): Border handling; 'reflect' reduces artificial rims on plates.
            cval (float): Fill value when `mode='constant'`.
            truncate (float): Gaussian support in standard deviations (advanced).
            preserve_range (bool): Keep the original intensity range; useful if
                subsequent steps or measurements assume a specific scaling.
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
        image.enh_gray[:] = image.enh_gray[:].copy() - background
        return image
