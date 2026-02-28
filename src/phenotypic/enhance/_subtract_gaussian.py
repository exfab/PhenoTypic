from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.filters import gaussian

from phenotypic.abc_ import ImageEnhancer


class SubtractGaussian(ImageEnhancer):
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
    - n_iter: Number of successive subtraction passes. Multiple passes progressively
      remove residual background not eliminated in the first pass, useful when
      illumination gradients are complex or vary at multiple spatial scales.

    Caveats:
    - If sigma is too low, colonies can be attenuated or produce halos.
    - Very large sigma can oversmooth and retain large shadows or plate rim effects.
    - Background subtraction may re-center intensities around zero; ensure later
      steps handle negative values or re-normalize if needed.
    - Multiple iterations compound clipping effects; n_iter > 3 rarely adds benefit.

    Attributes:
        sigma (float): Gaussian std for background scale; use > colony diameter.
        mode (str): Border handling: 'reflect', 'constant', 'nearest', 'mirror', 'wrap'.
        cval (float): Fill value if `mode='constant'`.
        truncate (float): Gaussian support in standard deviations.
        preserve_range (bool): Preserve input value range during filtering.
        n_iter (int): Number of successive subtraction passes to apply.
    """

    def __init__(
            self,
            sigma: float = 50.0,
            mode: str = "reflect",
            cval: float = 0.0,
            truncate: float = 4.0,
            preserve_range: bool = True,
            n_iter: int = 1,
    ):
        """
        Parameters:
            sigma (float): Background scale. Set larger than colony diameter so
                colonies are preserved while slow illumination is removed.
            mode (str): Border handling; 'reflect' reduces artificial rims on plates.
            cval (float): Fill value when `mode='constant'`.
            truncate (float): Gaussian support in standard deviations (advanced).
            preserve_range (bool): Keep the original intensity range; useful if
                subsequent steps or measurements assume a specific scaling.
            n_iter (int): Number of successive subtraction passes. Must be >= 1.
                One pass (default) removes a single background estimate. Multiple
                passes (2+) iteratively subtract residual background, useful for
                complex or multi-scale illumination gradients.
        """
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")

        self.sigma: float = sigma
        self.mode: str = mode
        self.cval: float = cval
        self.truncate: float = truncate
        self.preserve_range: bool = preserve_range
        self.n_iter: int = n_iter

    def _operate(self, image: Image) -> Image:
        for _ in range(self.n_iter):
            background = gaussian(
                    image=image.detect_mat[:],
                    sigma=self.sigma,
                    mode=self.mode,
                    cval=self.cval,
                    truncate=self.truncate,
                    preserve_range=self.preserve_range,
            )
            image.detect_mat[:] = np.clip((image.detect_mat[:].copy() - background),
                                          a_min=0.0,
                                          a_max=1.0)

        return image
