from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image
from skimage.filters import gaussian

from ..abc_ import ImageEnhancer


class GaussianBlur(ImageEnhancer):
    """
    Gaussian blur smoothing for plate images.

    Applies Gaussian smoothing to reduce high-frequency noise and minor texture
    on agar plates (e.g., scanner noise, agar granularity, condensation speckle).
    This can make colony boundaries more coherent and reduce false edges before
    edge detection or thresholding.

    Use cases (agar plates):
    - Suppress “salt-and-pepper” noise and minor agar texture so thresholding is
      driven by colony signal rather than noise.
    - Pre-filter before `SobelFilter` or Laplacian to avoid amplifying noise.
    - Slightly smooth within colonies to make segmentation more compact.

    Tuning and effects:
    - sigma: Controls blur strength. Choose below the typical colony radius to
      avoid merging close colonies. Too large sigma will wash out small colonies
      and narrow gaps between neighbors.
    - mode/cval: Define how edges are handled. For plates, 'reflect' usually
      avoids artificial dark/bright rims. 'constant' with a neutral `cval` may
      be useful for cropped regions.
    - truncate: Larger values include more of the Gaussian tail (slightly slower)
      with subtle effect on smoothness near edges.

    Caveats:
    - Excessive blur merges adjacent colonies and reduces edge sharpness.
    - Do not rely on blur to fix illumination gradients; prefer background
      subtraction (e.g., `GaussianSubtract` or `RollingBallRemoveBG`).

    Attributes:
        sigma (int): Standard deviation of the Gaussian kernel (blur strength).
        mode (str): Edge handling: 'reflect', 'constant', or 'nearest'.
        cval (float): Fill value when `mode='constant'`.
        truncate (float): Radius of kernel in standard deviations; kernel is
            truncated beyond this distance.
    """

    def __init__(self, sigma: int = 2,
                 *,
                 mode: str = 'reflect',
                 cval=0.0,
                 truncate: float = 4.0):
        """
        Parameters:
            sigma (int): Blur strength; start near 1–3 for high-resolution scans.
                Keep below the colony radius to avoid merging colonies.
            mode (str): Boundary handling. 'reflect' is a safe default for plates;
                'constant' may require setting `cval` close to background.
            cval (float): Constant fill value when `mode='constant'`.
            truncate (float): Kernel extent in standard deviations. Rarely needs
                adjustment; larger values slightly widen the effective kernel.
        """
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
