from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import gaussian

from ..abc_ import Smoothing
from ..tools_.typing_ import TuneSpec


class GaussianBlur(Smoothing):
    """Smooth noise in detect_mat using isotropic Gaussian convolution.

    Reduces high-frequency noise, scanner artifacts, and minor agar texture
    so that downstream thresholding responds to colony signal rather than
    noise. Colony edges become more coherent at the cost of some spatial
    sharpness.

    For a comparison of denoising approaches, see
    :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Standard deviation of the Gaussian kernel in pixels. Controls
            blur strength. Typical range: 0.5--5.0. Keep below the smallest
            colony radius to avoid merging adjacent colonies. Default: 2.0.
        mode: Boundary handling. Accepted values: ``'reflect'``, ``'constant'``,
            ``'nearest'``. Default: ``'reflect'``.
        cval: Fill value when ``mode='constant'``. Default: 0.0.
        truncate: Kernel extent in standard deviations. Rarely needs
            adjustment. Default: 4.0.

    Returns:
        Image: Input image with ``detect_mat`` smoothed by the Gaussian
        kernel. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values.

    Best For:
        - Plates with visible scanner noise or agar granularity.
        - Pre-filtering before edge-based detectors (Sobel, Canny).
        - Quick preprocessing when speed matters more than edge preservation.

    Consider Also:
        - :class:`MedianFilter` when salt-and-pepper noise dominates and edge
          preservation is important.
        - :class:`LocalEdgeDenoise` for smoothing within regions while keeping
          colony boundaries sharp.
        - :class:`StableDenoise` for highest-quality BM3D denoising on critical
          experiments.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of enhancement before detection.
        :doc:`/how_to/notebooks/denoise_low_light` for a comparison of
        denoising methods.
    """

    sigma: Annotated[float, TuneSpec(0.5, 5.0, log=True)] = 2.0
    mode: Literal["reflect", "constant", "nearest"] = "reflect"
    cval: Annotated[float, TuneSpec(tunable=False)] = 0.0
    truncate: Annotated[float, TuneSpec(tunable=False)] = 4.0

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = gaussian(
                image=image.detect_mat[:],
                sigma=self.sigma,
                mode=self.mode,
                truncate=self.truncate,
                cval=self.cval,
                channel_axis=-1,
        )
        return image
