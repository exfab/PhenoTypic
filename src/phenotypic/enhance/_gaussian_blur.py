from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from skimage.filters import gaussian

from ..abc_ import Smoothing


class GaussianBlur(Smoothing):
    """Smooth noise in ``detect_mat`` using isotropic Gaussian convolution.

    Reduces high-frequency noise, scanner artefacts, and minor agar texture
    so that downstream thresholding responds to colony signal rather than
    local pixel variation. Colony edges become more coherent at the cost of
    some spatial sharpness; keeping ``sigma`` below the smallest colony radius
    avoids merging adjacent colonies.

    For a comparison of denoising approaches, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates with visible scanner CCD noise or agar granularity that
          triggers false positives in thresholding.
        - Pre-filtering before edge-based detectors (Sobel, Canny) to reduce
          noise edges.
        - Quick preprocessing when speed matters more than edge preservation.

    Consider Also:
        - :class:`MedianFilter` when salt-and-pepper noise dominates and
          preserving sharp colony edges is important.
        - :class:`LocalEdgeDenoise` for bilateral smoothing within regions
          while keeping colony boundaries sharp.
        - :class:`StableDenoise` for highest-quality BM3D denoising on
          critical low-light experiments.

    Args:
        sigma: Standard deviation of the Gaussian kernel in pixels. Controls
            blur strength. Typical range: 0.5--5.0. Keep below the smallest
            colony radius to avoid merging adjacent colonies. Default: 2.0.
        mode: Boundary handling strategy. Accepted values: ``'reflect'``,
            ``'constant'``, ``'nearest'``. Default: ``'reflect'``.
        cval: Fill value used when ``mode='constant'``. Default: 0.0.
        truncate: Kernel half-width in standard deviations; the kernel extends
            ``truncate * sigma`` pixels from center. Larger values include more
            of the Gaussian tails with minimal quality improvement beyond 4.0.
            Default: 4.0.

    Returns:
        Image: Input image with ``detect_mat`` smoothed by the Gaussian
        kernel. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``mode`` is not one of the accepted values.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a visual
        walkthrough of enhancement before detection.
        :doc:`/how_to/notebooks/denoise_low_light` for a comparison of
        denoising methods on low-light plate images.
    """

    sigma: float = 2.0
    mode: Literal["reflect", "constant", "nearest"] = "reflect"
    cval: float = 0.0
    truncate: float = 4.0

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = gaussian(
                image=image.detect_mat[:],
                sigma=self.sigma,
                mode=self.mode,
                truncate=self.truncate,
                cval=self.cval,
                channel_axis=None,
        )
        return image
