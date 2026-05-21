from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from skimage.filters import gaussian

from phenotypic.abc_ import ImageEnhancer


class SubtractGaussian(ImageEnhancer):
    """Remove background from ``detect_mat`` by subtracting a Gaussian-blurred estimate.

    Estimates a smooth background via Gaussian blur and subtracts it,
    removing gradual illumination gradients (vignetting, agar thickness,
    scanner shading) while retaining sharp colony features. Improves
    downstream thresholding and edge detection.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        sigma: Gaussian standard deviation defining the background scale.
            Must be larger than the typical colony diameter. Typical range:
            20--100. Default: 50.0.
        mode: Border handling. Accepted values: ``'reflect'`` (default),
            ``'constant'``, ``'nearest'``, ``'mirror'``, ``'wrap'``.
        cval: Fill value when ``mode='constant'``. Default: 0.0.
        truncate: Gaussian support in standard deviations. Default: 4.0.
        preserve_range: Preserve the input value range during filtering.
            Default: ``True``.
        n_iter: Number of successive subtraction passes. Multiple passes
            remove residual background from complex gradients. Typical
            range: 1--3. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` background-subtracted and
        clipped to [0, 1]. ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Correcting uneven lighting across plates or scan beds.
        - Flattening background to enhance dark colonies on bright agar.
        - Normalizing batches captured with varying exposure or
          illumination profiles.

    Consider Also:
        - :class:`SubtractRollingBall` for parabolic background estimation
          that adapts to non-Gaussian intensity ramps.
        - :class:`OpeningSubtractBg` for faster morphological background
          subtraction in high-throughput pipelines.
        - :class:`BilateralDenoise` when the primary issue is noise rather
          than illumination gradients.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of background subtraction on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        illumination correction strategies.
    """

    sigma: float = 50.0
    mode: str = "reflect"
    cval: float = 0.0
    truncate: float = 4.0
    preserve_range: bool = True
    n_iter: int = 1

    @field_validator("n_iter")
    @classmethod
    def _check_n_iter(cls, n_iter: int) -> int:
        """Require at least one subtraction pass (matches the legacy guard)."""
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        return n_iter

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
