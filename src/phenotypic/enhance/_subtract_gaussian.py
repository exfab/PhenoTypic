from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import Field
from skimage.filters import gaussian

from phenotypic.abc_ import BackgroundSubtraction
from phenotypic.tools_.typing_ import TuneSpec


class SubtractGaussian(BackgroundSubtraction):
    """Remove background from ``detect_mat`` by subtracting a Gaussian-blurred estimate.

    Estimates a smooth background by blurring the image with a wide Gaussian
    kernel and subtracts it, removing gradual illumination gradients such as
    vignetting, agar thickness variation, and scanner shading while retaining
    sharp colony features. The result is clipped to [0, 1] and improves
    downstream thresholding and edge detection.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Correcting uneven illumination gradients across the scan bed or
          plate.
        - Flattening background intensity so bright colonies or bright
          colony features stand out against local background.
        - Normalizing batches captured with varying scanner exposure or
          lamp profiles.
        - Plates where illumination varies smoothly and a Gaussian is a
          reasonable background model.

    Consider Also:
        - :class:`SubtractRollingBall` for parabolic background estimation
          that handles non-Gaussian intensity ramps and sharp gradients
          more accurately.
        - :class:`SubtractOpening` for OpenCV-accelerated morphological
          background subtraction in high-throughput pipelines.
        - :class:`FlattenIllumination` for homomorphic filtering that
          separates illumination from reflectance in the frequency domain.
        - :class:`ImageInverter` before this operation when colonies are
          dark on bright agar and should be made bright explicitly.

    Args:
        sigma: Standard deviation of the Gaussian background kernel in
            pixels. Set larger than the typical colony diameter so that
            colonies are blurred into the background estimate rather than
            surviving it; too small a sigma subtracts colony signal along
            with the background. Typical range: 20--100 for standard plate
            images. A reasonable starting point is a value somewhat larger
            than the widest colony present. Default: 50.0.
        mode: Border-handling strategy for the Gaussian convolution.
            Accepted values: ``'reflect'`` (default), ``'constant'``,
            ``'nearest'``, ``'mirror'``, ``'wrap'``.
        cval: Constant fill value used when ``mode='constant'``. Has no
            effect for other border modes. Default: 0.0.
        truncate: Number of standard deviations at which the Gaussian
            kernel is truncated. Larger values are more accurate but
            increase compute time. Default: 4.0.
        preserve_range: Preserve the input pixel value range during
            Gaussian filtering. Default: ``True``.
        n_iter: Number of successive background-subtraction passes.
            Additional passes remove residual gradients left after the
            first subtraction. Typical range: 1--3. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` background-subtracted and
        clipped to [0, 1]. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``n_iter`` is less than 1.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of background subtraction on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        illumination correction strategies.
    """

    # TODO: review bound (unverified vs literature)
    sigma: Annotated[float, TuneSpec(20.0, 100.0)] = 50.0
    mode: str = "reflect"
    cval: Annotated[float, TuneSpec(tunable=False)] = 0.0
    truncate: Annotated[float, TuneSpec(tunable=False)] = 4.0
    preserve_range: bool = True
    n_iter: Annotated[int, TuneSpec(1, 3)] = Field(1, ge=1)

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
