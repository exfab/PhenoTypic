from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from skimage.restoration import rolling_ball

from phenotypic.abc_ import BackgroundSubtraction
from phenotypic.tools_.typing_ import NdArrayField, TuneSpec


class SubtractRollingBall(BackgroundSubtraction):
    """Remove background from ``detect_mat`` with ImageJ-style rolling-ball subtraction.

    Models the background as the surface traced by rolling a parabolic ball
    under the image intensity landscape, then subtracts it. Effectively
    removes slow illumination gradients and agar shading while preserving
    colony structures. Handles non-Gaussian intensity ramps better than
    :class:`SubtractGaussian`.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        radius: Rolling-ball radius in pixels. Must be larger than the
            typical colony diameter to avoid subtracting colony signal.
            Typical range: 50--200. Default: 100.
        kernel: Optional custom ball/shape array. When provided, overrides
            ``radius``. Default: ``None``.
        nansafe: If ``True``, treat NaNs as missing data to avoid artifacts
            when using masked images. Default: ``False``.

    Returns:
        Image: Input image with ``detect_mat`` background-subtracted.
        ``rgb`` and ``gray`` are unchanged.

    Best For:
        - Scanner vignetting, lid glare, or agar thickness variations.
        - Flattening backgrounds to improve segmentation of dark colonies
          on bright agar.
        - Images with non-linear illumination gradients where Gaussian
          subtraction leaves residual background.

    Consider Also:
        - :class:`SubtractGaussian` for faster Gaussian-based subtraction
          with continuous sigma control.
        - :class:`SubtractOpening` for OpenCV-accelerated morphological
          background removal in high-throughput pipelines.
        - :class:`WhiteTophatEnhance` when you want to isolate small
          bright structures rather than subtract background.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of background subtraction on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        rolling-ball and other illumination correction strategies.
    """

    # TODO: review bound (unverified vs literature)
    radius: Annotated[int, TuneSpec(50, 200, log=True)] = 100
    kernel: NdArrayField | None = None
    nansafe: bool = False

    def _operate(self, image: Image):
        image.detect_mat[:] = image.detect_mat[:] - rolling_ball(
                image=image.detect_mat[:],
                radius=self.radius,
                kernel=self.kernel,
                nansafe=self.nansafe,
        )
        return image
