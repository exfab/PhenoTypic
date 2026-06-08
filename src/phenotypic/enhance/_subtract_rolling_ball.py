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
    under the image intensity landscape, then subtracts it. Removes slow
    illumination gradients and agar shading while preserving colony
    structures, and handles non-Gaussian intensity ramps better than
    Gaussian-based subtraction.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Scanner vignetting, lid glare, or agar thickness variations that
          produce non-Gaussian illumination gradients.
        - Flattening backgrounds so bright colonies or bright colony
          features stand out against local background.
        - Images where Gaussian subtraction leaves residual background
          near plate edges or bright corners.

    Consider Also:
        - :class:`SubtractGaussian` for Gaussian-based subtraction with
          continuous sigma control when the background gradient is smooth
          and Gaussian-like.
        - :class:`SubtractOpening` for OpenCV-accelerated morphological
          background removal in high-throughput pipelines.
        - :class:`WhiteTophatEnhance` for isolating small bright
          structures rather than producing a corrected background-free
          image.
        - :class:`ImageInverter` before this operation when colonies are
          dark on bright agar and should be made bright explicitly.

    Args:
        radius: Rolling-ball radius in pixels. Must exceed the largest
            colony diameter so that colony pixels do not contribute to
            the background estimate. Typical range: 50--200. A reasonable
            starting point is a value larger than the widest colony on the
            plate. Default: 100.
        kernel: Optional custom ball or paraboloid array overriding the
            default parabolic shape. When provided, ``radius`` is ignored.
            Default: ``None``.
        nansafe: Treat NaN-valued pixels as missing data during background
            estimation, preventing NaN propagation in masked or padded
            images. Default: ``False``.

    Returns:
        Image: Input image with ``detect_mat`` background-subtracted.
        ``rgb`` and ``gray`` are unchanged.

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
