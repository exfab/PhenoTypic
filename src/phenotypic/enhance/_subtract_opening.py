from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import BackgroundSubtraction
from phenotypic.tools_.mixin import FootprintMixin


class SubtractOpening(BackgroundSubtraction, FootprintMixin):
    """Subtract background from ``detect_mat`` via OpenCV-accelerated morphological opening.

    Computes the white top-hat transform (original minus morphological
    opening) using OpenCV's C++/SIMD backend, isolating bright foreground
    structures smaller than the structuring element while removing
    slow-varying background intensity. Significantly faster than
    scikit-image equivalents for high-throughput workflows.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        shape: Structuring element geometry. ``'disk'`` (default) gives
            isotropic removal suited to round colonies; ``'square'`` is
            fastest; ``'diamond'`` is a compromise.
        width: Diameter of the structuring element in pixels. Must be
            larger than colony diameter to avoid subtracting colony
            signal. Typical range: 31--101. Default: 51.
        n_iter: Number of morphological iterations. Higher values
            intensify background removal. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` containing only foreground
        structures smaller than the structuring element. ``rgb`` and
        ``gray`` are unchanged.

    Best For:
        - Fast background subtraction for high-throughput plate screening.
        - Removing uneven illumination gradients and agar shading before
          colony detection.
        - Pipelines where speed matters (large batches, parameter sweeps).
        - Drop-in performance upgrade over :class:`SubtractRollingBall`
          when a flat structuring element is acceptable.

    Consider Also:
        - :class:`SubtractRollingBall` for parabolic background estimation
          that handles gradual intensity ramps more accurately.
        - :class:`SubtractGaussian` for Gaussian-based background
          subtraction with continuous control over the background scale.
        - :class:`WhiteTophatEnhance` when you want to keep only the
          extracted small bright structures.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of background subtraction on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        morphological background removal strategies.
    """

    shape: Literal["square", "diamond", "disk"] = "disk"
    width: int = 51
    n_iter: int = 1

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = cv2.morphologyEx(
                src=image.detect_mat[:],
                op=cv2.MORPH_TOPHAT,
                kernel=self._make_footprint(shape=self.shape, width=self.width),
                iterations=self.n_iter,
        )
        return image
