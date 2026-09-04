from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

# `cv2` is imported inside the methods that use it. A module-scope import
# would load OpenCV on `import phenotypic` for every entry point, though only
# a run reaching this operation needs it.

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import BackgroundSubtraction
from phenotypic.sdk_.mixin import FootprintMixin
from phenotypic.sdk_.typing_ import TuneSpec


class SubtractOpening(BackgroundSubtraction, FootprintMixin):
    """Subtract background from ``detect_mat`` via OpenCV-accelerated morphological opening.

    Computes the white top-hat transform (original minus morphological
    opening) using OpenCV's C++/SIMD backend, isolating bright foreground
    structures smaller than the structuring element while discarding
    slow-varying background intensity. Significantly faster than
    scikit-image equivalents for high-throughput workflows.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Fast background subtraction for high-throughput plate screening
          or parameter sweeps.
        - Removing uneven illumination gradients and agar shading before
          colony detection.
        - Large-batch pipelines where speed is a priority and a flat
          structuring element is acceptable.
        - Drop-in acceleration when :class:`SubtractRollingBall` is the
          accuracy reference but runtime is the constraint.

    Consider Also:
        - :class:`SubtractRollingBall` for parabolic background estimation
          that handles gradual, non-uniform intensity ramps more
          accurately.
        - :class:`SubtractGaussian` for Gaussian-based subtraction with
          continuous control over the background scale.
        - :class:`WhiteTophatEnhance` when you want to retain only the
          extracted small bright structures rather than a corrected image.

    Args:
        shape: Structuring element geometry. Accepted values: ``'disk'``
            (default) for isotropic removal suited to round colonies;
            ``'square'`` for fastest computation; ``'diamond'`` as a
            compromise between the two.
        width: Diameter of the structuring element in pixels. Must exceed
            the largest colony diameter to avoid including colony pixels
            in the background estimate. Typical range: 31--101. Default:
            51.
        n_iter: Number of morphological opening iterations. Additional
            iterations intensify background removal at the cost of
            eroding fine colony structure. Default: 1.

    Returns:
        Image: Input image with ``detect_mat`` containing only foreground
        structures smaller than the structuring element. ``rgb`` and
        ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of background subtraction on plate images.
        :doc:`/explanation/what_enhancement_does` for background on
        morphological background removal strategies.
    """

    shape: Literal["square", "diamond", "disk"] = "disk"
    # TODO: review bound (unverified vs literature)
    width: Annotated[int, TuneSpec(31, 101, step=2)] = 51
    n_iter: Annotated[int, TuneSpec(1, 3)] = 1

    def _operate(self, image: Image) -> Image:
        import cv2

        image.detect_mat[:] = cv2.morphologyEx(
                src=image.detect_mat[:],
                op=cv2.MORPH_TOPHAT,
                kernel=self._make_footprint(shape=self.shape, width=self.width),
                iterations=self.n_iter,
        )
        return image
