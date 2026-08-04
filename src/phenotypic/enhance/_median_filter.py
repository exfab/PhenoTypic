from __future__ import annotations

from typing import Annotated, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from phenotypic.abc_ import Smoothing
from phenotypic.sdk_.typing_ import TuneSpec

from skimage.filters import median


class MedianFilter(Smoothing):
    """Remove impulsive noise from ``detect_mat`` while preserving colony edges.

    Replaces each pixel with the median of its local neighbourhood, making it
    robust to outlier pixels such as condensation droplets, dust specks, and
    sensor noise spikes. Preserves colony boundaries more faithfully than
    Gaussian smoothing because the median operation does not average intensity
    across edges.

    For a comparison of denoising approaches, see
    :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Plates with salt-and-pepper noise or isolated bright/dark speckle
          artefacts from scanner CCD defects or condensation.
        - Preserving sharp colony boundary edges during denoising before
          edge-based detection.
        - Pre-filtering before edge-based detectors (Canny, Sobel) where noise
          edges must be suppressed without blurring colony margins.

    Consider Also:
        - :class:`BlurGauss` for faster, simpler smoothing when edge
          preservation is less critical and noise is Gaussian rather than
          impulsive.
        - :class:`LocalEdgeDenoise` for bilateral edge-preserving smoothing
          with continuous intensity gradients.
        - :class:`RankMedianEnhancer` for configurable rank-based filtering
          with explicit footprint control.

    Args:
        mode: Boundary handling strategy. Accepted values: ``'nearest'``,
            ``'reflect'``, ``'constant'``, ``'mirror'``, ``'wrap'``.
            Default: ``'nearest'``.
        shape: Structuring element shape for the median neighbourhood.
            Accepted values: ``'disk'``, ``'square'``, ``'diamond'``, or
            ``None`` for the library default footprint. Default: ``None``.
        width: Size of the structuring element in pixels. Larger values
            smooth more aggressively. Typical range: 3--9. Default: 5.
        cval: Fill value used when ``mode='constant'``. Default: 0.0.

    Returns:
        Image: Input image with ``detect_mat`` filtered. ``rgb`` and ``gray``
        are unchanged.

    See Also:
        :doc:`/how_to/notebooks/denoise_low_light` for a comparison of
        denoising methods on low-light plate images.
        :doc:`/explanation/what_enhancement_does` for how enhancement fits
        into the pipeline model.
    """

    mode: Literal["nearest", "reflect", "constant", "mirror", "wrap"] = "nearest"
    shape: Literal["disk", "square", "diamond"] | None = None
    width: Annotated[int, TuneSpec(3, 9, step=2)] = 5
    cval: Annotated[float, TuneSpec(tunable=False)] = 0.0

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = median(
                image=image.detect_mat[:],
                behavior="ndimage",
                footprint=(
                    self.shape
                    if self.shape is None
                    else self._make_footprint(shape=self.shape, width=self.width)
                ),
                mode=self.mode,
                cval=self.cval,
        )
        return image
