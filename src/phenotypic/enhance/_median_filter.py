from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from phenotypic.abc_ import ImageEnhancer

from skimage.filters import median


class MedianFilter(ImageEnhancer):
    """Remove impulsive noise from detect_mat while preserving colony edges.

    Replaces each pixel with the median of its local neighborhood, making it
    robust to outlier pixels (condensation droplets, dust specks, sensor noise).
    Preserves colony boundaries better than Gaussian smoothing because it does
    not average across edges.

    Args:
        mode: Boundary handling. Accepted values: ``'nearest'``, ``'reflect'``,
            ``'constant'``, ``'mirror'``, ``'wrap'``. Default: ``'nearest'``.
        shape: Structuring element shape. Accepted values: ``'disk'``,
            ``'square'``, ``'diamond'``, or ``None`` for library default.
            Default: ``None``.
        width: Size of the structuring element in pixels. Larger values
            smooth more aggressively. Typical range: 3--9. Default: 5.
        cval: Fill value when ``mode='constant'``. Default: 0.0.

    Returns:
        Image: Input image with ``detect_mat`` filtered. ``rgb`` and ``gray``
        are unchanged.

    Best For:
        - Plates with salt-and-pepper noise or bright/dark speckle artifacts.
        - Preserving sharp colony edges during denoising.
        - Pre-filtering before edge-based detection (Canny, Sobel).

    Consider Also:
        - :class:`GaussianBlur` for faster, simpler smoothing when edge
          preservation is less critical.
        - :class:`LocalEdgeDenoise` for edge-preserving smoothing with
          continuous intensity gradients.
        - :class:`RankMedianEnhancer` for configurable rank-based filtering
          with explicit footprint control.

    See Also:
        :doc:`/how_to/notebooks/denoise_low_light` for a comparison of
        denoising methods on low-light plates.
        :doc:`/explanation/what_enhancement_does` for how enhancement fits
        into the pipeline model.
    """

    mode: Literal["nearest", "reflect", "constant", "mirror", "wrap"] = "nearest"
    shape: Literal["disk", "square", "diamond"] | None = None
    width: int = 5
    cval: float = 0.0

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
