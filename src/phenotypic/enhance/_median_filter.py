from __future__ import annotations

from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from phenotypic.abc_ import ImageEnhancer

from skimage.filters import median


class MedianFilter(ImageEnhancer):
    “””Remove impulsive noise from detect_mat while preserving colony edges.

    Replaces each pixel with the median of its local neighborhood, making it
    robust to outlier pixels (condensation droplets, dust specks, sensor noise).
    Preserves colony boundaries better than Gaussian smoothing because it does
    not average across edges.

    Best For:
        - Plates with salt-and-pepper noise or bright/dark speckle artifacts.
        - Preserving sharp colony edges during denoising.
        - Pre-filtering before edge-based detection (Canny, Sobel).

    Consider Also:
        - :class:`GaussianBlur` for faster, simpler smoothing when edge
          preservation is less critical.
        - :class:`BilateralDenoise` for edge-preserving smoothing with
          continuous intensity gradients.
        - :class:`RankMedianEnhancer` for configurable rank-based filtering
          with explicit footprint control.

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

    See Also:
        :doc:`/how_to/notebooks/denoise_low_light` for a comparison of
        denoising methods on low-light plates.
        :doc:`/explanation/what_enhancement_does` for how enhancement fits
        into the pipeline model.
    “””

    def __init__(
            self,
            mode: Literal[
                "nearest", "reflect", "constant", "mirror", "wrap"] = "nearest",
            shape: Literal["disk", "square", "diamond"] | None = None,
            width: int = 5,
            cval: float = 0.0,
    ):
        """
        This class is designed to facilitate image processing tasks, particularly for analyzing microbe
        colonies on solid media agar. By adjusting the mode, shape, width, and cval attributes,
        users can modify the processing behavior and results to suit their specific requirements for
        studying spatial arrangements, colony boundaries, and other morphological features.

        Attributes:
            mode (Literal["nearest", "reflect", "constant", "mirror", "wrap"]):
                Determines how boundaries of the image are handled during processing.
                For instance, "reflect" can help minimize edge artifacts when analyzing
                colonies near the edge of the image by mirroring boundary pixels, while
                "constant" fills with a value (cval), which might highlight isolated colonies.
                Adjusting this can significantly affect how edge regions are interpreted.

            shape (Literal["disk", "square", "diamond"] | None):
                Specifies the shape of the structuring element used in morphological
                operations. For instance, "disk" simulates circular neighborhood which works
                well for circular colonies, whereas "square" gives a grid-like neighborhood.
                This can directly impact how structures are identified or segmented.

            width (int):
                Size of the structuring element. Larger widths result in broader neighborhoods
                being considered, which may smooth or connect distant colonies, while smaller
                widths preserve finer details but may miss larger structural relationships. Only
                if shape is not None.

            cval (float):
                Value used to fill borders when mode is set to "constant". This directly affects
                colony recognition at the edges; for example, setting a high cval compared to
                colony intensity might obscure colonies near the borders.
        """
        if mode in ["nearest", "reflect", "constant", "mirror", "wrap"]:
            self.mode = mode
            self.shape = shape
            self.width = width
            self.cval = cval
        else:
            raise ValueError(
                    'mode must be one of "nearest","reflect","constant","mirror","wrap"'
            )

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
