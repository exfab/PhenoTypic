from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
import numpy as np
from pydantic import field_validator
from skimage.filters.rank import median
from skimage.util import img_as_ubyte, img_as_float

from phenotypic.abc_ import ImageEnhancer


class RankMedianEnhancer(ImageEnhancer):
    """Suppress impulsive noise in ``detect_mat`` with rank-based median filtering.

    Applies a local median using rank filters with a configurable structuring
    element shape and size. Effectively removes salt-and-pepper noise, dust
    speckles, and pixel-level artifacts while preserving colony boundaries
    when the footprint is smaller than the minimum colony diameter.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Args:
        shape: Footprint geometry. ``'disk'`` for isotropic smoothing;
            ``'square'`` (default) to align with grid artifacts.
        width: Footprint width in pixels. Set smaller than the minimum
            colony diameter to preserve colony edges. ``None`` (default)
            derives a small value from image size.
        shift_x: Horizontal footprint offset. Typically 0. Default: 0.
        shift_y: Vertical footprint offset. Typically 0. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` median-filtered. ``rgb``
        and ``gray`` are unchanged.

    Best For:
        - Salt-and-pepper or impulsive noise from sensor defects.
        - Dust speckles and pixel-level artifacts on scanned plates.
        - Grid-like imaging artifacts when using a ``'square'`` footprint.

    Consider Also:
        - :class:`BilateralDenoise` for edge-preserving Gaussian noise
          removal without the intensity quantization of rank filters.
        - :class:`NonLocalMeansDenoiser` for patch-based denoising that
          preserves texture better on noisy plates.
        - :class:`GrayOpening` for morphological artifact removal that
          does not require uint8 conversion.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
    """

    shape: str = "square"
    width: int | None = None
    shift_x: int = 0
    shift_y: int = 0

    @field_validator("shape")
    @classmethod
    def _check_shape_supported(cls, shape: str) -> str:
        """Reject an unsupported footprint shape (matches the legacy guard)."""
        if shape not in ["disk", "square", "sphere", "cube"]:
            raise ValueError(f"shape shape {shape} is not supported")
        return shape

    def _operate(self, image: Image) -> Image:
        image.detect_mat[:] = img_as_float(
                median(
                        image=img_as_ubyte(image.detect_mat[:]),
                        footprint=self._get_footprint(
                                self._get_footprint_width(image.detect_mat[:])
                        ),
                )
        )
        return image

    def _get_footprint_width(self, detect_mat: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(detect_mat.shape) * 0.002)
        else:
            return self.width

    def _get_footprint(self, width: int) -> np.ndarray:
        match self.shape:
            # Use the central ImageEnhancer utility for common 2D shapes
            case "disk" | "square":
                return self._make_footprint(shape=self.shape, width=width)
            case _:
                raise TypeError("Unknown shape shape")
