from __future__ import annotations

from typing import Annotated, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image
import numpy as np
from pydantic import field_validator
from skimage.filters.rank import median
from skimage.util import img_as_ubyte, img_as_float

from phenotypic.abc_ import Smoothing
from phenotypic.sdk_.typing_ import TuneSpec


class RankMedianEnhancer(Smoothing):
    """Suppress impulsive noise in ``detect_mat`` with rank-based median filtering.

    Applies a local median using rank filters with a configurable structuring
    element shape and size. Effectively removes salt-and-pepper noise, dust
    speckles, and pixel-level artifacts while preserving colony boundaries
    when the footprint is smaller than the minimum colony diameter.

    For algorithm details, see :doc:`/explanation/what_enhancement_does`.

    Best For:
        - Salt-and-pepper or impulsive noise from sensor defects or scanner
          CCD artifacts.
        - Dust speckles and pixel-level artifacts on scanned plates.
        - Grid-like imaging artifacts where a ``'square'`` footprint aligns
          with the noise geometry.
        - Pre-detection cleanup before applying a threshold-based detector.

    Consider Also:
        - :class:`LocalEdgeDenoise` for edge-preserving smoothing of
          Gaussian noise without the uint8 conversion required by rank
          filters.
        - :class:`NonLocalMeansDenoiser` for patch-based denoising that
          preserves fine colony texture better on noisy plates.
        - :class:`GrayOpening` for morphological artifact removal that
          does not require uint8 quantization.

    Args:
        shape: Footprint geometry. Accepted values: ``'disk'`` for
            isotropic smoothing suited to round colonies; ``'square'``
            (default) to align with grid-pattern sensor artifacts.
        width: Footprint width in pixels. Set smaller than the minimum
            colony diameter to preserve colony edges. ``None`` (default)
            auto-derives a small value as approximately 0.2 % of the
            shorter image dimension.
        shift_x: Horizontal offset of the footprint centre in pixels.
            Non-zero values shift the filter kernel to correct for
            directional streak artefacts. Default: 0.
        shift_y: Vertical offset of the footprint centre in pixels.
            Non-zero values shift the filter kernel to correct for
            directional streak artefacts. Default: 0.

    Returns:
        Image: Input image with ``detect_mat`` median-filtered. ``rgb``
        and ``gray`` are unchanged.

    See Also:
        :doc:`/tutorials/notebooks/03_enhancing_before_detection` for a
        visual walkthrough of denoising pipelines on plate images.
        :doc:`/explanation/what_enhancement_does` for background on rank
        filtering and its role in colony detection pipelines.
    """

    shape: str = "square"
    width: Annotated[int | None, TuneSpec(3, 15, step=2)] = None
    shift_x: Annotated[int, TuneSpec(tunable=False)] = 0
    shift_y: Annotated[int, TuneSpec(tunable=False)] = 0

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
