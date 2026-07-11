"""HSV Saturation and Value detection modes."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    register_detection_mode,
)
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth


class _HsvChannelMode(DetectionMode):
    """Base for modes that extract a single HSV channel."""

    @property
    def requires_rgb(self) -> bool:
        return True

    @property
    @abstractmethod
    def _channel_index(self) -> int: ...

    def compute(self, image: Image) -> np.ndarray:
        assert image._data.rgb is not None
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Convert *rgb* to HSV and select one channel.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Unused; this mode needs no colour configuration.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
        from skimage.color import rgb2hsv

        return rgb2hsv(rgb)[:, :, self._channel_index].astype(np.float32)


@register_detection_mode
class HsvSaturationMode(_HsvChannelMode):
    """Detection matrix from the HSV saturation channel."""

    @property
    def name(self) -> str:
        return "HsvS"

    @property
    def _channel_index(self) -> int:
        return 1


@register_detection_mode
class HsvValueMode(_HsvChannelMode):
    """Detection matrix from the HSV value channel."""

    @property
    def name(self) -> str:
        return "HsvV"

    @property
    def _channel_index(self) -> int:
        return 2
