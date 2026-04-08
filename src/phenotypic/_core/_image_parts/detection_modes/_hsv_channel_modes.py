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


class _HsvChannelMode(DetectionMode):
    """Base for modes that extract a single HSV channel."""

    @property
    def requires_rgb(self) -> bool:
        return True

    @property
    @abstractmethod
    def _channel_index(self) -> int: ...

    def compute(self, image: Image) -> np.ndarray:
        return image.color.hsv[:, :, self._channel_index].astype(np.float32)


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
