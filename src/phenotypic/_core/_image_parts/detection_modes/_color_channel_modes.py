"""Red / Green / Blue single-channel detection modes."""

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


class _ColorChannelMode(DetectionMode):
    """Base for modes that extract a single RGB channel."""

    @property
    def requires_rgb(self) -> bool:
        return True

    @property
    @abstractmethod
    def _channel_index(self) -> int: ...

    def compute(self, image: Image) -> np.ndarray:
        assert image._data.rgb is not None
        rgb_normed = normalize_rgb_bitdepth(image._data.rgb)
        return rgb_normed[:, :, self._channel_index].astype(np.float32)


@register_detection_mode
class RedChannelMode(_ColorChannelMode):
    """Detection matrix from the red channel."""

    @property
    def name(self) -> str:
        return "red"

    @property
    def _channel_index(self) -> int:
        return 0


@register_detection_mode
class GreenChannelMode(_ColorChannelMode):
    """Detection matrix from the green channel."""

    @property
    def name(self) -> str:
        return "green"

    @property
    def _channel_index(self) -> int:
        return 1


@register_detection_mode
class BlueChannelMode(_ColorChannelMode):
    """Detection matrix from the blue channel."""

    @property
    def name(self) -> str:
        return "blue"

    @property
    def _channel_index(self) -> int:
        return 2
