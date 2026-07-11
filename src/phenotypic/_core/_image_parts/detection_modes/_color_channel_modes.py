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
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Select one channel of *rgb*.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Unused; this mode needs no colour configuration.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
        return rgb[:, :, self._channel_index].astype(np.float32)


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
