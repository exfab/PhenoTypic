"""CIE L*a*b* channel detection modes."""

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


class _LabChannelMode(DetectionMode):
    """Base for modes that extract a single CIE L*a*b* channel."""

    @property
    def requires_rgb(self) -> bool:
        return True

    @property
    @abstractmethod
    def _channel_index(self) -> int: ...

    @abstractmethod
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray: ...

    def compute(self, image: Image) -> np.ndarray:
        return self.compute_from_rgb(image.rgb.normed(), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Convert *rgb* to CIE L*a*b* and select one normalized channel.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Source image, consulted for ``gamma``, ``illuminant`` and
                ``_observer`` only.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
        import colour

        from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import (
            rgb_to_xyz,
        )

        xyz = rgb_to_xyz(
            rgb,
            gamma=image.gamma,
            illuminant=image.illuminant,
            observer=image._observer,
        )
        lab = colour.XYZ_to_Lab(
            XYZ=xyz,
            illuminant=colour.CCS_ILLUMINANTS[image._observer][image.illuminant],
        )
        return self._normalize_channel(lab[:, :, self._channel_index])


@register_detection_mode
class LabLightnessMode(_LabChannelMode):
    """Detection matrix from the L* (lightness) channel."""

    @property
    def name(self) -> str:
        return "LabL"

    @property
    def _channel_index(self) -> int:
        return 0

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        return np.clip(channel / 100.0, 0.0, 1.0).astype(np.float32)


@register_detection_mode
class LabAMode(_LabChannelMode):
    """Detection matrix from the a* (green-red) channel."""

    @property
    def name(self) -> str:
        return "LabA"

    @property
    def _channel_index(self) -> int:
        return 1

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        return np.clip((channel + 128.0) / 255.0, 0.0, 1.0).astype(np.float32)


@register_detection_mode
class LabBMode(_LabChannelMode):
    """Detection matrix from the b* (blue-yellow) channel."""

    @property
    def name(self) -> str:
        return "LabB"

    @property
    def _channel_index(self) -> int:
        return 2

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        return np.clip((channel + 128.0) / 255.0, 0.0, 1.0).astype(np.float32)
