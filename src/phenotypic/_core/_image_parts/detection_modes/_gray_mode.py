"""Gray detection mode — uses the luminance channel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    register_detection_mode,
)


@register_detection_mode
class GrayDetectionMode(DetectionMode):
    """Detection matrix sourced from the grayscale channel."""

    @property
    def name(self) -> str:
        return "gray"

    @property
    def requires_rgb(self) -> bool:
        return False

    def compute(self, image: Image) -> np.ndarray:
        assert image._data.gray is not None
        return image._data.gray.copy()
