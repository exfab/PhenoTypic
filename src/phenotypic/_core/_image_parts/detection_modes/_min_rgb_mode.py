"""Min-RGB detection mode — per-pixel minimum across R, G, B."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    register_detection_mode,
)
from phenotypic.tools_.funcs_ import normalize_rgb_bitdepth


@register_detection_mode
class MinRGBDetectionMode(DetectionMode):
    """Detection matrix from the per-pixel minimum of R, G, B channels."""

    @property
    def name(self) -> str:
        return "MinRGB"

    @property
    def requires_rgb(self) -> bool:
        return True

    def compute(self, image: Image) -> np.ndarray:
        assert image._data.rgb is not None
        return np.min(normalize_rgb_bitdepth(image._data.rgb), axis=2).astype(np.float32)
