"""Min-RGB detection mode — per-pixel minimum across R, G, B."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    register_detection_mode,
)
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth


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
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Take the per-pixel minimum across the channels of *rgb*.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Unused; this mode needs no colour configuration.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
        return np.min(rgb, axis=2).astype(np.float32)
