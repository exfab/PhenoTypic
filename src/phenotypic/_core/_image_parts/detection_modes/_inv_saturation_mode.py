"""Inverse saturation detection mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    register_detection_mode,
)


@register_detection_mode
class InverseSaturationMode(DetectionMode):
    """Detection matrix from inverted HSV saturation (1 - S).

    Useful for colony detection on colored agar where colonies are the
    least saturated regions. Inversion makes colonies bright in the
    detection matrix, aligning with the convention that high values
    indicate regions of interest.
    """

    @property
    def name(self) -> str:
        return "InvS"

    @property
    def requires_rgb(self) -> bool:
        return True

    def compute(self, image: Image) -> np.ndarray:
        sat = image.color.hsv[:, :, 1].astype(np.float32)
        return 1.0 - sat
