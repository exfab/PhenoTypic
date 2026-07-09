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
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth


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
        assert image._data.rgb is not None
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Invert the HSV saturation channel of *rgb*.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Unused; this mode needs no colour configuration.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
        from skimage.color import rgb2hsv

        return 1.0 - rgb2hsv(rgb)[:, :, 1].astype(np.float32)
