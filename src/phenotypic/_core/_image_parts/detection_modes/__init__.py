"""Registry-based detection mode system.

Importing this package triggers registration of all built-in modes via
the ``@register_detection_mode`` decorator.
"""

from phenotypic._core._image_parts.detection_modes._detection_mode import (
    DetectionMode,
    get_detection_mode,
    available_modes,
    register_detection_mode,
)

# Import mode modules to trigger registration decorators.
from phenotypic._core._image_parts.detection_modes._gray_mode import (
    GrayDetectionMode,
)
from phenotypic._core._image_parts.detection_modes._color_channel_modes import (
    RedChannelMode,
    GreenChannelMode,
    BlueChannelMode,
)
from phenotypic._core._image_parts.detection_modes._min_rgb_mode import (
    MinRGBDetectionMode,
)
from phenotypic._core._image_parts.detection_modes._lab_channel_modes import (
    LabLightnessMode,
    LabAMode,
    LabBMode,
)
from phenotypic._core._image_parts.detection_modes._hsv_channel_modes import (
    HsvSaturationMode,
    HsvValueMode,
)

__all__ = [
    "DetectionMode",
    "get_detection_mode",
    "available_modes",
    "register_detection_mode",
    "GrayDetectionMode",
    "RedChannelMode",
    "GreenChannelMode",
    "BlueChannelMode",
    "MinRGBDetectionMode",
    "LabLightnessMode",
    "LabAMode",
    "LabBMode",
    "HsvSaturationMode",
    "HsvValueMode",
]
