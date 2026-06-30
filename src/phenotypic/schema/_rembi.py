"""REMBI module taxonomy for classifying metadata columns.

REMBI (Recommended Metadata for Biological Images; Sarkans et al. 2021) groups
bioimage provenance into modules. Each metadata enum declares its module via
``MeasurementInfo.rembi_module()``; measurement/locator enums fall back to
``ANALYZED_DATA``. Definition order is the canonical manifest/section order.

Import-light: stdlib only (see schema package load-order rule).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._measurement_info import MeasurementInfo


class REMBI_MODULE(str, Enum):
    """REMBI metadata modules. Definition order is canonical."""

    STUDY = "Study"
    BIOSAMPLE = "Biosample"
    SPECIMEN_PREP = "SpecimenPreparation"
    IMAGE_ACQUISITION = "ImageAcquisition"
    IMAGE_DATA = "ImageData"
    ANALYZED_DATA = "AnalyzedData"
    UNCATEGORIZED = "Uncategorized"
