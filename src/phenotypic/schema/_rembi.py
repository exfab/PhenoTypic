"""REMBI module taxonomy for classifying metadata columns.

REMBI (Recommended Metadata for Biological Images; Sarkans et al. 2021) groups
bioimage provenance into modules. Each metadata enum declares its module via
``MeasurementInfo.rembi_module()``; measurement/locator enums fall back to
``ANALYZED_DATA``. Definition order is the canonical manifest/section order.

Import-light: stdlib only (see schema package load-order rule).
"""
from __future__ import annotations

from enum import Enum


class REMBI_MODULE(str, Enum):
    """REMBI metadata modules. Definition order is canonical."""

    STUDY = "Study"
    BIOSAMPLE = "Biosample"
    SPECIMEN_PREP = "SpecimenPreparation"
    IMAGE_ACQUISITION = "ImageAcquisition"
    IMAGE_DATA = "ImageData"
    ANALYZED_DATA = "AnalyzedData"
    UNCATEGORIZED = "Uncategorized"


def header_to_module() -> "dict[str, REMBI_MODULE]":
    """Map every known column header to its REMBI module.

    Walks every ``MeasurementInfo`` subclass exported from ``phenotypic.schema``
    and reads each member's ``resolved_rembi_module``. Built fresh on each call
    (cheap; <1k members). Used by the manifest builder's column router.
    """
    from . import __all__ as _names
    from . import _measurement_info as _mi
    import phenotypic.schema as _schema

    out: "dict[str, REMBI_MODULE]" = {}
    for name in _names:
        obj = getattr(_schema, name)
        if (isinstance(obj, type) and issubclass(obj, _mi.MeasurementInfo)
                and obj is not _mi.MeasurementInfo and list(obj)):
            for member in obj:
                out[member.value] = member.resolved_rembi_module
    return out
