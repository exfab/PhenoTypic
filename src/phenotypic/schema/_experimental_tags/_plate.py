"""Plate and array layout metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import IdentityInfo


class PLATE_METADATA(IdentityInfo):
    """Recommended ``Metadata_*`` tags describing the assay plate and its array.

    These capture plate-level grouping and physical layout (plate id, batch, array
    density, incubator position). Members render as ``Metadata_<Label>`` (e.g.
    ``Metadata_PlateID``) and share the ``Metadata_`` namespace with the other
    experimental-tag enums. Recommended vocabulary, not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.SPECIMEN_PREP

    PLATE_ID = Entry("PlateID", "Identifier of the assay/imaging plate.")
    BATCH = Entry("Batch", "Experimental batch grouping.")
    ARRAY_DENSITY = Entry(
        "ArrayDensity",
        "Colony array density, i.e. wells per plate (e.g. 96, 384, 1536).",
    )
    INCUBATOR_POSITION = Entry(
        "IncubatorPosition",
        "Physical position of the plate within the incubator.",
    )
