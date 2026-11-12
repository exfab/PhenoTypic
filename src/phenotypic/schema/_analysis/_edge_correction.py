"""Measurement info container for edge correction analysis results."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DerivedMeasure


class EDGE_CORRECTION(DerivedMeasure):
    """Measurement info container for edge correction analysis results.

    Provides metadata for measurement values produced by the EdgeCorrector,
    organizing corrected colony measurements under the "EdgeCorrection" metric family.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "EdgeCorrection"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.OF_INPUT

    CORRECTED_CAP = Entry(
        "Cap",
        "Ceiling applied to the measurement in this group: the mean of the "
        "top_n largest values among fully surrounded (interior) colonies at "
        "the last time point. When the group is not corrected it is the "
        "group's maximum, so nothing is clipped.",
        derivation_type="normalization",
        derives_from="SIZE",
    )
    NEW_VAL = Entry(
        "NewVal",
        "The measurement after edge correction: each value in the group "
        "clipped to between 0 and Cap, or the original value when the group "
        "is not corrected.",
        derivation_type="normalization",
        derives_from="SIZE",
    )
