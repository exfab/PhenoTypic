"""Measurement info container for edge correction analysis results."""

from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class EDGE_CORRECTION(DerivedMeasure):
    """Measurement info container for edge correction analysis results.

    Provides metadata for measurement values produced by the EdgeCorrector,
    organizing corrected colony measurements under the "EdgeCorrection" category.
    """

    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = Entry("Cap", "The carrying capacity for the target measurement",
                          derivation_type="normalization", derives_from="SIZE")
    NEW_VAL = Entry("NewVal", "The new value of the target measurement",
                    derivation_type="normalization", derives_from="SIZE")
