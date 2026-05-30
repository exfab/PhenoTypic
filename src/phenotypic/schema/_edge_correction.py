"""Measurement info container for edge correction analysis results."""

from ._measurement_info import MeasurementInfo


class EDGE_CORRECTION(MeasurementInfo):
    """Measurement info container for edge correction analysis results.

    Provides metadata for measurement values produced by the EdgeCorrector,
    organizing corrected colony measurements under the "EdgeCorrection" category.
    """

    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = "Cap", "The carrying capacity for the target measurement"
    NEW_VAL = "NewVal", "The new value of the target measurement"
