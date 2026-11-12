"""Measurement info container for replicate-agreement standard-error QC."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import QualityInfo


class QUALITY_SE(QualityInfo):
    """Measurement info for the replicate-agreement standard-error QC check.

    Carries the per-(group, time) summary statistics used to gauge how tightly
    biological replicates agree on a colony phenotype. Disagreement between
    replicates often signals contamination, edge artifacts, or imaging issues
    rather than real biology, so the SE and CV columns drive curation decisions
    downstream of the QC pipeline.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "QC_SE"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.OF_INPUT

    VALUE = Entry(
        "Value",
        "Standard error of the replicates' mean at this group and time point, "
        "before it is divided by the mean. It is the raw SE, stddev / "
        "sqrt(n), with the sample standard deviation (dividing by n - 1).",
    )
    MEAN = Entry("Mean", "Mean across replicates at this (group, time).")
    CV = Entry(
        "CV",
        "Coefficient of variation of the replicates at this group and time "
        "point: their standard deviation divided by the absolute value of "
        "their mean. It is stddev / |mean|, and NaN when the mean is within "
        "eps of 0.",
        unit=UNITS.UNITLESS,
    )
    NUM_REPLICATES = Entry("NumReplicates", "Replicate count contributing to the SE.", unit=UNITS.UNITLESS)
