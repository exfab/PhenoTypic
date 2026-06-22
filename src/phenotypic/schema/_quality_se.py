"""Measurement info container for replicate-agreement standard-error QC."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class QUALITY_SE(QualityInfo):
    """Measurement info for the replicate-agreement standard-error QC check.

    Carries the per-(group, time) summary statistics used to gauge how tightly
    biological replicates agree on a colony phenotype. Disagreement between
    replicates often signals contamination, edge artifacts, or imaging issues
    rather than real biology, so the SE and CV columns drive curation decisions
    downstream of the QC pipeline.
    """

    @classmethod
    def category(cls) -> str:
        return "QC_SE"

    VALUE = Entry("Value", "Raw SE = stddev / sqrt(n) across replicates.")
    MEAN = Entry("Mean", "Mean across replicates at this (group, time).")
    CV = Entry("CV", "Coefficient of variation, stddev / |mean|.")
    NUM_REPLICATES = Entry("NumReplicates", "Replicate count contributing to the SE.")
