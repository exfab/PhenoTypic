"""Measurement info container for replicate-agreement standard-error QC."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class QUALITY_SE(MeasurementInfo):
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

    VALUE = ("Value", "Raw SE = stddev / sqrt(n) across replicates.")
    MEAN = ("Mean", "Mean across replicates at this (group, time).")
    CV = ("CV", "Coefficient of variation, stddev / |mean|.")
    NUM_REPLICATES = ("NumReplicates", "Replicate count contributing to the SE.")
