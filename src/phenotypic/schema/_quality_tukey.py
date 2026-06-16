"""Measurement info container for replicate-agreement Tukey-fence QC."""

from ._measurement_info import Entry, MeasurementInfo


class QUALITY_TUKEY(MeasurementInfo):
    """Measurement info for the replicate-agreement Tukey-fence QC check.

    Documents the per-check extra columns behind the Tukey-fence agreement
    check, which flags colonies whose measurement falls outside Q1 - k*IQR or
    Q3 + k*IQR relative to their replicates or detection runs. Members beyond
    the fences are robust outliers that frequently reflect contamination, edge
    artifacts, or segmentation errors rather than real biology, so these
    columns drive curation decisions downstream of the QC pipeline.
    """

    @classmethod
    def category(cls) -> str:
        return "QC_Tukey"

    LOWER_FENCE = Entry("LowerFence", "Lower Tukey fence, Q1 - k*IQR.")
    UPPER_FENCE = Entry("UpperFence", "Upper Tukey fence, Q3 + k*IQR.")
    NUM_OUTLIERS = Entry("NumOutliers", "Members falling outside the fences.")
    NUM_MEMBERS = Entry("NumMembers", "Members contributing to the statistic.")
