"""Measurement info container for replicate-agreement modified Z-score QC."""

from ._measurement_info import Entry, MeasurementInfo


class QUALITY_ZMAX(MeasurementInfo):
    """Measurement info for the replicate-agreement modified Z-score QC check.

    Documents the per-check extra columns behind the modified Z-score (ZMax)
    agreement check, which scales each member's deviation from the group median
    by the MAD to spot the colony that disagrees most with its replicates or
    detection runs. A large modified Z-score often marks contamination, edge
    artifacts, or segmentation errors rather than real biology, so these
    columns drive curation decisions downstream of the QC pipeline.
    """

    @classmethod
    def category(cls) -> str:
        return "QC_ZMax"

    MEDIAN = Entry("Median", "Group median of the measurement.")
    MAD = Entry("MAD", "Median absolute deviation used to scale the modified Z-score.")
    NUM_MEMBERS = Entry("NumMembers", "Members contributing to the statistic.")
