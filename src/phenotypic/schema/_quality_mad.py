"""Measurement info container for replicate-agreement MAD QC."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class QUALITY_MAD(QualityInfo):
    """Measurement info for the replicate-agreement MAD QC check.

    Documents the per-check extra columns produced by the median absolute
    deviation (MAD) agreement check, which gauges how tightly biological
    replicates or detection runs agree on a colony phenotype. The MAD is a
    robust spread estimate that resists single contaminated or mis-segmented
    colonies, so these columns help flag groups whose disagreement signals
    imaging artifacts rather than real biology.
    """

    @classmethod
    def category(cls) -> str:
        return "QC_MAD"

    MEDIAN = Entry("Median", "Group median of the measurement.")
    MAD = Entry("MAD", "Median absolute deviation (raw, before normalization).")
    NUM_MEMBERS = Entry("NumMembers", "Members contributing to the statistic.")
