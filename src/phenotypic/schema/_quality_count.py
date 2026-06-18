"""Measurement info container for expected-vs-detected colony count QC."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class QUALITY_COUNT(QualityInfo):
    """Measurement info for the expected-vs-detected colony count QC check.

    Carries the per-group colony counts compared by the count quality check:
    how many colonies were detected on the plate versus how many the metadata
    declared, and their signed difference. Negative ``Delta`` values flag
    missing colonies (e.g. failed spots, dropouts) while positive values flag
    spurious detections (e.g. fragmentation, artifacts).
    """

    @classmethod
    def category(cls) -> str:
        return "QC_Count"

    DETECTED = Entry("Detected", "Detected colony count in the group.")
    EXPECTED = Entry("Expected", "Expected colony count from the metadata frame.")
    DELTA = Entry("Delta", "Detected − Expected (signed; negative = missing).")
