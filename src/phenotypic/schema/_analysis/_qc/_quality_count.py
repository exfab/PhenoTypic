"""Measurement info container for expected-vs-detected colony count QC."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import QualityInfo


class QUALITY_COUNT(QualityInfo):
    """Measurement info for the expected-vs-detected colony count QC check.

    Carries the per-group colony counts compared by the count quality check:
    how many colonies were detected on the plate versus how many the metadata
    declared, and their signed difference. Negative ``Delta`` values flag
    missing colonies (e.g. failed spots, dropouts) while positive values flag
    spurious detections (e.g. fragmentation, artifacts).
    """

    @classmethod
    def metric_family(cls) -> str:
        return "QC_Count"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    DETECTED = Entry("Detected", "Detected colony count in the group.")
    EXPECTED = Entry("Expected", "Expected colony count from the metadata frame.")
    DELTA = Entry(
        "Delta",
        "Detected minus expected colony count in the group, so it is negative "
        "when colonies are missing and positive when there are extra "
        "detections. It is Detected − Expected.",
    )
