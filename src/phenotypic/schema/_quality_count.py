"""Measurement info container for expected-vs-detected colony count QC."""

from ._measurement_info import MeasurementInfo


class QUALITY_COUNT(MeasurementInfo):
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

    DETECTED = ("Detected", "Detected colony count in the group.")
    EXPECTED = ("Expected", "Expected colony count from the metadata frame.")
    DELTA = ("Delta", "Detected − Expected (signed; negative = missing).")
