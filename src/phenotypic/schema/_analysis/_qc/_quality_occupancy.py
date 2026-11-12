"""Measurement info container for the grid-occupancy QC check."""

from ..._base._measurement_info import Entry
from ..._base._units import UNITS
from ..._base._tiers import QualityInfo


class QUALITY_OCCUPANCY(QualityInfo):
    """Measurement info for the grid-occupancy QC check.

    Carries the per-group cell-occupancy counts compared by the occupancy
    quality check: how many of a plate's expected grid cells are filled by
    at least one colony (doublets collapse to one), how many cells the
    metadata frame declared, and how many remain empty. A high ``Vacant``
    count flags plates with many missing pin positions (failed spots,
    dropouts) independent of any over-detection elsewhere on the plate.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "QC_Occupancy"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    FILLED = Entry(
        "Filled",
        "Distinct grid cells holding at least one colony (doublets count once).",
    )
    EXPECTED = Entry(
        "Expected",
        "Expected grid-cell count from the metadata frame (rows per group).",
    )
    VACANT = Entry(
        "Vacant",
        "Number of expected grid cells in the group that hold no colony. It "
        "is Expected - Filled, floored at 0.",
    )
