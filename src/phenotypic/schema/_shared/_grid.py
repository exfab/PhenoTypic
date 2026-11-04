"""Constants for grid structure in the PhenoTypic module."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import IdentityInfo


class GRID(IdentityInfo):
    """Constants for grid structure in the PhenoTypic module."""

    @classmethod
    def metric_family(cls) -> str:
        return "Grid"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.NOT_A_QUANTITY

    ROW_NUM = Entry(
        "RowNum",
        "Zero-based row number, from 0 at the top, of the grid cell that "
        "holds the colony's centroid (the mean position of its pixels). A "
        "colony whose centroid lies outside the grid is assigned to the "
        "nearest edge row.",
    )
    ROW_INTERVAL_START = Entry(
        "RowIntervalStart",
        "The start of the row interval of the object",
        unit=UNITS.PX,
    )
    ROW_INTERVAL_END = Entry("RowIntervalEnd", "The end of the row interval of the object", unit=UNITS.PX)

    COL_NUM = Entry(
        "ColNum",
        "Zero-based column number, from 0 at the left, of the grid cell that "
        "holds the colony's centroid (the mean position of its pixels). A "
        "colony whose centroid lies outside the grid is assigned to the "
        "nearest edge column.",
    )
    COL_INTERVAL_START = Entry(
        "ColIntervalStart",
        "The start of the column interval of the object",
        unit=UNITS.PX,
    )
    COL_INTERVAL_END = Entry("ColIntervalEnd", "The end of the column interval of the object", unit=UNITS.PX)

    ROW_MAJOR_IDX = Entry(
        "RowMajorIdx",
        "Zero-based number of the colony's grid cell, counted left to right "
        "along each row and then down the rows. It is RowNum × (number of "
        "grid columns) + ColNum. Row major is the standard in most "
        "programming and data science array libraries. Used for indexing into "
        "2D arrays.",
    )

    COL_MAJOR_IDX = Entry(
        "ColMajorIdx",
        "Zero-based number of the colony's grid cell, counted top to bottom "
        "down each column and then across the columns. It is ColNum × (number "
        "of grid rows) + RowNum. Lab automation logic uses column-major "
        "(column-wise) indexing for well plate operations because 96-well "
        "plates are physically arranged with 8 rows (labeled A-H) and 12 "
        "columns (numbered 1-12), and this layout maps directly to how "
        "multichannel pipettes operate."
    )
