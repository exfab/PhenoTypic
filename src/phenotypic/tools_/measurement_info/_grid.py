"""Constants for grid structure in the PhenoTypic module."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class GRID(MeasurementInfo):
    """Constants for grid structure in the PhenoTypic module."""

    @classmethod
    def category(cls) -> str:
        return "Grid"

    ROW_NUM = "RowNum", "The row idx of the object"
    ROW_INTERVAL_START = (
        "RowIntervalStart",
        "The start of the row interval of the object",
    )
    ROW_INTERVAL_END = "RowIntervalEnd", "The end of the row interval of the object"

    COL_NUM = "ColNum", "The column idx of the object"
    COL_INTERVAL_START = (
        "ColIntervalStart",
        "The start of the column interval of the object",
    )
    COL_INTERVAL_END = "ColIntervalEnd", "The end of the column interval of the object"

    ROW_MAJOR_IDX = (
        "RowMajorIdx",
        "The row-major index of the object. Row major is the standard in most"
        " programming and data science array libraries. Used for indexing into"
        " 2D arrays.",
    )

    COL_MAJOR_IDX = (
        "ColMajorIdx",
        "The col-major index of the object in an array. Lab automation logic uses"
        " column-major (column-wise) indexing for well plate operations because"
        " 96-well plates are physically arranged with 8 rows"
        " (labeled A-H) and 12 columns (numbered 1-12), and this layout maps directly"
        " to how multichannel pipettes operate."
    )
