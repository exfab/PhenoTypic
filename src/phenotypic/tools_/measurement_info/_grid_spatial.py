"""Measurement info for spatial information for grid pinned colonies."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class GRID_SPATIAL(MeasurementInfo):
    """Measurement info for spatial information for grid pinned colonies"""

    @classmethod
    def category(cls) -> str:
        return "GridSpatial"

    LEFT_NEIGHBOR_OBJ_LABEL = "LeftNeighborObjLabel", ("The object label of the left"
                                                       " neighbor colony")
    LEFT_DISTANCE = "LeftDistance", ("The distance of the left neighbor colony"
                                     " calculated using euclidean distance between"
                                     " bounding boxes")

    RIGHT_NEIGHBOR_OBJ_LABEL = "RightNeighborObjLabel", ("The object label of"
                                                         " the right neighbor colony")
    RIGHT_DISTANCE = (
        "RightDistance",
        "The distance of the right neighbor colony calculated"
        " using euclidean distance between bounding boxes"
    )
    ABOVE_NEIGHBOR_OBJ_LABEL = "AboveNeighborObjLabel", ("The object label of"
                                                         " the above neighbor colony")
    ABOVE_DISTANCE = (
        "AboveDistance",
        "The distance of the above neighbor colony calculated using euclidean"
        " distance between bounding boxes"
    )
    UNDER_NEIGHBOR_OBJ_LABEL = ("UnderNeighborObjLabel",
                                "The object label of the under neighbor colony")
    UNDER_DISTANCE = (
        "UnderDistance",
        "The distance of the under neighbor colony calculated using euclidean distance between bounding boxes"
    )
