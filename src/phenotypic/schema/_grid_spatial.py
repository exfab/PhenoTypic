"""Measurement info for spatial information for grid pinned colonies."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class GRID_SPATIAL(QualityInfo):
    """Measure pixel-to-pixel distances to neighbors in adjacent grid cells.

    For each detected colony, identify the nearest object in the left,
    right, above, and below grid cells and report the minimum Euclidean
    distance between their pixel masks. Distances are computed via a
    per-section distance transform over a local window covering the target
    cell and its immediate neighbors, so round colonies are not
    over-estimated by their bounding boxes. Edge and corner colonies report
    ``NaN`` for directions beyond the plate boundary.
    """

    @classmethod
    def category(cls) -> str:
        return "GridSpatial"

    LEFT_NEIGHBOR_OBJ_LABEL = Entry("LeftNeighborObjLabel", ("The object label of the left"
                                                       " neighbor colony"))
    LEFT_DISTANCE = Entry("LeftDistance", ("The minimum pixel-to-pixel distance to the left"
                                     " neighbor colony, computed via a Euclidean"
                                     " distance transform of object pixel masks"))

    RIGHT_NEIGHBOR_OBJ_LABEL = Entry("RightNeighborObjLabel", ("The object label of"
                                                         " the right neighbor colony"))
    RIGHT_DISTANCE = Entry(
        "RightDistance",
        "The minimum pixel-to-pixel distance to the right neighbor colony, computed"
        " via a Euclidean distance transform of object pixel masks"
    )
    ABOVE_NEIGHBOR_OBJ_LABEL = Entry("AboveNeighborObjLabel", ("The object label of"
                                                         " the above neighbor colony"))
    ABOVE_DISTANCE = Entry(
        "AboveDistance",
        "The minimum pixel-to-pixel distance to the above neighbor colony, computed"
        " via a Euclidean distance transform of object pixel masks"
    )
    UNDER_NEIGHBOR_OBJ_LABEL = Entry("UnderNeighborObjLabel",
                                "The object label of the under neighbor colony")
    UNDER_DISTANCE = Entry(
        "UnderDistance",
        "The minimum pixel-to-pixel distance to the under neighbor colony, computed"
        " via a Euclidean distance transform of object pixel masks"
    )
