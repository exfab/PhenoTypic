"""Measurement info for spatial information for grid pinned colonies."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class NEIGHBOR_DIST(QualityInfo):
    """Measure distances from each colony to its grid neighbours and its nearest object.

    Two families of columns. The directional columns report, for each colony,
    the nearest object in the left, right, above, and below grid cells and the
    minimum Euclidean distance between their pixel masks, computed via a
    per-section distance transform so round colonies are not over-estimated by
    their bounding boxes; edge and corner colonies report ``NaN`` beyond the
    plate boundary. The nearest-object columns report the single closest other
    object anywhere on the plate (same cell, diagonal, or further away) and its
    grid relation to the colony. On a plain ``Image`` only the nearest label and
    distance are populated.
    """

    @classmethod
    def category(cls) -> str:
        return "NeighborDist"

    LEFT_NEIGHBOR_OBJ_LABEL = Entry("LeftNeighborObjLabel",
                                    ("The object label of the left"
                                     " neighbor colony"))
    LEFT_DISTANCE = Entry("LeftDistance",
                          ("The minimum pixel-to-pixel distance to the left"
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
    NEAREST_OBJ_LABEL = Entry(
            "NearestObjLabel",
            "The object label of the closest other object anywhere on the plate,"
            " by minimum pixel-to-pixel distance between object masks. On a"
            " GridImage only objects assigned to a grid cell are considered;"
            " on a plain Image every object is. NaN when fewer than two"
            " eligible objects exist. Ties resolve to the smaller label."
    )
    NEAREST_DISTANCE = Entry(
            "NearestDistance",
            "The minimum Euclidean distance, in pixels, between the pixel centres"
            " of this object's mask and the nearest object's mask. Two objects"
            " whose masks share an edge report 1. Always less than or equal to"
            " every non-NaN directional distance for the same object."
    )
    NEAREST_RELATION = Entry(
            "NearestRelation",
            "Integer code for where the nearest object sits relative to this"
            " object's grid cell, from the absolute row offset dr and column"
            " offset dc between the two cells: 0 = same cell (dr = dc = 0);"
            " 1 = edge-adjacent cell (dr + dc = 1); 2 = diagonal cell"
            " (dr = dc = 1); 3 = any cell further away. NaN on a plain Image"
            " (no grid) or when there is no nearest object."
    )
