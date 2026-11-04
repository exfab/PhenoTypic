"""Bounding-box and centroid coordinate measurements."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import IdentityInfo


class BBOX(IdentityInfo):
    """Extract bounding box coordinates and centroids of detected colonies.

    Compute the axis-aligned bounding box and centroid (geometric and
    intensity-weighted) for each detected colony. These spatial
    measurements form the foundation for region-of-interest extraction,
    grid alignment assessment, and neighbor-distance calculations.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "Bbox"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.PX

    CENTER_RR = Entry(
        "CenterRR",
        "Row coordinate of the colony's centroid, the mean position of its "
        "pixels (skimage ``centroid``).",
    )
    MIN_RR = Entry("MinRR", "The smallest row coordinate of the bounding box.")
    MAX_RR = Entry(
        "MaxRR",
        "The largest row coordinate of the bounding box. It is one past the "
        "colony's last pixel row (exclusive, as in Python slicing), so rows "
        "MinRR to MaxRR - 1 hold the colony and ``image[MinRR:MaxRR]`` covers "
        "it.",
    )
    CENTER_CC = Entry(
        "CenterCC",
        "Column coordinate of the colony's centroid, the mean position of its "
        "pixels (skimage ``centroid``).",
    )
    MIN_CC = Entry(
        "MinCC",
        "The smallest column coordinate of the bounding box.",
    )
    MAX_CC = Entry(
        "MaxCC",
        "The largest column coordinate of the bounding box. It is one past "
        "the colony's last pixel column (exclusive, as in Python slicing), so "
        "columns MinCC to MaxCC - 1 hold the colony and ``image[:, "
        "MinCC:MaxCC]`` covers it.",
    )
    INTENSITY_WEIGHTED_CENTER_RR = Entry(
        "IntensityWeightedCenterRR",
        "The intensity-weighted center row coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    INTENSITY_WEIGHTED_CENTER_CC = Entry(
        "IntensityWeightedCenterCC",
        "The intensity-weighted center column coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    DIST_WEIGHTED_CENTER_RR = Entry(
        "DistWeightedCenterRR",
        "Row coordinate of the colony's distance-weighted centroid: the mean "
        "position of its pixels, each weighted by its distance to the "
        "colony's edge. It is Σ dt·row / Σ dt over the Euclidean distance "
        "transform dt, and the plain centroid for a colony with no interior "
        "pixels. Robust to thin filamentous extensions that pull "
        "intensity-weighted centroids off-body.",
    )
    DIST_WEIGHTED_CENTER_CC = Entry(
        "DistWeightedCenterCC",
        "Column coordinate of the colony's distance-weighted centroid: the "
        "mean position of its pixels, each weighted by its distance to the "
        "colony's edge. It is Σ dt·col / Σ dt over the Euclidean distance "
        "transform dt, and the plain centroid for a colony with no interior "
        "pixels. Robust to thin filamentous extensions that pull "
        "intensity-weighted centroids off-body.",
    )
