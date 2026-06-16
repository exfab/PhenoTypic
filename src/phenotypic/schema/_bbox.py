"""Bounding-box and centroid coordinate measurements."""

from ._measurement_info import Entry, MeasurementInfo


class BBOX(MeasurementInfo):
    """Extract bounding box coordinates and centroids of detected colonies.

    Compute the axis-aligned bounding box and centroid (geometric and
    intensity-weighted) for each detected colony. These spatial
    measurements form the foundation for region-of-interest extraction,
    grid alignment assessment, and neighbor-distance calculations.
    """

    @classmethod
    def category(cls) -> str:
        return "Bbox"

    CENTER_RR = Entry("CenterRR", "The row coordinate of the center of the bounding box.")
    MIN_RR = Entry("MinRR", "The smallest row coordinate of the bounding box.")
    MAX_RR = Entry("MaxRR", "The largest row coordinate of the bounding box.")
    CENTER_CC = Entry("CenterCC", " The column coordinate of the center of the bounding box.")
    MIN_CC = Entry("MinCC", " The smallest column coordinate of the bounding box.")
    MAX_CC = Entry("MaxCC", " The largest column coordinate of the bounding box.")
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
        "Row coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )
    DIST_WEIGHTED_CENTER_CC = Entry(
        "DistWeightedCenterCC",
        "Column coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )
