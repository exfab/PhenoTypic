"""Bounding-box and centroid coordinate measurements."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class BBOX(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "Bbox"

    CENTER_RR = "CenterRR", "The row coordinate of the center of the bounding box."
    MIN_RR = "MinRR", "The smallest row coordinate of the bounding box."
    MAX_RR = "MaxRR", "The largest row coordinate of the bounding box."
    CENTER_CC = "CenterCC", " The column coordinate of the center of the bounding box."
    MIN_CC = "MinCC", " The smallest column coordinate of the bounding box."
    MAX_CC = "MaxCC", " The largest column coordinate of the bounding box."
    INTENSITY_WEIGHTED_CENTER_RR = (
        "IntensityWeightedCenterRR",
        "The intensity-weighted center row coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    INTENSITY_WEIGHTED_CENTER_CC = (
        "IntensityWeightedCenterCC",
        "The intensity-weighted center column coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    DIST_WEIGHTED_CENTER_RR = (
        "DistWeightedCenterRR",
        "Row coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )
    DIST_WEIGHTED_CENTER_CC = (
        "DistWeightedCenterCC",
        "Column coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )
