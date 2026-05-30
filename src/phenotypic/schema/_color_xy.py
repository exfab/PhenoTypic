"""Per-object summary statistics in CIE chromaticity xy coordinates."""

from ._measurement_info import MeasurementInfo


class Colorxy(MeasurementInfo):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Covers the (x, y) chromaticity channels.
    """

    @classmethod
    def category(cls):
        return "Colorxy"

    x_MINIMUM = ("xMin", "The minimum chromaticity x coordinate of the object")
    x_Q1 = ("xQ1", "The lower quartile (Q1) chromaticity x coordinate of the object")
    x_MEAN = ("xMean", "The mean chromaticity x coordinate of the object")
    x_MEDIAN = ("xMedian", "The median chromaticity x coordinate of the object")
    x_Q3 = ("xQ3", "The upper quartile (Q3) chromaticity x coordinate of the object")
    x_MAXIMUM = ("xMax", "The maximum chromaticity x coordinate of the object")
    x_STDDEV = (
        "xStdDev",
        "The standard deviation of the chromaticity x coordinate of the object",
    )
    x_COEFF_VARIANCE = (
        "xCoeffVar",
        "The coefficient of variation of the chromaticity x coordinate of the object",
    )

    @classmethod
    def x_headers(cls):
        return [
            str(cls.x_MINIMUM),
            str(cls.x_Q1),
            str(cls.x_MEAN),
            str(cls.x_MEDIAN),
            str(cls.x_Q3),
            str(cls.x_MAXIMUM),
            str(cls.x_STDDEV),
            str(cls.x_COEFF_VARIANCE),
        ]

    y_MINIMUM = ("yMin", "The minimum chromaticity y coordinate of the object")
    y_Q1 = ("yQ1", "The lower quartile (Q1) chromaticity y coordinate of the object")
    y_MEAN = ("yMean", "The mean chromaticity y coordinate of the object")
    y_MEDIAN = ("yMedian", "The median chromaticity y coordinate of the object")
    y_Q3 = ("yQ3", "The upper quartile (Q3) chromaticity y coordinate of the object")
    y_MAXIMUM = ("yMax", "The maximum chromaticity y coordinate of the object")
    y_STDDEV = (
        "yStdDev",
        "The standard deviation of the chromaticity y coordinate of the object",
    )
    y_COEFF_VARIANCE = (
        "yCoeffVar",
        "The coefficient of variation of the chromaticity y coordinate of the object",
    )

    @classmethod
    def y_headers(cls):
        return [
            str(cls.y_MINIMUM),
            str(cls.y_Q1),
            str(cls.y_MEAN),
            str(cls.y_MEDIAN),
            str(cls.y_Q3),
            str(cls.y_MAXIMUM),
            str(cls.y_STDDEV),
            str(cls.y_COEFF_VARIANCE),
        ]
