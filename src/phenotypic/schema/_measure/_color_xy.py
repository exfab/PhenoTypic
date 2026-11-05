"""Per-object summary statistics in CIE chromaticity xy coordinates."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DiscriminativeFeature


class Colorxy(DiscriminativeFeature):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Covers the (x, y) chromaticity channels.
    """

    @classmethod
    def metric_family(cls):
        return "Colorxy"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    x_MINIMUM = Entry(
        "xMin",
        "Lowest CIE x chromaticity coordinate among the colony's pixels. Each "
        "pixel's x is X / (X + Y + Z).",
    )
    x_Q1 = Entry(
        "xQ1",
        "CIE x chromaticity coordinate that about a quarter of the colony's "
        "pixels are at or below. It is the 25th percentile, linearly "
        "interpolated. Each pixel's x is X / (X + Y + Z).",
    )
    x_MEAN = Entry(
        "xMean",
        "Average CIE x chromaticity coordinate of the colony's pixels. Each "
        "pixel's x is X / (X + Y + Z).",
    )
    x_MEDIAN = Entry(
        "xMedian",
        "Median CIE x chromaticity coordinate of the colony's pixels. Each "
        "pixel's x is X / (X + Y + Z).",
    )
    x_Q3 = Entry(
        "xQ3",
        "CIE x chromaticity coordinate that about three quarters of the "
        "colony's pixels are at or below. It is the 75th percentile, linearly "
        "interpolated. Each pixel's x is X / (X + Y + Z).",
    )
    x_MAXIMUM = Entry(
        "xMax",
        "Highest CIE x chromaticity coordinate among the colony's pixels. "
        "Each pixel's x is X / (X + Y + Z).",
    )
    x_STDDEV = Entry(
        "xStdDev",
        "How much the CIE x chromaticity coordinate varies across the "
        "colony's pixels, as its population standard deviation. Each pixel's "
        "x is X / (X + Y + Z).",
    )
    x_COEFF_VARIANCE = Entry(
        "xCoeffVar",
        "How much the CIE x chromaticity coordinate varies across the "
        "colony's pixels relative to its mean, as the standard deviation "
        "divided by the mean. The ratio is then multiplied by (1 + 1/n), "
        "where n is the colony's pixel count. Each pixel's x is X / (X + Y + "
        "Z).",
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

    y_MINIMUM = Entry(
        "yMin",
        "Lowest CIE y chromaticity coordinate among the colony's pixels. Each "
        "pixel's y is Y / (X + Y + Z).",
    )
    y_Q1 = Entry(
        "yQ1",
        "CIE y chromaticity coordinate that about a quarter of the colony's "
        "pixels are at or below. It is the 25th percentile, linearly "
        "interpolated. Each pixel's y is Y / (X + Y + Z).",
    )
    y_MEAN = Entry(
        "yMean",
        "Average CIE y chromaticity coordinate of the colony's pixels. Each "
        "pixel's y is Y / (X + Y + Z).",
    )
    y_MEDIAN = Entry(
        "yMedian",
        "Median CIE y chromaticity coordinate of the colony's pixels. Each "
        "pixel's y is Y / (X + Y + Z).",
    )
    y_Q3 = Entry(
        "yQ3",
        "CIE y chromaticity coordinate that about three quarters of the "
        "colony's pixels are at or below. It is the 75th percentile, linearly "
        "interpolated. Each pixel's y is Y / (X + Y + Z).",
    )
    y_MAXIMUM = Entry(
        "yMax",
        "Highest CIE y chromaticity coordinate among the colony's pixels. "
        "Each pixel's y is Y / (X + Y + Z).",
    )
    y_STDDEV = Entry(
        "yStdDev",
        "How much the CIE y chromaticity coordinate varies across the "
        "colony's pixels, as its population standard deviation. Each pixel's "
        "y is Y / (X + Y + Z).",
    )
    y_COEFF_VARIANCE = Entry(
        "yCoeffVar",
        "How much the CIE y chromaticity coordinate varies across the "
        "colony's pixels relative to its mean, as the standard deviation "
        "divided by the mean. The ratio is then multiplied by (1 + 1/n), "
        "where n is the colony's pixel count. Each pixel's y is Y / (X + Y + "
        "Z).",
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
