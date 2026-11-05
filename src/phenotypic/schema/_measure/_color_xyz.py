"""Per-object summary statistics in the CIE XYZ color space."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DiscriminativeFeature


class ColorXYZ(DiscriminativeFeature):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Covers CIE XYZ tristimulus values (X, Y, Z).
    """

    @classmethod
    def metric_family(cls):
        return "ColorXYZ"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    X_MINIMUM = Entry(
        "CieXMin",
        "Lowest CIE 1931 X tristimulus value among the colony's pixels.",
    )
    X_Q1 = Entry(
        "CieXQ1",
        "CIE 1931 X tristimulus value that about a quarter of the colony's "
        "pixels are at or below. It is the 25th percentile, linearly "
        "interpolated.",
    )
    X_MEAN = Entry(
        "CieXMean",
        "Average CIE 1931 X tristimulus value of the colony's pixels.",
    )
    X_MEDIAN = Entry(
        "CieXMedian",
        "Median CIE 1931 X tristimulus value of the colony's pixels.",
    )
    X_Q3 = Entry(
        "CieXQ3",
        "CIE 1931 X tristimulus value that about three quarters of the "
        "colony's pixels are at or below. It is the 75th percentile, linearly "
        "interpolated.",
    )
    X_MAXIMUM = Entry(
        "CieXMax",
        "Highest CIE 1931 X tristimulus value among the colony's pixels.",
    )
    X_STDDEV = Entry(
        "CieXStdDev",
        "How much the CIE 1931 X tristimulus value varies across the colony's "
        "pixels, as its population standard deviation.",
    )
    X_COEFF_VARIANCE = Entry(
        "CieXCoeffVar",
        "How much the CIE 1931 X tristimulus value varies across the colony's "
        "pixels relative to its mean, as the standard deviation divided by "
        "the mean. The ratio is then multiplied by (1 + 1/n), where n is the "
        "colony's pixel count.",
    )

    @classmethod
    def cieX_headers(cls):
        return [
            str(cls.X_MINIMUM),
            str(cls.X_Q1),
            str(cls.X_MEAN),
            str(cls.X_MEDIAN),
            str(cls.X_Q3),
            str(cls.X_MAXIMUM),
            str(cls.X_STDDEV),
            str(cls.X_COEFF_VARIANCE),
        ]

    Y_MINIMUM = Entry(
        "CieYMin",
        "Lowest CIE 1931 Y tristimulus value (relative luminance) among the "
        "colony's pixels.",
    )
    Y_Q1 = Entry(
        "CieYQ1",
        "CIE 1931 Y tristimulus value (relative luminance) that about a "
        "quarter of the colony's pixels are at or below. It is the 25th "
        "percentile, linearly interpolated.",
    )
    Y_MEAN = Entry(
        "CieYMean",
        "Average CIE 1931 Y tristimulus value (relative luminance) of the "
        "colony's pixels.",
    )
    Y_MEDIAN = Entry(
        "CieYMedian",
        "Median CIE 1931 Y tristimulus value (relative luminance) of the "
        "colony's pixels.",
    )
    Y_Q3 = Entry(
        "CieYQ3",
        "CIE 1931 Y tristimulus value (relative luminance) that about three "
        "quarters of the colony's pixels are at or below. It is the 75th "
        "percentile, linearly interpolated.",
    )
    Y_MAXIMUM = Entry(
        "CieYMax",
        "Highest CIE 1931 Y tristimulus value (relative luminance) among the "
        "colony's pixels.",
    )
    Y_STDDEV = Entry(
        "CieYStdDev",
        "How much the CIE 1931 Y tristimulus value (relative luminance) "
        "varies across the colony's pixels, as its population standard "
        "deviation.",
    )
    Y_COEFF_VARIANCE = Entry(
        "CieYCoeffVar",
        "How much the CIE 1931 Y tristimulus value (relative luminance) "
        "varies across the colony's pixels relative to its mean, as the "
        "standard deviation divided by the mean. The ratio is then multiplied "
        "by (1 + 1/n), where n is the colony's pixel count.",
    )

    @classmethod
    def cieY_headers(cls):
        return [
            str(cls.Y_MINIMUM),
            str(cls.Y_Q1),
            str(cls.Y_MEAN),
            str(cls.Y_MEDIAN),
            str(cls.Y_Q3),
            str(cls.Y_MAXIMUM),
            str(cls.Y_STDDEV),
            str(cls.Y_COEFF_VARIANCE),
        ]

    Z_MINIMUM = Entry(
        "CieZMin",
        "Lowest CIE 1931 Z tristimulus value among the colony's pixels.",
    )
    Z_Q1 = Entry(
        "CieZQ1",
        "CIE 1931 Z tristimulus value that about a quarter of the colony's "
        "pixels are at or below. It is the 25th percentile, linearly "
        "interpolated.",
    )
    Z_MEAN = Entry(
        "CieZMean",
        "Average CIE 1931 Z tristimulus value of the colony's pixels.",
    )
    Z_MEDIAN = Entry(
        "CieZMedian",
        "Median CIE 1931 Z tristimulus value of the colony's pixels.",
    )
    Z_Q3 = Entry(
        "CieZQ3",
        "CIE 1931 Z tristimulus value that about three quarters of the "
        "colony's pixels are at or below. It is the 75th percentile, linearly "
        "interpolated.",
    )
    Z_MAXIMUM = Entry(
        "CieZMax",
        "Highest CIE 1931 Z tristimulus value among the colony's pixels.",
    )
    Z_STDDEV = Entry(
        "CieZStdDev",
        "How much the CIE 1931 Z tristimulus value varies across the colony's "
        "pixels, as its population standard deviation.",
    )
    Z_COEFF_VARIANCE = Entry(
        "CieZCoeffVar",
        "How much the CIE 1931 Z tristimulus value varies across the colony's "
        "pixels relative to its mean, as the standard deviation divided by "
        "the mean. The ratio is then multiplied by (1 + 1/n), where n is the "
        "colony's pixel count.",
    )

    @classmethod
    def cieZ_headers(cls):
        return [
            str(cls.Z_MINIMUM),
            str(cls.Z_Q1),
            str(cls.Z_MEAN),
            str(cls.Z_MEDIAN),
            str(cls.Z_Q3),
            str(cls.Z_MAXIMUM),
            str(cls.Z_STDDEV),
            str(cls.Z_COEFF_VARIANCE),
        ]
