"""Per-object summary statistics in the CIE XYZ color space."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class ColorXYZ(MeasurementInfo):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Covers CIE XYZ tristimulus values (X, Y, Z).
    """

    @classmethod
    def category(cls):
        return "ColorXYZ"

    X_MINIMUM = ("CieXMin", "The minimum X value of the object in CIE XYZ color space")
    X_Q1 = (
        "CieXQ1",
        "The lower quartile (Q1) X value of the object in CIE XYZ color space",
    )
    X_MEAN = ("CieXMean", "The mean X value of the object in CIE XYZ color space")
    X_MEDIAN = ("CieXMedian", "The median X value of the object in CIE XYZ color space")
    X_Q3 = (
        "CieXQ3",
        "The upper quartile (Q3) X value of the object in CIE XYZ color space",
    )
    X_MAXIMUM = ("CieXMax", "The maximum X value of the object in CIE XYZ color space")
    X_STDDEV = (
        "CieXStdDev",
        "The standard deviation of the X value of the object in CIE XYZ color space",
    )
    X_COEFF_VARIANCE = (
        "CieXCoeffVar",
        "The coefficient of variation of the X value of the object in CIE XYZ color space",
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

    Y_MINIMUM = ("CieYMin", "The minimum Y value of the object in CIE XYZ color space")
    Y_Q1 = (
        "CieYQ1",
        "The lower quartile (Q1) Y value of the object in CIE XYZ color space",
    )
    Y_MEAN = ("CieYMean", "The mean Y value of the object in CIE XYZ color space")
    Y_MEDIAN = ("CieYMedian", "The median Y value of the object in CIE XYZ color space")
    Y_Q3 = (
        "CieYQ3",
        "The upper quartile (Q3) Y value of the object in CIE XYZ color space",
    )
    Y_MAXIMUM = ("CieYMax", "The maximum Y value of the object in CIE XYZ color space")
    Y_STDDEV = (
        "CieYStdDev",
        "The standard deviation of the Y value of the object in CIE XYZ color space",
    )
    Y_COEFF_VARIANCE = (
        "CieYCoeffVar",
        "The coefficient of variation of the Y value of the object in CIE XYZ color space",
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

    Z_MINIMUM = ("CieZMin", "The minimum Z value of the object in CIE XYZ color space")
    Z_Q1 = (
        "CieZQ1",
        "The lower quartile (Q1) Z value of the object in CIE XYZ color space",
    )
    Z_MEAN = ("CieZMean", "The mean Z value of the object in CIE XYZ color space")
    Z_MEDIAN = ("CieZMedian", "The median Z value of the object in CIE XYZ color space")
    Z_Q3 = (
        "CieZQ3",
        "The upper quartile (Q3) Z value of the object in CIE XYZ color space",
    )
    Z_MAXIMUM = ("CieZMax", "The maximum Z value of the object in CIE XYZ color space")
    Z_STDDEV = (
        "CieZStdDev",
        "The standard deviation of the Z value of the object in CIE XYZ color space",
    )
    Z_COEFF_VARIANCE = (
        "CieZCoeffVar",
        "The coefficient of variation of the Z value of the object in CIE XYZ color space",
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
