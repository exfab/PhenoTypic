"""Per-object summary statistics in the CIE L*a*b* color space."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class ColorLab(MeasurementInfo):
    @classmethod
    def category(cls):
        return "ColorLab"

    L_STAR_MINIMUM = ("L*Min", "The minimum L* value of the object")
    L_STAR_Q1 = ("L*Q1", "The lower quartile (Q1) L* value of the object")
    L_STAR_MEAN = ("L*Mean", "The mean L* value of the object")
    L_STAR_MEDIAN = ("L*Median", "The median L* value of the object")
    L_STAR_Q3 = ("L*Q3", "The upper quartile (Q3) L* value of the object")
    L_STAR_MAXIMUM = ("L*Max", "The maximum L* value of the object")
    L_STAR_STDDEV = ("L*StdDev", "The standard deviation of the L* value of the object")
    L_STAR_COEFF_VARIANCE = (
        "L*CoeffVar",
        "The coefficient of variation of the L* value of the object",
    )

    @classmethod
    def l_star_headers(cls):
        return [
            str(cls.L_STAR_MINIMUM),
            str(cls.L_STAR_Q1),
            str(cls.L_STAR_MEAN),
            str(cls.L_STAR_MEDIAN),
            str(cls.L_STAR_Q3),
            str(cls.L_STAR_MAXIMUM),
            str(cls.L_STAR_STDDEV),
            str(cls.L_STAR_COEFF_VARIANCE),
        ]

    A_STAR_MINIMUM = ("a*Min", "The minimum a* value of the object")
    A_STAR_Q1 = ("a*Q1", "The lower quartile (Q1) a* value of the object")
    A_STAR_MEAN = ("a*Mean", "The mean a* value of the object")
    A_STAR_MEDIAN = ("a*Median", "The median a* value of the object")
    A_STAR_Q3 = ("a*Q3", "The upper quartile (Q3) a* value of the object")
    A_STAR_MAXIMUM = ("a*Max", "The maximum a* value of the object")
    A_STAR_STDDEV = ("a*StdDev", "The standard deviation of the a* value of the object")
    A_STAR_COEFF_VARIANCE = (
        "a*CoeffVar",
        "The coefficient of variation of the a* value of the object",
    )

    @classmethod
    def a_star_headers(cls):
        return [
            str(cls.A_STAR_MINIMUM),
            str(cls.A_STAR_Q1),
            str(cls.A_STAR_MEAN),
            str(cls.A_STAR_MEDIAN),
            str(cls.A_STAR_Q3),
            str(cls.A_STAR_MAXIMUM),
            str(cls.A_STAR_STDDEV),
            str(cls.A_STAR_COEFF_VARIANCE),
        ]

    B_STAR_MINIMUM = ("b*Min", "The minimum b* value of the object")
    B_STAR_Q1 = ("b*Q1", "The lower quartile (Q1) b* value of the object")
    B_STAR_MEAN = ("b*Mean", "The mean b* value of the object")
    B_STAR_MEDIAN = ("b*Median", "The median b* value of the object")
    B_STAR_Q3 = ("b*Q3", "The upper quartile (Q3) b* value of the object")
    B_STAR_MAXIMUM = ("b*Max", "The maximum b* value of the object")
    B_STAR_STDDEV = ("b*StdDev", "The standard deviation of the b* value of the object")
    B_STAR_COEFF_VARIANCE = (
        "b*CoeffVar",
        "The coefficient of variation of the b* value of the object",
    )

    @classmethod
    def b_star_headers(cls):
        return [
            str(cls.B_STAR_MINIMUM),
            str(cls.B_STAR_Q1),
            str(cls.B_STAR_MEAN),
            str(cls.B_STAR_MEDIAN),
            str(cls.B_STAR_Q3),
            str(cls.B_STAR_MAXIMUM),
            str(cls.B_STAR_STDDEV),
            str(cls.B_STAR_COEFF_VARIANCE),
        ]

    CHROMA_EST_MEAN = (
        "ChromaEstimatedMean",
        r"The mean chroma estimation of the object calculated using :math:`\(sqrt(a^{*}_{mean}^2 + b^{*}_{mean})^2}`",
    )
    CHROMA_EST_MEDIAN = (
        "ChromaEstimatedMedian",
        r"The median chroma estimation of the object using :math:`\sqrt({a*_{median}^2 + b*_{median})^2}`",
    )
