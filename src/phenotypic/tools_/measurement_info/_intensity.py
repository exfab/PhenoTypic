"""Per-object grayscale intensity summary statistics."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class INTENSITY(MeasurementInfo):
    """Measure grayscale intensity statistics of detected colonies.

    Compute per-colony intensity metrics from the grayscale channel:
    integrated intensity, percentiles (min, Q1, median, Q3, max),
    standard deviation, coefficient of variation, and area-normalized
    density. These statistics reflect colony optical density, biomass
    accumulation, and internal heterogeneity.
    """

    @classmethod
    def category(cls):
        return "Intensity"

    INTEGRATED_INTENSITY = ("IntegratedIntensity", "The sum of the object's pixels")
    DENSITY = ("Density", "The ratio of the object's intensity to the max possible "
                          "intensity of the object")
    CONVEX_DENSITY = ("ConvexDensity", "The ratio of the objects intensity to the max "
                                       "possible intensity of the object's convex hull")
    MINIMUM_INTENSITY = ("MinimumIntensity", "The minimum intensity of the object")
    MAXIMUM_INTENSITY = ("MaximumIntensity", "The maximum intensity of the object")
    MEAN_INTENSITY = ("MeanIntensity", "The mean intensity of the object")
    MEDIAN_INTENSITY = ("MedianIntensity", "The median intensity of the object")
    STANDARD_DEVIATION_INTENSITY = (
        "StandardDeviationIntensity",
        "The standard deviation of the object",
    )
    COEFFICIENT_VARIANCE_INTENSITY = (
        "CoefficientVarianceIntensity",
        "The coefficient of variation of the object",
    )
    Q1_INTENSITY = (
        "LowerQuartileIntensity",
        "The lower quartile intensity of the object",
    )
    Q3_INTENSITY = (
        "UpperQuartileIntensity",
        "The upper quartile intensity of the object",
    )
    IQR_INTENSITY = (
        "InterquartileRangeIntensity",
        "The interquartile range of the object",
    )
