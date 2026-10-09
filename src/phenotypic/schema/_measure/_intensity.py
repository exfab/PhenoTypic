"""Per-object grayscale intensity summary statistics."""

from .._base._categories import CATEGORIES
from .._base._change_notes import SINGLE_CHANNEL_PRODUCED_NOTE, append_change_note
from .._base._measurement_info import Entry
from .._base._tiers import DirectPhenotype


class INTENSITY(DirectPhenotype):
    """Measure grayscale intensity statistics of detected colonies.

    Compute per-colony intensity metrics from the grayscale channel:
    integrated intensity, percentiles (min, Q1, median, Q3, max),
    standard deviation, coefficient of variation, and area-normalized
    density. These statistics reflect colony optical density, biomass
    accumulation, and internal heterogeneity.
    """

    @classmethod
    def metric_family(cls):
        return "Intensity"

    @classmethod
    def change_note(cls) -> str:
        return SINGLE_CHANNEL_PRODUCED_NOTE

    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        "The sum of the object's pixels",
        categories=CATEGORIES.STARTING_METRICS,
    )
    DENSITY = Entry("Density", "The ratio of the object's intensity to the max possible "
                          "intensity of the object")
    CONVEX_DENSITY = Entry("ConvexDensity", "The ratio of the objects intensity to the max "
                                       "possible intensity of the object's convex hull")
    MINIMUM_INTENSITY = Entry("MinimumIntensity", "The minimum intensity of the object")
    MAXIMUM_INTENSITY = Entry("MaximumIntensity", "The maximum intensity of the object")
    MEAN_INTENSITY = Entry("MeanIntensity", "The mean intensity of the object")
    MEDIAN_INTENSITY = Entry("MedianIntensity", "The median intensity of the object")
    STANDARD_DEVIATION_INTENSITY = Entry(
        "StandardDeviationIntensity",
        "The standard deviation of the object",
    )
    COEFFICIENT_VARIANCE_INTENSITY = Entry(
        "CoefficientVarianceIntensity",
        "The coefficient of variation of the object",
    )
    Q1_INTENSITY = Entry(
        "LowerQuartileIntensity",
        "The lower quartile intensity of the object",
    )
    Q3_INTENSITY = Entry(
        "UpperQuartileIntensity",
        "The upper quartile intensity of the object",
    )
    IQR_INTENSITY = Entry(
        "InterquartileRangeIntensity",
        "The interquartile range of the object",
    )


INTENSITY.__doc__ = append_change_note(INTENSITY.__doc__, INTENSITY.change_note())
