"""Per-object grayscale intensity summary statistics."""

from .._base._categories import CATEGORIES
from .._base._change_notes import SINGLE_CHANNEL_PRODUCED_NOTE, append_change_note
from .._base._measurement_info import Entry
from .._base._units import UNITS
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
    def default_unit(cls) -> UNITS:
        return UNITS.INTENSITY

    @classmethod
    def change_note(cls) -> str:
        return SINGLE_CHANNEL_PRODUCED_NOTE

    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        "Sum of the grayscale values of all the colony's pixels, so it grows with "
        "both the colony's area and how bright its pixels are.",
        unit=UNITS.INTENSITY_PX2,
    )
    DENSITY = Entry(
        "Density",
        "The colony's integrated intensity divided by its area in pixels: the "
        "mean grayscale value of its pixels, equal to MeanIntensity.",
    )
    CONVEX_DENSITY = Entry(
        "ConvexDensity",
        "The colony's integrated intensity divided by the area of its convex "
        "hull, so indentations and gaps in the colony lower it relative to "
        "Density. The hull passes through pixel centers, so its area is "
        "smaller than the pixel count: slightly for a convex colony, where "
        "this value can slightly exceed Density, and substantially for an "
        "object of only a few pixels (a 2 x 2 pixel object has a hull area of "
        "1, so this value is four times Density). It is NaN for a single "
        "pixel or a one-pixel-wide line, which has no hull.",
    )
    MINIMUM_INTENSITY = Entry(
        "MinimumIntensity",
        "Lowest grayscale value among the colony's pixels.",
    )
    MAXIMUM_INTENSITY = Entry(
        "MaximumIntensity",
        "Highest grayscale value among the colony's pixels.",
    )
    MEAN_INTENSITY = Entry(
        "MeanIntensity",
        "Average grayscale value of the colony's pixels, read from the image's "
        "gray layer.",
        categories=CATEGORIES.STARTING_METRICS,
    )
    MEDIAN_INTENSITY = Entry(
        "MedianIntensity",
        "Middle grayscale value of the colony's pixels: half of them are at or "
        "below it, so a few very bright or very dark pixels move it less than "
        "they move the mean.",
    )
    STANDARD_DEVIATION_INTENSITY = Entry(
        "StandardDeviationIntensity",
        "How much the grayscale values of the colony's pixels vary around their "
        "mean, as their standard deviation. It is the population standard "
        "deviation, dividing by the pixel count n.",
    )
    COEFFICIENT_VARIANCE_INTENSITY = Entry(
        "CoefficientVarianceIntensity",
        "How much the colony's grayscale values vary relative to their mean, as "
        "the standard deviation divided by the mean. The ratio is then "
        "multiplied by (1 + 1/n), where n is the colony's pixel count.",
        unit=UNITS.UNITLESS,
    )
    Q1_INTENSITY = Entry(
        "LowerQuartileIntensity",
        "Grayscale value that about a quarter of the colony's pixels are at or "
        "below. It is the 25th percentile, linearly interpolated.",
    )
    Q3_INTENSITY = Entry(
        "UpperQuartileIntensity",
        "Grayscale value that about three quarters of the colony's pixels are "
        "at or below. It is the 75th percentile, linearly interpolated.",
    )
    IQR_INTENSITY = Entry(
        "InterquartileRangeIntensity",
        "Spread of the middle half of the colony's grayscale values, from the "
        "lower to the upper quartile. It is UpperQuartileIntensity minus "
        "LowerQuartileIntensity.",
    )


INTENSITY.__doc__ = append_change_note(INTENSITY.__doc__, INTENSITY.change_note())
