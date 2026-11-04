"""Per-object robust colorimetric statistics in the CIE L*a*b* color space."""

from .._base._categories import CATEGORIES
from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DescriptiveTrait


class ColorLab(DescriptiveTrait):
    """Robust CIE L*a*b* colorimetric summary for a colony.

    Reports two robust center colors -- the ΔE76 (Euclidean) geometric median
    and the ΔE2000 medoid -- plus ΔE2000 within-colony consistency scalars, the
    total Euclidean color variance, and an sRGB hex swatch (plot-only) derived
    from the medoid.
    """

    @classmethod
    def metric_family(cls):
        return "ColorLab"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.CIELAB

    # -- ΔE76 geometric-median center (continuous, 0.5 breakdown) --
    L_STAR_GEOMEDIAN = Entry(
        "L*GeoMedian",
        "Lightness (CIE L*, from 0 for black to 100 for white) of the "
        "colony's geometric-median color: the L*a*b* point with the smallest "
        "total distance (ΔE76) to the colony's pixels.",
    )
    A_STAR_GEOMEDIAN = Entry(
        "a*GeoMedian",
        "Green-red coordinate (CIE a*: negative is green, positive is red) of "
        "the colony's geometric-median color: the L*a*b* point with the "
        "smallest total distance (ΔE76) to the colony's pixels.",
    )
    B_STAR_GEOMEDIAN = Entry(
        "b*GeoMedian",
        "Blue-yellow coordinate (CIE b*: negative is blue, positive is "
        "yellow) of the colony's geometric-median color: the L*a*b* point "
        "with the smallest total distance (ΔE76) to the colony's pixels.",
    )

    # -- ΔE2000 medoid center (real pixel, perceptually-corrected) --
    L_STAR_MEDOID = Entry(
        "L*Medoid",
        "Lightness (CIE L*, from 0 for black to 100 for white) of the "
        "colony's medoid color: the colony pixel with the smallest total "
        "ΔE2000 color difference to all the colony's pixels. Found "
        "deterministically (no random sampling): the candidates are the "
        "object pixels nearest a tightly converged L*a*b* geometric median, "
        "each scored against every object pixel, which recovers the "
        "exhaustive medoid on a unimodal colour cloud. That candidate centre "
        "is computed separately and more tightly than the reported "
        "``*GeoMedian`` columns, so it need not equal them. Only the object's "
        "own pixels are used, never a neighbouring object's",
        categories=CATEGORIES.STARTING_METRICS,
    )
    A_STAR_MEDOID = Entry(
        "a*Medoid",
        "Green-red coordinate (CIE a*: negative is green, positive is red) of "
        "the colony's medoid color: the colony pixel with the smallest total "
        "ΔE2000 color difference to all the colony's pixels. Found "
        "deterministically (no random sampling): the candidates are the "
        "object pixels nearest a tightly converged L*a*b* geometric median, "
        "each scored against every object pixel, which recovers the "
        "exhaustive medoid on a unimodal colour cloud. That candidate centre "
        "is computed separately and more tightly than the reported "
        "``*GeoMedian`` columns, so it need not equal them. Only the object's "
        "own pixels are used, never a neighbouring object's",
        categories=CATEGORIES.STARTING_METRICS,
    )
    B_STAR_MEDOID = Entry(
        "b*Medoid",
        "Blue-yellow coordinate (CIE b*: negative is blue, positive is "
        "yellow) of the colony's medoid color: the colony pixel with the "
        "smallest total ΔE2000 color difference to all the colony's pixels. "
        "Found deterministically (no random sampling): the candidates are the "
        "object pixels nearest a tightly converged L*a*b* geometric median, "
        "each scored against every object pixel, which recovers the "
        "exhaustive medoid on a unimodal colour cloud. That candidate centre "
        "is computed separately and more tightly than the reported "
        "``*GeoMedian`` columns, so it need not equal them. Only the object's "
        "own pixels are used, never a neighbouring object's",
        categories=CATEGORIES.STARTING_METRICS,
    )

    # -- ΔE2000 within-colony consistency, measured from the medoid --
    DELTA_E2000_MEDIAN = Entry("DeltaE2000MedianFromMedoid", "Median ΔE2000 of object pixels from the ΔE2000 medoid center (robust perceptual MAD)")
    DELTA_E2000_MEAN = Entry("DeltaE2000MeanFromMedoid", "Mean ΔE2000 of object pixels from the ΔE2000 medoid center (color-uniformity standard)")
    DELTA_E2000_P95 = Entry("DeltaE2000P95FromMedoid", "95th-percentile ΔE2000 of object pixels from the ΔE2000 medoid center (worst-case / sectoring flag)")

    # -- classical Euclidean spread --
    LAB_TOTAL_VARIANCE = Entry(
        "LabTotalVariance",
        "How spread out the colony's pixel colors are around their average "
        "color, as the sum of the variances of L*, a* and b*. It is the trace "
        "of the 3x3 L*a*b* covariance (var L* + var a* + var b*); "
        "mean-squared ΔE76 spread about the arithmetic mean (NOT about the "
        "reported GeoMedian/Medoid center)",
        unit=UNITS.CIELAB2,
    )

    # -- plot-only swatch --
    MEDOID_COLOR_HEX = Entry("MedoidColorHex", "sRGB hex string of the ΔE2000 medoid color; for plot visualization only (not a numeric measurement)", unit=UNITS.NOT_A_QUANTITY)

    @classmethod
    def robust_headers(cls):
        return [
            str(cls.L_STAR_GEOMEDIAN),
            str(cls.A_STAR_GEOMEDIAN),
            str(cls.B_STAR_GEOMEDIAN),
            str(cls.L_STAR_MEDOID),
            str(cls.A_STAR_MEDOID),
            str(cls.B_STAR_MEDOID),
            str(cls.DELTA_E2000_MEDIAN),
            str(cls.DELTA_E2000_MEAN),
            str(cls.DELTA_E2000_P95),
            str(cls.LAB_TOTAL_VARIANCE),
            str(cls.MEDOID_COLOR_HEX),
        ]
