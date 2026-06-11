"""Per-object robust colorimetric statistics in the CIE L*a*b* color space."""

from ._measurement_info import MeasurementInfo


class ColorLab(MeasurementInfo):
    """Robust CIE L*a*b* colorimetric summary for a colony.

    Reports two robust center colors -- the ΔE76 (Euclidean) geometric median
    and the ΔE2000 medoid -- plus ΔE2000 within-colony consistency scalars, the
    total Euclidean color variance, and an sRGB hex swatch (plot-only) derived
    from the medoid.
    """

    @classmethod
    def category(cls):
        return "ColorLab"

    # -- ΔE76 geometric-median center (continuous, 0.5 breakdown) --
    L_STAR_GEOMEDIAN = ("L*GeoMedian", "L* of the ΔE76 (Euclidean) geometric-median center color of the object")
    A_STAR_GEOMEDIAN = ("a*GeoMedian", "a* of the ΔE76 (Euclidean) geometric-median center color of the object")
    B_STAR_GEOMEDIAN = ("b*GeoMedian", "b* of the ΔE76 (Euclidean) geometric-median center color of the object")

    # -- ΔE2000 medoid center (real pixel, perceptually-corrected) --
    L_STAR_MEDOID = ("L*Medoid", "L* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")
    A_STAR_MEDOID = ("a*Medoid", "a* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")
    B_STAR_MEDOID = ("b*Medoid", "b* of the ΔE2000 medoid center color (real pixel minimizing total ΔE2000)")

    # -- ΔE2000 within-colony consistency, measured from the medoid --
    DELTA_E2000_MEDIAN = ("DeltaE2000MedianFromMedoid", "Median ΔE2000 of object pixels from the ΔE2000 medoid center (robust perceptual MAD)")
    DELTA_E2000_MEAN = ("DeltaE2000MeanFromMedoid", "Mean ΔE2000 of object pixels from the ΔE2000 medoid center (color-uniformity standard)")
    DELTA_E2000_P95 = ("DeltaE2000P95FromMedoid", "95th-percentile ΔE2000 of object pixels from the ΔE2000 medoid center (worst-case / sectoring flag)")

    # -- classical Euclidean spread --
    LAB_TOTAL_VARIANCE = ("LabTotalVariance", "Trace of the 3x3 L*a*b* covariance (var L* + var a* + var b*); mean-squared ΔE76 spread about the arithmetic mean (NOT about the reported GeoMedian/Medoid center)")

    # -- plot-only swatch --
    MEDOID_COLOR_HEX = ("MedoidColorHex", "sRGB hex string of the ΔE2000 medoid color; for plot visualization only (not a numeric measurement)")

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
