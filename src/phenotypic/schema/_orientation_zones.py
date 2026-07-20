"""Public header schema for MeasureOrientationZones (category ``OrientZones``).

Existing absolute-orientation headers use
``OrientZones_<Metric>-<Variant>-<Zone>``. Radial-relative headers use the
detected-structure selector explicitly:
``OrientZones_<Metric>-Mask-<Region>``. Region is usually one of
{Overall, Dense, Sparse}; longer-range rotation also includes the paired
DenseToSparse transition.
"""

from __future__ import annotations

from ._measurement_info import Entry
from ._tiers import DescriptiveTrait, QualityInfo


class ORIENTATION_ZONE_PRIMARY(DescriptiveTrait):
    """Primary full-length outward-rotation traits by radial zone."""

    @classmethod
    def category(cls) -> str:
        return "OrientZones"

    OUTWARD_ROTATION_SUSTAINED_PEAK_OVERALL = Entry(
        "OutwardRotationSustainedPeak-Mask-Overall",
        "Largest rolling-median absolute cumulative rotation across consecutive "
        "supported literal skeleton-ring crossings from the inferred inoculum "
        "boundary through the full detected object extent. Reported in degrees, "
        "nonnegative with no fixed upper bound. The default three-ring window "
        "rejects a one-ring spike; NaN means no complete supported window.",
    )
    OUTWARD_ROTATION_SUSTAINED_PEAK_DENSE = Entry(
        "OutwardRotationSustainedPeak-Mask-Dense",
        "Largest rolling-median absolute cumulative rotation across consecutive "
        "supported literal skeleton-ring crossings from the inferred inoculum "
        "boundary to dense_end. Reported in degrees, nonnegative with no fixed "
        "upper bound. The default three-ring window rejects a one-ring spike; "
        "NaN means no complete supported window.",
    )
    OUTWARD_ROTATION_SUSTAINED_PEAK_SPARSE = Entry(
        "OutwardRotationSustainedPeak-Mask-Sparse",
        "Largest rolling-median absolute cumulative rotation across consecutive "
        "supported literal skeleton-ring crossings from dense_end through the "
        "full detected object extent. Reported in degrees, nonnegative with no "
        "fixed upper bound. The default three-ring window rejects a one-ring "
        "spike; NaN means no complete supported window. Cumulative rotation is "
        "not rebased at dense_end, so this Sparse magnitude may include rotation "
        "accumulated while the profile was in Dense.",
    )
    OUTWARD_ROTATION_NET_OVERALL = Entry(
        "OutwardRotationNet-Mask-Overall",
        "Outer endpoint median minus inner endpoint median of cumulative literal "
        "skeleton-ring rotation along the longest supported run from the inferred "
        "inoculum boundary through the full detected object extent. Reported in "
        "signed degrees with no fixed bounds; positive is clockwise and negative "
        "is counterclockwise while moving outward in image coordinates. NaN means "
        "the run is shorter than the configured minimum.",
    )
    OUTWARD_ROTATION_NET_DENSE = Entry(
        "OutwardRotationNet-Mask-Dense",
        "Outer endpoint median minus inner endpoint median of cumulative literal "
        "skeleton-ring rotation along the longest supported run from the inferred "
        "inoculum boundary to dense_end. Reported in signed degrees with no fixed "
        "bounds; positive is clockwise and negative is counterclockwise while "
        "moving outward in image coordinates. NaN means the run is shorter than "
        "the configured minimum.",
    )
    OUTWARD_ROTATION_NET_SPARSE = Entry(
        "OutwardRotationNet-Mask-Sparse",
        "Outer endpoint median minus inner endpoint median of cumulative literal "
        "skeleton-ring rotation along the longest supported run from dense_end "
        "through the full detected object extent. Reported in signed degrees with "
        "no fixed bounds; positive is clockwise and negative is counterclockwise "
        "while moving outward in image coordinates. NaN means the run is shorter "
        "than the configured minimum. Because this is a difference, cumulative "
        "rotation carried into Sparse cancels from the result.",
    )
    OUTWARD_ROTATION_RATE_OVERALL = Entry(
        "OutwardRotationRate-Mask-Overall",
        "Median of all pairwise cumulative-rotation slopes along the longest "
        "supported literal skeleton-ring run from the inferred inoculum boundary "
        "through the full detected object extent. Reported in signed degrees per "
        "pixel; positive is clockwise and negative is counterclockwise while "
        "moving outward in image coordinates. NaN means the run is shorter than "
        "the configured minimum.",
    )
    OUTWARD_ROTATION_RATE_DENSE = Entry(
        "OutwardRotationRate-Mask-Dense",
        "Median of all pairwise cumulative-rotation slopes along the longest "
        "supported literal skeleton-ring run from the inferred inoculum boundary "
        "to dense_end. Reported in signed degrees per pixel; positive is clockwise "
        "and negative is counterclockwise while moving outward in image "
        "coordinates. NaN means the run is shorter than the configured minimum.",
    )
    OUTWARD_ROTATION_RATE_SPARSE = Entry(
        "OutwardRotationRate-Mask-Sparse",
        "Median of all pairwise cumulative-rotation slopes along the longest "
        "supported literal skeleton-ring run from dense_end through the full "
        "detected object extent. Reported in signed degrees per pixel; positive "
        "is clockwise and negative is counterclockwise while moving outward in "
        "image coordinates. NaN means the run is shorter than the configured "
        "minimum. The slope is unchanged by cumulative rotation carried into "
        "Sparse.",
    )
    OUTWARD_ROTATION_CONSISTENCY_OVERALL = Entry(
        "OutwardRotationConsistency-Mask-Overall",
        "Absolute Kendall tau-b association between radius and cumulative literal "
        "skeleton-ring rotation along the longest supported run from the inferred "
        "inoculum boundary through the full detected object extent. Dimensionless "
        "in [0, 1]; 1 is fully monotonic and values near 0 indicate reversals, "
        "plateaus, or no ordered trend. Direction is carried by the rotation "
        "rate. NaN means the run is shorter than the configured minimum.",
    )
    OUTWARD_ROTATION_CONSISTENCY_DENSE = Entry(
        "OutwardRotationConsistency-Mask-Dense",
        "Absolute Kendall tau-b association between radius and cumulative literal "
        "skeleton-ring rotation along the longest supported run from the inferred "
        "inoculum boundary to dense_end. Dimensionless in [0, 1]; 1 is fully "
        "monotonic and values near 0 indicate reversals, plateaus, or no ordered "
        "trend. Direction is carried by the rotation rate. NaN means the run is "
        "shorter than the configured minimum.",
    )
    OUTWARD_ROTATION_CONSISTENCY_SPARSE = Entry(
        "OutwardRotationConsistency-Mask-Sparse",
        "Absolute Kendall tau-b association between radius and cumulative literal "
        "skeleton-ring rotation along the longest supported run from dense_end "
        "through the full detected object extent. Dimensionless in [0, 1]; 1 is "
        "fully monotonic and values near 0 indicate reversals, plateaus, or no "
        "ordered trend. Direction is carried by the rotation rate, and the rank "
        "association is unchanged by rotation carried into Sparse. NaN means the "
        "run is shorter than the configured minimum.",
    )


class ORIENTATION_ZONE_DIAGNOSTIC(QualityInfo):
    """Opt-in validation, comparator, and legacy orientation-zone columns.

    The literal skeleton-ring diagnostics use the detected-object mask from the
    inferred inoculum boundary through the full object extent. The retained
    legacy field diagnostics use the structure-tensor field in symmetric-radius
    zones and include ``Radial`` (all tile pixels) and ``Mask`` variants. See
    :class:`MeasureOrientationZones` for parameters and method.
    """

    @classmethod
    def category(cls) -> str:
        return "OrientZones"

    OUTWARD_ROTATION_RAW_PEAK_OVERALL = Entry(
        "OutwardRotationRawPeak-Mask-Overall",
        "Largest absolute cumulative literal skeleton-ring rotation from the "
        "inferred inoculum boundary through the full detected object extent. "
        "Reported in degrees, nonnegative with no fixed upper bound. This raw "
        "maximum is outlier-sensitive and should be compared with the primary "
        "sustained peak.",
    )
    OUTWARD_ROTATION_RAW_PEAK_DENSE = Entry(
        "OutwardRotationRawPeak-Mask-Dense",
        "Largest absolute cumulative literal skeleton-ring rotation from the "
        "inferred inoculum boundary to dense_end. Reported in degrees, "
        "nonnegative with no fixed upper bound. This raw maximum is "
        "outlier-sensitive and should be compared with the primary sustained "
        "peak.",
    )
    OUTWARD_ROTATION_RAW_PEAK_SPARSE = Entry(
        "OutwardRotationRawPeak-Mask-Sparse",
        "Largest absolute cumulative literal skeleton-ring rotation from "
        "dense_end through the full detected object extent. Reported in degrees, "
        "nonnegative with no fixed upper bound. This raw maximum is "
        "outlier-sensitive. Cumulative rotation is not rebased at dense_end, so "
        "this Sparse magnitude may include rotation accumulated while the profile "
        "was in Dense.",
    )
    OUTWARD_ROTATION_P90_OVERALL = Entry(
        "OutwardRotationP90-Mask-Overall",
        "90th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from the inferred inoculum boundary through the "
        "full detected object extent. Reported in degrees, nonnegative with no "
        "fixed upper bound. This validation comparator ignores radial order.",
    )
    OUTWARD_ROTATION_P90_DENSE = Entry(
        "OutwardRotationP90-Mask-Dense",
        "90th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from the inferred inoculum boundary to dense_end. "
        "Reported in degrees, nonnegative with no fixed upper bound. This "
        "validation comparator ignores radial order.",
    )
    OUTWARD_ROTATION_P90_SPARSE = Entry(
        "OutwardRotationP90-Mask-Sparse",
        "90th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from dense_end through the full detected object "
        "extent. Reported in degrees, nonnegative with no fixed upper bound. This "
        "comparator ignores radial order. Cumulative rotation is not rebased at "
        "dense_end, so this Sparse magnitude may include rotation accumulated "
        "while the profile was in Dense.",
    )
    OUTWARD_ROTATION_P95_OVERALL = Entry(
        "OutwardRotationP95-Mask-Overall",
        "95th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from the inferred inoculum boundary through the "
        "full detected object extent. Reported in degrees, nonnegative with no "
        "fixed upper bound. With few supported rings it remains close to the raw "
        "maximum and is not the primary robust metric.",
    )
    OUTWARD_ROTATION_P95_DENSE = Entry(
        "OutwardRotationP95-Mask-Dense",
        "95th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from the inferred inoculum boundary to dense_end. "
        "Reported in degrees, nonnegative with no fixed upper bound. With few "
        "supported rings it remains close to the raw maximum.",
    )
    OUTWARD_ROTATION_P95_SPARSE = Entry(
        "OutwardRotationP95-Mask-Sparse",
        "95th percentile of absolute cumulative literal skeleton-ring rotation "
        "over supported rings from dense_end through the full detected object "
        "extent. Reported in degrees, nonnegative with no fixed upper bound. "
        "Cumulative rotation is not rebased at dense_end, so this Sparse "
        "magnitude may include rotation accumulated while the profile was in "
        "Dense.",
    )
    OUTWARD_ROTATION_MEDIAN_MAGNITUDE_OVERALL = Entry(
        "OutwardRotationMedianMagnitude-Mask-Overall",
        "Median absolute cumulative literal skeleton-ring rotation over supported "
        "rings from the inferred inoculum boundary through the full detected "
        "object extent. Reported in degrees, nonnegative with no fixed upper "
        "bound. This validation comparator describes typical magnitude but "
        "ignores radial order and localized outer turns.",
    )
    OUTWARD_ROTATION_MEDIAN_MAGNITUDE_DENSE = Entry(
        "OutwardRotationMedianMagnitude-Mask-Dense",
        "Median absolute cumulative literal skeleton-ring rotation over supported "
        "rings from the inferred inoculum boundary to dense_end. Reported in "
        "degrees, nonnegative with no fixed upper bound. This validation "
        "comparator describes typical magnitude but ignores radial order.",
    )
    OUTWARD_ROTATION_MEDIAN_MAGNITUDE_SPARSE = Entry(
        "OutwardRotationMedianMagnitude-Mask-Sparse",
        "Median absolute cumulative literal skeleton-ring rotation over supported "
        "rings from dense_end through the full detected object extent. Reported "
        "in degrees, nonnegative with no fixed upper bound. Cumulative rotation "
        "is not rebased at dense_end, so this Sparse magnitude may include "
        "rotation accumulated while the profile was in Dense.",
    )
    OUTWARD_ROTATION_ABSOLUTE_AREA_OVERALL = Entry(
        "OutwardRotationAbsoluteArea-Mask-Overall",
        "Trapezoidal area under absolute cumulative rotation within supported "
        "runs from the inferred inoculum boundary through the full detected "
        "object extent, divided by their total supported radial span. Reported "
        "in degrees, nonnegative with no fixed upper bound. It describes "
        "radially persistent rotation but can remain high after one early step.",
    )
    OUTWARD_ROTATION_ABSOLUTE_AREA_DENSE = Entry(
        "OutwardRotationAbsoluteArea-Mask-Dense",
        "Trapezoidal area under absolute cumulative rotation within supported "
        "runs from the inferred inoculum boundary to dense_end, divided by their "
        "total supported radial span. Reported in degrees, nonnegative with no "
        "fixed upper bound. It describes radially persistent rotation but can "
        "remain high after one early step.",
    )
    OUTWARD_ROTATION_ABSOLUTE_AREA_SPARSE = Entry(
        "OutwardRotationAbsoluteArea-Mask-Sparse",
        "Trapezoidal area under absolute cumulative rotation within supported "
        "runs from dense_end through the full detected object extent, divided by "
        "their total supported radial span. Reported in degrees, nonnegative with "
        "no fixed upper bound. Cumulative rotation is not rebased at dense_end, "
        "so this Sparse magnitude may include rotation accumulated while the "
        "profile was in Dense.",
    )
    OUTWARD_ROTATION_TOTAL_VARIATION_OVERALL = Entry(
        "OutwardRotationTotalVariation-Mask-Overall",
        "Sum of absolute adjacent cumulative-rotation changes within supported "
        "runs from the inferred inoculum boundary through the full detected "
        "object extent, without bridging gaps. Reported in degrees, nonnegative "
        "with no fixed upper bound. It is sensitive to oscillation, profile "
        "length, and noise and is diagnostic only.",
    )
    OUTWARD_ROTATION_TOTAL_VARIATION_DENSE = Entry(
        "OutwardRotationTotalVariation-Mask-Dense",
        "Sum of absolute adjacent cumulative-rotation changes within supported "
        "runs from the inferred inoculum boundary to dense_end, without bridging "
        "gaps. Reported in degrees, nonnegative with no fixed upper bound. It is "
        "sensitive to oscillation, profile length, and noise and is diagnostic "
        "only.",
    )
    OUTWARD_ROTATION_TOTAL_VARIATION_SPARSE = Entry(
        "OutwardRotationTotalVariation-Mask-Sparse",
        "Sum of absolute adjacent cumulative-rotation changes within supported "
        "runs from dense_end through the full detected object extent, without "
        "bridging gaps. Reported in degrees, nonnegative with no fixed upper "
        "bound. It is sensitive to oscillation, profile length, and noise and is "
        "diagnostic only.",
    )
    OUTWARD_ROTATION_RATE_GRADIENT_OVERALL = Entry(
        "OutwardRotationRateGradient-Mask-Overall",
        "Outer-half robust rotation rate minus inner-half robust rotation rate, "
        "divided by the separation of their median radii, along the longest "
        "supported run from the inferred inoculum boundary through the full "
        "detected object extent. Reported in signed degrees per pixel squared. "
        "This is a spatial rate gradient, not temporal acceleration, and is NaN "
        "unless the run meets both the configured minimum and eight rings.",
    )
    OUTWARD_ROTATION_RATE_GRADIENT_DENSE = Entry(
        "OutwardRotationRateGradient-Mask-Dense",
        "Outer-half robust rotation rate minus inner-half robust rotation rate, "
        "divided by the separation of their median radii, along the longest "
        "supported run from the inferred inoculum boundary to dense_end. Reported "
        "in signed degrees per pixel squared. This is a spatial rate gradient, "
        "not temporal acceleration, and is NaN unless the run meets both the "
        "configured minimum and eight rings.",
    )
    OUTWARD_ROTATION_RATE_GRADIENT_SPARSE = Entry(
        "OutwardRotationRateGradient-Mask-Sparse",
        "Outer-half robust rotation rate minus inner-half robust rotation rate, "
        "divided by the separation of their median radii, along the longest "
        "supported run from dense_end through the full detected object extent. "
        "Reported in signed degrees per pixel squared. This is a spatial rate "
        "gradient, not temporal acceleration, and is NaN for runs shorter than "
        "the larger of eight rings and the configured minimum.",
    )
    OUTWARD_ROTATION_RING_SUPPORT_OVERALL = Entry(
        "OutwardRotationRingSupport-Mask-Overall",
        "Fraction of candidate literal sampling rings from the inferred inoculum "
        "boundary through the full detected object extent with a finite "
        "cumulative-rotation state. Dimensionless in [0, 1]. This is a "
        "density-sensitive quality diagnostic, not an orientation phenotype.",
    )
    OUTWARD_ROTATION_RING_SUPPORT_DENSE = Entry(
        "OutwardRotationRingSupport-Mask-Dense",
        "Fraction of candidate literal sampling rings from the inferred inoculum "
        "boundary to dense_end with a finite cumulative-rotation state. "
        "Dimensionless in [0, 1]. This is a density-sensitive quality diagnostic, "
        "not an orientation phenotype.",
    )
    OUTWARD_ROTATION_RING_SUPPORT_SPARSE = Entry(
        "OutwardRotationRingSupport-Mask-Sparse",
        "Fraction of candidate literal sampling rings from dense_end through the "
        "full detected object extent with a finite cumulative-rotation state. "
        "Dimensionless in [0, 1]. This is a density-sensitive quality diagnostic, "
        "not an orientation phenotype.",
    )
    OUTWARD_ROTATION_RUN_SPAN_SUPPORT_OVERALL = Entry(
        "OutwardRotationRunSpanSupport-Mask-Overall",
        "Radial span of the longest contiguous supported literal-crossing run "
        "divided by the full candidate span from the inferred inoculum boundary "
        "through the full detected object extent. Dimensionless in [0, 1]. This "
        "reports continuity of usable evidence, not orientation.",
    )
    OUTWARD_ROTATION_RUN_SPAN_SUPPORT_DENSE = Entry(
        "OutwardRotationRunSpanSupport-Mask-Dense",
        "Radial span of the longest contiguous supported literal-crossing run "
        "divided by the full candidate span from the inferred inoculum boundary "
        "to dense_end. Dimensionless in [0, 1]. This reports continuity of usable "
        "evidence, not orientation.",
    )
    OUTWARD_ROTATION_RUN_SPAN_SUPPORT_SPARSE = Entry(
        "OutwardRotationRunSpanSupport-Mask-Sparse",
        "Radial span of the longest contiguous supported literal-crossing run "
        "divided by the full candidate span from dense_end through the full "
        "detected object extent. Dimensionless in [0, 1]. This reports continuity "
        "of usable evidence, not orientation.",
    )
    OUTWARD_ROTATION_MEDIAN_RESULTANT_OVERALL = Entry(
        "OutwardRotationMedianResultant-Mask-Overall",
        "Median doubled-angle resultant of crossing orientations among eligible "
        "literal rings from the inferred inoculum boundary through the full "
        "detected object extent. Dimensionless in [0, 1]; larger values mean "
        "stronger within-ring axial agreement. This is an orientation-reliability "
        "diagnostic, not a rotation phenotype.",
    )
    OUTWARD_ROTATION_MEDIAN_RESULTANT_DENSE = Entry(
        "OutwardRotationMedianResultant-Mask-Dense",
        "Median doubled-angle resultant of crossing orientations among eligible "
        "literal rings from the inferred inoculum boundary to dense_end. "
        "Dimensionless in [0, 1]; larger values mean stronger within-ring axial "
        "agreement. This is an orientation-reliability diagnostic, not a rotation "
        "phenotype.",
    )
    OUTWARD_ROTATION_MEDIAN_RESULTANT_SPARSE = Entry(
        "OutwardRotationMedianResultant-Mask-Sparse",
        "Median doubled-angle resultant of crossing orientations among eligible "
        "literal rings from dense_end through the full detected object extent. "
        "Dimensionless in [0, 1]; larger values mean stronger within-ring axial "
        "agreement. This is an orientation-reliability diagnostic, not a rotation "
        "phenotype.",
    )

    CONCENTRATION_RADIAL_OVERALL = Entry(
        "Concentration-Radial-Overall",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over all tile pixels in the full symmetric disk (0 .. symmetric_radius) "
        "region. Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = "
        "isotropic. NaN when the summed coherence over the selector is ~0 or the zone "
        "has zero width.",
    )
    CONCENTRATION_RADIAL_DENSE = Entry(
        "Concentration-Radial-Dense",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over all tile pixels in the dense ring (core_end .. dense_end radii) region. "
        "Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = isotropic. NaN "
        "when the summed coherence over the selector is ~0 or the zone has zero "
        "width.",
    )
    CONCENTRATION_RADIAL_SPARSE = Entry(
        "Concentration-Radial-Sparse",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over all tile pixels in the sparse ring (dense_end .. sparse_end radii) "
        "region. Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = "
        "isotropic. NaN when the summed coherence over the selector is ~0 or the zone "
        "has zero width.",
    )
    CONCENTRATION_MASK_OVERALL = Entry(
        "Concentration-Mask-Overall",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over detected-object pixels in the full symmetric disk (0 .. "
        "symmetric_radius) region. Dimensionless in [0, 1]; 1 = perfectly aligned "
        "hyphae, 0 = isotropic. NaN when the summed coherence over the selector is ~0 "
        "or the zone has zero width.",
    )
    CONCENTRATION_MASK_DENSE = Entry(
        "Concentration-Mask-Dense",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over detected-object pixels in the dense ring (core_end .. dense_end radii) "
        "region. Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = "
        "isotropic. NaN when the summed coherence over the selector is ~0 or the zone "
        "has zero width.",
    )
    CONCENTRATION_MASK_SPARSE = Entry(
        "Concentration-Mask-Sparse",
        "Coherence-weighted resultant length R of the doubled-angle orientation field "
        "over detected-object pixels in the sparse ring (dense_end .. sparse_end "
        "radii) region. Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = "
        "isotropic. NaN when the summed coherence over the selector is ~0 or the zone "
        "has zero width.",
    )
    TURNING_RADIAL_OVERALL = Entry(
        "Turning-Radial-Overall",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over all "
        "tile pixels in the full symmetric disk (0 .. symmetric_radius) region, in "
        "degrees per pixel. Higher values indicate curving/fanning hyphae; ~0 "
        "indicates straight parallel growth.",
    )
    TURNING_RADIAL_DENSE = Entry(
        "Turning-Radial-Dense",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over all "
        "tile pixels in the dense ring (core_end .. dense_end radii) region, in "
        "degrees per pixel. Higher values indicate curving/fanning hyphae; ~0 "
        "indicates straight parallel growth.",
    )
    TURNING_RADIAL_SPARSE = Entry(
        "Turning-Radial-Sparse",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over all "
        "tile pixels in the sparse ring (dense_end .. sparse_end radii) region, in "
        "degrees per pixel. Higher values indicate curving/fanning hyphae; ~0 "
        "indicates straight parallel growth.",
    )
    TURNING_MASK_OVERALL = Entry(
        "Turning-Mask-Overall",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over "
        "detected-object pixels in the full symmetric disk (0 .. symmetric_radius) "
        "region, in degrees per pixel. Higher values indicate curving/fanning hyphae; "
        "~0 indicates straight parallel growth.",
    )
    TURNING_MASK_DENSE = Entry(
        "Turning-Mask-Dense",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over "
        "detected-object pixels in the dense ring (core_end .. dense_end radii) "
        "region, in degrees per pixel. Higher values indicate curving/fanning hyphae; "
        "~0 indicates straight parallel growth.",
    )
    TURNING_MASK_SPARSE = Entry(
        "Turning-Mask-Sparse",
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> over "
        "detected-object pixels in the sparse ring (dense_end .. sparse_end radii) "
        "region, in degrees per pixel. Higher values indicate curving/fanning hyphae; "
        "~0 indicates straight parallel growth.",
    )
    COHERENCE_RADIAL_OVERALL = Entry(
        "Coherence-Radial-Overall",
        "Mean structure-tensor coherence C over all tile pixels in the full symmetric "
        "disk (0 .. symmetric_radius) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    COHERENCE_RADIAL_DENSE = Entry(
        "Coherence-Radial-Dense",
        "Mean structure-tensor coherence C over all tile pixels in the dense ring "
        "(core_end .. dense_end radii) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    COHERENCE_RADIAL_SPARSE = Entry(
        "Coherence-Radial-Sparse",
        "Mean structure-tensor coherence C over all tile pixels in the sparse ring "
        "(dense_end .. sparse_end radii) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    COHERENCE_MASK_OVERALL = Entry(
        "Coherence-Mask-Overall",
        "Mean structure-tensor coherence C over detected-object pixels in the full "
        "symmetric disk (0 .. symmetric_radius) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    COHERENCE_MASK_DENSE = Entry(
        "Coherence-Mask-Dense",
        "Mean structure-tensor coherence C over detected-object pixels in the dense "
        "ring (core_end .. dense_end radii) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    COHERENCE_MASK_SPARSE = Entry(
        "Coherence-Mask-Sparse",
        "Mean structure-tensor coherence C over detected-object pixels in the sparse "
        "ring (dense_end .. sparse_end radii) region. Dimensionless in [0, 1]; a "
        "confidence/QC readout for how well orientation is defined there (low where "
        "texture is isotropic).",
    )
    RADIAL_TILT_MASK_OVERALL = Entry(
        "RadialTilt-Mask-Overall",
        "Equal-angular-sector mean of the coherence-weighted absolute axial "
        "difference between the local fiber axis and the outward radial spoke at the "
        "same pixel, over detected-object pixels in the full symmetric disk (0 .. "
        "symmetric_radius) region. Reported in degrees in [0, 90]; 0 = locally radial "
        "and 90 = locally tangential. Each occupied 10-degree angular sector "
        "contributes equally. For a fixed set of reliable sectors, multiplying branch "
        "evidence without changing within-sector tilt distributions leaves the result "
        "unchanged. A support-threshold crossing can add a newly reliable sector and "
        "change the estimate; mixed orientations within one sector remain "
        "pixel-weighted.",
    )
    RADIAL_TILT_MASK_DENSE = Entry(
        "RadialTilt-Mask-Dense",
        "Equal-angular-sector mean of the coherence-weighted absolute axial "
        "difference between the local fiber axis and the outward radial spoke at the "
        "same pixel, over detected-object pixels in the dense ring (core_end .. "
        "dense_end radii) region. Reported in degrees in [0, 90]; 0 = locally radial "
        "and 90 = locally tangential. Each occupied 10-degree angular sector "
        "contributes equally. For a fixed set of reliable sectors, multiplying branch "
        "evidence without changing within-sector tilt distributions leaves the result "
        "unchanged. A support-threshold crossing can add a newly reliable sector and "
        "change the estimate; mixed orientations within one sector remain "
        "pixel-weighted.",
    )
    RADIAL_TILT_MASK_SPARSE = Entry(
        "RadialTilt-Mask-Sparse",
        "Equal-angular-sector mean of the coherence-weighted absolute axial "
        "difference between the local fiber axis and the outward radial spoke at the "
        "same pixel, over detected-object pixels in the sparse ring (dense_end .. "
        "sparse_end radii) region. Reported in degrees in [0, 90]; 0 = locally radial "
        "and 90 = locally tangential. Each occupied 10-degree angular sector "
        "contributes equally. For a fixed set of reliable sectors, multiplying branch "
        "evidence without changing within-sector tilt distributions leaves the result "
        "unchanged. A support-threshold crossing can add a newly reliable sector and "
        "change the estimate; mixed orientations within one sector remain "
        "pixel-weighted.",
    )
    OUTWARD_TURNING_MASK_OVERALL = Entry(
        "OutwardTurning-Mask-Overall",
        "Equal-angular-sector mean radial derivative magnitude of the radial-relative "
        "fiber tilt, over detected-object pixels in the full symmetric disk (0 .. "
        "symmetric_radius) region. Reported in degrees per pixel. 0 means the tilt "
        "stays constant while moving outward; larger values mean the local fiber "
        "field rotates relative to its radial spoke. The aggregation gives each "
        "occupied 10-degree angular sector equal weight. This is a field-level "
        "curvature measure, not parent-to-daughter branch tracking.",
    )
    OUTWARD_TURNING_MASK_DENSE = Entry(
        "OutwardTurning-Mask-Dense",
        "Equal-angular-sector mean radial derivative magnitude of the radial-relative "
        "fiber tilt, over detected-object pixels in the dense ring (core_end .. "
        "dense_end radii) region. Reported in degrees per pixel. 0 means the tilt "
        "stays constant while moving outward; larger values mean the local fiber "
        "field rotates relative to its radial spoke. The aggregation gives each "
        "occupied 10-degree angular sector equal weight. This is a field-level "
        "curvature measure, not parent-to-daughter branch tracking.",
    )
    OUTWARD_TURNING_MASK_SPARSE = Entry(
        "OutwardTurning-Mask-Sparse",
        "Equal-angular-sector mean radial derivative magnitude of the radial-relative "
        "fiber tilt, over detected-object pixels in the sparse ring (dense_end .. "
        "sparse_end radii) region. Reported in degrees per pixel. 0 means the tilt "
        "stays constant while moving outward; larger values mean the local fiber "
        "field rotates relative to its radial spoke. The aggregation gives each "
        "occupied 10-degree angular sector equal weight. This is a field-level "
        "curvature measure, not parent-to-daughter branch tracking.",
    )
    RADIAL_SECTOR_SUPPORT_MASK_OVERALL = Entry(
        "RadialSectorSupport-Mask-Overall",
        "Fraction of the 36 fixed 10-degree sectors in the full symmetric disk (0 .. "
        "symmetric_radius) region that contain at least three detected-structure "
        "pixels with structure-tensor coherence C >= 0.15. Dimensionless in [0, 1]. "
        "This is a density-sensitive quality diagnostic for interpreting radial tilt "
        "and outward turning, not an orientation phenotype.",
    )
    RADIAL_SECTOR_SUPPORT_MASK_DENSE = Entry(
        "RadialSectorSupport-Mask-Dense",
        "Fraction of the 36 fixed 10-degree sectors in the dense ring (core_end .. "
        "dense_end radii) region that contain at least three detected-structure "
        "pixels with structure-tensor coherence C >= 0.15. Dimensionless in [0, 1]. "
        "This is a density-sensitive quality diagnostic for interpreting radial tilt "
        "and outward turning, not an orientation phenotype.",
    )
    RADIAL_SECTOR_SUPPORT_MASK_SPARSE = Entry(
        "RadialSectorSupport-Mask-Sparse",
        "Fraction of the 36 fixed 10-degree sectors in the sparse ring (dense_end .. "
        "sparse_end radii) region that contain at least three detected-structure "
        "pixels with structure-tensor coherence C >= 0.15. Dimensionless in [0, 1]. "
        "This is a density-sensitive quality diagnostic for interpreting radial tilt "
        "and outward turning, not an orientation phenotype.",
    )
    LONG_RANGE_ROTATION_MASK_OVERALL = Entry(
        "LongRangeRotation-Mask-Overall",
        "Equal-cell mean absolute seam-safe axial change between matching 10-degree "
        "sectors in configured-width Sholl-style annular bands (8 pixels by default) "
        "whose centres are separated by the configured long-range lag (16 pixels by "
        "default). Ring pairs are assigned to the core-excluded symmetric region "
        "(core_end .. min(sparse_end, symmetric_radius)) region by their midpoint. "
        "Reported in degrees in [0, 90]. Annular bands begin outside the inferred "
        "inoculum core. Each reliable ring-sector comparison contributes equally, so "
        "multiplying same-orientation branch evidence within an already reliable cell "
        "does not change its contribution.",
    )
    LONG_RANGE_ROTATION_MASK_DENSE = Entry(
        "LongRangeRotation-Mask-Dense",
        "Equal-cell mean absolute seam-safe axial change between matching 10-degree "
        "sectors in configured-width Sholl-style annular bands (8 pixels by default) "
        "whose centres are separated by the configured long-range lag (16 pixels by "
        "default). Ring pairs are assigned to the dense ring (core_end .. dense_end "
        "radii) region by their midpoint. Reported in degrees in [0, 90]. Annular "
        "bands begin outside the inferred inoculum core. Each reliable ring-sector "
        "comparison contributes equally, so multiplying same-orientation branch "
        "evidence within an already reliable cell does not change its contribution.",
    )
    LONG_RANGE_ROTATION_MASK_SPARSE = Entry(
        "LongRangeRotation-Mask-Sparse",
        "Equal-cell mean absolute seam-safe axial change between matching 10-degree "
        "sectors in configured-width Sholl-style annular bands (8 pixels by default) "
        "whose centres are separated by the configured long-range lag (16 pixels by "
        "default). Ring pairs are assigned to the sparse ring (dense_end .. "
        "sparse_end radii) region by their midpoint. Reported in degrees in [0, 90]. "
        "Annular bands begin outside the inferred inoculum core. Each reliable "
        "ring-sector comparison contributes equally, so multiplying same-orientation "
        "branch evidence within an already reliable cell does not change its "
        "contribution.",
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_OVERALL = Entry(
        "SignedLongRangeRotation-Mask-Overall",
        "Signed counterpart of LongRangeRotation over the core-excluded symmetric "
        "region (core_end .. min(sparse_end, symmetric_radius)) region, in degrees in "
        "[-90, 90]. Positive means the radial-relative fiber axis rotates clockwise "
        "and negative means counterclockwise while moving outward in image "
        "coordinates. Opposing reliable ring-sector changes cancel in this "
        "directional summary; inspect the absolute metric and support alongside it.",
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_DENSE = Entry(
        "SignedLongRangeRotation-Mask-Dense",
        "Signed counterpart of LongRangeRotation over the dense ring (core_end .. "
        "dense_end radii) region, in degrees in [-90, 90]. Positive means the "
        "radial-relative fiber axis rotates clockwise and negative means "
        "counterclockwise while moving outward in image coordinates. Opposing "
        "reliable ring-sector changes cancel in this directional summary; inspect the "
        "absolute metric and support alongside it.",
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_SPARSE = Entry(
        "SignedLongRangeRotation-Mask-Sparse",
        "Signed counterpart of LongRangeRotation over the sparse ring (dense_end .. "
        "sparse_end radii) region, in degrees in [-90, 90]. Positive means the "
        "radial-relative fiber axis rotates clockwise and negative means "
        "counterclockwise while moving outward in image coordinates. Opposing "
        "reliable ring-sector changes cancel in this directional summary; inspect the "
        "absolute metric and support alongside it.",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_OVERALL = Entry(
        "LongRangeRotationSupport-Mask-Overall",
        "Fraction of fixed-lag ring-sector comparison cells assigned to the "
        "core-excluded symmetric region (core_end .. min(sparse_end, "
        "symmetric_radius)) region that have reliable orientation estimates at both "
        "radii. Dimensionless in [0, 1]. This is a density-sensitive quality "
        "diagnostic, not an orientation phenotype.",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE = Entry(
        "LongRangeRotationSupport-Mask-Dense",
        "Fraction of fixed-lag ring-sector comparison cells assigned to the dense "
        "ring (core_end .. dense_end radii) region that have reliable orientation "
        "estimates at both radii. Dimensionless in [0, 1]. This is a "
        "density-sensitive quality diagnostic, not an orientation phenotype.",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_SPARSE = Entry(
        "LongRangeRotationSupport-Mask-Sparse",
        "Fraction of fixed-lag ring-sector comparison cells assigned to the sparse "
        "ring (dense_end .. sparse_end radii) region that have reliable orientation "
        "estimates at both radii. Dimensionless in [0, 1]. This is a "
        "density-sensitive quality diagnostic, not an orientation phenotype.",
    )
    LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE = Entry(
        "LongRangeRotation-Mask-DenseToSparse",
        "Equal-sector mean absolute seam-safe axial difference between the broad "
        "Dense-zone and Sparse-zone radial-relative fiber means. Reported in degrees "
        "in [0, 90]. Only 10-degree sectors reliable in both zones contribute, and "
        "each paired sector receives equal weight. This measures accumulated "
        "zone-to-zone rotation without detecting individual branches.",
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE = Entry(
        "SignedLongRangeRotation-Mask-DenseToSparse",
        "Signed mean Dense-to-Sparse axial change over paired reliable 10-degree "
        "sectors, in degrees in [-90, 90]. Positive means clockwise and negative "
        "means counterclockwise radial-relative rotation while moving outward in "
        "image coordinates. Opposing sector rotations cancel.",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE_TO_SPARSE = Entry(
        "LongRangeRotationSupport-Mask-DenseToSparse",
        "Fraction of the 36 fixed 10-degree sectors with reliable radial-relative "
        "orientation estimates in both the Dense and Sparse zones. Dimensionless in "
        "[0, 1]. This is a density-sensitive quality diagnostic, not an orientation "
        "phenotype.",
    )


# Preserve member-level access for callers that imported the former legacy enum.
# The diagnostic class contains every former member plus the new opt-in fields.
# New code should use the explicit primary/diagnostic names above.
ORIENTATION_ZONES = ORIENTATION_ZONE_DIAGNOSTIC
