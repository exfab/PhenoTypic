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
from ._tiers import DescriptiveTrait

_METRIC_DESC = {
    "Concentration": (
        "Coherence-weighted resultant length R of the doubled-angle "
        "orientation field over the {variant} selector of the {zone} region. "
        "Dimensionless in [0, 1]; 1 = perfectly aligned hyphae, 0 = isotropic. "
        "NaN when the summed coherence over the selector is ~0 or the zone has "
        "zero width."
    ),
    "Turning": (
        "Coherence-weighted mean orientation-gradient magnitude <|grad phi|> "
        "over the {variant} selector of the {zone} region, in degrees per "
        "pixel. Higher values indicate "
        "curving/fanning hyphae; ~0 indicates straight parallel growth."
    ),
    "Coherence": (
        "Mean structure-tensor coherence C over the {variant} selector of the "
        "{zone} region. Dimensionless in [0, 1]; a confidence/QC readout for how "
        "well orientation is defined there (low where texture is isotropic)."
    ),
    "RadialTilt": (
        "Equal-angular-sector mean of the coherence-weighted absolute axial "
        "difference between the local fiber axis and the outward radial spoke "
        "at the same pixel, over the {variant} selector of the {zone} region. "
        "Reported in degrees in [0, 90]; 0 = locally radial and 90 = locally "
        "tangential. Each occupied 10-degree angular sector contributes "
        "equally. For a fixed set of reliable sectors, multiplying branch "
        "evidence without changing within-sector tilt distributions leaves the "
        "result unchanged. A support-threshold crossing can add a newly reliable "
        "sector and change the estimate; mixed orientations within one sector "
        "remain pixel-weighted."
    ),
    "OutwardTurning": (
        "Equal-angular-sector mean radial derivative magnitude of the "
        "radial-relative fiber tilt, over the {variant} selector of the {zone} "
        "region. Reported in degrees per pixel. 0 means the tilt stays constant "
        "while moving outward; larger values mean the local fiber field rotates "
        "relative to its radial spoke. The aggregation gives each occupied "
        "10-degree angular sector equal "
        "weight. This is a field-level curvature measure, not parent-to-daughter "
        "branch tracking."
    ),
    "RadialSectorSupport": (
        "Fraction of the 36 fixed 10-degree sectors in the {zone} region that "
        "contain at least three detected-structure pixels with structure-tensor "
        "coherence C >= 0.15. Dimensionless in [0, 1]. This is a "
        "density-sensitive quality diagnostic for interpreting radial tilt and "
        "outward turning, not an orientation phenotype."
    ),
    "LongRangeRotation": (
        "Equal-cell mean absolute seam-safe axial change between matching "
        "10-degree sectors in configured-width Sholl-style annular bands "
        "(8 pixels by default) whose centres are separated by the configured "
        "long-range lag (16 pixels by default). Ring pairs are assigned to the "
        "{zone} region by their "
        "midpoint. Reported in degrees in [0, 90]. Annular bands begin outside "
        "the inferred inoculum core. Each reliable ring-sector comparison "
        "contributes equally, so multiplying same-orientation branch evidence "
        "within an already reliable cell does not change its contribution."
    ),
    "SignedLongRangeRotation": (
        "Signed counterpart of LongRangeRotation over the {zone} region, in "
        "degrees in [-90, 90]. Positive means the radial-relative fiber axis "
        "rotates clockwise and negative means counterclockwise while moving "
        "outward in image coordinates. Opposing reliable ring-sector changes "
        "cancel in this directional summary; inspect the absolute metric and "
        "support alongside it."
    ),
    "LongRangeRotationSupport": (
        "Fraction of fixed-lag ring-sector comparison cells assigned to the "
        "{zone} region that have reliable orientation estimates at both radii. "
        "Dimensionless in [0, 1]. This is a density-sensitive quality "
        "diagnostic, not an orientation phenotype."
    ),
}
_ZONE_MEANING = {
    "Overall": "the full symmetric disk (0 .. symmetric_radius)",
    "Dense": "the dense ring (core_end .. dense_end radii)",
    "Sparse": "the sparse ring (dense_end .. sparse_end radii)",
    "DenseToSparse": (
        "paired reliable 10-degree sectors between the broad Dense and Sparse "
        "zones"
    ),
}
_VARIANT_MEANING = {
    "Radial": "all tile pixels in the radial region (mask-free)",
    "Mask": "the radial region intersected with the detected object mask",
}
_LONG_RANGE_OVERALL_MEANING = (
    "the core-excluded symmetric region (core_end .. min(sparse_end, "
    "symmetric_radius))"
)


def _desc(metric: str, variant: str, zone: str) -> str:
    zone_meaning = _ZONE_MEANING[zone]
    if metric in {
        "LongRangeRotation",
        "SignedLongRangeRotation",
        "LongRangeRotationSupport",
    } and zone == "Overall":
        zone_meaning = _LONG_RANGE_OVERALL_MEANING
    return _METRIC_DESC[metric].format(
        variant=_VARIANT_MEANING[variant], zone=zone_meaning
    )


class ORIENTATION_ZONES(DescriptiveTrait):
    """Per-zone absolute and radial-relative hyphal orientation traits.

    Computed from the structure-tensor orientation field over a mask-free tile,
    aggregated coherence-weighted over radially-defined zones bounded by the
    symmetric radius, in both a ``Radial`` (all tile pixels) and a raw ``Mask``
    variant. See :class:`MeasureOrientationZones` for parameters and method.
    """

    @classmethod
    def category(cls) -> str:
        return "OrientZones"

    CONCENTRATION_RADIAL_OVERALL = Entry(
        "Concentration-Radial-Overall",
        _desc("Concentration", "Radial", "Overall"),
    )
    CONCENTRATION_RADIAL_DENSE = Entry(
        "Concentration-Radial-Dense", _desc("Concentration", "Radial", "Dense")
    )
    CONCENTRATION_RADIAL_SPARSE = Entry(
        "Concentration-Radial-Sparse",
        _desc("Concentration", "Radial", "Sparse"),
    )
    CONCENTRATION_MASK_OVERALL = Entry(
        "Concentration-Mask-Overall", _desc("Concentration", "Mask", "Overall")
    )
    CONCENTRATION_MASK_DENSE = Entry(
        "Concentration-Mask-Dense", _desc("Concentration", "Mask", "Dense")
    )
    CONCENTRATION_MASK_SPARSE = Entry(
        "Concentration-Mask-Sparse", _desc("Concentration", "Mask", "Sparse")
    )
    TURNING_RADIAL_OVERALL = Entry(
        "Turning-Radial-Overall", _desc("Turning", "Radial", "Overall")
    )
    TURNING_RADIAL_DENSE = Entry(
        "Turning-Radial-Dense", _desc("Turning", "Radial", "Dense")
    )
    TURNING_RADIAL_SPARSE = Entry(
        "Turning-Radial-Sparse", _desc("Turning", "Radial", "Sparse")
    )
    TURNING_MASK_OVERALL = Entry(
        "Turning-Mask-Overall", _desc("Turning", "Mask", "Overall")
    )
    TURNING_MASK_DENSE = Entry(
        "Turning-Mask-Dense", _desc("Turning", "Mask", "Dense")
    )
    TURNING_MASK_SPARSE = Entry(
        "Turning-Mask-Sparse", _desc("Turning", "Mask", "Sparse")
    )
    COHERENCE_RADIAL_OVERALL = Entry(
        "Coherence-Radial-Overall", _desc("Coherence", "Radial", "Overall")
    )
    COHERENCE_RADIAL_DENSE = Entry(
        "Coherence-Radial-Dense", _desc("Coherence", "Radial", "Dense")
    )
    COHERENCE_RADIAL_SPARSE = Entry(
        "Coherence-Radial-Sparse", _desc("Coherence", "Radial", "Sparse")
    )
    COHERENCE_MASK_OVERALL = Entry(
        "Coherence-Mask-Overall", _desc("Coherence", "Mask", "Overall")
    )
    COHERENCE_MASK_DENSE = Entry(
        "Coherence-Mask-Dense", _desc("Coherence", "Mask", "Dense")
    )
    COHERENCE_MASK_SPARSE = Entry(
        "Coherence-Mask-Sparse", _desc("Coherence", "Mask", "Sparse")
    )
    RADIAL_TILT_MASK_OVERALL = Entry(
        "RadialTilt-Mask-Overall", _desc("RadialTilt", "Mask", "Overall")
    )
    RADIAL_TILT_MASK_DENSE = Entry(
        "RadialTilt-Mask-Dense", _desc("RadialTilt", "Mask", "Dense")
    )
    RADIAL_TILT_MASK_SPARSE = Entry(
        "RadialTilt-Mask-Sparse", _desc("RadialTilt", "Mask", "Sparse")
    )
    OUTWARD_TURNING_MASK_OVERALL = Entry(
        "OutwardTurning-Mask-Overall",
        _desc("OutwardTurning", "Mask", "Overall"),
    )
    OUTWARD_TURNING_MASK_DENSE = Entry(
        "OutwardTurning-Mask-Dense", _desc("OutwardTurning", "Mask", "Dense")
    )
    OUTWARD_TURNING_MASK_SPARSE = Entry(
        "OutwardTurning-Mask-Sparse", _desc("OutwardTurning", "Mask", "Sparse")
    )
    RADIAL_SECTOR_SUPPORT_MASK_OVERALL = Entry(
        "RadialSectorSupport-Mask-Overall",
        _desc("RadialSectorSupport", "Mask", "Overall"),
        derivation_type="diagnostic",
    )
    RADIAL_SECTOR_SUPPORT_MASK_DENSE = Entry(
        "RadialSectorSupport-Mask-Dense",
        _desc("RadialSectorSupport", "Mask", "Dense"),
        derivation_type="diagnostic",
    )
    RADIAL_SECTOR_SUPPORT_MASK_SPARSE = Entry(
        "RadialSectorSupport-Mask-Sparse",
        _desc("RadialSectorSupport", "Mask", "Sparse"),
        derivation_type="diagnostic",
    )
    LONG_RANGE_ROTATION_MASK_OVERALL = Entry(
        "LongRangeRotation-Mask-Overall",
        _desc("LongRangeRotation", "Mask", "Overall"),
    )
    LONG_RANGE_ROTATION_MASK_DENSE = Entry(
        "LongRangeRotation-Mask-Dense",
        _desc("LongRangeRotation", "Mask", "Dense"),
    )
    LONG_RANGE_ROTATION_MASK_SPARSE = Entry(
        "LongRangeRotation-Mask-Sparse",
        _desc("LongRangeRotation", "Mask", "Sparse"),
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_OVERALL = Entry(
        "SignedLongRangeRotation-Mask-Overall",
        _desc("SignedLongRangeRotation", "Mask", "Overall"),
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_DENSE = Entry(
        "SignedLongRangeRotation-Mask-Dense",
        _desc("SignedLongRangeRotation", "Mask", "Dense"),
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_SPARSE = Entry(
        "SignedLongRangeRotation-Mask-Sparse",
        _desc("SignedLongRangeRotation", "Mask", "Sparse"),
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_OVERALL = Entry(
        "LongRangeRotationSupport-Mask-Overall",
        _desc("LongRangeRotationSupport", "Mask", "Overall"),
        derivation_type="diagnostic",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE = Entry(
        "LongRangeRotationSupport-Mask-Dense",
        _desc("LongRangeRotationSupport", "Mask", "Dense"),
        derivation_type="diagnostic",
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_SPARSE = Entry(
        "LongRangeRotationSupport-Mask-Sparse",
        _desc("LongRangeRotationSupport", "Mask", "Sparse"),
        derivation_type="diagnostic",
    )
    LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE = Entry(
        "LongRangeRotation-Mask-DenseToSparse",
        (
            "Equal-sector mean absolute seam-safe axial difference between "
            "the broad Dense-zone and Sparse-zone radial-relative fiber means. "
            "Reported in degrees in [0, 90]. Only 10-degree sectors reliable "
            "in both zones contribute, and each paired sector receives equal "
            "weight. This measures accumulated zone-to-zone rotation without "
            "detecting individual branches."
        ),
    )
    SIGNED_LONG_RANGE_ROTATION_MASK_DENSE_TO_SPARSE = Entry(
        "SignedLongRangeRotation-Mask-DenseToSparse",
        (
            "Signed mean Dense-to-Sparse axial change over paired reliable "
            "10-degree sectors, in degrees in [-90, 90]. Positive means "
            "clockwise and negative means counterclockwise radial-relative "
            "rotation while moving outward in image coordinates. Opposing "
            "sector rotations cancel."
        ),
    )
    LONG_RANGE_ROTATION_SUPPORT_MASK_DENSE_TO_SPARSE = Entry(
        "LongRangeRotationSupport-Mask-DenseToSparse",
        (
            "Fraction of the 36 fixed 10-degree sectors with reliable "
            "radial-relative orientation estimates in both the Dense and "
            "Sparse zones. Dimensionless in [0, 1]. This is a "
            "density-sensitive quality diagnostic, not an orientation "
            "phenotype."
        ),
        derivation_type="diagnostic",
    )
