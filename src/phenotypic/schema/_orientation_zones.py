"""Public header schema for MeasureOrientationZones (category ``OrientZones``).

Header pattern: ``OrientZones_<Metric>-<Variant>-<Zone>`` — single underscore
after the category, then Metric/Variant/Zone hyphen-joined. Metric in
{Concentration, Turning, Coherence}; Variant in {Radial, Mask}; Zone in
{Overall, Dense, Sparse}.
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
        "over the {variant} selector of the {zone} region, in radians per pixel "
        "(radians per micron when a pixel scale is set). Higher values indicate "
        "curving/fanning hyphae; ~0 indicates straight parallel growth."
    ),
    "Coherence": (
        "Mean structure-tensor coherence C over the {variant} selector of the "
        "{zone} region. Dimensionless in [0, 1]; a confidence/QC readout for how "
        "well orientation is defined there (low where texture is isotropic)."
    ),
}
_ZONE_MEANING = {
    "Overall": "the full symmetric disk (0 .. symmetric_radius)",
    "Dense": "the dense ring (core_end .. dense_end radii)",
    "Sparse": "the sparse ring (dense_end .. sparse_end radii)",
}
_VARIANT_MEANING = {
    "Radial": "all tile pixels in the radial region (mask-free)",
    "Mask": "the radial region intersected with the detected object mask",
}


def _desc(metric: str, variant: str, zone: str) -> str:
    return (
        _METRIC_DESC[metric].format(variant=_VARIANT_MEANING[variant], zone=_ZONE_MEANING[zone])
    )


class ORIENTATION_ZONES(DescriptiveTrait):
    """Per-zone hyphal orientation traits (concentration, turning, coherence).

    Computed from the structure-tensor orientation field over a mask-free tile,
    aggregated coherence-weighted over radially-defined zones bounded by the
    symmetric radius, in both a ``Radial`` (all tile pixels) and a raw ``Mask``
    variant. See :class:`MeasureOrientationZones` for parameters and method.
    """

    @classmethod
    def category(cls) -> str:
        return "OrientZones"

    CONCENTRATION_RADIAL_OVERALL = Entry("Concentration-Radial-Overall", _desc("Concentration", "Radial", "Overall"))
    CONCENTRATION_RADIAL_DENSE = Entry("Concentration-Radial-Dense", _desc("Concentration", "Radial", "Dense"))
    CONCENTRATION_RADIAL_SPARSE = Entry("Concentration-Radial-Sparse", _desc("Concentration", "Radial", "Sparse"))
    CONCENTRATION_MASK_OVERALL = Entry("Concentration-Mask-Overall", _desc("Concentration", "Mask", "Overall"))
    CONCENTRATION_MASK_DENSE = Entry("Concentration-Mask-Dense", _desc("Concentration", "Mask", "Dense"))
    CONCENTRATION_MASK_SPARSE = Entry("Concentration-Mask-Sparse", _desc("Concentration", "Mask", "Sparse"))
    TURNING_RADIAL_OVERALL = Entry("Turning-Radial-Overall", _desc("Turning", "Radial", "Overall"))
    TURNING_RADIAL_DENSE = Entry("Turning-Radial-Dense", _desc("Turning", "Radial", "Dense"))
    TURNING_RADIAL_SPARSE = Entry("Turning-Radial-Sparse", _desc("Turning", "Radial", "Sparse"))
    TURNING_MASK_OVERALL = Entry("Turning-Mask-Overall", _desc("Turning", "Mask", "Overall"))
    TURNING_MASK_DENSE = Entry("Turning-Mask-Dense", _desc("Turning", "Mask", "Dense"))
    TURNING_MASK_SPARSE = Entry("Turning-Mask-Sparse", _desc("Turning", "Mask", "Sparse"))
    COHERENCE_RADIAL_OVERALL = Entry("Coherence-Radial-Overall", _desc("Coherence", "Radial", "Overall"))
    COHERENCE_RADIAL_DENSE = Entry("Coherence-Radial-Dense", _desc("Coherence", "Radial", "Dense"))
    COHERENCE_RADIAL_SPARSE = Entry("Coherence-Radial-Sparse", _desc("Coherence", "Radial", "Sparse"))
    COHERENCE_MASK_OVERALL = Entry("Coherence-Mask-Overall", _desc("Coherence", "Mask", "Overall"))
    COHERENCE_MASK_DENSE = Entry("Coherence-Mask-Dense", _desc("Coherence", "Mask", "Dense"))
    COHERENCE_MASK_SPARSE = Entry("Coherence-Mask-Sparse", _desc("Coherence", "Mask", "Sparse"))
