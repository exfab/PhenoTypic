"""Mask-based radial expansion measurements for colonies on solid media."""

from ._measurement_info import MeasurementInfo


class SYMMETRIC_ZONES(MeasurementInfo):
    """Mask-based radial expansion measurements for colonies on solid media.

    Summarises each colony's radial growth through four scalars derived
    directly from the binary object mask: the inoculum core radius (shared
    with :class:`RADIAL_EXPANSION`), the radius at which growth stops being
    angularly symmetric, and mean / maximum radial extent beyond the core.

    Unlike :class:`RADIAL_EXPANSION`, no skeletonization, branch tracing, or
    runner-outlier statistics are involved — all values are computed from
    per-annulus mask geometry and a circular-statistics angular resultant
    length. Intended for users who care about colony-level expansion and the
    symmetry of that expansion, not individual hyphae.
    """

    @classmethod
    def category(cls) -> str:
        return "SymZones"

    CORE_RADIUS = (
        "CoreRadius",
        "Radius of the dense inoculum core, determined by PELT changepoint "
        "detection on the radial mask-density profile centered on the "
        "inoculum. Growth measurements are reported relative to this boundary.",
    )
    SYMMETRIC_RADIUS = (
        "SymmetricRadius",
        "Radial distance from the inoculum centroid at which colony growth "
        "ceases to be angularly uniform. Computed as the first radius past "
        "the core where the smoothed per-annulus circular mean resultant "
        "length of mask-boundary pixels exceeds the symmetry threshold. "
        "Equals the colony outer envelope when growth remains symmetric "
        "throughout.",
    )
    MEAN_EXPANSION = (
        "MeanExpansion",
        "Mean distance of mask-boundary pixels from the inoculum centroid, "
        "measured from the core boundary outward. Captures the typical "
        "radial extent of growth past the inoculum, averaged over all "
        "angular directions.",
    )
    MAX_EXPANSION = (
        "MaxExpansion",
        "Maximum distance of any mask pixel from the inoculum centroid, "
        "measured from the core boundary outward. Captures the farthest "
        "extent of growth past the inoculum.",
    )
    CORE_END_RADIUS = (
        "CoreEndRadius",
        "Mean radius of the inoculum core boundary derived from the per-angle "
        "bright/background ratio walk. Each of 360 1° angular sectors finds the "
        "outer edge of the contiguous core run (bright fraction >= tau_core); the "
        "reported value is the mean across sectors. Compare with CoreRadius (the "
        "global PELT changepoint) — close agreement indicates a well-formed core.",
    )
    DENSE_END_RADIUS = (
        "DenseEndRadius",
        "Mean outer radius of the dense branching zone, where mask-bright pixels "
        "dominate (bright fraction >= tau_sparse). Per-angle radii are capped at "
        "the SymmetricRadius and angularly median-smoothed before averaging.",
    )
    SPARSE_END_RADIUS = (
        "SparseEndRadius",
        "Mean outer radius of the sparse branching zone (= colony envelope inside "
        "the symmetric growth front). Equals min(objmask outer envelope, "
        "SymmetricRadius) per angle, averaged across 360 sectors.",
    )
    CORE_AREA = (
        "CoreArea",
        "Pixel^2 area of the inoculum core zone, integrated across the 360-sector "
        "polar polygon defined by the per-angle core radii.",
    )
    DENSE_AREA = (
        "DenseArea",
        "Pixel^2 area of the dense branching zone, the annular region between the "
        "per-angle core boundary and dense-branching boundary.",
    )
    SPARSE_AREA = (
        "SparseArea",
        "Pixel^2 area of the sparse branching zone, the annular region between the "
        "per-angle dense boundary and the symmetric-envelope outer boundary.",
    )
