"""Radial expansion, symmetry, and orientation-zone measurements."""

from ._measurement_info import Entry
from ._tiers import DescriptiveTrait


class SYMMETRIC_ZONES(DescriptiveTrait):
    """Radial expansion, symmetry, and branch-orientation zone geometry.

    The first four measurements are independent summaries derived directly
    from the binary object mask: PELT core radius, angular symmetry radius,
    and mean and maximum expansion. Canonical CoreZone, DenseZone, and
    SparseZone geometry is resolved by Method B from the target mask,
    detection matrix, structure-tensor field, and literal skeleton crossings.
    Historical colony-ness geometry remains available through
    ``legacy_mode=True``.
    """

    @classmethod
    def category(cls) -> str:
        return "SymZones"

    CORE_RADIUS = Entry(
        "CoreRadius",
        "Radius of the dense inoculum core, determined by PELT changepoint "
        "detection on the radial mask-density profile centered on the "
        "inoculum. Growth measurements are reported relative to this boundary.",
    )
    SYMMETRIC_RADIUS = Entry(
        "SymmetricRadius",
        "Radial distance from the inoculum centroid at which colony growth "
        "ceases to be angularly uniform. Computed as the first radius past "
        "the core where the smoothed per-annulus circular mean resultant "
        "length of mask-boundary pixels exceeds the symmetry threshold. "
        "Equals the colony outer envelope when growth remains symmetric "
        "throughout.",
    )
    MEAN_EXPANSION = Entry(
        "MeanExpansion",
        "Mean distance of mask-boundary pixels from the inoculum centroid, "
        "measured from the core boundary outward. Captures the typical "
        "radial extent of growth past the inoculum, averaged over all "
        "angular directions.",
    )
    MAX_EXPANSION = Entry(
        "MaxExpansion",
        "Maximum distance of any mask pixel from the inoculum centroid, "
        "measured from the core boundary outward. Captures the farthest "
        "extent of growth past the inoculum.",
    )
    CORE_END_RADIUS = Entry(
        "CoreEndRadius",
        "Outer radius in pixels of CoreZone. Canonical Method B defines CoreZone "
        "as the inoculum plus any inner region without resolvable branch "
        "orientation. With legacy_mode=True, the historical colony-ness "
        "threshold defines this boundary. CoreRadius remains the independent "
        "mask-density PELT estimate.",
    )
    DENSE_END_RADIUS = Entry(
        "DenseEndRadius",
        "Outer radius in pixels of DenseZone. Canonical Method B uses its second "
        "change point; a collapsed one-change solution makes this equal to "
        "CoreEndRadius. Legacy mode uses the historical colony-ness threshold.",
    )
    SPARSE_END_RADIUS = Entry(
        "SparseEndRadius",
        "Outer radius in pixels of SparseZone. Canonical Method B uses the exact "
        "target-mask radius selected by outer_zone_percentile; legacy mode uses "
        "the historical colony-ness and symmetric-envelope boundary.",
    )
    CORE_AREA = Entry(
        "CoreArea",
        "Pixel-squared area of CoreZone using the concentric CoreEndRadius circle.",
    )
    DENSE_AREA = Entry(
        "DenseArea",
        "Pixel-squared area of DenseZone between CoreEndRadius and "
        "DenseEndRadius. It is zero for a collapsed canonical solution.",
    )
    SPARSE_AREA = Entry(
        "SparseArea",
        "Pixel-squared area of SparseZone between DenseEndRadius and "
        "SparseEndRadius.",
    )
