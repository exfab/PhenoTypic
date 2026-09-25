"""Orientation-field transforms, ring profiles, and literal ring crossings.

The helpers in this package compute or operate on already-computed
orientation fields and object masks. They do not detect branches, infer
parent-child correspondence, or apply image enhancement. Operation-specific
aggregation into measurement columns stays private to the operation.
"""

from ._aggregates import (
    LiteralCrossingZoneMetrics,
    aggregate_literal_crossing_zone,
)
from ._axial import (
    axial_change,
    axial_difference,
    is_orthogonal_change,
    signed_axial_mean,
)
from ._bend import fiber_bend_field
from ._constants import (
    CROSSING_HALF_WIDTH,
    FIBER_AXIS_OFFSET,
    MIN_AXIAL_RESULTANT,
    MIN_PIXELS_PER_SECTOR,
    N_SECTORS,
    RELIABLE_PIXEL_COHERENCE,
)
from ._field import orientation_field
from ._literal_crossings import (
    LiteralCrossingRingProfile,
    LiteralSkeletonRingCrossing,
    LiteralSkeletonRingCrossingTransform,
    literal_crossing_ring_profile,
    literal_skeleton_ring_crossings,
)
from ._matched_rings import (
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)
from ._plots import (
    plot_literal_crossing_map,
    plot_literal_crossing_outward_profile,
    plot_literal_crossing_population,
)
from ._ring_profiles import (
    axial_sector_means,
    cumulative_ring_rotation_profile,
    long_range_ring_rotation_profile,
    radial_ring_orientation_profile,
    radial_ring_sector_field,
    signed_radial_relative_field,
)

__all__ = [
    "CROSSING_HALF_WIDTH",
    "FIBER_AXIS_OFFSET",
    "MIN_AXIAL_RESULTANT",
    "MIN_PIXELS_PER_SECTOR",
    "N_SECTORS",
    "RELIABLE_PIXEL_COHERENCE",
    "LiteralCrossingZoneMetrics",
    "LiteralCrossingRingProfile",
    "LiteralSkeletonRingCrossing",
    "LiteralSkeletonRingCrossingTransform",
    "aggregate_literal_crossing_zone",
    "axial_change",
    "axial_difference",
    "axial_sector_means",
    "cumulative_ring_rotation_profile",
    "fiber_bend_field",
    "is_orthogonal_change",
    "literal_crossing_ring_profile",
    "literal_skeleton_ring_crossings",
    "long_range_ring_rotation_profile",
    "matched_ring_cumulative_rotation_profile",
    "matched_tracks_to_ring_sector_values",
    "orientation_field",
    "plot_literal_crossing_map",
    "plot_literal_crossing_outward_profile",
    "plot_literal_crossing_population",
    "radial_ring_orientation_profile",
    "radial_ring_sector_field",
    "signed_axial_mean",
    "signed_radial_relative_field",
]
