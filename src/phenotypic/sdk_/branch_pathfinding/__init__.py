"""Multi-source Dijkstra branch pathfinding over image cost surfaces.

General-purpose machinery extracted from
:class:`~phenotypic.detect.FilamentousFungiDetector`. Composes a composite
cost surface from orientation/texture features, prescreens candidate
regions, runs seeded multi-source Dijkstra, extracts minimum-cost paths
back to the source regions, and (optionally) filters paths by structure
quality.

Cost surfaces are the caller's responsibility — this subpackage is
algorithm-agnostic and does not know about phase congruency, skeletons,
or any other domain concept. Callers (e.g. the fungi detector or
:class:`~phenotypic.measure.MeasureRadialExpansion`) assemble their own
cost surface from whatever image features they have on hand.
"""

from ._cost_surface import (
    _apply_border_penalty_inplace,  # noqa: F401 — re-exported for back-compat
    _apply_distance_gap_penalty_inplace,  # noqa: F401 — re-exported for back-compat
    _apply_structure_mask_inplace,  # noqa: F401 — re-exported for back-compat
    apply_border_penalty,
    apply_distance_gap_penalty,
    apply_structure_mask,
    assemble_composite_cost,
    compute_anisotropy,
    compute_local_mad_map,
    compute_orientation_coherence,
)
from ._dataclasses import (
    CalibrationData,
    DijkstraResult,
    FilterResult,
    FilterThresholds,
    FragmentAssignment,
    FragmentPath,
    PathMetrics,
    PrescreenResult,
)
from ._diagnostics import (
    paths_metrics_dataframe,
    plot_cost_distance_heatmap,
    plot_paths_over_image,
)
from ._dijkstra_kernels import (
    assemble_connected_mask,
    assign_fragments_to_colonies,
    backtrack_path,
    extract_fragment_paths,
    run_multisource_dijkstra,
)
from ._fragment_prescreening import (
    _compute_screening_envelope,  # noqa: F401 — re-exported for back-compat
    calibrate_screening_threshold,
    compute_min_cost_envelope,
    prescreen_fragments,
)
from ._path_quality import (
    apply_filter_cascade,
    calibrate_thresholds,
    compute_path_metrics,
    extract_calibration_branches,
    filter_paths,
)
from ._voronoi_partition import (
    connectivity_correct_labels,
    euclidean_voronoi_assign,
)

__all__ = [
    # Dataclasses
    "CalibrationData",
    "DijkstraResult",
    "FilterResult",
    "FilterThresholds",
    "FragmentAssignment",
    "FragmentPath",
    "PathMetrics",
    "PrescreenResult",
    # Cost surface
    "apply_border_penalty",
    "apply_distance_gap_penalty",
    "apply_structure_mask",
    "assemble_composite_cost",
    "compute_anisotropy",
    "compute_local_mad_map",
    "compute_orientation_coherence",
    # Dijkstra
    "assemble_connected_mask",
    "assign_fragments_to_colonies",
    "backtrack_path",
    "extract_fragment_paths",
    "run_multisource_dijkstra",
    # Prescreening
    "calibrate_screening_threshold",
    "compute_min_cost_envelope",
    "prescreen_fragments",
    # Path quality
    "apply_filter_cascade",
    "calibrate_thresholds",
    "compute_path_metrics",
    "extract_calibration_branches",
    "filter_paths",
    # Voronoi partition
    "connectivity_correct_labels",
    "euclidean_voronoi_assign",
    # Diagnostics
    "paths_metrics_dataframe",
    "plot_cost_distance_heatmap",
    "plot_paths_over_image",
]
