"""Filamentous fungi detection and Dijkstra-based branch reconnection.

Internal subpackage for :class:`~phenotypic.detect.FilamentousFungiDetector`.
Re-exports key functions so the main detector can import with a single line.
"""

from ._cost_surface import (
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
from ._dijkstra_kernels import (
    assemble_connected_mask,
    assign_fragments_to_colonies,
    backtrack_path,
    extract_fragment_paths,
    run_multisource_dijkstra,
)
from ._fragment_prescreening import (
    calibrate_screening_threshold,
    compute_min_cost_envelope,
    prescreen_fragments,
)
from ._path_quality import (
    apply_quality_filters,
    calibrate_quality_thresholds,
    compute_path_metrics,
    extract_calibration_branches,
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
    "apply_quality_filters",
    "calibrate_quality_thresholds",
    "compute_path_metrics",
    "extract_calibration_branches",
]
