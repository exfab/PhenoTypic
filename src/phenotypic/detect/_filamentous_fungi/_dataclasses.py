"""Data structures for the filamentous fungi reconnection pipeline.

Dataclasses used across Dijkstra propagation, fragment assignment,
path extraction, pre-screening, and metric-based filtering stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DijkstraResult:
    """Output of multi-source Dijkstra propagation.

    Attributes:
        cost_distance: Float64 array (H, W). Accumulated travel cost from
            the nearest colony boundary pixel. 0 inside colonies, inf for
            unreached pixels.
        colony_id: Int32 array (H, W). Colony label that owns each pixel.
            -1 for unreached pixels.
        predecessor: Int32 array (H, W). Flattened index of the preceding
            pixel on the shortest path back to the colony. -1 for colony
            pixels and unreached pixels.
        colony_centroids: Dict mapping colony_id to (row, col) centroid.
    """

    cost_distance: np.ndarray
    colony_id: np.ndarray
    predecessor: np.ndarray
    colony_centroids: dict[int, tuple[float, float]]


@dataclass
class FragmentAssignment:
    """Colony assignment for a single fragment.

    Attributes:
        fragment_id: Label of this fragment.
        colony_id: Assigned colony label (majority vote).
        is_bridge: True if minority colony fraction exceeds bridge_threshold,
            indicating the fragment spans a colony boundary.
        majority_fraction: Fraction of fragment pixels assigned to the
            majority colony. 1.0 means unambiguous assignment.
        mean_cost: Mean cost-distance across the fragment pixels.
    """

    fragment_id: int
    colony_id: int
    is_bridge: bool
    majority_fraction: float
    mean_cost: float


@dataclass
class FragmentPath:
    """Path from a fragment back to its assigned colony.

    Attributes:
        fragment_id: Label of the source fragment.
        colony_id: Colony reached by this path.
        coords: (N, 2) int32 array of (row, col) path coordinates, ordered
            from the fragment seed pixel to the colony boundary.
        cost_profile: (N,) float64 array of per-pixel cost-distance values
            along the path.
        total_cost: Accumulated cost from fragment seed to colony.
        path_length: Number of pixels in the path.
    """

    fragment_id: int
    colony_id: int
    coords: np.ndarray
    cost_profile: np.ndarray
    total_cost: float
    path_length: int


@dataclass
class PrescreenResult:
    """Output of the fragment pre-screening stage.

    Attributes:
        screened_fragment_labels: Int32 label array (H, W) with rejected
            fragments zeroed out.
        passed_ids: Set of fragment IDs that passed pre-screening.
        rejected_ids: Set of fragment IDs rejected by pre-screening.
        threshold_used: The size threshold applied during screening.
    """

    screened_fragment_labels: np.ndarray
    passed_ids: set[int]
    rejected_ids: set[int]
    threshold_used: float


@dataclass
class PathMetrics:
    """Per-path quality metrics for filtering.

    Attributes:
        cost_per_length: Mean cost per pixel along the path. Lower values
            indicate the path follows low-cost (high-evidence) terrain.
        efficiency: Ratio of Euclidean distance to path length.
            1.0 is a perfectly straight path; lower values indicate
            meandering.
        min_windowed_displacement: Minimum net displacement over a
            sliding window along the path. Near-zero values indicate
            the path doubles back on itself.
        max_windowed_variance: Maximum directional variance over a
            sliding window. High values indicate erratic direction
            changes.
    """

    cost_per_length: float
    efficiency: float
    min_windowed_displacement: float
    max_windowed_variance: float


@dataclass
class CalibrationData:
    """Calibration metric arrays collected from known-good paths.

    Attributes:
        cpl_values: Cost-per-length values from calibration paths.
        efficiency_values: Efficiency values from calibration paths.
        displacement_values: Minimum windowed displacement values.
        variance_values: Maximum windowed variance values.
    """

    cpl_values: np.ndarray
    efficiency_values: np.ndarray
    displacement_values: np.ndarray
    variance_values: np.ndarray


@dataclass
class FilterThresholds:
    """Threshold values for each path quality filter.

    Attributes:
        tau_cpl: Maximum acceptable cost-per-length.
        tau_efficiency: Minimum acceptable path efficiency.
        tau_displacement: Minimum acceptable windowed displacement.
        tau_variance: Maximum acceptable windowed variance.
        percentile: Percentile used to derive thresholds from
            calibration data.
    """

    tau_cpl: float
    tau_efficiency: float
    tau_displacement: float
    tau_variance: float
    percentile: float


@dataclass
class FilterResult:
    """Output of the metric-based path filter.

    Attributes:
        passed_ids: Set of fragment IDs whose paths passed all filters.
        rejected_ids: Set of fragment IDs rejected by at least one filter.
        per_filter_rejections: Dict mapping filter name to the set of
            fragment IDs rejected by that specific filter.
        metrics: Dict mapping fragment_id to its computed PathMetrics.
        thresholds: The FilterThresholds applied during filtering.
    """

    passed_ids: set[int]
    rejected_ids: set[int]
    per_filter_rejections: dict[str, set[int]]
    metrics: dict[int, PathMetrics]
    thresholds: FilterThresholds
