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
    """Per-path structure-based quality metrics for filtering.

    Attributes:
        median_raw_cost: Median of raw cost surface values along the path.
            Low values indicate the path follows real structure; high
            values indicate the path crosses background.
        max_window_cost: Maximum of windowed median raw cost along the
            path. Catches paths that are mostly on structure but have
            one segment crossing background.
        band_cost_variance: Variance of cost values in a dilated band
            around the path. Low values indicate uniform cost (real
            structure); high values indicate lucky low-cost pixels
            surrounded by high-cost noise.
        pct_energy_band_median: Median PCT energy in the dilated band.
            High values indicate real structure with strong PCT response
            across full width; low values indicate the path threads
            through a weak-PCT region.
        gray_band_snr: Local grayscale SNR using PCT-masked
            double-dilation background. High values indicate the path
            stands out from unstructured surroundings; low values
            indicate the path intensity matches background noise.
    """

    median_raw_cost: float
    max_window_cost: float
    band_cost_variance: float
    pct_energy_band_median: float
    gray_band_snr: float


@dataclass
class CalibrationData:
    """Calibration metric arrays collected from known-good colony skeleton branches.

    Attributes:
        median_cost_values: Median raw cost for each calibration branch.
        max_window_cost_values: Max windowed cost for each calibration branch.
        band_variance_values: Band cost variance for each calibration branch.
        pct_energy_median_values: PCT energy band median for each branch.
        gray_snr_values: Local grayscale SNR for each branch.
    """

    median_cost_values: np.ndarray
    max_window_cost_values: np.ndarray
    band_variance_values: np.ndarray
    pct_energy_median_values: np.ndarray
    gray_snr_values: np.ndarray


@dataclass
class FilterThresholds:
    """Threshold values for each path quality filter.

    Attributes:
        tau_median_cost: Maximum allowed median raw cost.
        tau_window_cost: Maximum allowed max windowed cost.
        tau_band_variance: Maximum allowed band cost variance.
        tau_pct_energy_median: Minimum allowed PCT energy band median.
            Paths below this lack sufficient PCT response.
        tau_gray_snr: Minimum allowed local grayscale SNR.
            Paths below this don't stand out from background.
        k_iqr: IQR multiplier used for calibration (0 if overridden).
    """

    tau_median_cost: float
    tau_window_cost: float
    tau_band_variance: float
    tau_pct_energy_median: float
    tau_gray_snr: float
    k_iqr: float


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
