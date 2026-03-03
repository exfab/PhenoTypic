"""Post-hoc path quality filtering for fragment-to-colony connections.

Applies a calibrated filter cascade to reject biologically implausible
paths that exploit noise corridors, salt-and-pepper bridges, or
CED-enhanced texture in the cost surface. Thresholds are derived from
known-good colony skeleton branches, making the approach
self-calibrating per image.

Filter cascade:
    F1: Cost-per-length -- reject abnormally expensive paths
    F2: Efficiency -- reject tortuous paths (low endpoint-to-arclength ratio)
    F3: Windowed displacement -- reject paths with locally stagnant segments
    F4: Windowed cost variance -- reject paths with erratic per-step costs
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import skeletonize

from ._dataclasses import (
    CalibrationData,
    FilterResult,
    FilterThresholds,
    PathMetrics,
)


# ── Metric computation ───────────────────────────────────────────────


def compute_path_metrics(
    path: Any,
    window_disp: int = 40,
    window_var: int = 20,
) -> PathMetrics:
    """Compute quality metrics for a single path.

    Args:
        path: A duck-typed path object with ``.coords``, ``.cost_profile``,
            ``.total_cost``, and ``.path_length`` attributes. Typically a
            ``FragmentPath`` or a calibration proxy.
        window_disp: Window size in pixels for the displacement filter.
            Paths shorter than this skip F3 (metric set to inf, always
            passes any threshold).
        window_var: Window size in pixels for the cost-variance filter.
            Paths shorter than this skip F4 (metric set to 0, always
            passes any threshold).

    Returns:
        PathMetrics with all four metric values computed from the path.
    """
    coords = path.coords  # (N, 2) int array
    n = path.path_length

    # F1: cost per length
    cost_per_length = path.total_cost / max(n, 1)

    # F2: efficiency = euclidean(endpoints) / arc_length
    endpoint_dist = float(
        np.sqrt(
            np.sum(
                (coords[0].astype(np.float64) - coords[-1].astype(np.float64))
                ** 2
            )
        )
    )
    efficiency = endpoint_dist / max(n, 1)

    # F3: min windowed displacement
    if n <= window_disp:
        min_windowed_displacement = float("inf")
    else:
        starts = coords[0 : n - window_disp].astype(np.float64)
        ends = coords[window_disp:n].astype(np.float64)
        displacements = np.sqrt(np.sum((ends - starts) ** 2, axis=1))
        ratios = displacements / window_disp
        min_windowed_displacement = float(np.min(ratios))

    # F4: max windowed cost variance via cumsum trick
    if n <= window_var:
        max_windowed_variance = 0.0
    else:
        step_costs = np.abs(np.diff(path.cost_profile))
        if len(step_costs) <= window_var:
            max_windowed_variance = 0.0
        else:
            cs = np.cumsum(step_costs)
            cs_sq = np.cumsum(step_costs**2)
            # Prepend 0 for prefix-sum indexing
            cs = np.concatenate([[0.0], cs])
            cs_sq = np.concatenate([[0.0], cs_sq])
            w = window_var
            n_windows = len(step_costs) - w + 1
            win_sum = cs[w : w + n_windows] - cs[0:n_windows]
            win_sum_sq = cs_sq[w : w + n_windows] - cs_sq[0:n_windows]
            win_mean = win_sum / w
            win_var = win_sum_sq / w - win_mean**2
            # Clamp numerical noise
            win_var = np.maximum(win_var, 0.0)
            max_windowed_variance = float(np.max(win_var))

    return PathMetrics(
        cost_per_length=cost_per_length,
        efficiency=efficiency,
        min_windowed_displacement=min_windowed_displacement,
        max_windowed_variance=max_windowed_variance,
    )


# ── Skeleton tracing for calibration ─────────────────────────────────


def _trace_skeleton_segment(
    skeleton: np.ndarray,
    start_r: int,
    start_c: int,
) -> np.ndarray:
    """Walk along a skeleton segment from a starting pixel.

    Follows 8-connected neighbors in the skeleton, never revisiting
    pixels. Returns ordered (row, col) coordinates of the traced path.

    Args:
        skeleton: Boolean skeleton image (H, W).
        start_r: Starting row coordinate.
        start_c: Starting column coordinate.

    Returns:
        (N, 2) int32 array of (row, col) coordinates in walk order.
    """
    h, w = skeleton.shape
    visited = np.zeros((h, w), dtype=np.bool_)
    path_r = [start_r]
    path_c = [start_c]
    visited[start_r, start_c] = True

    r, c = start_r, start_c
    while True:
        found = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and skeleton[nr, nc]
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    path_r.append(nr)
                    path_c.append(nc)
                    r, c = nr, nc
                    found = True
                    break
            if found:
                break
        if not found:
            break

    return np.column_stack([path_r, path_c]).astype(np.int32)


# ── Calibration from colony skeleton ─────────────────────────────────


def extract_calibration_branches(
    colony_labels: np.ndarray,
    unmasked_cost_surface: np.ndarray,
    min_branch_length: int = 10,
    window_disp: int = 40,
    window_var: int = 20,
) -> CalibrationData:
    """Extract quality metrics from known-good colony skeleton branches.

    Args:
        colony_labels: Int32 array (H, W) of labeled colonies. 0 is
            background.
        unmasked_cost_surface: Float32 array (H, W) of raw cost values
            before colony pixels were masked to epsilon. Needed to sample
            real cost values along skeleton branches.
        min_branch_length: Minimum pixel count for a branch to be
            included in calibration. Shorter branches are ignored.
        window_disp: Window size for the displacement metric.
        window_var: Window size for the variance metric.

    Returns:
        CalibrationData with metric arrays from all qualifying branches.
            Arrays may be empty if no branches meet *min_branch_length*.

    Longer description:
        Skeletonizes the colony mask, identifies branch points (pixels
        with more than two skeleton neighbors), removes them to split the
        skeleton into linear segments, then traces each segment and
        computes the four quality metrics. The resulting distributions
        define what "normal" intra-colony paths look like.
    """
    from scipy.ndimage import label as ndi_label

    colony_mask = colony_labels > 0
    skel = skeletonize(colony_mask)

    # Find branch points: skeleton pixels with >2 skeleton neighbors
    h, w = skel.shape
    neighbor_count = np.zeros((h, w), dtype=np.int32)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(skel)
            r_src = slice(max(0, -dr), min(h, h - dr))
            c_src = slice(max(0, -dc), min(w, w - dc))
            r_dst = slice(max(0, dr), min(h, h + dr))
            c_dst = slice(max(0, dc), min(w, w + dc))
            shifted[r_dst, c_dst] = skel[r_src, c_src]
            neighbor_count += shifted.astype(np.int32)

    branch_points = skel & (neighbor_count > 2)

    # Remove branch points to split skeleton into segments
    skel_segments = skel.copy()
    skel_segments[branch_points] = False

    # Label connected components of the remaining skeleton
    labeled_segments, n_segments = ndi_label(skel_segments)  # type: ignore[misc]

    cpl_list: list[float] = []
    eff_list: list[float] = []
    disp_list: list[float] = []
    var_list: list[float] = []

    for seg_id in range(1, n_segments + 1):
        seg_mask = labeled_segments == seg_id
        seg_pixels = np.argwhere(seg_mask)
        n_pixels = len(seg_pixels)

        if n_pixels < min_branch_length:
            continue

        # Default starting pixel (first in raster order)
        start_r, start_c = int(seg_pixels[0, 0]), int(seg_pixels[0, 1])

        # Find actual endpoint: pixel in segment with fewest neighbors
        seg_neighbor_count = np.zeros_like(seg_mask, dtype=np.int32)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                shifted = np.zeros_like(seg_mask)
                r_src = slice(max(0, -dr), min(h, h - dr))
                c_src = slice(max(0, -dc), min(w, w - dc))
                r_dst = slice(max(0, dr), min(h, h + dr))
                c_dst = slice(max(0, dc), min(w, w + dc))
                shifted[r_dst, c_dst] = seg_mask[r_src, c_src]
                seg_neighbor_count += shifted.astype(np.int32)

        # Endpoints have <=1 neighbor within the segment
        endpoint_mask = seg_mask & (seg_neighbor_count <= 1)
        endpoints = np.argwhere(endpoint_mask)
        if len(endpoints) > 0:
            start_r, start_c = int(endpoints[0, 0]), int(endpoints[0, 1])

        # Trace ordered coordinates along the segment
        coords = _trace_skeleton_segment(seg_mask, start_r, start_c)
        n_traced = len(coords)
        if n_traced < min_branch_length:
            continue

        # Sample cost along the skeleton branch
        cost_values = unmasked_cost_surface[coords[:, 0], coords[:, 1]].astype(
            np.float64
        )
        # Build a synthetic cumulative cost profile (cumsum of per-step costs)
        cumulative_cost = np.cumsum(cost_values[::-1])[::-1]

        # Create a lightweight path-like proxy for compute_path_metrics
        class _CalibPath:
            pass

        cal_path = _CalibPath()
        cal_path.coords = coords  # type: ignore[attr-defined]
        cal_path.cost_profile = cumulative_cost  # type: ignore[attr-defined]
        cal_path.total_cost = float(cumulative_cost[0])  # type: ignore[attr-defined]
        cal_path.path_length = n_traced  # type: ignore[attr-defined]

        metrics = compute_path_metrics(cal_path, window_disp, window_var)
        cpl_list.append(metrics.cost_per_length)
        eff_list.append(metrics.efficiency)
        disp_list.append(metrics.min_windowed_displacement)
        var_list.append(metrics.max_windowed_variance)

    return CalibrationData(
        cpl_values=np.array(cpl_list, dtype=np.float64),
        efficiency_values=np.array(eff_list, dtype=np.float64),
        displacement_values=np.array(disp_list, dtype=np.float64),
        variance_values=np.array(var_list, dtype=np.float64),
    )


# ── Threshold calibration ────────────────────────────────────────────


def calibrate_quality_thresholds(
    calibration: CalibrationData,
    percentile: float = 95.0,
) -> FilterThresholds:
    """Derive filter thresholds from calibration branch metrics.

    Args:
        calibration: Metric arrays from ``extract_calibration_branches``.
        percentile: Calibration stringency. 95 means reject paths worse
            than the worst 5% of known-good branches.

    Returns:
        FilterThresholds with calibrated cutoffs for each metric.

    Longer description:
        High cost-per-length and high variance are bad (reject above
        *percentile*). Low efficiency and low displacement are bad
        (reject below ``100 - percentile``).
    """
    tau_cpl = float(np.percentile(calibration.cpl_values, percentile))
    tau_efficiency = float(
        np.percentile(calibration.efficiency_values, 100 - percentile)
    )
    tau_displacement = float(
        np.percentile(calibration.displacement_values, 100 - percentile)
    )
    tau_variance = float(
        np.percentile(calibration.variance_values, percentile)
    )

    return FilterThresholds(
        tau_cpl=tau_cpl,
        tau_efficiency=tau_efficiency,
        tau_displacement=tau_displacement,
        tau_variance=tau_variance,
        percentile=percentile,
    )


# ── Filter cascade ───────────────────────────────────────────────────


def apply_quality_filters(
    paths: dict[int, Any],
    thresholds: FilterThresholds,
    window_disp: int = 40,
    window_var: int = 20,
) -> FilterResult:
    """Apply the four-stage filter cascade to candidate paths.

    Args:
        paths: Dict mapping fragment_id to a path object (typically
            ``FragmentPath``). Each path must support the duck-typed
            interface expected by ``compute_path_metrics``.
        thresholds: Calibrated ``FilterThresholds`` from
            ``calibrate_quality_thresholds``.
        window_disp: Window size for the displacement metric.
        window_var: Window size for the cost-variance metric.

    Returns:
        FilterResult with per-filter breakdown, passed/rejected ID sets,
        computed metrics for every path, and the thresholds applied.

    Longer description:
        Filters are applied sequentially: each filter only considers
        paths that passed all previous filters, making the cascade
        monotonically reducing. The four stages are:

        - **F1** cost-per-length (high is bad)
        - **F2** efficiency (low is bad)
        - **F3** windowed displacement (low is bad)
        - **F4** windowed cost variance (high is bad)
    """
    # Compute metrics for all paths up front
    all_metrics: dict[int, PathMetrics] = {}
    for fid, path in paths.items():
        all_metrics[fid] = compute_path_metrics(path, window_disp, window_var)

    # Start with all path IDs as candidates
    remaining = set(paths.keys())
    per_filter: dict[str, set[int]] = {}

    # F1: cost-per-length (high is bad)
    f1_reject: set[int] = set()
    for fid in remaining:
        if all_metrics[fid].cost_per_length > thresholds.tau_cpl:
            f1_reject.add(fid)
    per_filter["F1_cost_per_length"] = f1_reject
    remaining -= f1_reject

    # F2: efficiency (low is bad)
    f2_reject: set[int] = set()
    for fid in remaining:
        if all_metrics[fid].efficiency < thresholds.tau_efficiency:
            f2_reject.add(fid)
    per_filter["F2_efficiency"] = f2_reject
    remaining -= f2_reject

    # F3: windowed displacement (low is bad)
    f3_reject: set[int] = set()
    for fid in remaining:
        if all_metrics[fid].min_windowed_displacement < thresholds.tau_displacement:
            f3_reject.add(fid)
    per_filter["F3_displacement"] = f3_reject
    remaining -= f3_reject

    # F4: windowed cost variance (high is bad)
    f4_reject: set[int] = set()
    for fid in remaining:
        if all_metrics[fid].max_windowed_variance > thresholds.tau_variance:
            f4_reject.add(fid)
    per_filter["F4_variance"] = f4_reject
    remaining -= f4_reject

    all_rejected: set[int] = set()
    for s in per_filter.values():
        all_rejected |= s

    return FilterResult(
        passed_ids=remaining,
        rejected_ids=all_rejected,
        per_filter_rejections=per_filter,
        metrics=all_metrics,
        thresholds=thresholds,
    )
