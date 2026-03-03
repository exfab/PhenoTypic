"""Post-hoc path quality filtering for fragment-to-colony connections.

Applies a calibrated filter cascade to reject paths that don't follow real
structure in the cost surface. Thresholds are derived from known-good colony
skeleton branches, making the approach self-calibrating per image.

Filter cascade (structure-based):
    F1: Median raw cost -- reject paths with high median cost along the path
    F2: Max windowed cost -- reject paths with any segment crossing background
    F3: Band cost variance -- reject paths with high cost variance in dilated band
    F4: PCT energy band median -- reject paths with weak PCT response in band
    F5: Local grayscale SNR -- reject paths that don't stand out from
        unstructured surroundings (PCT-masked double-dilation background)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import disk, skeletonize

from ._dataclasses import (
    CalibrationData,
    FilterResult,
    FilterThresholds,
    PathMetrics,
)


# ── Band sampling helpers ────────────────────────────────────────────


def _sample_dilated_band(
    coords: np.ndarray,
    image: np.ndarray,
    dilation_radius: int,
) -> np.ndarray:
    """Sample values in a dilated band around path coordinates.

    Args:
        coords: (N, 2) int32 array of (row, col) path coordinates.
        image: (H, W) array to sample from.
        dilation_radius: Radius of the structuring element (disk).

    Returns:
        1-D float64 array of all sampled values in the dilated band.
    """
    H, W = image.shape
    selem = disk(dilation_radius)
    offsets = np.argwhere(selem) - dilation_radius  # (K, 2) relative offsets

    # Broadcast: (N, 1) + (1, K) -> (N, K)
    rows = coords[:, 0:1] + offsets[:, 0:1].T
    cols = coords[:, 1:2] + offsets[:, 1:2].T

    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)

    return image[rows.ravel(), cols.ravel()].astype(np.float64)


def _compute_local_snr(
    coords: np.ndarray,
    grayscale: np.ndarray,
    pct_energy: np.ndarray,
    inner_radius: int,
    outer_radius: int,
    pct_noise_ceil: float,
) -> float:
    """Compute local SNR with PCT-masked background ring.

    Signal = grayscale in the inner dilated region (hypha width).
    Background = grayscale in the annular ring (outer - inner),
    filtered to keep only unstructured pixels (PCT < pct_noise_ceil).

    Args:
        coords: (N, 2) int32 array of (row, col) path coordinates.
        grayscale: (H, W) grayscale image.
        pct_energy: (H, W) PCT energy map.
        inner_radius: Disk radius for signal region.
        outer_radius: Disk radius for outer edge of background ring.
        pct_noise_ceil: PCT energy threshold -- ring pixels below this
            are considered unstructured background.

    Returns:
        Local SNR value. 0.0 if insufficient unstructured background pixels.
    """
    H, W = grayscale.shape

    def _linear_indices(radius: int) -> np.ndarray:
        selem = disk(radius)
        offsets = np.argwhere(selem) - radius
        rows = np.clip(coords[:, 0:1] + offsets[:, 0:1].T, 0, H - 1).ravel()
        cols = np.clip(coords[:, 1:2] + offsets[:, 1:2].T, 0, W - 1).ravel()
        return np.unique(rows * W + cols)

    inner_idx = _linear_indices(inner_radius)
    outer_idx = _linear_indices(outer_radius)

    # Ring = outer - inner
    ring_idx = np.setdiff1d(outer_idx, inner_idx)

    # Signal from inner region
    signal_gray = grayscale.ravel()[inner_idx].astype(np.float64)

    # Background ring: filter to unstructured pixels only
    ring_pct = pct_energy.ravel()[ring_idx]
    unstructured_mask = ring_pct < pct_noise_ceil
    unstructured_gray = grayscale.ravel()[ring_idx[unstructured_mask]].astype(
        np.float64
    )

    if len(unstructured_gray) < 5:
        return 0.0

    return float(
        abs(np.median(signal_gray) - np.median(unstructured_gray))
        / (np.std(unstructured_gray) + 1e-8)
    )


# ── Metric computation ───────────────────────────────────────────────


def compute_path_metrics(
    path: Any,
    cost_surface: np.ndarray,
    window_cost: int = 30,
    dilation_radius: int = 2,
    pct_energy: np.ndarray | None = None,
    grayscale: np.ndarray | None = None,
    snr_margin: int = 3,
    pct_noise_ceil: float = 0.0,
) -> PathMetrics:
    """Compute structure-based quality metrics for a single path.

    Args:
        path: A duck-typed path object with ``.coords`` attribute
            ((N, 2) int array of (row, col) coordinates). Typically a
            ``FragmentPath`` or a calibration proxy.
        cost_surface: Float32 (H, W) raw cost surface (before colony masking).
        window_cost: Window size (pixels) for the windowed cost metric.
        dilation_radius: Radius for the dilated band sampling (F3/F4/F5).
        pct_energy: Float (H, W) PCT energy map for F4. None to skip.
        grayscale: Float (H, W) grayscale image for F5. None to skip.
        snr_margin: Extra radius beyond *dilation_radius* for the SNR
            background ring.
        pct_noise_ceil: PCT energy threshold for F5 background masking.

    Returns:
        PathMetrics with all five metric values.
    """
    raw_costs = cost_surface[path.coords[:, 0], path.coords[:, 1]].astype(
        np.float64
    )

    # F1: median raw cost
    median_raw_cost = float(np.median(raw_costs))

    # F2: max windowed median cost
    if len(raw_costs) <= window_cost:
        max_window_cost = median_raw_cost
    else:
        n_windows = len(raw_costs) - window_cost + 1
        window_medians = np.array(
            [np.median(raw_costs[i : i + window_cost]) for i in range(n_windows)]
        )
        max_window_cost = float(np.max(window_medians))

    # F3: band cost variance
    band_values = _sample_dilated_band(path.coords, cost_surface, dilation_radius)
    band_cost_variance = float(np.var(band_values))

    # F4: PCT energy band median (low is bad)
    if pct_energy is not None:
        pct_band = _sample_dilated_band(path.coords, pct_energy, dilation_radius)
        pct_energy_band_median = float(np.median(pct_band))
    else:
        pct_energy_band_median = 0.0

    # F5: local grayscale SNR (low is bad)
    if grayscale is not None and pct_energy is not None:
        gray_band_snr = _compute_local_snr(
            path.coords,
            grayscale,
            pct_energy,
            inner_radius=dilation_radius,
            outer_radius=dilation_radius + snr_margin,
            pct_noise_ceil=pct_noise_ceil,
        )
    else:
        gray_band_snr = 0.0

    return PathMetrics(
        median_raw_cost=median_raw_cost,
        max_window_cost=max_window_cost,
        band_cost_variance=band_cost_variance,
        pct_energy_band_median=pct_energy_band_median,
        gray_band_snr=gray_band_snr,
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


def _count_8connected_neighbors(mask: np.ndarray) -> np.ndarray:
    """Count 8-connected neighbors for each pixel in a boolean mask.

    Args:
        mask: Boolean array (H, W).

    Returns:
        Int32 array (H, W) with neighbor counts (0-8) at each pixel.
    """
    h, w = mask.shape
    neighbor_count = np.zeros((h, w), dtype=np.int32)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask)
            r_src = slice(max(0, -dr), min(h, h - dr))
            c_src = slice(max(0, -dc), min(w, w - dc))
            r_dst = slice(max(0, dr), min(h, h + dr))
            c_dst = slice(max(0, dc), min(w, w + dc))
            shifted[r_dst, c_dst] = mask[r_src, c_src]
            neighbor_count += shifted.astype(np.int32)
    return neighbor_count


def extract_calibration_branches(
    colony_labels: np.ndarray,
    unmasked_cost_surface: np.ndarray,
    min_branch_length: int = 10,
    window_cost: int = 30,
    dilation_radius: int = 2,
    pct_energy: np.ndarray | None = None,
    grayscale: np.ndarray | None = None,
    snr_margin: int = 3,
    pct_noise_ceil: float = 0.0,
) -> CalibrationData:
    """Extract quality metrics from known-good colony skeleton branches.

    Args:
        colony_labels: Int32 array (H, W) of labeled colonies. 0 is
            background.
        unmasked_cost_surface: Float32 array (H, W) of raw cost values
            before colony pixels were masked to epsilon.
        min_branch_length: Minimum pixel count for a branch to be
            included in calibration.
        window_cost: Window size for the windowed cost metric.
        dilation_radius: Radius for the dilated band sampling (F3/F4/F5).
        pct_energy: Float (H, W) PCT energy map for F4. None to skip.
        grayscale: Float (H, W) grayscale image for F5. None to skip.
        snr_margin: Extra radius beyond *dilation_radius* for the SNR
            background ring.
        pct_noise_ceil: PCT energy threshold for F5 background masking.

    Returns:
        CalibrationData with metric arrays from all qualifying branches.
            Arrays may be empty if no branches meet *min_branch_length*.

    Longer description:
        Skeletonizes the colony mask, identifies branch points (pixels
        with more than two skeleton neighbors), removes them to split the
        skeleton into linear segments, then traces each segment and
        computes the five quality metrics. The resulting distributions
        define what "normal" intra-colony paths look like.
    """
    from scipy.ndimage import label as ndi_label

    colony_mask = colony_labels > 0
    skel = skeletonize(colony_mask)

    # Find branch points: skeleton pixels with >2 skeleton neighbors
    neighbor_count = _count_8connected_neighbors(skel)
    branch_points = skel & (neighbor_count > 2)

    # Remove branch points to split skeleton into segments
    skel_segments = skel.copy()
    skel_segments[branch_points] = False

    # Label connected components of the remaining skeleton
    labeled_segments, n_segments = ndi_label(skel_segments)  # type: ignore[misc]

    # Compute neighbor counts once on the branch-point-removed skeleton;
    # endpoints (<=1 neighbor) are the same whether counted globally or
    # per-segment, because branch points have already been removed.
    seg_neighbor_count = _count_8connected_neighbors(skel_segments)

    median_cost_list: list[float] = []
    window_cost_list: list[float] = []
    band_variance_list: list[float] = []
    pct_energy_median_list: list[float] = []
    gray_snr_list: list[float] = []

    for seg_id in range(1, n_segments + 1):
        seg_mask = labeled_segments == seg_id
        seg_pixels = np.argwhere(seg_mask)
        n_pixels = len(seg_pixels)

        if n_pixels < min_branch_length:
            continue

        # Default starting pixel (first in raster order)
        start_r, start_c = int(seg_pixels[0, 0]), int(seg_pixels[0, 1])

        # Endpoints have <=1 neighbor within the branch-point-removed skeleton
        endpoint_mask = seg_mask & (seg_neighbor_count <= 1)
        endpoints = np.argwhere(endpoint_mask)
        if len(endpoints) > 0:
            start_r, start_c = int(endpoints[0, 0]), int(endpoints[0, 1])

        # Trace ordered coordinates along the segment
        coords = _trace_skeleton_segment(seg_mask, start_r, start_c)
        n_traced = len(coords)
        if n_traced < min_branch_length:
            continue

        # Sample raw cost along the skeleton branch
        raw_costs = unmasked_cost_surface[coords[:, 0], coords[:, 1]].astype(
            np.float64
        )

        # F1: median raw cost
        median_cost = float(np.median(raw_costs))

        # F2: max windowed median cost
        if len(raw_costs) <= window_cost:
            max_win_cost = median_cost
        else:
            n_windows = len(raw_costs) - window_cost + 1
            window_medians = np.array(
                [
                    np.median(raw_costs[i : i + window_cost])
                    for i in range(n_windows)
                ]
            )
            max_win_cost = float(np.max(window_medians))

        # F3: band cost variance
        band_values = _sample_dilated_band(
            coords, unmasked_cost_surface, dilation_radius
        )
        band_var = float(np.var(band_values))

        # F4: PCT energy band median
        if pct_energy is not None:
            pct_band = _sample_dilated_band(coords, pct_energy, dilation_radius)
            pct_energy_median_list.append(float(np.median(pct_band)))
        else:
            pct_energy_median_list.append(0.0)

        # F5: local grayscale SNR
        if grayscale is not None and pct_energy is not None:
            snr = _compute_local_snr(
                coords,
                grayscale,
                pct_energy,
                inner_radius=dilation_radius,
                outer_radius=dilation_radius + snr_margin,
                pct_noise_ceil=pct_noise_ceil,
            )
            gray_snr_list.append(snr)
        else:
            gray_snr_list.append(0.0)

        median_cost_list.append(median_cost)
        window_cost_list.append(max_win_cost)
        band_variance_list.append(band_var)

    return CalibrationData(
        median_cost_values=np.array(median_cost_list, dtype=np.float64),
        max_window_cost_values=np.array(window_cost_list, dtype=np.float64),
        band_variance_values=np.array(band_variance_list, dtype=np.float64),
        pct_energy_median_values=np.array(pct_energy_median_list, dtype=np.float64),
        gray_snr_values=np.array(gray_snr_list, dtype=np.float64),
    )


# ── Threshold calibration ────────────────────────────────────────────


def calibrate_thresholds(
    calibration: CalibrationData,
    k: float = 3.0,
) -> FilterThresholds:
    """Derive filter thresholds from calibration branch metrics via IQR.

    F1-F3 use upper thresholds (median + k * IQR, high is bad).
    F4, F5 use lower thresholds (median - k * IQR, low is bad).

    Args:
        calibration: Metric arrays from ``extract_calibration_branches``.
        k: IQR multiplier. Higher values are more permissive.

    Returns:
        FilterThresholds with calibrated cutoffs for each metric.
    """

    def _upper(values: np.ndarray) -> float:
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        return float(med + k * (q75 - q25))

    def _lower(values: np.ndarray) -> float:
        q25, med, q75 = np.percentile(values, [25, 50, 75])
        return float(med - k * (q75 - q25))

    return FilterThresholds(
        tau_median_cost=_upper(calibration.median_cost_values),
        tau_window_cost=_upper(calibration.max_window_cost_values),
        tau_band_variance=_upper(calibration.band_variance_values),
        tau_pct_energy_median=_lower(calibration.pct_energy_median_values),
        tau_gray_snr=_lower(calibration.gray_snr_values),
        k_iqr=k,
    )


# ── Filter cascade ───────────────────────────────────────────────────


def apply_filter_cascade(
    paths: dict[int, Any],
    cost_surface: np.ndarray,
    thresholds: FilterThresholds,
    window_cost: int = 30,
    dilation_radius: int = 2,
    pct_energy: np.ndarray | None = None,
    grayscale: np.ndarray | None = None,
    snr_margin: int = 3,
    pct_noise_ceil: float = 0.0,
) -> FilterResult:
    """Apply the five-stage filter cascade to candidate paths.

    Args:
        paths: Dict mapping fragment_id to a path object (typically
            ``FragmentPath``). Each path must have a ``.coords`` attribute.
        cost_surface: Float32 (H, W) raw cost surface.
        thresholds: Calibrated ``FilterThresholds`` from
            ``calibrate_thresholds``.
        window_cost: Window size for the windowed cost metric.
        dilation_radius: Radius for the dilated band sampling.
        pct_energy: Float (H, W) PCT energy map for F4. None to skip.
        grayscale: Float (H, W) grayscale image for F5. None to skip.
        snr_margin: Extra radius beyond *dilation_radius* for the SNR
            background ring.
        pct_noise_ceil: PCT energy threshold for F5 background masking.

    Returns:
        FilterResult with per-filter breakdown, passed/rejected ID sets,
        computed metrics for every path, and the thresholds applied.

    Longer description:
        Filters are applied sequentially: each filter only considers
        paths that passed all previous filters, making the cascade
        monotonically reducing. The five stages are:

        - **F1** median raw cost (high is bad)
        - **F2** max windowed cost (high is bad)
        - **F3** band cost variance (high is bad)
        - **F4** PCT energy band median (low is bad)
        - **F5** local grayscale SNR (low is bad)
    """
    # Compute metrics for all paths up front
    all_metrics: dict[int, PathMetrics] = {}
    for fid, path in paths.items():
        all_metrics[fid] = compute_path_metrics(
            path,
            cost_surface,
            window_cost,
            dilation_radius,
            pct_energy=pct_energy,
            grayscale=grayscale,
            snr_margin=snr_margin,
            pct_noise_ceil=pct_noise_ceil,
        )

    # Start with all path IDs as candidates
    remaining = set(paths.keys())
    per_filter: dict[str, set[int]] = {}

    # F1: median raw cost (high is bad)
    f1_reject: set[int] = {
        fid
        for fid in remaining
        if all_metrics[fid].median_raw_cost > thresholds.tau_median_cost
    }
    per_filter["F1_median_cost"] = f1_reject
    remaining -= f1_reject

    # F2: max windowed cost (high is bad)
    f2_reject: set[int] = {
        fid
        for fid in remaining
        if all_metrics[fid].max_window_cost > thresholds.tau_window_cost
    }
    per_filter["F2_window_cost"] = f2_reject
    remaining -= f2_reject

    # F3: band cost variance (high is bad)
    f3_reject: set[int] = {
        fid
        for fid in remaining
        if all_metrics[fid].band_cost_variance > thresholds.tau_band_variance
    }
    per_filter["F3_band_variance"] = f3_reject
    remaining -= f3_reject

    # F4: PCT energy band median (low is bad)
    f4_reject: set[int] = {
        fid
        for fid in remaining
        if all_metrics[fid].pct_energy_band_median < thresholds.tau_pct_energy_median
    }
    per_filter["F4_pct_energy"] = f4_reject
    remaining -= f4_reject

    # F5: local grayscale SNR (low is bad)
    f5_reject: set[int] = {
        fid
        for fid in remaining
        if all_metrics[fid].gray_band_snr < thresholds.tau_gray_snr
    }
    per_filter["F5_gray_snr"] = f5_reject
    remaining -= f5_reject

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


# ── Orchestrator ─────────────────────────────────────────────────────


def filter_paths(
    paths: dict[int, Any],
    colony_labels: np.ndarray,
    unmasked_cost_surface: np.ndarray,
    k: float = 3.0,
    window_cost: int = 30,
    dilation_radius: int = 2,
    thresholds_override: FilterThresholds | None = None,
    pct_energy: np.ndarray | None = None,
    grayscale: np.ndarray | None = None,
    snr_margin: int = 3,
) -> FilterResult:
    """Run the full path quality filtering pipeline.

    1. Extract calibration metrics from colony skeleton branches
    2. Derive thresholds as median +/- k * IQR
    3. Apply five-stage filter cascade to candidate paths

    Args:
        paths: Dict mapping fragment_id to a path object from Stage 3.
        colony_labels: Int32 (H, W) labeled colony mask. 0 is background.
        unmasked_cost_surface: Float32 (H, W) raw cost surface (before
            colony masking).
        k: IQR multiplier for calibration. Higher is more permissive.
        window_cost: Window size for the windowed cost metric.
        dilation_radius: Radius for the dilated band sampling.
        thresholds_override: If provided, skip calibration and use these
            thresholds directly.
        pct_energy: Float (H, W) PCT energy map for F4/F5. None to skip.
        grayscale: Float (H, W) grayscale image for F5. None to skip.
        snr_margin: Extra radius beyond *dilation_radius* for the SNR
            background ring.

    Returns:
        FilterResult with passed/rejected IDs, per-filter breakdown,
        metrics, and thresholds.
    """
    from skimage.filters import threshold_otsu

    # Compute PCT noise ceiling via Otsu for F5 background masking
    if pct_energy is not None:
        pct_noise_ceil = float(threshold_otsu(pct_energy))
    else:
        pct_noise_ceil = 0.0

    if thresholds_override is not None:
        thresholds = thresholds_override
    else:
        calibration = extract_calibration_branches(
            colony_labels,
            unmasked_cost_surface,
            min_branch_length=10,
            window_cost=window_cost,
            dilation_radius=dilation_radius,
            pct_energy=pct_energy,
            grayscale=grayscale,
            snr_margin=snr_margin,
            pct_noise_ceil=pct_noise_ceil,
        )
        thresholds = calibrate_thresholds(calibration, k)

    return apply_filter_cascade(
        paths,
        unmasked_cost_surface,
        thresholds,
        window_cost,
        dilation_radius,
        pct_energy=pct_energy,
        grayscale=grayscale,
        snr_margin=snr_margin,
        pct_noise_ceil=pct_noise_ceil,
    )
