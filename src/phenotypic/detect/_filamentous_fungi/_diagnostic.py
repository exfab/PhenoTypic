"""Diagnostic plotting for FilamentousFungiDetector pathfinding and filtering.

Provides ``collect_diagnostic_state`` to capture all Phase-4 intermediates
(cost surfaces, Dijkstra output, paths, quality filters) and six plotting
functions that visualize them.  Intended for rapid parameter tuning of the
reconnection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from phenotypic._core._image import Image

from ._dataclasses import (
    CalibrationData,
    DijkstraResult,
    FilterResult,
    FragmentAssignment,
    FragmentPath,
)


# ── Diagnostic state ────────────────────────────────────────────────


@dataclass
class DiagnosticState:
    """All Phase-4 intermediates needed by the diagnostic plots.

    Attributes:
        enhanced_arr: Contrast-stretched detect_mat.
        enhanced_gray: Contrast-stretched grayscale.
        colony_labels: Voronoi colony assignment (pre-reconnection).
        central_mask: Branch pixels overlapping colonies.
        screened_frags: Pre-screened fragment labels.
        cost_surface: Masked composite cost (Dijkstra input).
        unmasked_cost: Unmasked composite cost (quality reference).
        dijkstra: Dijkstra propagation result.
        paths: Dict mapping fragment_id to FragmentPath.
        unconnected_ids: Fragment IDs that failed path extraction.
        assignments: Dict mapping fragment_id to FragmentAssignment.
        calibration: Calibration data from colony skeleton branches.
        filter_result: Filter cascade output with metrics and rejections.
        pct_noise_ceil: PCT energy Otsu threshold for F5.
    """

    enhanced_arr: np.ndarray
    enhanced_gray: np.ndarray
    colony_labels: np.ndarray
    central_mask: np.ndarray
    screened_frags: np.ndarray
    cost_surface: np.ndarray
    unmasked_cost: np.ndarray
    dijkstra: DijkstraResult
    paths: dict[int, FragmentPath]
    unconnected_ids: list[int]
    assignments: dict[int, FragmentAssignment]
    calibration: CalibrationData
    filter_result: FilterResult
    pct_noise_ceil: float


# ── State collector ──────────────────────────────────────────────────


def collect_diagnostic_state(
    detector: 'FilamentousFungiDetector',  # noqa: F821
    image: 'Image',
) -> DiagnosticState:
    """Run Phases 1-4 of detection and capture all intermediates.

    Mirrors the logic in ``_operate`` but processes the full image as a
    single tile (no tiling) so that all Dijkstra and filter outputs are
    globally coherent for diagnostic inspection.

    Args:
        detector: Configured FilamentousFungiDetector instance.
        image: Input image (not modified).

    Returns:
        DiagnosticState with all Phase-4 intermediates populated.

    Raises:
        ValueError: If detection produces no centers, no branches, or
            no overlapping structure.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    from phenotypic import ImagePipeline
    from phenotypic.enhance import ContrastStretching

    from . import (
        _compute_screening_envelope,
        apply_filter_cascade,
        assign_fragments_to_colonies,
        calibrate_screening_threshold,
        calibrate_thresholds,
        euclidean_voronoi_assign,
        extract_calibration_branches,
        extract_fragment_paths,
        prescreen_fragments,
        run_multisource_dijkstra,
    )

    # ── Phase 1: Inoculum detection ──────────────────────────────
    if isinstance(detector.inoculum_detector, ImagePipeline):
        center_result = detector.inoculum_detector.apply(
            image, inplace=False, reset=False,
        )
    else:
        center_result = detector.inoculum_detector.apply(image, inplace=False)

    center_objmask = center_result.objmask[:]
    center_objmap = center_result.objmap[:]

    if center_objmap.max() == 0:
        raise ValueError("No centers detected by inoculum_detector.")

    # ── Phase 2: Branch detection ────────────────────────────────
    enhanced_work = image.copy()
    ContrastStretching().apply(enhanced_work, inplace=True)
    enhanced_arr = enhanced_work.detect_mat[:]
    enhanced_gray = enhanced_work.gray[:]

    gauss_labels = detector._detect_gauss_branches(enhanced_work)
    del enhanced_work

    pct_mask, pct_result = detector._detect_pct_branches(enhanced_arr)
    branch_labels = detector._filter_gauss_by_pct_overlap(gauss_labels, pct_mask)
    overall_objmask = branch_labels > 0

    # ── Phase 3: Center filtering + Voronoi ──────────────────────
    filtered_center_objmask = detector._filter_mask_by_overlap(
        mask=center_objmask, reference_mask=overall_objmask,
    )
    overlap_objmap = label(filtered_center_objmask)

    if overlap_objmap.max() == 0:
        raise ValueError(
            "No centers overlap with overall structure after filtering.",
        )

    region_markers = detector._create_markers_from_centers(center_objmap)
    mask_m = overall_objmask | filtered_center_objmask
    colony_labels = euclidean_voronoi_assign(region_markers, mask_m)

    # ── Phase 4: Dijkstra reconnection (full image, no tiling) ───
    central_mask, fragment_labels = detector._separate_central_and_fragments(
        branch_labels, colony_labels,
    )

    unmasked_cost, cost_surface = detector._build_cost_surface(
        pct_result, enhanced_arr, colony_labels, central_mask,
    )

    # Pre-screen fragments
    colony_branch_mask = (colony_labels > 0).astype(np.int32)
    min_cost_envelope, _ = _compute_screening_envelope(
        cost_surface, colony_branch_mask, detector.r_screen,
    )
    tau_screen, _ = calibrate_screening_threshold(
        cost_surface, colony_branch_mask, r_screen=detector.r_screen,
        min_cost_envelope=min_cost_envelope,
    )
    screen_result = prescreen_fragments(
        cost_surface, fragment_labels,
        r_screen=detector.r_screen,
        tau_screen=tau_screen,
        colony_branch_mask=colony_branch_mask,
        min_cost_envelope=min_cost_envelope,
    )
    screened_frags = screen_result.screened_fragment_labels

    pct_energy = pct_result.pc_sum.astype(np.float32)
    pct_noise_ceil = float(threshold_otsu(pct_energy))

    # Dijkstra on full image
    dijkstra = run_multisource_dijkstra(
        cost_surface, colony_labels, detector.delta,
    )

    assignments = assign_fragments_to_colonies(
        screened_frags, dijkstra.colony_id, dijkstra.cost_distance,
    )

    paths, unconnected_ids = extract_fragment_paths(
        screened_frags, assignments, dijkstra, cost_surface,
    )

    # Quality filtering
    calibration = extract_calibration_branches(
        colony_labels, unmasked_cost,
        window_cost=detector.window_cost,
        dilation_radius=detector.path_dilation_radius,
        pct_energy=pct_energy,
        grayscale=enhanced_gray,
        snr_margin=detector.snr_margin,
        pct_noise_ceil=pct_noise_ceil,
    )

    if calibration.median_cost_values.size > 0:
        thresholds = calibrate_thresholds(calibration, k=detector.quality_k)
        filter_result = apply_filter_cascade(
            paths, unmasked_cost, thresholds,
            window_cost=detector.window_cost,
            dilation_radius=detector.path_dilation_radius,
            pct_energy=pct_energy,
            grayscale=enhanced_gray,
            snr_margin=detector.snr_margin,
            pct_noise_ceil=pct_noise_ceil,
        )
    else:
        from ._dataclasses import FilterResult, FilterThresholds

        filter_result = FilterResult(
            passed_ids=set(paths.keys()),
            rejected_ids=set(),
            per_filter_rejections={},
            metrics={},
            thresholds=FilterThresholds(
                tau_median_cost=0, tau_window_cost=0,
                tau_band_variance=0, tau_pct_energy_median=0,
                tau_gray_snr=0, k_iqr=0,
            ),
        )

    return DiagnosticState(
        enhanced_arr=enhanced_arr,
        enhanced_gray=enhanced_gray,
        colony_labels=colony_labels,
        central_mask=central_mask,
        screened_frags=screened_frags,
        cost_surface=cost_surface,
        unmasked_cost=unmasked_cost,
        dijkstra=dijkstra,
        paths=paths,
        unconnected_ids=unconnected_ids,
        assignments=assignments,
        calibration=calibration,
        filter_result=filter_result,
        pct_noise_ceil=pct_noise_ceil,
    )


# ── Plotting functions ───────────────────────────────────────────────


def plot_cost_distance(state: DiagnosticState) -> 'plt.Figure':
    """Visualize Dijkstra cost-distance, colony territory, and clipped cost.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with 1x3 panel layout.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cd = state.dijkstra.cost_distance.copy()
    cd[cd == np.inf] = np.nan

    # Left: log1p cost distance
    ax = axes[0]
    im = ax.imshow(np.log1p(np.nan_to_num(cd, nan=0.0)), cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("log1p(cost_distance)")
    ax.axis("off")

    # Center: colony territory
    ax = axes[1]
    cid = state.dijkstra.colony_id.astype(np.float64)
    cid[cid < 0] = np.nan
    n_cols = int(np.nanmax(cid)) + 1 if not np.all(np.isnan(cid)) else 1
    cmap = plt.get_cmap("tab20", max(n_cols, 1))
    ax.imshow(cid, cmap=cmap, interpolation="nearest")
    ax.set_title("Colony territory (Dijkstra)")
    ax.axis("off")

    # Right: cost-distance clipped at p50
    ax = axes[2]
    valid = cd[~np.isnan(cd)]
    vmax = float(np.percentile(valid, 50)) if len(valid) > 0 else 1.0
    im = ax.imshow(np.clip(np.nan_to_num(cd, nan=0.0), 0, vmax), cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Cost-distance clipped at p50 ({vmax:.1f})")
    ax.axis("off")

    fig.tight_layout()
    return fig


def plot_fragment_overlay(state: DiagnosticState) -> 'plt.Figure':
    """Overlay colony, fragment, and path information on the enhanced image.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with single panel showing colony/fragment/path overlay.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Background
    ax.imshow(state.enhanced_arr, cmap="gray", alpha=0.6)

    # Build RGBA overlay
    h, w = state.enhanced_arr.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    # Blue: colony pixels
    colony_px = state.colony_labels > 0
    overlay[colony_px] = [0.2, 0.4, 1.0, 0.5]

    # Determine passed/rejected fragments
    passed_ids = state.filter_result.passed_ids
    all_frag_ids = set(int(i) for i in np.unique(state.screened_frags) if i > 0)
    rejected_ids = (all_frag_ids - passed_ids) | set(state.unconnected_ids)

    # Green: fragments with paths that passed
    for fid in passed_ids:
        overlay[state.screened_frags == fid] = [0.0, 0.8, 0.0, 0.6]

    # Red: rejected / unconnected fragments
    for fid in rejected_ids:
        if fid > 0:
            overlay[state.screened_frags == fid] = [1.0, 0.0, 0.0, 0.6]

    ax.imshow(overlay)

    # Lime lines: all Dijkstra paths (before filtering)
    for path in state.paths.values():
        ax.plot(
            path.coords[:, 1], path.coords[:, 0],
            color="lime", lw=0.8, alpha=0.8,
        )

    n_paths = len(state.paths)
    n_passed = len(passed_ids)
    n_rejected = len(rejected_ids)
    ax.set_title(
        f"Fragments: {len(all_frag_ids)} total | "
        f"Paths: {n_paths} extracted, {n_passed} passed, "
        f"{n_rejected} rejected/unconnected"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_path_metrics(state: DiagnosticState) -> 'plt.Figure':
    """Histograms of path cost, length, and cost-per-pixel.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with 1x3 histogram layout.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    costs = [p.total_cost for p in state.paths.values()]
    lengths = [p.path_length for p in state.paths.values()]
    ratios = [
        p.total_cost / max(p.path_length, 1) for p in state.paths.values()
    ]

    ax = axes[0]
    ax.hist(costs, bins=40, edgecolor="black", alpha=0.7, color="tab:blue")
    ax.set_xlabel("Total cost")
    ax.set_ylabel("Count")
    ax.set_title("Path total cost")

    ax = axes[1]
    ax.hist(lengths, bins=40, edgecolor="black", alpha=0.7, color="tab:orange")
    ax.set_xlabel("Path length (px)")
    ax.set_ylabel("Count")
    ax.set_title("Path length")

    ax = axes[2]
    ax.hist(ratios, bins=40, edgecolor="black", alpha=0.7, color="tab:green")
    ax.set_xlabel("Cost / length")
    ax.set_ylabel("Count")
    ax.set_title("Cost per pixel")

    fig.tight_layout()
    return fig


def plot_cost_profiles(state: DiagnosticState) -> 'plt.Figure':
    """Line plots of cumulative cost along cheapest and most expensive paths.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with 1x2 layout showing cheapest and most expensive paths.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    if not state.paths:
        for ax in axes:
            ax.set_title("No paths")
            ax.set_xlabel("Step along path")
            ax.set_ylabel("Cumulative cost-distance")
        fig.tight_layout()
        return fig

    sorted_paths = sorted(state.paths.values(), key=lambda p: p.total_cost)

    # Left: 3 cheapest
    ax = axes[0]
    for p in sorted_paths[:3]:
        ax.plot(p.cost_profile, label=f"frag {p.fragment_id} (cost={p.total_cost:.1f})")
    ax.set_xlabel("Step along path")
    ax.set_ylabel("Cumulative cost-distance")
    ax.set_title("3 cheapest paths")
    ax.legend(fontsize=8)

    # Right: 3 most expensive
    ax = axes[1]
    for p in sorted_paths[-3:]:
        ax.plot(p.cost_profile, label=f"frag {p.fragment_id} (cost={p.total_cost:.1f})")
    ax.set_xlabel("Step along path")
    ax.set_ylabel("Cumulative cost-distance")
    ax.set_title("3 most expensive paths")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_filter_dashboard(state: DiagnosticState) -> 'plt.Figure':
    """Scatter plots and bar chart of filter metrics and rejections.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with 2x3 layout of filter analysis panels.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fr = state.filter_result
    metrics = fr.metrics
    thresholds = fr.thresholds

    if not metrics:
        for row in axes:
            for ax in row:
                ax.set_title("No metrics")
        fig.tight_layout()
        return fig

    passed = fr.passed_ids

    def _scatter(ax, x_attr, y_attr, x_label, y_label):
        for fid, m in metrics.items():
            color = "steelblue" if fid in passed else "red"
            marker = "o" if fid in passed else "x"
            s = 10 if fid in passed else 15
            ax.scatter(
                getattr(m, x_attr), getattr(m, y_attr),
                c=color, marker=marker, s=s,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # [0,0]: median_raw_cost vs max_window_cost
    ax = axes[0, 0]
    _scatter(ax, "median_raw_cost", "max_window_cost",
             "Median raw cost (F1)", "Max window cost (F2)")
    ax.axvline(thresholds.tau_median_cost, color="orange", ls="--", label="F1 threshold")
    ax.axhline(thresholds.tau_window_cost, color="green", ls="--", label="F2 threshold")
    ax.legend(fontsize=7)

    # [0,1]: median_raw_cost vs band_cost_variance
    ax = axes[0, 1]
    _scatter(ax, "median_raw_cost", "band_cost_variance",
             "Median raw cost (F1)", "Band cost variance (F3)")
    ax.axvline(thresholds.tau_median_cost, color="orange", ls="--", label="F1 threshold")
    ax.axhline(thresholds.tau_band_variance, color="purple", ls="--", label="F3 threshold")
    ax.legend(fontsize=7)

    # [0,2]: bar chart of per-filter rejection counts
    ax = axes[0, 2]
    filter_names = list(fr.per_filter_rejections.keys())
    counts = [len(fr.per_filter_rejections[k]) for k in filter_names]
    bar_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71", "#3498db"]
    colors = bar_colors[: len(filter_names)]
    ax.bar(range(len(filter_names)), counts, color=colors, edgecolor="black")
    ax.set_xticks(range(len(filter_names)))
    ax.set_xticklabels(filter_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rejection count")
    ax.set_title("Per-filter rejections")

    # [1,0]: median_raw_cost vs pct_energy_band_median
    ax = axes[1, 0]
    _scatter(ax, "median_raw_cost", "pct_energy_band_median",
             "Median raw cost (F1)", "PCT energy band median (F4)")
    ax.axhline(thresholds.tau_pct_energy_median, color="green", ls="--", label="F4 threshold")
    ax.legend(fontsize=7)

    # [1,1]: median_raw_cost vs gray_band_snr
    ax = axes[1, 1]
    _scatter(ax, "median_raw_cost", "gray_band_snr",
             "Median raw cost (F1)", "Gray band SNR (F5)")
    ax.axhline(thresholds.tau_gray_snr, color="purple", ls="--", label="F5 threshold")
    ax.legend(fontsize=7)

    # [1,2]: pct_energy_band_median vs gray_band_snr
    ax = axes[1, 2]
    _scatter(ax, "pct_energy_band_median", "gray_band_snr",
             "PCT energy band median (F4)", "Gray band SNR (F5)")
    ax.axvline(thresholds.tau_pct_energy_median, color="green", ls="--", label="F4 threshold")
    ax.axhline(thresholds.tau_gray_snr, color="purple", ls="--", label="F5 threshold")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_filter_spatial(state: DiagnosticState) -> 'plt.Figure':
    """Spatial map of per-filter rejections overlaid on the enhanced image.

    Args:
        state: Populated DiagnosticState from ``collect_diagnostic_state``.

    Returns:
        Figure with dynamic grid: one panel per filter + summary.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fr = state.filter_result
    filter_names = list(fr.per_filter_rejections.keys())
    n_panels = len(filter_names) + 1  # +1 for summary
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows),
        squeeze=False,
    )

    filter_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71", "#3498db"]

    # Per-filter panels
    for idx, fname in enumerate(filter_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        ax.imshow(state.enhanced_arr, cmap="gray", alpha=0.5)

        this_rejected = fr.per_filter_rejections[fname]
        other_rejected = fr.rejected_ids - this_rejected

        # Draw paths
        for fid, path in state.paths.items():
            if fid in this_rejected:
                color = "red"
            elif fid in other_rejected:
                color = "gray"
            else:
                color = "lime"
            ax.plot(path.coords[:, 1], path.coords[:, 0],
                    color=color, lw=0.8, alpha=0.7)

        ax.set_title(f"{fname} ({len(this_rejected)} rejected)")
        ax.axis("off")

    # Summary panel
    summary_idx = len(filter_names)
    row, col = divmod(summary_idx, n_cols)
    ax = axes[row, col]
    ax.imshow(state.enhanced_arr, cmap="gray", alpha=0.5)

    legend_handles = []
    for idx, fname in enumerate(filter_names):
        color = filter_colors[idx % len(filter_colors)]
        for fid in fr.per_filter_rejections[fname]:
            if fid in state.paths:
                path = state.paths[fid]
                ax.plot(path.coords[:, 1], path.coords[:, 0],
                        color=color, lw=0.8, alpha=0.7)
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=fname),
        )

    # Draw passed paths in lime
    for fid in fr.passed_ids:
        if fid in state.paths:
            path = state.paths[fid]
            ax.plot(path.coords[:, 1], path.coords[:, 0],
                    color="lime", lw=0.8, alpha=0.7)
    legend_handles.append(
        Line2D([0], [0], color="lime", lw=2, label="Passed"),
    )

    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_title("Summary: all filters")
    ax.axis("off")

    # Hide unused axes
    for idx in range(summary_idx + 1, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig
