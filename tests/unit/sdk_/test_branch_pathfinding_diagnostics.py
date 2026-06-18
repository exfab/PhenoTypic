"""Smoke tests for branch pathfinding diagnostic plotting helpers.

Exercises :func:`plot_paths_over_image`, :func:`plot_cost_distance_heatmap`,
and :func:`paths_metrics_dataframe` against hand-crafted tiny inputs. The
plotly backend is optional, so the full module is skipped when plotly is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go  # noqa: E402

from phenotypic.tools_.branch_pathfinding import (  # noqa: E402
    DijkstraResult,
    FilterResult,
    FilterThresholds,
    FragmentPath,
    PathMetrics,
    paths_metrics_dataframe,
    plot_cost_distance_heatmap,
    plot_paths_over_image,
)


# =====================================================================
# Fixtures
# =====================================================================


def _make_fragment_path(
    *,
    fragment_id: int = 1,
    colony_id: int = 1,
    coords: np.ndarray | None = None,
    cost_profile: np.ndarray | None = None,
) -> FragmentPath:
    """Build a minimal FragmentPath with a 5-pixel diagonal trajectory."""
    if coords is None:
        coords = np.array(
            [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]],
            dtype=np.int32,
        )
    if cost_profile is None:
        cost_profile = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    return FragmentPath(
        fragment_id=fragment_id,
        colony_id=colony_id,
        coords=coords,
        cost_profile=cost_profile,
        total_cost=float(cost_profile.sum()),
        path_length=len(coords),
    )


def _make_path_metrics() -> PathMetrics:
    return PathMetrics(
        median_raw_cost=0.3,
        max_window_cost=0.5,
        band_cost_variance=0.01,
        pct_energy_band_median=0.7,
        gray_band_snr=2.5,
    )


def _make_filter_result(passed_ids: set[int]) -> FilterResult:
    thresholds = FilterThresholds(
        tau_median_cost=1.0,
        tau_window_cost=1.0,
        tau_band_variance=0.5,
        tau_pct_energy_median=0.1,
        tau_gray_snr=1.0,
        k_iqr=1.5,
    )
    return FilterResult(
        passed_ids=passed_ids,
        rejected_ids=set(),
        per_filter_rejections={},
        metrics={1: _make_path_metrics()},
        thresholds=thresholds,
    )


# =====================================================================
# plot_paths_over_image
# =====================================================================


def test_plot_paths_over_image_scalar_bg():
    """Grayscale (H, W) background renders via go.Heatmap."""
    background = np.linspace(0.0, 1.0, 20 * 20, dtype=np.float64).reshape(20, 20)
    paths = {1: _make_fragment_path()}
    fig = plot_paths_over_image(background, paths)
    assert isinstance(fig, go.Figure)
    trace_types = [type(tr).__name__ for tr in fig.data]
    assert "Heatmap" in trace_types


def test_plot_paths_over_image_rgb_bg():
    """RGB (H, W, 3) background renders via go.Image."""
    background = np.zeros((20, 20, 3), dtype=np.uint8)
    background[..., 0] = 128
    paths = [_make_fragment_path()]  # list form exercises the list branch too
    fig = plot_paths_over_image(background, paths)
    assert isinstance(fig, go.Figure)
    trace_types = [type(tr).__name__ for tr in fig.data]
    assert "Image" in trace_types


def test_plot_paths_over_image_with_colony_labels():
    """Providing colony_labels adds an extra boundary trace."""
    background = np.zeros((20, 20), dtype=np.float64)
    paths = {1: _make_fragment_path()}

    fig_without = plot_paths_over_image(background, paths, colony_labels=None)

    colony_labels = np.zeros((20, 20), dtype=np.int32)
    colony_labels[10:18, 10:18] = 1
    fig_with = plot_paths_over_image(background, paths, colony_labels=colony_labels)

    assert isinstance(fig_with, go.Figure)
    assert len(fig_with.data) > len(fig_without.data)


# =====================================================================
# plot_cost_distance_heatmap
# =====================================================================


def test_plot_cost_distance_heatmap():
    """Builds a minimal DijkstraResult with some unreached pixels."""
    cost = np.full((10, 10), 1.5, dtype=np.float64)
    cost[0, 0] = np.inf  # unreached -> NaN
    cost[9, 9] = np.inf
    colony_id = np.zeros((10, 10), dtype=np.int32)
    colony_id[4:7, 4:7] = 1
    predecessor = np.full((10, 10), -1, dtype=np.int32)
    centroids = {1: (5.0, 5.0)}

    result = DijkstraResult(
        cost_distance=cost,
        colony_id=colony_id,
        predecessor=predecessor,
        colony_centroids=centroids,
    )

    fig = plot_cost_distance_heatmap(result)
    assert isinstance(fig, go.Figure)

    # With colony_labels overlay
    fig2 = plot_cost_distance_heatmap(result, colony_labels=colony_id)
    assert isinstance(fig2, go.Figure)
    assert len(fig2.data) >= len(fig.data)


# =====================================================================
# paths_metrics_dataframe
# =====================================================================


def test_paths_metrics_dataframe_minimal():
    """One path, no metrics/filter -> 7 always-present columns, 1 row."""
    pd = pytest.importorskip("pandas")

    paths = {1: _make_fragment_path()}
    df = paths_metrics_dataframe(paths)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    expected = {
        "path_id",
        "colony_id",
        "n_pixels",
        "total_cost",
        "mean_cost_profile",
        "max_cost_profile",
        "min_cost_profile",
    }
    assert expected.issubset(df.columns)
    assert set(df.columns) == expected
    row = df.iloc[0]
    assert int(row["path_id"]) == 1
    assert int(row["colony_id"]) == 1
    assert int(row["n_pixels"]) == 5


def test_paths_metrics_dataframe_with_metrics_and_filter():
    """Metrics + filter -> 13 columns (7 base + 5 metrics + passed)."""
    pd = pytest.importorskip("pandas")

    paths = {1: _make_fragment_path()}
    metrics = {1: _make_path_metrics()}
    filter_result = _make_filter_result(passed_ids={1})

    df = paths_metrics_dataframe(paths, metrics=metrics, filter_result=filter_result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert len(df.columns) == 13
    for col in (
        "median_raw_cost",
        "max_window_cost",
        "band_cost_variance",
        "pct_energy_band_median",
        "gray_band_snr",
        "passed",
    ):
        assert col in df.columns
    assert bool(df.iloc[0]["passed"]) is True
