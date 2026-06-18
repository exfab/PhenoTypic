"""Diagnostic plotting helpers for branch pathfinding.

Interactive plotly overlays and tabular summaries for Dijkstra-based
branch reconstruction. All plotting functions import plotly lazily so
that the pathfinding subpackage (hot-path for
:class:`~phenotypic.detect.FilamentousFungiDetector`) does not pull in
plotly at import time.

The three public functions cover the common diagnostic cases:

- :func:`plot_paths_over_image` overlays fragment-to-colony paths on an
  image background.
- :func:`plot_cost_distance_heatmap` renders the accumulated Dijkstra
  cost surface with optional colony boundaries.
- :func:`paths_metrics_dataframe` summarizes paths (and optional
  structure-quality metrics / filter results) as a per-path DataFrame.

Style follows the Okabe-Ito palette used throughout PhenoTypic
diagnostics (see ``_measure_radial_expansion.py`` for the reference
pattern). Many-path scatters are built as a single NaN-separated
``go.Scattergl`` trace to keep browsers responsive at 100+ paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._dataclasses import DijkstraResult, FilterResult, FragmentPath, PathMetrics

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import plotly.graph_objects as go  # noqa: F401


def _iter_paths(
    paths: dict[int, FragmentPath] | list[FragmentPath],
):
    """Yield ``FragmentPath`` objects regardless of container type.

    Args:
        paths: Either a dict keyed by path id or an iterable list.

    Yields:
        Each :class:`FragmentPath` in the container.
    """
    if isinstance(paths, dict):
        yield from paths.values()
    else:
        yield from paths


def _colony_boundary_coords(colony_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) of inner colony boundary pixels.

    Args:
        colony_labels: Integer label array (H, W). Non-zero labels denote
            colonies; zero is background.

    Returns:
        Tuple ``(xs, ys)`` of column/row coordinates along the inner
        boundary of ``colony_labels > 0``.
    """
    from skimage.segmentation import find_boundaries

    boundary = find_boundaries(colony_labels > 0, mode="inner", connectivity=2)
    coords = np.argwhere(boundary)
    if coords.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return coords[:, 1].astype(np.float64), coords[:, 0].astype(np.float64)


def plot_paths_over_image(
    background: np.ndarray,
    paths: dict[int, FragmentPath] | list[FragmentPath],
    *,
    colony_labels: np.ndarray | None = None,
    title: str = "Dijkstra paths",
    path_color: str = "#E69F00",  # Okabe-Ito orange
    colony_boundary_color: str = "#56B4E9",  # Okabe-Ito sky
    show_fragment_seeds: bool = True,
    fig=None,
):
    """Overlay Dijkstra paths on a background image.

    ``background`` is rendered via ``go.Image`` if it is an RGB array
    (H, W, 3) and via ``go.Heatmap`` if it is scalar (H, W). Paths are
    drawn as a single NaN-separated ``go.Scattergl`` trace (keeps the
    browser responsive at 100+ paths). When ``colony_labels`` is given,
    inner colony boundaries are overlaid in ``colony_boundary_color``.
    When ``show_fragment_seeds`` is True, the first coord of each path
    (the fragment seed) is marked.

    Args:
        background: Either an (H, W, 3) RGB uint8 array or an (H, W)
            scalar array (e.g. grayscale, PCT energy, cost surface).
        paths: Fragment-to-colony paths, either as a dict keyed by
            fragment id or a list of :class:`FragmentPath`.
        colony_labels: Optional (H, W) integer label array. When given,
            inner colony boundaries are drawn.
        title: Figure title.
        path_color: Hex colour for path lines (default Okabe-Ito orange).
        colony_boundary_color: Hex colour for colony boundary markers
            (default Okabe-Ito sky blue).
        show_fragment_seeds: If True, mark the first coord of every path
            (the fragment seed) with a distinct marker.
        fig: Optional existing ``plotly.graph_objects.Figure`` to add the
            traces to. When ``None`` a new figure is created.

    Returns:
        A ``plotly.graph_objects.Figure`` with the background, path
        scatter, optional colony boundary, and optional fragment-seed
        markers.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If ``background`` is neither (H, W) nor (H, W, 3).
    """
    from phenotypic.sdk_._plotly_helpers import _require_plotly

    _require_plotly()
    import plotly.graph_objects as go

    if fig is None:
        fig = go.Figure()

    # Background layer
    if background.ndim == 3 and background.shape[-1] == 3:
        fig.add_trace(
            go.Image(
                z=background,
                name="Background",
                hoverinfo="skip",
            )
        )
    elif background.ndim == 2:
        fig.add_trace(
            go.Heatmap(
                z=background,
                colorscale="Gray",
                showscale=False,
                name="Background",
                hoverinfo="skip",
            )
        )
    else:
        raise ValueError(
            f"background must be (H, W) or (H, W, 3); got shape {background.shape}"
        )

    # Path polylines — single NaN-separated Scattergl trace
    path_xs: list[float] = []
    path_ys: list[float] = []
    seed_xs: list[float] = []
    seed_ys: list[float] = []
    for path in _iter_paths(paths):
        coords = path.coords
        if coords is None or len(coords) == 0:
            continue
        path_xs.extend(coords[:, 1].astype(np.float64).tolist())
        path_ys.extend(coords[:, 0].astype(np.float64).tolist())
        path_xs.append(float("nan"))
        path_ys.append(float("nan"))
        if show_fragment_seeds:
            seed_xs.append(float(coords[0, 1]))
            seed_ys.append(float(coords[0, 0]))

    if path_xs:
        fig.add_trace(
            go.Scattergl(
                x=path_xs,
                y=path_ys,
                mode="lines",
                line=dict(color=path_color, width=2),
                name="Paths",
                hoverinfo="skip",
            )
        )

    # Colony boundary overlay (inner boundary as scatter of sparse pixels)
    if colony_labels is not None:
        bx, by = _colony_boundary_coords(colony_labels)
        if bx.size > 0:
            fig.add_trace(
                go.Scattergl(
                    x=bx,
                    y=by,
                    mode="markers",
                    marker=dict(color=colony_boundary_color, size=2),
                    name="Colony boundary",
                    hoverinfo="skip",
                )
            )

    # Fragment seed markers
    if show_fragment_seeds and seed_xs:
        fig.add_trace(
            go.Scatter(
                x=seed_xs,
                y=seed_ys,
                mode="markers",
                marker=dict(
                    color=path_color,
                    size=7,
                    symbol="circle",
                    line=dict(color="white", width=0.5),
                ),
                name="Fragment seeds",
                hoverinfo="skip",
            )
        )

    # Layout: image-style axes (y reversed so origin is top-left)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_cost_distance_heatmap(
    dijkstra: DijkstraResult,
    *,
    colony_labels: np.ndarray | None = None,
    log_scale: bool = True,
    colorscale: str = "Viridis",
    title: str = "Cost-distance map",
):
    """Render the Dijkstra cost-distance map as a heatmap.

    Unreached pixels (``cost_distance == inf``) are masked to NaN so
    plotly colours them transparently. When ``log_scale`` is True the
    values are ``np.log1p``-transformed for visibility (colour-bar ticks
    are left in the transformed scale — document this in the annotation).
    When ``colony_labels`` is given, inner colony boundaries are overlaid.

    Args:
        dijkstra: :class:`DijkstraResult` produced by
            :func:`run_multisource_dijkstra`.
        colony_labels: Optional (H, W) integer label array. When given,
            inner boundaries are overlaid in Okabe-Ito sky blue.
        log_scale: If True, apply ``np.log1p`` to the finite cost values
            before plotting (recommended — raw Dijkstra cost ranges are
            typically long-tailed).
        colorscale: Plotly colorscale name (default ``"Viridis"``).
        title: Figure title.

    Returns:
        A ``plotly.graph_objects.Figure`` containing the heatmap and
        optional colony-boundary overlay.

    Raises:
        ImportError: If plotly is not installed.
    """
    from phenotypic.sdk_._plotly_helpers import _require_plotly

    _require_plotly()
    import plotly.graph_objects as go

    cost = np.asarray(dijkstra.cost_distance, dtype=np.float64).copy()
    # Mask unreached pixels to NaN for transparent rendering.
    cost[~np.isfinite(cost)] = np.nan

    colorbar_title = "cost"
    if log_scale:
        # log1p handles zero safely; NaNs propagate.
        with np.errstate(invalid="ignore"):
            cost = np.log1p(cost)
        colorbar_title = "log1p(cost)"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=cost,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            name="Cost distance",
            hoverinfo="z",
        )
    )

    if colony_labels is not None:
        bx, by = _colony_boundary_coords(colony_labels)
        if bx.size > 0:
            fig.add_trace(
                go.Scattergl(
                    x=bx,
                    y=by,
                    mode="markers",
                    marker=dict(color="#56B4E9", size=2),
                    name="Colony boundary",
                    hoverinfo="skip",
                )
            )

    annotation_text = (
        "Values shown on log1p(cost) scale; colorbar ticks are in log1p units."
        if log_scale
        else "Unreached pixels rendered transparent (NaN)."
    )
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        annotations=[
            dict(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.02,
                showarrow=False,
                font=dict(size=10),
            )
        ],
    )
    return fig


def paths_metrics_dataframe(
    paths: dict[int, FragmentPath] | list[FragmentPath],
    metrics: dict[int, PathMetrics] | None = None,
    filter_result: FilterResult | None = None,
) -> pd.DataFrame:
    """Summarize paths as a per-path DataFrame.

    Columns (always): path_id, colony_id, n_pixels, total_cost,
    mean_cost_profile, max_cost_profile, min_cost_profile.

    Additional columns when ``metrics`` is given: median_raw_cost,
    max_window_cost, band_cost_variance, pct_energy_band_median,
    gray_band_snr.

    Additional column when ``filter_result`` is given: passed (bool).

    Args:
        paths: Fragment-to-colony paths, either as a dict keyed by
            fragment id or a list of :class:`FragmentPath`.
        metrics: Optional mapping ``fragment_id -> PathMetrics`` (e.g.
            the ``metrics`` field of a :class:`FilterResult`). When
            provided, per-path structure-quality columns are appended.
        filter_result: Optional :class:`FilterResult`. When provided, a
            ``passed`` boolean column is appended (True if the fragment
            id is in ``filter_result.passed_ids``).

    Returns:
        A :class:`pandas.DataFrame` with one row per input path.
    """
    import pandas as pd

    records: list[dict] = []
    for path in _iter_paths(paths):
        cost_profile = np.asarray(path.cost_profile, dtype=np.float64)
        if cost_profile.size > 0:
            mean_cost = float(np.mean(cost_profile))
            max_cost = float(np.max(cost_profile))
            min_cost = float(np.min(cost_profile))
        else:
            mean_cost = float("nan")
            max_cost = float("nan")
            min_cost = float("nan")

        row: dict = {
            "path_id": int(path.fragment_id),
            "colony_id": int(path.colony_id),
            "n_pixels": int(path.path_length),
            "total_cost": float(path.total_cost),
            "mean_cost_profile": mean_cost,
            "max_cost_profile": max_cost,
            "min_cost_profile": min_cost,
        }

        if metrics is not None:
            pm = metrics.get(int(path.fragment_id))
            if pm is not None:
                row["median_raw_cost"] = float(pm.median_raw_cost)
                row["max_window_cost"] = float(pm.max_window_cost)
                row["band_cost_variance"] = float(pm.band_cost_variance)
                row["pct_energy_band_median"] = float(pm.pct_energy_band_median)
                row["gray_band_snr"] = float(pm.gray_band_snr)
            else:
                row["median_raw_cost"] = float("nan")
                row["max_window_cost"] = float("nan")
                row["band_cost_variance"] = float("nan")
                row["pct_energy_band_median"] = float("nan")
                row["gray_band_snr"] = float("nan")

        if filter_result is not None:
            row["passed"] = int(path.fragment_id) in filter_result.passed_ids

        records.append(row)

    return pd.DataFrame(records)
