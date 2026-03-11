"""Plotly helper functions for interactive image visualization.

Provides shared utilities for rendering images, colormaps, gridlines,
section boxes, and object labels using Plotly. Used by accessor-level
show methods throughout the codebase.
"""

from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    import plotly.graph_objects as go


# Mapping from matplotlib colormap names to plotly colorscale names
_MPL_TO_PLOTLY = {
    "gray": "gray",
    "viridis": "Viridis",
    "inferno": "Inferno",
    "plasma": "Plasma",
    "magma": "Magma",
    "hot": "Hot",
    "jet": "Jet",
    "RdBu": "RdBu",
    "coolwarm": "RdBu",
}

PLOTLY_CONFIG = {"scrollZoom": True}


def _require_plotly() -> None:
    """Raise ImportError if plotly is not installed."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install it with: pip install plotly>=6.0.0  "
            "or install the gui extras: pip install phenotypic[gui]"
        )


def mpl_cmap_to_plotly(cmap_name: str) -> str | list:
    """Convert a matplotlib colormap name to a plotly colorscale.

    Args:
        cmap_name: Name of a matplotlib colormap.

    Returns:
        A plotly colorscale string for known names, or a list of
        ``[position, "rgb(r,g,b)"]`` pairs sampled from the matplotlib
        colormap for unknown names.
    """
    _require_plotly()
    if cmap_name in _MPL_TO_PLOTLY:
        return _MPL_TO_PLOTLY[cmap_name]

    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    n = 256
    scale = []
    for i in range(n):
        rgba = cmap(i / (n - 1))
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        scale.append([i / (n - 1), f"rgb({r},{g},{b})"])
    return scale


def _auto_figsize(arr: np.ndarray) -> tuple[int, int]:
    """Calculate figure size in inches from array aspect ratio.

    Args:
        arr: Image array (2D or 3D).

    Returns:
        Tuple of (width_inches, height_inches).
    """
    height, width = arr.shape[:2]
    aspect_ratio = width / height

    best_figsize = (6, 6)
    best_error = float("inf")

    for h in range(6, 31):
        w = round(h * aspect_ratio)
        w = max(6, min(30, w))
        error = abs(w / h - aspect_ratio)
        if error < best_error or (
            error == best_error and w * h < best_figsize[0] * best_figsize[1]
        ):
            best_error = error
            best_figsize = (w, h)

    return best_figsize


def plotly_imshow(
    arr: np.ndarray,
    title: str | None = None,
    cmap: str = "gray",
    figsize: tuple[int, int] | None = None,
) -> go.Figure:
    """Render an image array as an interactive Plotly figure.

    Args:
        arr: 2D (grayscale) or 3D (RGB) image array. Supports uint8,
            uint16, and float dtypes.
        title: Figure title. Defaults to None.
        cmap: Matplotlib colormap name for 2D arrays. Ignored for 3D.
            Defaults to ``"gray"``.
        figsize: Figure size as ``(width_inches, height_inches)``.
            Converted to pixels at 100 dpi. If None, auto-calculated
            from the array aspect ratio.

    Returns:
        A ``plotly.graph_objects.Figure`` with zoom-friendly defaults.
    """
    _require_plotly()
    if figsize is None:
        figsize = _auto_figsize(arr)

    width_px = figsize[0] * 100
    height_px = figsize[1] * 100

    if arr.ndim == 3:
        fig = px.imshow(arr, binary_string=True)
    else:
        colorscale = mpl_cmap_to_plotly(cmap)
        fig = px.imshow(arr, color_continuous_scale=colorscale)

    fig.update_layout(
        width=width_px,
        height=height_px,
        dragmode="zoom",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
        title=title,
    )

    return fig


def add_plotly_gridlines(
    fig: go.Figure,
    col_edges: np.ndarray,
    row_edges: np.ndarray,
    ncols: int,
    nrows: int,
) -> None:
    """Add cyan dashed gridlines and row/column labels to a Plotly figure.

    Args:
        fig: Plotly figure to modify in-place.
        col_edges: Column edge pixel positions.
        row_edges: Row edge pixel positions.
        ncols: Number of grid columns.
        nrows: Number of grid rows.
    """
    _require_plotly()
    if len(col_edges) == 0 or len(row_edges) == 0:
        return

    y_min = float(row_edges.min())
    y_max = float(row_edges.max())
    x_min = float(col_edges.min())
    x_max = float(col_edges.max())

    # Vertical gridlines
    for x in col_edges:
        fig.add_shape(
            type="line",
            x0=float(x), x1=float(x),
            y0=y_min, y1=y_max,
            line=dict(color="cyan", dash="dash", width=1),
        )

    # Horizontal gridlines
    for y in row_edges:
        fig.add_shape(
            type="line",
            x0=x_min, x1=x_max,
            y0=float(y), y1=float(y),
            line=dict(color="cyan", dash="dash", width=1),
        )

    # Column number labels (top)
    if len(col_edges) > 1:
        upper = col_edges[1:]
        lower = col_edges[:-1]
        col_centers = ((upper - lower) // 2) + lower
        for i, cx in enumerate(col_centers):
            if i >= ncols:
                break
            fig.add_annotation(
                x=float(cx), y=y_min,
                text=str(i),
                showarrow=False,
                yshift=-15,
                font=dict(color="cyan", size=10),
            )

    # Row number labels (right)
    if len(row_edges) > 1:
        upper = row_edges[1:]
        lower = row_edges[:-1]
        row_centers = ((upper - lower) // 2) + lower
        for i, cy in enumerate(row_centers):
            if i >= nrows:
                break
            fig.add_annotation(
                x=x_max, y=float(cy),
                text=str(i),
                showarrow=False,
                xshift=15,
                font=dict(color="cyan", size=10),
            )


def add_plotly_section_boxes(fig: go.Figure, root_image) -> None:
    """Add colored bounding-box rectangles around grid sections.

    Args:
        fig: Plotly figure to modify in-place.
        root_image: A ``GridImage`` instance with ``grid`` accessor
            and ``objmap`` support.
    """
    _require_plotly()
    from phenotypic.measure import MeasureBounds
    from phenotypic.tools_.measurement_info_ import BBOX

    import matplotlib
    tab20 = matplotlib.colormaps["tab20"]
    colors = cycle(tab20(i) for i in range(tab20.N))

    img = root_image.copy()
    img.objmap = root_image.grid.get_section_map()
    gs_table = MeasureBounds().measure(img)

    for obj_label in gs_table.index.unique():
        subtable = gs_table.loc[obj_label, :]
        min_rr = subtable.loc[str(BBOX.MIN_RR)]
        max_rr = subtable.loc[str(BBOX.MAX_RR)]
        min_cc = subtable.loc[str(BBOX.MIN_CC)]
        max_cc = subtable.loc[str(BBOX.MAX_CC)]

        rgba = next(colors)
        color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"

        fig.add_shape(
            type="rect",
            x0=float(min_cc), x1=float(max_cc),
            y0=float(min_rr), y1=float(max_rr),
            line=dict(color=color_str, width=2),
        )


def add_plotly_obj_labels(
    fig: go.Figure,
    root_image,
    object_label: int | None = None,
    color: str = "white",
    size: int = 10,
    bgcolor: str = "black",
) -> None:
    """Add centroid labels for detected objects to a Plotly figure.

    Args:
        fig: Plotly figure to modify in-place.
        root_image: Image instance with ``objects`` accessor providing
            ``props`` and ``labels``.
        object_label: If set, only label this specific object.
            If None, label all objects.
        color: Label text color.
        size: Label font size.
        bgcolor: Label background color.
    """
    _require_plotly()
    props = root_image.objects.props
    for i, label in enumerate(root_image.objects.labels):
        if object_label is not None and object_label != label:
            continue
        rr, cc = props[i].centroid
        fig.add_annotation(
            x=cc,
            y=rr,
            text=str(label),
            showarrow=False,
            font=dict(color=color, size=size),
            bgcolor=bgcolor,
            opacity=0.6,
        )
