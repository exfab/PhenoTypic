"""Pure good-vs-error distribution figure for the Error-analysis tab.

A vertical box+strip of the *good* baseline against the focused error
category for one measurement (value on the y-axis), with an **editable**
horizontal line shape at the suggested cutoff. The drag emits
``relayoutData`` carrying ``shapes[0].y0``/``y1`` so the readout callback
can recompute recall/specificity at any dragged position.

Palette rule (DESIGN.md): the good series uses the neutral data tone
``COLOR_INFO`` (OI_SKY); the error series uses
``category_color(category, custom_index)`` from the Okabe-Ito data
palette. :func:`phenotypic.viz.figures.apply_theme` is applied before the
figure is returned.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from phenotypic.gui._design import COLOR_INFO, category_color
from phenotypic.viz.figures import apply_theme

#: Display name of the good-baseline box trace.
_GOOD_TRACE_NAME = "Good kept"


def build_distribution_figure(
    good_values: np.ndarray,
    error_values: np.ndarray,
    measurement: str,
    category: str,
    cutoff: float,
    custom_index: int = 0,
) -> go.Figure:
    """Build the good-vs-error distribution figure for one measurement.

    Args:
        good_values: Good-baseline measurement values (NaN-tolerant).
        error_values: Error-category measurement values (NaN-tolerant).
        measurement: The focused measurement column name (y-axis title).
        category: The focused error-category token (error trace name +
            color).
        cutoff: The suggested cutoff — drawn as an editable horizontal
            line at ``y=cutoff``.
        custom_index: Registration index for a custom category, forwarded
            to :func:`category_color`. Ignored for core categories.

    Returns:
        A themed :class:`plotly.graph_objects.Figure` with two box traces
        (good + error) and a single editable horizontal cutoff line shape
        at index 0.
    """
    good = np.asarray(good_values, dtype=float)
    error = np.asarray(error_values, dtype=float)
    error_color = category_color(category, custom_index)

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=good,
            name=_GOOD_TRACE_NAME,
            marker_color=COLOR_INFO,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
        )
    )
    fig.add_trace(
        go.Box(
            y=error,
            name=category,
            marker_color=error_color,
            boxpoints="all",
            jitter=0.4,
            pointpos=0,
        )
    )

    # Editable horizontal cutoff line spanning the categorical x range.
    # Index 0 so the drag callback reads ``relayoutData["shapes[0].y0"]``.
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0.0,
        x1=1.0,
        yref="y",
        y0=cutoff,
        y1=cutoff,
        line={"color": error_color, "width": 2, "dash": "dash"},
        editable=True,
    )

    apply_theme(fig)
    fig.update_layout(
        yaxis={"title": {"text": measurement}},
        xaxis={"type": "category"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
    )
    return fig


__all__ = ["build_distribution_figure"]
