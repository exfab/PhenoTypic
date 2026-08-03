"""Private normalization for plot-provider runtime values."""

from __future__ import annotations

from typing import Any

from phenotypic.abc_.plotting import PlotOutput, PlotPage


def normalize_plot_output(value: Any | PlotOutput) -> PlotOutput:
    """Normalize a raw figure to the one-page runtime output contract."""
    if isinstance(value, PlotOutput):
        return value
    if value is None:
        return PlotOutput(pages=())
    return PlotOutput(pages=(PlotPage(key="default", figure=value),))


__all__ = ["normalize_plot_output"]
