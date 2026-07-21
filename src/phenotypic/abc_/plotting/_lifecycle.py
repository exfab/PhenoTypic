"""Fieldless lifecycle capabilities for plot producers."""

from __future__ import annotations

from ._pht_plot import PhtPlot

__all__ = ["PlotAnalysis", "PlotImage", "PlotMeas", "PlotQc"]


class PlotImage(PhtPlot):
    """Mark a plot for refresh after one image completes its pipeline stage."""

    _weakly_bind_subject = True


class PlotMeas(PhtPlot):
    """Mark a plot for refresh after the measurements mirror is updated."""


class PlotAnalysis(PhtPlot):
    """Mark a plot for refresh after its selected analysis input is updated."""


class PlotQc(PhtPlot):
    """Mark a plot for refresh after its selected input or QC state changes."""
