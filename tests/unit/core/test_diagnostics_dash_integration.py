"""Integration: the diagnostics provider through the new protocol surfaces.

Exercises the Phase 2 wiring end-to-end — ``image.plot.dash.diagnostics()`` (the
sub-accessor → ``FigureProvider.dash()`` → ipywidgets shell, since diagnostics
declares controls) and ``DiagnosticsPlotter.inspect()`` (the primary figure).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
    DiagnosticsPlotter,
)
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture(scope="module")
def image():
    return load_synth_yeast_plate()


def test_plot_dash_diagnostics_returns_ipywidget(image):
    widgets = pytest.importorskip("ipywidgets")
    dashboard = image.plot.dash.diagnostics()
    assert isinstance(dashboard, widgets.Widget)


def test_diagnostics_inspect_returns_primary_figure(image):
    fig = DiagnosticsPlotter(image).inspect()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_diagnostics_matplotlib_path_preserved(image):
    fig, metrics = image.plot.diagnostics()
    # still the matplotlib dual renderer: a 2-tuple of (Figure, metrics dict)
    assert isinstance(metrics, dict)
    assert "quality_scores" in metrics
    plt.close(fig)  # free the matplotlib figure (per CLAUDE.md)
