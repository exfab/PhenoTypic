"""Unit tests for the ``image.plot.dash`` sub-accessor (Phase 2, strand B).

Uses a temporarily-registered dummy control-free ``FigureProvider`` plotter so
the dispatch logic is exercised without depending on any concrete plotter.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from phenotypic.abc_ import FigureProvider, figure
from phenotypic._core._image_parts.plot_accessor._base_plotter import BasePlotter
from phenotypic.data import load_synth_yeast_plate
from phenotypic.tools_.register import register_plotter
from phenotypic.tools_.register._plotter_registry import PlotterRegistry


class _DummyDashPlotter(BasePlotter, FigureProvider):
    """A control-free FigureProvider plotter for dispatch tests."""

    call_name = "dummy_dash"

    @figure(title="Dummy")
    def dummy_dash(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))


@pytest.fixture
def registered_dummy():
    register_plotter(_DummyDashPlotter)
    try:
        yield
    finally:
        PlotterRegistry._REGISTRY.pop("dummy_dash", None)


@pytest.fixture(scope="module")
def image():
    return load_synth_yeast_plate()


def test_dash_dispatches_to_control_free_provider(image, registered_dummy):
    fig = image.plot.dash.dummy_dash()
    assert isinstance(fig, go.Figure)  # control-free → composed go.Figure
    assert len(fig.data) == 1


def test_dash_accessor_is_cached(image):
    assert image.plot.dash is image.plot.dash


def test_dash_rejects_non_figure_provider(image):
    # 'all' → AllDataPlotter, a plain BasePlotter (not a FigureProvider)
    with pytest.raises(AttributeError, match="not a\n?.*FigureProvider|does not support"):
        _ = image.plot.dash.all


def test_dash_unknown_name_raises(image):
    with pytest.raises(AttributeError, match="no interactive plotter"):
        _ = image.plot.dash.definitely_not_a_plotter


def test_dash_dir_lists_dashable(image, registered_dummy):
    assert "dummy_dash" in dir(image.plot.dash)
