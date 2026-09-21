"""Backend identification shared by the decorator and the publisher."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
from matplotlib.figure import Figure as MplFigure

from phenotypic.abc_.plotting import figure_backend_of


def test_identifies_a_plotly_figure() -> None:
    assert figure_backend_of(go.Figure()) == "plotly"


def test_identifies_a_matplotlib_figure() -> None:
    assert figure_backend_of(MplFigure()) == "mpl"


def test_returns_none_for_an_unsupported_object() -> None:
    assert figure_backend_of(object()) is None
    assert figure_backend_of(None) is None
    assert figure_backend_of("not a figure") is None


def test_agrees_with_the_publisher_vocabulary() -> None:
    """The manifest keeps 'matplotlib'; the decorator uses 'mpl'."""
    from phenotypic.plotting._pipeline import FigureAdapter

    assert FigureAdapter.backend_name(go.Figure()) == "plotly"
    assert FigureAdapter.backend_name(MplFigure()) == "matplotlib"
