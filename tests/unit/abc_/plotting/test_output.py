"""Backend identification shared by the decorator and the publisher."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
import pytest
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


def test_a_plotly_or_mpl_object_that_is_not_a_figure_is_rejected() -> None:
    """T1: the class-name half of the predicate, which had zero coverage.

    The three cases above are all `builtins`, so the MODULE half screens them
    and the name half is never reached. Deleting `type(figure).__name__ !=
    "Figure"` left the whole suite green. Without it every plotly and
    matplotlib object classifies as a figure -- and since _require_backend and
    FigureAdapter both delegate here, the one module both layers share would
    hand both a wrong answer silently.
    """
    assert figure_backend_of(go.Scatter()) is None
    assert figure_backend_of(MplFigure().add_subplot()) is None


def test_agrees_with_the_publisher_vocabulary() -> None:
    """The manifest keeps 'matplotlib'; the decorator uses 'mpl'."""
    from phenotypic.plotting._pipeline import FigureAdapter

    assert FigureAdapter.backend_name(go.Figure()) == "plotly"
    assert FigureAdapter.backend_name(MplFigure()) == "matplotlib"

    # T3: spec §5 asks for three arms; the unsupported one was missing.
    with pytest.raises(TypeError, match="unsupported figure type"):
        FigureAdapter.backend_name(object())
