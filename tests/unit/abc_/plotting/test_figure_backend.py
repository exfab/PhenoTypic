"""The @figure backend declaration: required, checked, and routed."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MplFigure

from phenotypic.abc_.plotting import Control, PhtPlot, figure


def test_backend_is_required() -> None:
    with pytest.raises(TypeError, match="backend"):
        # NO backend= ON PURPOSE: omitting it is what this test asserts.
        # Annotating this line makes the test pass while proving nothing.
        # This is the one unannotated @figure site in the repo, by design.
        @figure(title="No backend")  # type: ignore[call-arg]
        def _fig(self, subject):  # pragma: no cover - never called
            return go.Figure()


def test_an_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        @figure(title="Bad", backend="svg")  # type: ignore[arg-type]
        def _fig(self, subject):  # pragma: no cover - never called
            return go.Figure()


class _PlotlyDeclaredReturnsMpl(PhtPlot):
    @figure(title="Wrong way round", backend="plotly", primary=True)
    def wrong(self, subject):
        return MplFigure()


class _MplDeclaredReturnsPlotly(PhtPlot):
    @figure(title="Also wrong", backend="mpl", primary=True)
    def wrong(self, subject):
        return go.Figure()


class _ReturnsNothing(PhtPlot):
    @figure(title="Nothing", backend="plotly", primary=True)
    def nothing(self, subject):
        return None


def test_plotly_declared_method_returning_matplotlib_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        _PlotlyDeclaredReturnsMpl().inspect(object())
    message = str(excinfo.value)
    assert "wrong" in message           # names the method
    assert "'plotly'" in message        # names the declaration
    assert "matplotlib.figure.Figure" in message   # names what came back
    # T2: the REMEDY half. Every assertion above is satisfied by the diagnosis
    # alone, so `other = declared` left this test green -- telling the author to
    # declare the backend they already declared.
    assert "Declare backend='mpl'" in message


def test_mpl_declared_method_returning_plotly_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        _MplDeclaredReturnsPlotly().inspect(object())
    message = str(excinfo.value)
    assert "'mpl'" in message
    assert "plotly" in message
    # T2, mirror: "plotly" above matches the actual-type clause and says
    # nothing about the remedy.
    assert "Declare backend='plotly'" in message


def test_a_non_figure_return_raises_the_same_error() -> None:
    with pytest.raises(TypeError, match="NoneType"):
        _ReturnsNothing().inspect(object())


class _Plotly(PhtPlot):
    @figure(title="Good plotly", backend="plotly", primary=True)
    def good(self, subject):
        return go.Figure(go.Scatter(y=[1, 2, 3]))


class _Mpl(PhtPlot):
    """Asserts the rcParams are live DURING construction, not after."""

    observed_cycle: object = None

    @figure(title="Good mpl", backend="mpl", primary=True)
    def good(self, subject):
        import matplotlib as mpl

        type(self).observed_cycle = mpl.rcParams["axes.prop_cycle"]
        return MplFigure()


def test_a_plotly_figure_is_themed_after_construction() -> None:
    """The assertion must FAIL if apply_theme is skipped.

    `fig.layout.template is not None` is true of any go.Figure, themed or
    not -- measured: `[Q7] untheme template is None? False`. The font family
    is the discriminator: None when raw, the DESIGN.md stack when themed.
    """
    from phenotypic.sdk_.viz.figures import FONT_FAMILY_MONO

    # Control: an untouched figure carries no base font family at all.
    assert go.Figure().layout.template.layout.font.family is None

    fig = _Plotly().inspect(object())
    # FONT_FAMILY_MONO, not FONT_FAMILY: _theme.py:171 sets the template's BASE
    # font to the mono stack per DESIGN.md "02", and applies FONT_FAMILY to
    # titles and legend separately (:174, :180, :187, :189). Measured.
    assert fig.layout.template.layout.font.family == FONT_FAMILY_MONO


def test_the_mpl_theme_is_live_while_the_figure_is_built() -> None:
    import matplotlib as mpl
    from phenotypic.sdk_.viz.figures import phenotypic_rc

    before = mpl.rcParams["axes.prop_cycle"]
    _Mpl().inspect(object())
    after = mpl.rcParams["axes.prop_cycle"]

    # Themed inside the method body...
    assert _Mpl.observed_cycle == phenotypic_rc()["axes.prop_cycle"]
    # ...and scoped: the caller's global rcParams are untouched.
    assert before == after


class _AllMpl(PhtPlot):
    @figure(title="One", backend="mpl", primary=True)
    def one(self, subject):
        return MplFigure()


class _Mixed(PhtPlot):
    @figure(title="P", backend="plotly", primary=True)
    def p(self, subject):
        return go.Figure()

    @figure(title="M", backend="mpl")
    def m(self, subject):
        return MplFigure()


def test_report_refuses_an_all_matplotlib_provider() -> None:
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _AllMpl().report(object())


def test_report_refuses_a_mixed_provider() -> None:
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _Mixed().report(object())


def test_inspect_still_works_on_a_matplotlib_provider() -> None:
    """The limitation is composition, not rendering."""
    assert isinstance(_AllMpl().inspect(object()), MplFigure)


class _MplWithControls(PhtPlot):
    """B6: this provider never reaches _compose_control_free_figure."""

    @figure(
        title="Controlled",
        backend="mpl",
        primary=True,
        controls={"sigma": Control(label="s", kind="float", default=1.0,
                                   bounds=(0.0, 2.0))},
    )
    def controlled(self, subject, *, sigma: float = 1.0):
        return MplFigure()


def test_report_refuses_a_matplotlib_provider_that_declares_controls() -> None:
    """Guards the notebook-dashboard path, which bypasses the composer."""
    with pytest.raises(TypeError, match="cannot compose matplotlib"):
        _MplWithControls().report(object())
