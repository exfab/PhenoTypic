"""Tests for the public plotting capability and lifecycle mixins."""

from __future__ import annotations

import inspect
import gc

import plotly.graph_objects as go
import pytest
from pydantic import BaseModel

from phenotypic.abc_.plotting import (
    BoundFigures,
    Control,
    FigureSpec,
    PhtPlot,
    PlotAnalysis,
    PlotImage,
    PlotMeas,
    PlotQc,
    figure,
)


SIGMA = Control(
    label="Sigma",
    kind="float",
    default=1.0,
    bounds=(0.0, 5.0),
)


class _MultiPlot(PhtPlot):
    @figure(title="First")
    def first(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[1], y=[1]))

    @figure(title="Primary", primary=True)
    def primary(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[2], y=[2]))


class _PlotModel(BaseModel, PlotImage):
    scale: int = 3

    @figure(title="Model", primary=True)
    def model_figure(self) -> go.Figure:
        return go.Figure(go.Bar(y=[self.scale]))


def test_public_lifecycle_hierarchy_is_fieldless() -> None:
    lifecycle_types = (PlotImage, PlotMeas, PlotAnalysis, PlotQc)

    assert all(issubclass(lifecycle, PhtPlot) for lifecycle in lifecycle_types)
    assert all(
        "__init__" not in vars(lifecycle) for lifecycle in lifecycle_types
    )
    assert all(
        "__annotations__" not in vars(lifecycle)
        for lifecycle in lifecycle_types
    )
    assert "__init__" not in vars(PhtPlot)
    assert "__annotations__" not in vars(PhtPlot)


def test_pydantic_schema_and_dump_ignore_lifecycle_mixin() -> None:
    model = _PlotModel(scale=5)

    assert set(_PlotModel.model_fields) == {"scale"}
    assert set(_PlotModel.model_json_schema()["properties"]) == {"scale"}
    assert model.model_dump() == {"scale": 5}
    assert _PlotModel.model_validate(model.model_dump()).scale == 5


def test_pht_plot_exposes_report_without_legacy_aliases() -> None:
    assert callable(PhtPlot.inspect)
    assert callable(PhtPlot.report)
    assert not hasattr(PhtPlot, "dash")
    assert not hasattr(PhtPlot, "dashboard")

    report_signature = inspect.signature(PhtPlot.report)
    assert list(report_signature.parameters) == [
        "self",
        "subject",
        "overrides",
    ]
    assert (
        report_signature.parameters["overrides"].kind
        is inspect.Parameter.VAR_KEYWORD
    )


def test_figure_validates_controls_and_detects_subject() -> None:
    with pytest.raises(ValueError, match="not a parameter"):

        class _Invalid(PhtPlot):
            @figure(title="Invalid", controls={"missing": SIGMA})
            def render(self, *, sigma: float = 1.0) -> go.Figure:
                return go.Figure()

    @figure(title="Subject", controls={"sigma": SIGMA})
    def render_subject(
        self: PhtPlot,
        image: object,
        *,
        sigma: float = 1.0,
    ) -> go.Figure:
        return go.Figure()

    spec = render_subject.__figure_spec__
    assert isinstance(spec, FigureSpec)
    assert spec.wants_subject is True
    assert spec.subject_param == "image"


def test_iter_figures_preserves_definition_and_override_order() -> None:
    class _Derived(_MultiPlot):
        @figure(title="Overridden", primary=True)
        def primary(self) -> go.Figure:
            return go.Figure(go.Scatter(x=[9], y=[9]))

    specs = _Derived().iter_figures()

    assert [spec.name for spec in specs] == ["first", "primary"]
    assert specs[1].title == "Overridden"


def test_inspect_uses_primary_and_control_defaults() -> None:
    class _Controlled(PhtPlot):
        @figure(title="Controlled", controls={"sigma": SIGMA}, primary=True)
        def controlled(self, *, sigma: float) -> go.Figure:
            return go.Figure(go.Scatter(y=[sigma]))

    plot = _Controlled()

    assert _MultiPlot().inspect().data[0].x == (2,)
    assert plot.inspect().data[0].y == (1.0,)
    assert plot.inspect(sigma=4.0).data[0].y == (4.0,)
    with pytest.raises(ValueError, match="unknown override"):
        plot.inspect(no_such_control=1)


def test_report_composes_control_free_figures() -> None:
    report = _MultiPlot().report()

    assert isinstance(report, go.Figure)
    assert len(report.data) == 2


def test_report_delegates_controlled_figures_to_notebook_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phenotypic.sdk_.viz.notebook import _adapter

    sentinel = object()
    subject = object()

    class _Controlled(PhtPlot):
        @figure(title="Controlled", controls={"sigma": SIGMA})
        def controlled(self, image: object, *, sigma: float) -> go.Figure:
            return go.Figure()

    def _fake_dashboard(provider: PhtPlot, bound_subject: object) -> object:
        assert isinstance(provider, _Controlled)
        assert bound_subject is subject
        return sentinel

    monkeypatch.setattr(_adapter, "build_notebook_dashboard", _fake_dashboard)

    assert _Controlled().report(subject) is sentinel


def test_bound_figures_render_without_retaining_figure_cache() -> None:
    plot = _MultiPlot()
    bound = plot.figures()
    spec = bound.specs()[0]

    assert isinstance(bound, BoundFigures)
    assert bound.render(spec) is not bound.render(spec)


def test_bound_image_subject_is_weak() -> None:
    class _Subject:
        pass

    class _ImageProvider(PlotImage):
        @figure(title="Image", primary=True)
        def image(self, subject: object) -> go.Figure:
            return go.Figure()

    subject = _Subject()
    bound = _ImageProvider().figures(subject)
    spec = bound.specs()[0]
    del subject
    gc.collect()

    with pytest.raises(RuntimeError, match="has been released"):
        bound.render(spec)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "float", "default": 1.0}, "float requires bounds"),
        (
            {"kind": "select", "default": "missing", "options": ("a",)},
            "not in options",
        ),
        ({"kind": "bool", "default": "yes"}, "bool default"),
        ({"kind": "text", "default": 1}, "text default"),
    ],
)
def test_control_rejects_invalid_kind_specific_defaults(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Control(label="Invalid", **kwargs)  # type: ignore[arg-type]
