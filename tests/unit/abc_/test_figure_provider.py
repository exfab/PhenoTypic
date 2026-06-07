"""Unit tests for the figure protocol contract (``abc_/_figure_provider.py``).

Covers: Control validation, @figure signature/control-key validation,
wants_subject detection, theme auto-application, iter_figures ordering,
identity-based control dedup, inspect() primary selection, and — critically —
pydantic-safety (the mixin adds no fields and leaves schema/serialization intact).
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
from pydantic import BaseModel

from phenotypic.abc_ import (
    BoundFigures,
    Control,
    FigureProvider,
    FigureSpec,
    figure,
)
from phenotypic.viz.figures._theme import OKABE_ITO


# -- Control validation -----------------------------------------------------


class TestControl:
    def test_float_requires_bounds(self):
        with pytest.raises(ValueError, match="float requires bounds"):
            Control(label="sigma", kind="float", default=1.0)

    def test_float_default_within_bounds(self):
        with pytest.raises(ValueError, match="outside bounds"):
            Control(
                label="sigma", kind="float", default=9.0, bounds=(0.0, 5.0)
            )
        # valid
        Control(label="sigma", kind="float", default=1.0, bounds=(0.0, 5.0))

    def test_select_requires_options_and_member_default(self):
        with pytest.raises(ValueError, match="requires non-empty options"):
            Control(label="m", kind="select", default="a", options=())
        with pytest.raises(ValueError, match="not in options"):
            Control(label="m", kind="select", default="z", options=("a", "b"))
        Control(label="m", kind="select", default="a", options=("a", "b"))

    def test_bool_and_text_defaults(self):
        with pytest.raises(ValueError, match="bool default"):
            Control(label="b", kind="bool", default="yes")
        with pytest.raises(ValueError, match="text default"):
            Control(label="t", kind="text", default=3)
        Control(label="b", kind="bool", default=True)
        Control(label="t", kind="text", default="hi")

    def test_frozen(self):
        c = Control(label="b", kind="bool", default=True)
        with pytest.raises(Exception):
            c.default = False  # type: ignore[misc]


# -- @figure decorator ------------------------------------------------------


SIGMA = Control(label="Sigma", kind="float", default=1.0, bounds=(0.0, 5.0))
METHOD = Control(
    label="Method", kind="select", default="a", options=("a", "b")
)


class TestFigureDecorator:
    def test_rejects_unknown_control_key(self):
        with pytest.raises(ValueError, match="not a parameter"):

            class _Bad:  # noqa: D401
                @figure(title="x", controls={"nope": SIGMA})
                def plot(self, *, sigma) -> go.Figure:  # missing 'nope'
                    return go.Figure()

    def test_wants_subject_detection_operation_shape(self):
        @figure(title="overlay", controls={"base_layer": METHOD})
        def inspect(
            self, image=None, *, base_layer="a", for_save=False
        ) -> go.Figure:
            return go.Figure()

        spec = inspect.__figure_spec__
        assert spec.wants_subject is True
        assert spec.subject_param == "image"

    def test_wants_subject_detection_helper_shape(self):
        @figure(title="ridge", controls={"sigma": SIGMA})
        def plot_ridge(self, *, sigma) -> go.Figure:
            return go.Figure()

        spec = plot_ridge.__figure_spec__
        assert spec.wants_subject is False
        assert spec.subject_param is None

    def test_applies_theme_on_direct_call(self):
        class Helper(FigureProvider):
            @figure(title="t")
            def plot(self) -> go.Figure:
                return go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))

        fig = Helper().plot()
        assert isinstance(fig, go.Figure)
        # theme applied → the merged template carries the phenotypic colorway
        assert tuple(fig.layout.template.layout.colorway) == OKABE_ITO
        # traces preserved through the wrapper
        assert len(fig.data) == 1


# -- FigureProvider introspection / rendering -------------------------------


class _MultiHelper(FigureProvider):
    """Helper with several figures in a non-alphabetical definition order."""

    @figure(title="Zeta", section="s1")
    def zeta(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[1], y=[1]))

    @figure(title="Alpha", section="s1", primary=True)
    def alpha(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[2], y=[2]))

    @figure(title="Mu", section="s2")
    def mu(self) -> go.Figure:
        return go.Figure(go.Scatter(x=[3], y=[3]))


class TestFigureProvider:
    def test_iter_figures_in_definition_order(self):
        names = [s.name for s in _MultiHelper().iter_figures()]
        assert names == ["zeta", "alpha", "mu"]  # definition, not alphabetical

    def test_override_keeps_position(self):
        class Base(FigureProvider):
            @figure(title="A")
            def a(self) -> go.Figure:
                return go.Figure(go.Scatter(x=[1], y=[1]))

            @figure(title="B")
            def b(self) -> go.Figure:
                return go.Figure(go.Scatter(x=[2], y=[2]))

            @figure(title="C")
            def c(self) -> go.Figure:
                return go.Figure(go.Scatter(x=[3], y=[3]))

        class Derived(Base):
            @figure(title="B-override")
            def b(self) -> go.Figure:  # overrides Base.b, defined much later
                return go.Figure(go.Scatter(x=[9], y=[9]))

        specs = Derived().iter_figures()
        assert [s.name for s in specs] == ["a", "b", "c"]  # b stays in slot 2
        by_name = {s.name: s for s in specs}
        assert by_name["b"].title == "B-override"  # most-derived wins

    def test_undecorated_override_removes_base_figure(self):
        class Base(FigureProvider):
            @figure(title="A", controls={"method": METHOD})
            def a(self, *, method) -> go.Figure:
                return go.Figure(go.Scatter(name=method))

        class Derived(Base):
            def a(self) -> go.Figure:
                return go.Figure(go.Scatter(name="plain override"))

        derived = Derived()
        assert derived.iter_figures() == []
        with pytest.raises(RuntimeError, match="declares no @figure"):
            derived.inspect()

    def test_multiple_inheritance_order_uses_selected_base_slot(self):
        class Left(FigureProvider):
            @figure(title="left-plot")
            def plot(self) -> go.Figure:
                return go.Figure()

            @figure(title="left-after")
            def after(self) -> go.Figure:
                return go.Figure()

        class Right(FigureProvider):
            @figure(title="right-plot")
            def plot(self) -> go.Figure:
                return go.Figure()

        class Derived(Left, Right):
            pass

        specs = Derived().iter_figures()

        assert [(spec.name, spec.title) for spec in specs] == [
            ("plot", "left-plot"),
            ("after", "left-after"),
        ]

    def test_inspect_picks_primary(self):
        fig = _MultiHelper().inspect()
        assert isinstance(fig, go.Figure)
        # primary is 'alpha' → its single point x=2
        assert fig.data[0].x == (2,)

    def test_inspect_single_figure_without_primary(self):
        class One(FigureProvider):
            @figure(title="only")
            def only(self) -> go.Figure:
                return go.Figure(go.Scatter(x=[7], y=[7]))

        assert One().inspect().data[0].x == (7,)

    def test_inspect_ambiguous_raises(self):
        class Two(FigureProvider):
            @figure(title="a")
            def a(self) -> go.Figure:
                return go.Figure()

            @figure(title="b")
            def b(self) -> go.Figure:
                return go.Figure()

        with pytest.raises(RuntimeError, match="primary"):
            Two().inspect()

    def test_inspect_rejects_non_control_overrides(self):
        """inspect() overrides must be declared controls; a stray kwarg (incl.
        the subject param name) raises ValueError, not a cryptic TypeError."""

        class Prov(FigureProvider):
            @figure(title="Main", primary=True)
            def fig_main(
                self, image=None
            ) -> go.Figure:  # image == subject param
                return go.Figure()

        prov = Prov()
        with pytest.raises(ValueError, match="unknown override"):
            prov.inspect(not_a_control=1)
        with pytest.raises(ValueError, match="unknown override"):
            prov.inspect(
                image="passing the subject by keyword is not an override"
            )

    def test_dash_control_free_returns_composed_figure(self):
        fig = _MultiHelper().dash()
        assert isinstance(fig, go.Figure)
        # all three single-point traces composed into subplots
        assert len(fig.data) == 3

    def test_dash_single_control_free_returns_figure_as_is(self):
        class One(FigureProvider):
            @figure(title="only")
            def only(self) -> go.Figure:
                fig = go.Figure(go.Scatter(x=[1], y=[1]))
                fig.update_layout(title="kept")
                return fig

        out = One().dash()
        # single figure returned directly → its own layout survives (not re-wrapped)
        assert out.layout.title.text == "kept"

    def test_figures_returns_bound_caching_object(self):
        helper = _MultiHelper()
        bound = helper.figures()
        assert isinstance(bound, BoundFigures)
        spec = bound.specs()[0]
        f1 = bound.render(spec)
        f2 = bound.render(spec)
        assert f1 is f2  # cached by (name, values)


# -- identity-based control dedup -------------------------------------------


class TestControlIdentity:
    def test_shared_instance_is_one_control(self):
        shared = Control(
            label="S", kind="float", default=1.0, bounds=(0.0, 2.0)
        )

        class P(FigureProvider):
            @figure(title="a", controls={"sigma": shared})
            def a(self, *, sigma) -> go.Figure:
                return go.Figure()

            @figure(title="b", controls={"sigma": shared})
            def b(self, *, sigma) -> go.Figure:
                return go.Figure()

        from phenotypic.viz.notebook._adapter import unique_controls

        specs = P().iter_figures()
        assert len(unique_controls(specs)) == 1

    def test_distinct_instances_same_fields_are_two_controls(self):
        c1 = Control(label="S", kind="float", default=1.0, bounds=(0.0, 2.0))
        c2 = Control(label="S", kind="float", default=1.0, bounds=(0.0, 2.0))

        class P(FigureProvider):
            @figure(title="a", controls={"sigma": c1})
            def a(self, *, sigma) -> go.Figure:
                return go.Figure()

            @figure(title="b", controls={"sigma": c2})
            def b(self, *, sigma) -> go.Figure:
                return go.Figure()

        from phenotypic.viz.notebook._adapter import unique_controls

        assert len(unique_controls(P().iter_figures())) == 2


# -- pydantic safety --------------------------------------------------------


class _PydModel(BaseModel, FigureProvider):
    """A pydantic model that mixes in FigureProvider and declares a figure."""

    scale: int = 3

    @figure(title="pm", primary=True)
    def plot(self) -> go.Figure:
        return go.Figure(go.Bar(y=[self.scale]))


class TestPydanticSafety:
    def test_mixin_adds_no_fields(self):
        assert set(_PydModel.model_fields) == {"scale"}

    def test_schema_only_declared_fields(self):
        props = _PydModel.model_json_schema()["properties"]
        assert set(props) == {"scale"}

    def test_dump_and_roundtrip(self):
        m = _PydModel(scale=5)
        assert m.model_dump() == {"scale": 5}
        again = _PydModel.model_validate(m.model_dump())
        assert again.scale == 5

    def test_figure_methods_work_on_model(self):
        m = _PydModel(scale=9)
        fig = m.inspect()
        assert isinstance(fig, go.Figure)
        assert fig.data[0].y == (9,)
        assert isinstance(m.iter_figures()[0], FigureSpec)
