"""Unit tests for the analysis sub-app's plotting-controls module.

Covers :func:`plotting_params` signature introspection, the
:func:`collect_plot_kwargs` store-to-kwargs assembly, and the
``render_plot`` kwarg pass-through.
"""
from __future__ import annotations

from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402

from phenotypic.analysis import (  # noqa: E402
    EdgeCorrector,
    LogGrowthModel,
)
from phenotypic.gui.analysis._plot_controls import (  # noqa: E402
    collect_plot_kwargs,
    plotting_params,
)
from phenotypic.gui.analysis._render import render_plot  # noqa: E402


def _edge() -> EdgeCorrector:
    return EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"])


def _model() -> LogGrowthModel:
    return LogGrowthModel(
        on="Shape_Area",
        groupby=["Metadata_Strain"],
        time_label="Metadata_Time",
    )


class TestPlottingParams:
    """Signature introspection picks the right method and params."""

    def test_edge_corrector_specs(self) -> None:
        # EdgeCorrector does not override ``dash`` -> ``show`` is used.
        specs = {s.name: s for s in plotting_params(_edge())}
        assert set(specs) == {"figsize", "max_groups", "collapsed"}
        assert specs["figsize"].dtype == "tuple"
        assert specs["max_groups"].dtype == "number"
        assert specs["collapsed"].dtype == "bool"

    def test_edge_corrector_excludes_ax_and_criteria(self) -> None:
        names = {s.name for s in plotting_params(_edge())}
        assert "ax" not in names
        assert "criteria" not in names

    def test_model_specs_from_dash_signature(self) -> None:
        # LogGrowthModel overrides ``dash`` -> the dash signature is used.
        specs = {s.name: s for s in plotting_params(_model())}
        assert set(specs) == {"tmax", "figsize", "cmap", "legend"}
        assert specs["figsize"].dtype == "tuple"   # default (6, 4)
        assert specs["cmap"].dtype == "str"        # default "tab20"
        assert specs["legend"].dtype == "bool"     # default True
        assert specs["tmax"].dtype == "number"     # None default, int|float ann


class TestCollectPlotKwargs:
    """Store-keyed prefs assemble into render_plot kwargs."""

    def test_empty_prefs_yields_no_kwargs(self) -> None:
        assert collect_plot_kwargs("filter", 0, _edge(), None) == {}
        assert collect_plot_kwargs("filter", 0, _edge(), {}) == {}

    def test_scalar_and_tuple_params_assembled(self) -> None:
        prefs = {
            "filter-0-collapsed": False,
            "filter-0-max_groups": 5,
            "filter-0-figsize__0": 8,
            "filter-0-figsize__1": 4,
        }
        kwargs = collect_plot_kwargs("filter", 0, _edge(), prefs)
        assert kwargs == {
            "collapsed": False,
            "max_groups": 5,
            "figsize": (8, 4),
        }

    def test_partial_tuple_is_dropped(self) -> None:
        # Only one axis set -> figsize omitted so the analyzer default wins.
        prefs = {"filter-0-figsize__0": 8}
        assert "figsize" not in collect_plot_kwargs("filter", 0, _edge(), prefs)

    def test_keys_for_other_sections_ignored(self) -> None:
        prefs = {"filter-1-max_groups": 99, "model-0-max_groups": 7}
        assert collect_plot_kwargs("filter", 0, _edge(), prefs) == {}


class _FakeDashNode:
    """Node whose ``dash`` records kwargs and returns a plotly figure."""

    def __init__(self) -> None:
        self.dash_kwargs: dict[str, Any] | None = None

    def dash(self, **kwargs: Any) -> Any:
        import plotly.graph_objects as go

        self.dash_kwargs = kwargs
        return go.Figure()

    def show(self, **kwargs: Any) -> Any:  # pragma: no cover - dash path wins
        raise AssertionError("show() should not run when dash() succeeds")


class _FakeShowNode:
    """Node whose ``dash`` defers and whose ``show`` records kwargs."""

    def __init__(self) -> None:
        self.show_kwargs: dict[str, Any] | None = None

    def dash(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def show(self, **kwargs: Any) -> Any:
        self.show_kwargs = kwargs
        return matplotlib.figure.Figure()


class TestRenderPlotKwargPassthrough:
    """``render_plot`` forwards kwargs to whichever viz method runs."""

    def test_kwargs_reach_dash(self) -> None:
        node = _FakeDashNode()
        render_plot(node, figsize=(3, 2), cmap="viridis")
        assert node.dash_kwargs == {"figsize": (3, 2), "cmap": "viridis"}

    def test_kwargs_reach_show_on_fallback(self) -> None:
        node = _FakeShowNode()
        render_plot(node, figsize=(3, 2), collapsed=False)
        assert node.show_kwargs == {"figsize": (3, 2), "collapsed": False}

    def test_no_kwargs_is_still_valid(self) -> None:
        node = _FakeDashNode()
        render_plot(node)
        assert node.dash_kwargs == {}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
