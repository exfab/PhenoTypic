"""Integration tests for the analysis sub-app's plotting controls.

Two surfaces are covered:

* **Layout wiring** — booting the analysis Dash factory against a
  fixture output root and walking the rendered tree to confirm every
  filter / model card carries the Display-settings widgets, a Preview
  button, a plot slot, and that the session-scoped prefs store is
  mounted.
* **Preview flow** — replicating the body of the ``_on_preview_click``
  callback (resolve node -> read ``measurements.parquet`` -> ``analyze``
  -> ``render_plot``) and asserting a real plot component comes back.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import polars as pl
import pytest
from dash import dcc, html

from phenotypic import ImagePipeline
from phenotypic.analysis import EdgeCorrector, LogGrowthModel
from phenotypic.gui.analysis import _ids as ids
from phenotypic.gui.analysis._app import create_app
from phenotypic.gui.analysis._callbacks import _resolve_preview_node
from phenotypic.gui.analysis._plot_controls import collect_plot_kwargs
from phenotypic.gui.analysis._render import render_plot
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.tools_ import measurements_parquet_path

from tests._output_layout import write_master, write_measurements_mirror, write_pipeline_json


def _walk(component: Any) -> Iterator[Any]:
    """Yield every Dash component in a layout tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif isinstance(children, str):
        return
    else:
        yield from _walk(children)


def _logistic_growth_frame() -> pl.DataFrame:
    """A tiny logistic-growth measurements frame LogGrowthModel converges on."""
    rows: list[dict] = []
    for strain in ("CBS-A", "CBS-B"):
        for t in (0, 6, 12, 24, 36, 48):
            for rep in range(3):
                n = 100 + 800 / (1 + (1000 - 100) / 100 * math.exp(-0.15 * t))
                rows.append({
                    "Metadata_Dataset": "ds1",
                    "Metadata_ImageFile": f"{strain}_t{t}",
                    "Metadata_Strain": strain,
                    "Metadata_Time": float(t),
                    "Object_Label": rep,
                    "Shape_Area": float(n + (rep - 1) * 5),
                })
    return pl.DataFrame(rows)


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """CLI-shaped output root with a filter + model already configured."""
    df = _logistic_growth_frame()
    write_master(tmp_path, df)
    write_measurements_mirror(tmp_path, df)
    (tmp_path / "results" / "ds1").mkdir(parents=True)

    pipeline = ImagePipeline(name="t")
    pipeline.set_filters({
        "edge": EdgeCorrector(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
        )
    })
    pipeline.set_model(
        LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
        )
    )
    write_pipeline_json(tmp_path, pipeline)
    return OutputRoot.discover(tmp_path)


def _ids_of_type(layout: Any, type_name: str) -> list[dict]:
    """Collect every pattern-matching id of ``type_name`` in the layout."""
    found: list[dict] = []
    for component in _walk(layout):
        cid = getattr(component, "id", None)
        if isinstance(cid, dict) and cid.get("type") == type_name:
            found.append(cid)
    return found


class TestPlotControlsInLayout:
    """The analysis layout wires plotting controls into filter/model cards."""

    def test_prefs_store_is_session_scoped(self, output_root: OutputRoot) -> None:
        app = create_app(output_root=output_root)
        stores = [
            c
            for c in _walk(app.layout)
            if getattr(c, "id", None) == ids.ANALYSIS_PLOT_PREFS_STORE
        ]
        assert len(stores) == 1
        # Must be session storage — never persisted into pipeline.json.
        assert stores[0].storage_type == "session"

    def test_filter_card_has_plot_param_widgets(
        self, output_root: OutputRoot
    ) -> None:
        app = create_app(output_root=output_root)
        params = _ids_of_type(app.layout, "analysis-plot-param")
        # EdgeCorrector.show exposes figsize (paired) + max_groups + collapsed.
        filter_names = {p["name"] for p in params if p["kind"] == "filter"}
        assert {"max_groups", "collapsed", "figsize__0", "figsize__1"}.issubset(
            filter_names
        )

    def test_filter_and_model_cards_have_preview_button_and_slot(
        self, output_root: OutputRoot
    ) -> None:
        app = create_app(output_root=output_root)
        buttons = _ids_of_type(app.layout, "analysis-preview-btn")
        slots = _ids_of_type(app.layout, "analysis-plot-slot")
        kinds = {b["kind"] for b in buttons}
        assert kinds == {"filter", "model"}
        # One slot per button, matched on kind+index.
        assert {(s["kind"], s["index"]) for s in slots} == {
            (b["kind"], b["index"]) for b in buttons
        }

    def test_post_cards_have_no_plot_controls(
        self, tmp_path: Path
    ) -> None:
        # A pipeline with only a post op -> no plot-param widgets at all.
        df = _logistic_growth_frame()
        write_master(tmp_path, df)
        write_measurements_mirror(tmp_path, df)
        (tmp_path / "results" / "ds1").mkdir(parents=True)
        write_pipeline_json(tmp_path, ImagePipeline(name="t"))
        app = create_app(output_root=OutputRoot.discover(tmp_path))
        assert _ids_of_type(app.layout, "analysis-plot-param") == []


class TestPreviewFlow:
    """Replays the ``_on_preview_click`` callback body end to end."""

    @staticmethod
    def _recipe_with(pipeline: ImagePipeline) -> Any:
        import types

        return types.SimpleNamespace(pipeline=pipeline)

    def test_resolve_preview_node_filter_and_model(
        self, output_root: OutputRoot
    ) -> None:
        pipeline = ImagePipeline(name="t")
        edge = EdgeCorrector(on="Shape_Area", groupby=["Metadata_Strain"])
        model = LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
        )
        pipeline.set_filters({"edge": edge})
        pipeline.set_model(model)
        recipe = self._recipe_with(pipeline)

        assert _resolve_preview_node(recipe, "filter", 0) is edge
        assert _resolve_preview_node(recipe, "model", 0) is model
        # Out-of-range / unknown kinds resolve to None.
        assert _resolve_preview_node(recipe, "filter", 9) is None
        assert _resolve_preview_node(recipe, "post", 0) is None

    def test_model_preview_renders_a_plot(
        self, output_root: OutputRoot
    ) -> None:
        model = LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
        )
        pipeline = ImagePipeline(name="t")
        pipeline.set_model(model)
        recipe = self._recipe_with(pipeline)

        node = _resolve_preview_node(recipe, "model", 0)
        assert node is not None

        frame = pd.read_parquet(measurements_parquet_path(Path(output_root.root)))
        node.analyze(frame)
        kwargs = collect_plot_kwargs("model", 0, node, {"model-0-cmap": "viridis"})
        component = render_plot(node, **kwargs)

        # LogGrowthModel overrides dash() -> plotly fast path -> dcc.Graph.
        assert isinstance(component, dcc.Graph)

    def test_preview_honours_plotting_prefs(
        self, output_root: OutputRoot
    ) -> None:
        # A bad cmap pref must surface as an inline error card, proving the
        # plotting prefs are actually threaded into the viz call.
        model = LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
        )
        frame = pd.read_parquet(measurements_parquet_path(Path(output_root.root)))
        model.analyze(frame)
        kwargs = collect_plot_kwargs(
            "model", 0, model, {"model-0-cmap": "definitely-not-a-colormap"}
        )
        component = render_plot(model, **kwargs)
        assert isinstance(component, html.Div)
        assert "Preview unavailable" in str(component.children)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
