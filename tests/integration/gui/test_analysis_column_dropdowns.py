"""Integration tests for column-aware dropdowns in the analysis sub-app.

Boots the analysis Dash factory against a fixture output root that
carries a `measurements.parquet` with a known schema, then walks the
rendered layout to assert that filter / model column-ref params resolved
to dropdowns whose options match the parquet columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.gui.analysis._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot

from tests._output_layout import write_master, write_measurements_mirror, write_pipeline_json
from phenotypic.schema import METADATA


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


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """Build a CLI-shaped output root with a known measurements schema."""
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d"] * 4,
            str(METADATA.IMAGE_NAME): ["a", "b", "c", "d"],
            "MetadataGenetic_Strain": ["WT", "KO", "WT", "KO"],
            "MetadataCulture_Time": [0, 1, 2, 3],
            "Object_Label": [1, 1, 1, 1],
            "Shape_Area": [100.0, 200.0, 150.0, 250.0],
            "Intensity_MeanIntensity": [10.0, 20.0, 15.0, 25.0],
        }
    )
    write_master(tmp_path, df)
    write_measurements_mirror(tmp_path, df)
    (tmp_path / "results" / "d").mkdir(parents=True)
    write_pipeline_json(tmp_path, ImagePipeline(name="t"))
    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


@pytest.fixture()
def output_root_with_filter(tmp_path: Path) -> OutputRoot:
    """Output root whose pipeline.json already configures a TukeyOutlierRemover filter."""
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d"] * 4,
            str(METADATA.IMAGE_NAME): ["a", "b", "c", "d"],
            "MetadataGenetic_Strain": ["WT", "KO", "WT", "KO"],
            "MetadataCulture_Time": [0, 1, 2, 3],
            "Object_Label": [1, 1, 1, 1],
            "Shape_Area": [100.0, 200.0, 150.0, 250.0],
        }
    )
    write_master(tmp_path, df)
    write_measurements_mirror(tmp_path, df)
    (tmp_path / "results" / "d").mkdir(parents=True)

    pipeline = ImagePipeline(name="t")
    pipeline.set_filters({
        "tukey": TukeyOutlierRemover(
            on="Shape_Area",
            groupby=["MetadataGenetic_Strain"],
        )
    })
    pipeline.set_model(
        LogGrowthModel(
            on="Shape_Area",
            groupby=["MetadataGenetic_Strain"],
            time_label="MetadataCulture_Time",
        )
    )
    write_pipeline_json(tmp_path, pipeline)
    return OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
    )


class TestSchemaWiredIntoApp:
    def test_create_app_stashes_measurement_schema(self, output_root):
        from phenotypic.gui._config import CFG_MEASUREMENT_SCHEMA

        app = create_app(output_root=output_root)
        schema = app.server.config.get(CFG_MEASUREMENT_SCHEMA)
        assert schema is not None
        cols = schema.columns_for("measurements")
        assert "Shape_Area" in cols
        assert "MetadataGenetic_Strain" in cols


class TestColumnDropdownsRender:
    def test_filter_form_renders_column_dropdown_for_on(
        self, output_root_with_filter
    ):
        app = create_app(output_root=output_root_with_filter)
        # Find the TukeyOutlierRemover's `on` widget and verify it's a dbc.Select
        # (dropdown) populated from measurements.parquet — not a text input.
        cols = {"MetadataGenetic_Strain", "MetadataCulture_Time", "Shape_Area"}
        seen_on_dropdown = False
        for component in _walk(app.layout):
            cid = getattr(component, "id", None)
            if not isinstance(cid, dict):
                continue
            if (
                cid.get("type") == "param-column-scalar"
                and cid.get("name") == "on"
                and cid.get("prefix", "").startswith("analysis-filter")
            ):
                option_values = {o["value"] for o in component.options}
                assert cols.issubset(option_values), (
                    f"dropdown options {option_values} missing schema columns"
                )
                assert component.value == "Shape_Area"
                seen_on_dropdown = True
                break
        assert seen_on_dropdown, "TukeyOutlierRemover.on did not render as a dropdown"

    def test_model_form_renders_kmax_label_two_button_toggle(
        self, output_root_with_filter
    ):
        app = create_app(output_root=output_root_with_filter)
        seen_mode_toggle = False
        for component in _walk(app.layout):
            cid = getattr(component, "id", None)
            if not isinstance(cid, dict):
                continue
            if (
                cid.get("type") == "param-column-mode"
                and cid.get("name") == "Kmax_label"
                and cid.get("prefix", "").startswith("analysis-model")
            ):
                values = {o["value"] for o in component.options}
                assert {"column", "none"}.issubset(values)
                # Default analyzer was constructed with Kmax_label=None.
                assert component.value == "none"
                seen_mode_toggle = True
                break
        assert seen_mode_toggle, "Kmax_label did not render mode toggle"

    def test_groupby_renders_multi_dropdown(self, output_root_with_filter):
        app = create_app(output_root=output_root_with_filter)
        for component in _walk(app.layout):
            cid = getattr(component, "id", None)
            if not isinstance(cid, dict):
                continue
            if (
                cid.get("type") == "param-column-multi"
                and cid.get("name") == "groupby"
                and cid.get("prefix", "").startswith("analysis-filter")
            ):
                assert component.multi is True
                assert component.value == ["MetadataGenetic_Strain"]
                return
        pytest.fail("groupby did not render as multi-dropdown")
