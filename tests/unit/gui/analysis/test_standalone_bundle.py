"""Standalone deliverables-bundle path resolution for the analysis sub-app.

A standalone bundle's ``OutputRoot.root`` IS the deliverables folder
(``layout.output_root is None``), so any helper that internally joins
``deliverables/`` would double-join. These tests pin that the analysis
sub-app resolves its recipe (pipeline config) and measurement schema from
*inside* the bundle, routed through :class:`~phenotypic.sdk_.BundleLayout`
rather than re-joining ``root``.

The bundle is deliberately NOT named ``deliverables`` so a latent
double-join lands in a non-existent ``<base>/deliverables/`` directory and
is caught (an empty fallback recipe / empty column list).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.gui._config import CFG_MEASUREMENT_SCHEMA, CFG_RECIPE_STATE
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.analysis._app import create_app
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.measure import MeasureShape
from phenotypic.sdk_ import PIPELINE_JSON, BundleLayout
from phenotypic.sdk_._io_constants import _LEGACY_PIPELINE_JSON
from phenotypic.schema import METADATA


def _seed_standalone_bundle(base: Path, *, pipeline_filename: str | None) -> None:
    """Seed a renamed standalone deliverables bundle.

    Writes master + mirror parquet (both required ``Metadata_*`` columns plus a
    measurement column) and, when ``pipeline_filename`` is given, a serialized
    pipeline config under that name *directly in the bundle base* (NOT under a
    ``deliverables/`` subdir).
    """
    base.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["plate1", "plate1"],
            str(METADATA.IMAGE_NAME): ["img001", "img001"],
            "Object_Label": [1, 2],
            "Shape_Area": [12.0, 34.0],
        }
    )
    df.write_parquet(base / "master_measurements.parquet")
    df.write_parquet(base / "measurements.parquet")
    if pipeline_filename is not None:
        pipe = ImagePipeline(
            ops=[OtsuDetector()], meas=[MeasureShape()], name="seeded-analysis"
        )
        (base / pipeline_filename).write_text(pipe.to_json() or "", encoding="utf-8")


def test_recipe_state_from_layout_canonical_config_no_double_join(
    tmp_path: Path,
) -> None:
    """``RecipeState.from_layout`` reads the canonical config inside the bundle."""
    base = tmp_path / "my_export"  # renamed standalone bundle (NOT "deliverables")
    _seed_standalone_bundle(base, pipeline_filename=PIPELINE_JSON)
    layout = BundleLayout.detect(base)
    assert layout.output_root is None  # standalone

    state = RecipeState.from_layout(layout)

    assert [type(op).__name__ for op in state.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]
    # Future writes target the canonical typed path inside the bundle.
    assert state.path == layout.pipeline_config_path


def test_recipe_state_from_layout_legacy_json_fallback(tmp_path: Path) -> None:
    """A legacy ``pipeline.json`` inside the bundle still seeds the recipe."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=_LEGACY_PIPELINE_JSON)
    layout = BundleLayout.detect(base)

    state = RecipeState.from_layout(layout)

    assert [type(op).__name__ for op in state.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]
    # Read from the legacy file, but the write target stays canonical.
    assert state.path == layout.pipeline_config_path
    assert state.source_path == base / _LEGACY_PIPELINE_JSON


def test_recipe_state_from_layout_missing_config_empty_pipeline(
    tmp_path: Path,
) -> None:
    """No config in the bundle -> an empty pipeline, not a crash/double-join."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=None)
    layout = BundleLayout.detect(base)

    state = RecipeState.from_layout(layout)

    assert list(state.pipeline.get_ops().values()) == []
    assert state.seed_mtime_ns is None
    assert state.path == layout.pipeline_config_path


def test_measurement_schema_from_layout_standalone_resolves_columns(
    tmp_path: Path,
) -> None:
    """Column schema resolves from inside the bundle (no ``deliverables/`` join)."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=None)
    layout = BundleLayout.detect(base)

    schema = MeasurementSchema.from_layout(layout)

    assert "Shape_Area" in schema.columns_for("measurements")
    assert "Shape_Area" in schema.columns_for("master_measurements")


def test_create_app_standalone_bundle_loads_recipe_and_schema(
    tmp_path: Path,
) -> None:
    """End-to-end: ``create_app`` on a standalone bundle wires recipe + schema
    from inside the bundle without double-joining ``deliverables/``."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=PIPELINE_JSON)
    root = OutputRoot.discover(base)
    assert root.layout.output_root is None  # standalone

    app = create_app(output_root=root)

    recipe = app.server.config[CFG_RECIPE_STATE]
    assert [type(op).__name__ for op in recipe.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]
    schema = app.server.config[CFG_MEASUREMENT_SCHEMA]
    assert "Shape_Area" in schema.columns_for("measurements")
