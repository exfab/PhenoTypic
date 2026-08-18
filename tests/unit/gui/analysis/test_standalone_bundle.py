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

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl

from phenotypic import ImagePipeline
from phenotypic.analysis import LogGrowthModel
from phenotypic.detect import OtsuDetector
from phenotypic.gui._config import CFG_MEASUREMENT_SCHEMA, CFG_RECIPE_STATE
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.analysis import _ids
from phenotypic.gui.analysis._app import create_app
from phenotypic.gui.analysis._callbacks import _run_inline
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.measure import MeasureShape
from phenotypic.sdk_ import PIPELINE_JSON, BundleLayout
from phenotypic.sdk_._io_constants import _LEGACY_PIPELINE_JSON
from phenotypic.schema import IMAGE


def _walk(component: Any) -> Iterator[Any]:
    """Yield every component in a Dash layout tree."""
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)


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
            "Metadata_Dataset": ["plate1", "plate1"],
            str(IMAGE.IMAGE_NAME): ["img001", "img001"],
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
    root = OutputRoot.discover(
        base,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )
    assert root.layout.output_root is None  # standalone

    app = create_app(output_root=root)

    recipe = app.server.config[CFG_RECIPE_STATE]
    assert [type(op).__name__ for op in recipe.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]
    schema = app.server.config[CFG_MEASUREMENT_SCHEMA]
    assert "Shape_Area" in schema.columns_for("measurements")


def test_analysis_layout_includes_pipeline_gate_ack_store(
    tmp_path: Path,
) -> None:
    """The hydrated layout exposes the monotonic gate acknowledgment."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=PIPELINE_JSON)
    root = OutputRoot.discover(
        base,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )

    app = create_app(output_root=root)
    ack_store = next(
        component
        for component in _walk(app.layout)
        if getattr(component, "id", None)
        == _ids.ANALYSIS_PIPELINE_GATE_ACK_STORE
    )

    assert ack_store.data is None


def test_create_app_recipe_save_blocks_changed_processing_generation(
    tmp_path: Path,
) -> None:
    """The app-installed recipe guard rechecks processing under save lock."""
    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=PIPELINE_JSON)
    root = OutputRoot.discover(
        base,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )
    app = create_app(output_root=root)
    recipe = app.server.config[CFG_RECIPE_STATE]
    original = recipe.path.read_bytes()
    replacement = pl.read_parquet(root.layout.master_parquet).with_columns(
        (pl.col("Shape_Area") + 1).alias("Shape_Area")
    )
    replacement.write_parquet(root.layout.master_parquet)
    recipe.pipeline.name = "stale-edit"

    assert recipe.save() is False
    assert recipe.path.read_bytes() == original


def test_run_inline_blocks_external_recipe_replacement_with_preserved_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A content-changed recipe cannot publish from the stale in-memory model."""
    import phenotypic._cli._cli_output_manager as output_manager

    base = tmp_path / "my_export"
    _seed_standalone_bundle(base, pipeline_filename=PIPELINE_JSON)
    root = OutputRoot.discover(
        base,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )
    app = create_app(output_root=root)
    recipe = app.server.config[CFG_RECIPE_STATE]
    recipe.pipeline.set_model(
        LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Dataset"],
            time_label="Object_Label",
            n_jobs=1,
        )
    )
    original_mtime = recipe.path.stat().st_mtime_ns
    replacement = ImagePipeline(name="external-recipe").to_json() or ""
    recipe.path.write_text(replacement, encoding="utf-8")
    os.utime(recipe.path, ns=(original_mtime, original_mtime))

    called = False

    def _unexpected_emit(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("stale recipe reached publication")

    monkeypatch.setattr(
        output_manager,
        "_emit_analysis_outputs",
        _unexpected_emit,
    )

    status = _run_inline(recipe, root)

    assert called is False
    assert "pipeline configuration changed" in str(status.children)
