"""Generation checks spanning complete Results and Analysis construction."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.gui._config import CFG_OUTPUT_ROOT
from phenotypic.gui.analysis import _app as analysis_app
from phenotypic.gui.results_viewer import _app as results_app
from phenotypic.gui.results_viewer._output_root import (
    OutputRoot,
    OutputSnapshotChangedError,
)
from phenotypic.schema import METADATA
from tests._output_layout import seed_output_dir


def _bound_output(tmp_path: Path) -> OutputRoot:
    """Create and discover one minimal shared Results/Analysis output."""
    output = tmp_path / "output"
    frame = pl.DataFrame({
        "MetadataExperiment_Dataset": ["dataset"],
        str(METADATA.IMAGE_NAME): ["plate"],
        "Object_Label": [1],
        "Shape_Area": [100.0],
    })
    seed_output_dir(
        output,
        frame,
        mirror=frame,
        pipeline=ImagePipeline(name="snapshot-test"),
    )
    (output / "results" / "dataset" / "measurements").mkdir(
        parents=True,
        exist_ok=True,
    )
    overlay = output / "deliverables" / "overlays" / "dataset" / "plate.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"overlay")
    return OutputRoot.discover(
        output,
        cache_root=tmp_path / "viewer-cache",
    )


def _rewrite_master(output_root: OutputRoot) -> None:
    """Atomically publish a distinct master generation."""
    master_path = output_root.layout.master_parquet
    replacement = master_path.with_name("master-replacement.parquet")
    (
        pl.read_parquet(master_path)
        .with_columns((pl.col("Shape_Area") + 1.0).alias("Shape_Area"))
        .write_parquet(replacement)
    )
    replacement.replace(master_path)


def test_shared_apps_accept_the_same_verified_revision(tmp_path: Path) -> None:
    """Clean construction keeps Results and Analysis on one descriptor."""
    output_root = _bound_output(tmp_path)

    results = results_app.create_app(output_root=output_root)
    analysis = analysis_app.create_app(output_root=output_root)

    assert results.server.config[CFG_OUTPUT_ROOT].snapshot == (
        analysis.server.config[CFG_OUTPUT_ROOT].snapshot
    )


@pytest.mark.parametrize(
    "factory",
    [results_app.create_app, analysis_app.create_app],
)
def test_session_construction_rejects_state_changed_before_read(
    tmp_path: Path,
    factory,
) -> None:
    """Neither app can begin from a descriptor already changed on disk."""
    output_root = _bound_output(tmp_path)
    output_root.layout.custom_categories_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_root.layout.custom_categories_json.write_text(
        '{"categories": ["new"]}',
        encoding="utf-8",
    )

    with pytest.raises(OutputSnapshotChangedError, match="pre-read"):
        factory(output_root=output_root)


def test_results_rejects_change_during_curation_state_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results cannot combine old measurements with new curation state."""
    output_root = _bound_output(tmp_path)
    real_load = results_app.CurationLabels.load

    def _load_then_mutate(layout, frame):
        labels = real_load(layout, frame)
        layout.custom_categories_json.parent.mkdir(parents=True, exist_ok=True)
        layout.custom_categories_json.write_text(
            '{"categories": ["raced"]}',
            encoding="utf-8",
        )
        return labels

    monkeypatch.setattr(
        results_app.CurationLabels,
        "load",
        staticmethod(_load_then_mutate),
    )

    with pytest.raises(OutputSnapshotChangedError, match="post-read"):
        results_app.create_app(output_root=output_root)


def test_analysis_rejects_change_during_recipe_state_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis cannot combine a recipe from one revision with another schema."""
    output_root = _bound_output(tmp_path)
    real_load = analysis_app.RecipeState.from_layout

    def _load_then_mutate(layout):
        recipe = real_load(layout)
        layout.qc_review_state_path.parent.mkdir(parents=True, exist_ok=True)
        layout.qc_review_state_path.write_text(
            '{"revision": "raced"}',
            encoding="utf-8",
        )
        return recipe

    monkeypatch.setattr(
        analysis_app.RecipeState,
        "from_layout",
        staticmethod(_load_then_mutate),
    )

    with pytest.raises(OutputSnapshotChangedError, match="post-read"):
        analysis_app.create_app(output_root=output_root)


def test_results_rejects_master_rewrite_during_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results cannot finish against a changed processing generation."""
    output_root = _bound_output(tmp_path)
    real_load = results_app.CurationLabels.load

    def _load_then_rewrite_master(layout, frame):
        labels = real_load(layout, frame)
        _rewrite_master(output_root)
        return labels

    monkeypatch.setattr(
        results_app.CurationLabels,
        "load",
        staticmethod(_load_then_rewrite_master),
    )

    with pytest.raises(OutputSnapshotChangedError, match="post-read"):
        results_app.create_app(output_root=output_root)


def test_analysis_rejects_master_rewrite_during_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis cannot finish against a changed processing generation."""
    output_root = _bound_output(tmp_path)
    real_load = analysis_app.RecipeState.from_layout

    def _load_then_rewrite_master(layout):
        recipe = real_load(layout)
        _rewrite_master(output_root)
        return recipe

    monkeypatch.setattr(
        analysis_app.RecipeState,
        "from_layout",
        staticmethod(_load_then_rewrite_master),
    )

    with pytest.raises(OutputSnapshotChangedError, match="post-read"):
        analysis_app.create_app(output_root=output_root)
