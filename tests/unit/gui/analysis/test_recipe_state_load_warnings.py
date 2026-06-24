"""Unit tests for the analysis GUI's unknown-analyzer skip + banner path.

When the CLI seeds ``pipeline.json`` referencing a filter/model class that
has since been renamed or removed, the analysis page must still load. The
contract under test:

1. ``ImagePipeline.from_json(..., skip_unknown_analyzers=True,
   load_warnings=sink)`` drops the unresolved entries and appends a
   :class:`PipelineLoadWarning` per skipped entry.
2. ``RecipeState.load`` exposes the warnings via
   :attr:`RecipeState.load_warnings`.
3. The on-disk JSON is **not** modified by load; only a subsequent
   explicit save would prune the entries.
4. The banner builder emits a visible div when warnings exist and a
   hidden placeholder otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic._core._pipeline_parts._serializable_pipeline import (
    PipelineLoadWarning,
)
from phenotypic.detect import OtsuDetector
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._layout import _build_load_warnings_banner
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.measure import MeasureShape
from phenotypic.sdk_ import pipeline_json_path


def _write_pipeline_with_unknown_classes(output_dir: Path) -> Path:
    """Seed ``<output_dir>/deliverables/pipeline.json`` with one good + two
    bad analyzer entries: a renamed filter and a renamed model.
    """
    payload = {
        "version": "test",
        "name": "test-pipeline",
        "desc": None,
        "reset": False,
        "pipe_cfgs": {},
        "meas": {},
        "post": {},
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
            },
            "stale_filter": {
                # Doesn't exist in the live namespace -- should be skipped.
                "class": "LegacyZScoreFilter",
                "params": {"threshold": 3.0},
            },
        },
        "model": {
            # Pre-rename class name -- matches the user's report.
            "class": "LinearLagModelModel",
            "params": {},
        },
    }
    path = pipeline_json_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_from_json_skip_mode_collects_unknown_analyzers(tmp_path: Path) -> None:
    """``from_json(skip_unknown_analyzers=True)`` returns a partial pipeline
    plus a populated warnings sink; the original JSON is untouched."""
    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    seed_bytes_before = seed_path.read_bytes()

    warnings: list[PipelineLoadWarning] = []
    pipeline = ImagePipeline.from_json(
        seed_path,
        skip_unknown_analyzers=True,
        load_warnings=warnings,
    )

    # Good entry survives; bad entry is gone.
    assert "edge" in pipeline.get_filters()
    assert "stale_filter" not in pipeline.get_filters()
    assert pipeline.get_model() is None

    # One warning per dropped entry, regardless of slot.
    classes = {w.class_name for w in warnings}
    assert classes == {"LegacyZScoreFilter", "LinearLagModelModel"}
    slots = {w.slot for w in warnings}
    assert slots == {"filter", "model"}

    # On-disk file is untouched -- the user must explicitly save to prune.
    assert seed_path.read_bytes() == seed_bytes_before


def test_from_json_default_still_raises_on_unknown(tmp_path: Path) -> None:
    """Non-skip callers (the historical contract) still get an
    ``AttributeError`` so CLI loads don't silently degrade."""
    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    with pytest.raises(AttributeError, match="not found"):
        ImagePipeline.from_json(seed_path)


def test_recipe_state_load_records_unknown_analyzer(tmp_path: Path) -> None:
    """``RecipeState.load`` exposes warnings + the disk file stays put."""
    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    seed_bytes_before = seed_path.read_bytes()

    state = RecipeState.load(tmp_path)

    assert len(state.load_warnings) == 2
    assert {w.class_name for w in state.load_warnings} == {
        "LegacyZScoreFilter",
        "LinearLagModelModel",
    }
    # Disk artifact must be byte-identical: opening the page is read-only.
    assert seed_path.read_bytes() == seed_bytes_before


def test_recipe_state_load_no_warnings_when_all_classes_resolve(
    tmp_path: Path,
) -> None:
    """Clean pipeline -> empty ``load_warnings``; banner builder hides."""
    payload = {
        "version": "test",
        "name": "clean",
        "desc": None,
        "reset": False,
        "pipe_cfgs": {},
        "meas": {},
        "post": {},
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {"on": "Shape_Area", "groupby": ["Metadata_Strain"]},
            }
        },
        "model": None,
    }
    clean_path = pipeline_json_path(tmp_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(json.dumps(payload))

    state = RecipeState.load(tmp_path)
    assert state.load_warnings == []

    banner = _build_load_warnings_banner(state)
    # Banner exists as a hidden placeholder so callbacks can target the id.
    assert banner.id == analysis_ids.ANALYSIS_LOAD_WARNINGS_BANNER
    assert banner.style == {"display": "none"}


def test_recipe_state_reload_preserves_legacy_pipeline_config(tmp_path: Path) -> None:
    """Legacy-only pipeline configs remain loaded across a recipe reload."""
    legacy_path = tmp_path / "deliverables" / "pipeline.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
    legacy_path.write_text(pipe.to_json() or "")

    state = RecipeState.load(tmp_path)
    assert [type(op).__name__ for op in state.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]

    state.reload()

    assert [type(op).__name__ for op in state.pipeline.get_ops().values()] == [
        "OtsuDetector"
    ]


def test_build_load_warnings_banner_renders_visible_div_when_warnings(
    tmp_path: Path,
) -> None:
    """With warnings the banner renders a visible div listing the entries."""
    _write_pipeline_with_unknown_classes(tmp_path)
    state = RecipeState.load(tmp_path)
    banner = _build_load_warnings_banner(state)

    assert banner.id == analysis_ids.ANALYSIS_LOAD_WARNINGS_BANNER
    assert "display" not in banner.style  # visible (no display:none override)
    # Heading + paragraph + bulleted list = 3 children.
    assert len(banner.children) == 3
