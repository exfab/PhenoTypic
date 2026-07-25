"""Unit tests for the analysis GUI's unknown-analyzer skip + banner path.

When the CLI seeds ``pipeline.json`` referencing a filter/model class that
has since been renamed or removed, the analysis page must still load. The
contract under test:

1. ``ImagePipeline.from_json(..., skip_unknown_analyzers=True,
   load_warnings=sink)`` drops the unresolved entries and appends a
   :class:`PipelineLoadWarning` per skipped entry.
2. ``RecipeState.load`` exposes the warnings via
   :attr:`RecipeState.load_warnings`.
3. The on-disk JSON is **not** modified by load, and an unrelated explicit
   save preserves every opaque entry.
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
from phenotypic.gui.analysis._recipe_state import (
    RecipeState,
    _merge_opaque_pipeline_payload,
    _pipeline_validation_payload,
)
from phenotypic.measure import MeasureShape
from phenotypic.plotting._bindings import AnalysisInput, MeasurementInput
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


def _known_pipeline_with_extensions_payload() -> dict:
    """Return a current-schema pipeline carrying future nested fields."""
    return {
        "version": "test",
        "name": "before",
        "desc": None,
        "reset": False,
        "pipe_cfgs": {},
        "meas": {},
        "post": {},
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {
                    "on": "Shape_Area",
                    "groupby": ["Metadata_Strain"],
                    "future_filter_param": {"revision": 3},
                },
                "future_filter_envelope": {"revision": 2},
            }
        },
        "model": {
            "class": "LinearLagModel",
            "params": {
                "on": "Shape_Area",
                "groupby": ["Metadata_Strain"],
            },
        },
        "plots": [
            {
                "id": "growth",
                "ref": {
                    "slot": "model",
                    "key": None,
                    "future_ref_field": {"revision": 4},
                },
                "input": {
                    "kind": "analysis",
                    "analysis_id": "growth-table",
                    "future_input_field": {"revision": 6},
                },
                "future_plot_envelope": {"revision": 5},
            }
        ],
        "future_top_level": {"schema": "vNext"},
    }


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

    # On-disk file is untouched.
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


def test_unrelated_save_preserves_unknown_nodes_in_original_slots(
    tmp_path: Path,
) -> None:
    """A name edit cannot serialize only the successfully loaded subset."""
    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    original = json.loads(seed_path.read_text(encoding="utf-8"))
    original["extension_state"] = {
        "future_schema": ["kept", {"exact": True}]
    }
    seed_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
    state = RecipeState.load(tmp_path)

    state.pipeline.name = "edited-name"
    assert state.save() is True

    saved = json.loads(seed_path.read_text(encoding="utf-8"))
    assert saved["name"] == "edited-name"
    assert list(saved["filters"]) == ["edge", "stale_filter"]
    assert saved["filters"]["stale_filter"] == (
        original["filters"]["stale_filter"]
    )
    assert saved["model"] == original["model"]
    assert saved["extension_state"] == original["extension_state"]
    assert json.loads(state.last_json) == saved


def test_unsafe_processing_guard_blocks_save_without_losing_opaque_source(
    tmp_path: Path,
) -> None:
    """Active/stale bindings cannot publish a tolerant-load subset."""
    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    original = seed_path.read_bytes()
    state = RecipeState.load(tmp_path)
    state.publication_guard = lambda: False
    state.pipeline.name = "must-not-publish"

    assert state.save() is False
    assert seed_path.read_bytes() == original
    assert state.source_payload is not None
    assert state.source_payload["filters"]["stale_filter"]["class"] == (
        "LegacyZScoreFilter"
    )
    assert state.source_payload["model"]["class"] == "LinearLagModelModel"


def test_known_filter_edit_keeps_opaque_sibling_exactly(
    tmp_path: Path,
) -> None:
    """Editing a live filter merges around its unknown sibling."""
    from phenotypic.analysis import EdgeCorrector

    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    original = json.loads(seed_path.read_text(encoding="utf-8"))
    state = RecipeState.load(tmp_path)
    state.pipeline.set_filters({
        "edge": EdgeCorrector(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            top_n=7,
        )
    })

    assert state.save() is True

    saved = json.loads(seed_path.read_text(encoding="utf-8"))
    assert saved["filters"]["edge"]["params"]["top_n"] == 7
    assert saved["filters"]["stale_filter"] == (
        original["filters"]["stale_filter"]
    )
    assert saved["model"] == original["model"]


def test_explicit_live_model_replaces_opaque_model_node(tmp_path: Path) -> None:
    """A known model selection is an explicit replacement, not preservation."""
    from phenotypic.analysis import LinearLagModel

    seed_path = _write_pipeline_with_unknown_classes(tmp_path)
    state = RecipeState.load(tmp_path)
    state.pipeline.set_model(
        LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        )
    )

    assert state.save() is True

    saved = json.loads(seed_path.read_text(encoding="utf-8"))
    assert saved["model"]["class"] == "LinearLagModel"
    assert [warning.slot for warning in state.load_warnings] == ["filter"]


def test_name_only_edit_preserves_known_envelope_extensions_without_warnings(
    tmp_path: Path,
) -> None:
    """Known nested fields validate via a copy and round-trip exactly."""
    payload = _known_pipeline_with_extensions_payload()
    path = pipeline_json_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state = RecipeState.load(tmp_path)
    assert state.load_warnings == []

    state.pipeline.name = "after"
    assert state.save() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["name"] == "after"
    assert saved["future_top_level"] == payload["future_top_level"]
    assert saved["filters"]["edge"]["future_filter_envelope"] == {
        "revision": 2
    }
    assert saved["filters"]["edge"]["params"]["future_filter_param"] == {
        "revision": 3
    }
    assert saved["plots"][0]["ref"]["future_ref_field"] == {
        "revision": 4
    }
    assert saved["plots"][0]["future_plot_envelope"] == {
        "revision": 5
    }
    assert saved["plots"][0]["input"]["future_input_field"] == {
        "revision": 6
    }


def test_legacy_implicit_measurements_input_round_trips_unrelated_edit(
    tmp_path: Path,
) -> None:
    """Omitted legacy input remains stable with plot extensions intact."""
    payload = _known_pipeline_with_extensions_payload()
    del payload["plots"][0]["input"]
    path = pipeline_json_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state = RecipeState.load(tmp_path)
    assert isinstance(
        state.pipeline.get_plots()[0].input,
        MeasurementInput,
    )

    state.pipeline.name = "after"
    assert state.save() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "input" not in saved["plots"][0]
    assert saved["plots"][0]["ref"]["future_ref_field"] == {
        "revision": 4
    }
    assert saved["plots"][0]["future_plot_envelope"] == {
        "revision": 5
    }
    reloaded = RecipeState.load(tmp_path)
    assert isinstance(
        reloaded.pipeline.get_plots()[0].input,
        MeasurementInput,
    )
    strict = ImagePipeline.from_json(_pipeline_validation_payload(saved))
    assert isinstance(strict.get_plots()[0].input, MeasurementInput)


def test_explicit_known_replacement_drops_nested_extensions_after_load(
    tmp_path: Path,
) -> None:
    """Changed classes replace future fields bound to the prior nodes."""
    from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover

    payload = _known_pipeline_with_extensions_payload()
    path = pipeline_json_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state = RecipeState.load(tmp_path)

    state.pipeline.set_filters({
        "edge": TukeyOutlierRemover(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        )
    })
    state.pipeline.set_model(
        LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        )
    )
    assert state.save() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["filters"]["edge"]["class"] == "TukeyOutlierRemover"
    assert "future_filter_param" not in saved["filters"]["edge"]["params"]
    assert "future_filter_envelope" not in saved["filters"]["edge"]
    assert saved["model"]["class"] == "LogGrowthModel"
    assert "future_ref_field" not in saved["plots"][0]["ref"]
    assert "future_plot_envelope" not in saved["plots"][0]
    assert "future_input_field" not in saved["plots"][0]["input"]


@pytest.mark.parametrize(
    ("source_input", "replacement"),
    [
        (
            {
                "kind": "analysis",
                "analysis_id": "growth-table",
                "future_input_field": {"revision": 1},
            },
            MeasurementInput(),
        ),
        (
            {
                "kind": "measurements",
                "future_input_field": {"revision": 2},
            },
            AnalysisInput(analysis_id="new-analysis"),
        ),
        (
            {
                "kind": "analysis",
                "analysis_id": "old-analysis",
                "future_input_field": {"revision": 3},
            },
            AnalysisInput(analysis_id="new-analysis"),
        ),
        (
            None,
            AnalysisInput(analysis_id="new-analysis"),
        ),
    ],
)
def test_plot_input_replacement_does_not_resurrect_prior_variant(
    tmp_path: Path,
    source_input: dict | None,
    replacement: AnalysisInput | MeasurementInput,
) -> None:
    """Variant or stable-ID changes replace the prior input envelope."""
    payload = _known_pipeline_with_extensions_payload()
    del payload["filters"]["edge"]["params"]["future_filter_param"]
    if source_input is None:
        del payload["plots"][0]["input"]
    else:
        payload["plots"][0]["input"] = source_input
    path = pipeline_json_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state = RecipeState.load(tmp_path)
    binding = state.pipeline.get_plots()[0].model_copy(
        update={"input": replacement}
    )
    state.pipeline.set_plots([binding])

    assert state.save() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["plots"][0]["input"] == replacement.model_dump(mode="json")
    assert "future_input_field" not in saved["plots"][0]["input"]
    assert "future_ref_field" not in saved["plots"][0]["ref"]
    assert "future_plot_envelope" not in saved["plots"][0]
    strict = ImagePipeline.from_json(saved)
    assert strict.get_plots()[0].input == replacement


def test_stable_serialized_identity_preserves_known_envelope_extensions() -> None:
    """Owned edits retain future fields when stable IDs and classes match."""
    original = {
        "name": "before",
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {"on": "Size_Area"},
                "future_filter": {"kept": True},
            }
        },
        "model": {
            "class": "LinearLagModel",
            "params": {"on": "Size_Area", "groupby": ["strain"]},
            "future_model": {"kept": True},
        },
        "qc": [
            {
                "instance_id": "qc-count",
                "class": "ExpectedVsDetectedCount",
                "enabled": True,
                "params": {},
                "future_qc": {"kept": True},
            }
        ],
        "plots": [
            {
                "id": "growth",
                "ref": {
                    "slot": "model",
                    "key": None,
                    "future_ref": {"kept": True},
                },
                "input": {
                    "kind": "analysis",
                    "analysis_id": "growth-table",
                    "future_input": {"kept": True},
                },
                "future_plot": {"kept": True},
            }
        ],
        "future_top": {"kept": True},
    }
    current = {
        "name": "after",
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {"on": "Shape_Circularity"},
            }
        },
        "model": {
            "class": "LinearLagModel",
            "params": {"on": "Shape_Circularity", "groupby": ["strain"]},
        },
        "qc": [
            {
                "instance_id": "qc-count",
                "class": "ExpectedVsDetectedCount",
                "enabled": True,
                "params": {"tolerance": 2},
            }
        ],
        "plots": [
            {
                "id": "growth",
                "ref": {"slot": "model", "key": None},
                "input": {
                    "kind": "analysis",
                    "analysis_id": "growth-table",
                },
            }
        ],
    }

    merged = _merge_opaque_pipeline_payload(current, original, [])

    assert merged["future_top"] == {"kept": True}
    assert merged["filters"]["edge"]["future_filter"] == {"kept": True}
    assert merged["model"]["future_model"] == {"kept": True}
    assert merged["qc"][0]["future_qc"] == {"kept": True}
    assert merged["plots"][0]["future_plot"] == {"kept": True}
    assert merged["plots"][0]["ref"]["future_ref"] == {"kept": True}
    assert merged["plots"][0]["input"]["future_input"] == {"kept": True}


def test_explicit_replacement_drops_prior_nested_extensions() -> None:
    """A different live class or plot target replaces the whole envelope."""
    original = {
        "filters": {
            "edge": {
                "class": "EdgeCorrector",
                "params": {},
                "future_filter": "old",
            }
        },
        "model": {
            "class": "LinearLagModel",
            "params": {},
            "future_model": "old",
        },
        "qc": [
            {
                "instance_id": "qc-1",
                "class": "ExpectedVsDetectedCount",
                "enabled": True,
                "params": {},
                "future_qc": "old",
            }
        ],
        "plots": [
            {
                "id": "plot-1",
                "ref": {"slot": "model", "key": None},
                "future_plot": "old",
            }
        ],
    }
    current = {
        "filters": {
            "edge": {"class": "TukeyOutlierRemover", "params": {}}
        },
        "model": {"class": "LogGrowthModel", "params": {}},
        "qc": [
            {
                "instance_id": "qc-1",
                "class": "TukeyOutlierFraction",
                "enabled": True,
                "params": {},
            }
        ],
        "plots": [
            {
                "id": "plot-1",
                "ref": {"slot": "filters", "key": "edge"},
            }
        ],
    }

    merged = _merge_opaque_pipeline_payload(current, original, [])

    assert "future_filter" not in merged["filters"]["edge"]
    assert "future_model" not in merged["model"]
    assert "future_qc" not in merged["qc"][0]
    assert "future_plot" not in merged["plots"][0]


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
