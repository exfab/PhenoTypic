"""Transactional Analysis recipe mutation regressions."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import (
    EdgeCorrector,
    LinearLagModel,
    LogGrowthModel,
    TukeyOutlierRemover,
)
from phenotypic.gui.analysis._callbacks import (
    _apply_param_edit,
    _semantic_values_equal,
)
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.plotting._bindings import PlotBinding, PipelineObjectRef
from phenotypic.sdk_ import pipeline_json_path


def _edge(*, top_n: int = 1) -> EdgeCorrector:
    """Build one deterministic edge corrector."""
    return EdgeCorrector(
        on="Shape_Area",
        groupby=["Metadata_Strain"],
        top_n=top_n,
    )


def _seed_recipe(output_dir: Path) -> RecipeState:
    """Write and load a recipe containing editable analyzer sections."""
    pipeline = ImagePipeline(
        name="before",
        filters={
            "edge": _edge(),
            "tukey": TukeyOutlierRemover(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
            ),
        },
        model=LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        ),
        plots=[
            PlotBinding(
                id="model",
                ref=PipelineObjectRef(slot="model"),
            )
        ],
    )
    path = pipeline_json_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipeline.to_json() or "", encoding="utf-8")
    return RecipeState.load(output_dir)


def _add_filter(pipeline: ImagePipeline) -> None:
    filters = pipeline.get_filters()
    filters["added"] = _edge(top_n=3)
    pipeline.set_filters(filters)


def _remove_filter(pipeline: ImagePipeline) -> None:
    filters = pipeline.get_filters()
    filters.pop("edge")
    pipeline.set_filters(filters)


def _reorder_filters(pipeline: ImagePipeline) -> None:
    pipeline.set_filters(dict(reversed(pipeline.get_filters().items())))


def _replace_model(pipeline: ImagePipeline) -> None:
    pipeline.set_model(
        LogGrowthModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
            time_label="Metadata_Time",
            n_jobs=1,
        )
    )


def _replace_plot_input(pipeline: ImagePipeline) -> None:
    binding = pipeline.get_plots()[0].model_copy(update={"id": "model-preference"})
    pipeline.set_plots([binding])


@pytest.mark.parametrize(
    "mutation",
    [
        _add_filter,
        _remove_filter,
        _reorder_filters,
        _replace_model,
        _replace_plot_input,
    ],
    ids=["add", "remove", "reorder", "model", "plotting-preference"],
)
def test_guard_rejection_rolls_back_every_recipe_edit(
    tmp_path: Path,
    mutation: Callable[[ImagePipeline], None],
) -> None:
    """Rejected edits leave neither memory nor later saves contaminated."""
    state = _seed_recipe(tmp_path)
    original_bytes = state.path.read_bytes()
    original_pipeline = state.pipeline.to_json()
    state.publication_guard = lambda: False

    assert state.mutate_and_save(mutation) is False

    assert state.path.read_bytes() == original_bytes
    assert state.pipeline.to_json() == original_pipeline

    state.publication_guard = lambda: True
    assert state.mutate_and_save(
        lambda pipeline: setattr(pipeline, "name", "unrelated")
    ) is True
    saved = json.loads(state.path.read_text(encoding="utf-8"))
    baseline = json.loads(original_bytes)
    assert saved["name"] == "unrelated"
    for section in ("filters", "model", "plots"):
        assert saved[section] == baseline[section]


def test_external_source_cas_rejection_reloads_before_later_save(
    tmp_path: Path,
) -> None:
    """A stale edit cannot leak around a content-fingerprint refusal."""
    state = _seed_recipe(tmp_path)
    original_mtime = state.path.stat().st_mtime_ns
    external = ImagePipeline(name="external")
    external_payload = external.to_json() or ""
    state.path.write_text(external_payload, encoding="utf-8")
    os.utime(state.path, ns=(original_mtime, original_mtime))

    assert state.mutate_and_save(_add_filter) is False
    assert state.pipeline.name == "external"
    assert state.pipeline.get_filters() == {}

    assert state.mutate_and_save(
        lambda pipeline: setattr(pipeline, "name", "external-unrelated")
    ) is True
    saved = json.loads(state.path.read_text(encoding="utf-8"))
    assert saved["name"] == "external-unrelated"
    assert saved["filters"] == {}


class _ParamEditContext:
    """Small callback-context stand-in for one scalar parameter edit."""

    def __init__(
        self,
        *,
        name: str,
        value: object,
        kind: str = "edge",
    ) -> None:
        self.triggered_id = {
            "type": "param-num",
            "prefix": f"analysis-{kind}-0",
            "name": name,
        }
        self.triggered = [{"value": value}]
        self.inputs: dict[str, object] = {}


def test_programmatic_same_value_param_feedback_skips_save(
    tmp_path: Path,
) -> None:
    """A rebuilt form's current value cannot publish a second transaction."""
    state = _seed_recipe(tmp_path)
    before = state.path.read_bytes()
    save_calls = 0
    real_save = state.save

    def _counted_save() -> bool:
        nonlocal save_calls
        save_calls += 1
        return real_save()

    state.save = _counted_save  # type: ignore[method-assign]

    outcome = _apply_param_edit(
        state,
        _ParamEditContext(name="n_jobs", value=1, kind="model"),
    )

    assert outcome == (False, "model", True)
    assert save_calls == 0
    assert state.path.read_bytes() == before


def test_matching_nan_arrays_are_semantically_equal() -> None:
    """Array-valued feedback treats paired NaNs as an unchanged value."""
    left = np.array([1.0, np.nan, 3.0])
    right = np.array([1.0, np.nan, 3.0])

    assert _semantic_values_equal(left, right) is True
    assert _semantic_values_equal(
        {"nested": [left]},
        {"nested": [right]},
    ) is True


def test_concurrent_parameter_edits_rebuild_from_live_locked_instance(
    tmp_path: Path,
) -> None:
    """Two admitted edits cannot both succeed while losing one update."""
    state = _seed_recipe(tmp_path)
    real_mutate_and_save = state.mutate_and_save
    both_ready = threading.Barrier(2)

    def _coordinated_mutation(
        mutation: Callable[[ImagePipeline], bool | None],
    ) -> bool:
        both_ready.wait(timeout=5.0)
        return real_mutate_and_save(mutation)

    state.mutate_and_save = _coordinated_mutation  # type: ignore[method-assign]
    outcomes: list[tuple[bool, str | None, bool]] = []
    outcomes_lock = threading.Lock()

    def _edit(name: str, value: object) -> None:
        outcome = _apply_param_edit(
            state,
            _ParamEditContext(name=name, value=value),
        )
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [
        threading.Thread(target=_edit, args=("top_n", 7)),
        threading.Thread(target=_edit, args=("pvalue", 0.25)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert outcomes == [(True, "edge", False)] * 2
    edge = state.pipeline.get_filters()["edge"]
    assert edge.top_n == 7
    assert edge.pvalue == 0.25
    saved = json.loads(state.path.read_text(encoding="utf-8"))
    assert saved["filters"]["edge"]["params"]["top_n"] == 7
    assert saved["filters"]["edge"]["params"]["pvalue"] == 0.25
