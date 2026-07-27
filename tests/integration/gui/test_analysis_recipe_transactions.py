"""Transport-level regressions for Analysis recipe edit refusal."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import (
    EdgeCorrector,
    LinearLagModel,
    TukeyOutlierRemover,
)
from phenotypic.gui._config import CFG_RECIPE_STATE
from phenotypic.gui.analysis import _callbacks, _ids
from phenotypic.gui.analysis._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.post import AppendString
from phenotypic.schema import METADATA
from phenotypic.sdk_ import resolve_manifest_json_path
from tests._output_layout import seed_output_dir


def _seed_output(
    tmp_path: Path,
    *,
    pipeline: ImagePipeline | None = None,
) -> OutputRoot:
    """Build one complete output accepted by the Analysis app."""
    output = tmp_path / "output"
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["dataset"],
            str(METADATA.IMAGE_NAME): ["plate"],
            "Object_Label": [1],
            "Shape_Area": [100.0],
        }
    )
    seed_output_dir(
        output,
        frame,
        mirror=frame,
        pipeline=pipeline or ImagePipeline(name="before"),
    )
    manifest = resolve_manifest_json_path(output)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            }
        ),
        encoding="utf-8",
    )
    return OutputRoot.discover(
        output,
        cache_root=tmp_path / "viewer-cache",
    )


def _outputs_from_key(output_key: str) -> list[dict[str, str]]:
    """Parse one Dash callback-map key into response output records."""
    outputs: list[dict[str, str]] = []
    for segment in re.split(r"\.\.\.", output_key.strip(".")):
        component = segment.strip(".").split("@", 1)[0]
        component_id, prop = component.rsplit(".", 1)
        outputs.append({"id": component_id, "property": prop})
    return outputs


def _single_output_from_key(output_key: str) -> dict[str, str]:
    """Return the sole Dash output record for a mutation callback."""
    outputs = _outputs_from_key(output_key)
    assert len(outputs) == 1
    return outputs[0]


def _response_value(response: Any, component_id: str, prop: str) -> Any:
    """Read one component property from a Dash callback response."""
    payload = response.get_json()["response"]
    return payload[component_id][prop]


def _callback_key_for_input(app: Any, component_id: str) -> str:
    """Return the callback key whose inputs include one concrete component."""
    return next(
        key
        for key, callback in app.callback_map.items()
        if any(
            callback_input["id"] == component_id
            for callback_input in callback["inputs"]
        )
    )


def _post_value_mutation(
    app: Any,
    component_id: str,
    value: Any,
) -> Any:
    """Post one ordinary value-input mutation through Dash transport."""
    output_key = _callback_key_for_input(app, component_id)
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": [
                {
                    "id": component_id,
                    "property": "value",
                    "value": value,
                }
            ],
            "state": [],
            "changedPropIds": [f"{component_id}.value"],
        },
    )


def _post_reconciliation(app: Any, trigger: dict[str, Any]) -> Any:
    """Run the centralized full-page renderer for one transaction trigger."""
    output_key = next(
        key
        for key, callback in app.callback_map.items()
        if (
            any(
                callback_input["id"] == _ids.ANALYSIS_PIPELINE_STORE
                for callback_input in callback["inputs"]
            )
            and _ids.ANALYSIS_POST_STACK in key
            and _ids.ANALYSIS_MODEL_SECTION in key
        )
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _outputs_from_key(output_key),
            "inputs": [
                {
                    "id": _ids.ANALYSIS_PIPELINE_STORE,
                    "property": "data",
                    "value": trigger,
                }
            ],
            "state": [
                {
                    "id": _ids.ANALYSIS_PLOT_PREFS_STORE,
                    "property": "data",
                    "value": {},
                }
            ],
            "changedPropIds": [f"{_ids.ANALYSIS_PIPELINE_STORE}.data"],
        },
    )


def _post_pattern_param(
    app: Any,
    *,
    kind: str,
    name: str,
    widget_type: str,
    value: Any,
) -> Any:
    """Post one concrete input through the parameter fan-in callback."""
    output_key, callback = next(
        (key, callback)
        for key, callback in app.callback_map.items()
        if (
            _ids.ANALYSIS_PIPELINE_EVENT_STORE in key
            and any(
                widget_type in str(callback_input["id"])
                for callback_input in callback["inputs"]
            )
        )
    )
    concrete_id = {
        "name": name,
        "prefix": f"analysis-{kind}-0",
        "type": widget_type,
    }
    inputs: list[dict[str, Any]] = []
    for callback_input in callback["inputs"]:
        if widget_type in str(callback_input["id"]):
            inputs.append(
                {
                    "id": concrete_id,
                    "property": callback_input["property"],
                    "value": value,
                }
            )
        else:
            inputs.append(
                {
                    "id": callback_input["id"],
                    "property": callback_input["property"],
                    "value": [],
                }
            )
    concrete_prop = f"{json.dumps(concrete_id, separators=(',', ':'))}.value"
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": inputs,
            "state": [],
            "changedPropIds": [concrete_prop],
        },
    )


def test_pipeline_gate_ack_callback_wiring(tmp_path: Path) -> None:
    """The clientside monotonic gate writes applied state and one ack."""
    app = create_app(output_root=_seed_output(tmp_path))
    output_key, callback = next(
        (key, callback)
        for key, callback in app.callback_map.items()
        if any(
            callback_input["id"] == _ids.ANALYSIS_PIPELINE_EVENT_STORE
            for callback_input in callback["inputs"]
        )
    )

    assert _outputs_from_key(output_key) == [
        {"id": _ids.ANALYSIS_PIPELINE_STORE, "property": "data"},
        {
            "id": _ids.ANALYSIS_PIPELINE_GATE_ACK_STORE,
            "property": "data",
        },
    ]
    assert callback["inputs"] == [
        {"id": _ids.ANALYSIS_PIPELINE_EVENT_STORE, "property": "data"}
    ]
    assert callback["state"] == [
        {"id": _ids.ANALYSIS_PIPELINE_STORE, "property": "data"}
    ]


def test_model_rebuild_same_value_feedback_returns_no_update(
    tmp_path: Path,
) -> None:
    """One model selection saves once; rebuilt current values return 204."""
    app = create_app(output_root=_seed_output(tmp_path))
    recipe = app.server.config[CFG_RECIPE_STATE]
    save_calls = 0
    real_save = recipe.save

    def _counted_save() -> bool:
        nonlocal save_calls
        save_calls += 1
        return real_save()

    recipe.save = _counted_save  # type: ignore[method-assign]
    selected = _post_value_mutation(
        app,
        _ids.ANALYSIS_MODEL_DROPDOWN,
        "LinearLagModel",
    )
    assert selected.status_code == 200
    trigger = _response_value(
        selected,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    rendered = _post_reconciliation(app, trigger)
    assert rendered.status_code == 200
    assert save_calls == 1
    before_feedback = recipe.path.read_bytes()

    feedback = _post_pattern_param(
        app,
        kind="model",
        name="n_jobs",
        widget_type="param-num",
        value=1,
    )

    assert feedback.status_code == 204
    assert save_calls == 1
    assert recipe.path.read_bytes() == before_feedback


def test_nan_parameter_feedback_returns_no_update_without_save(
    tmp_path: Path,
) -> None:
    """Equal NaN values are a semantic no-op through real Dash transport."""
    app = create_app(
        output_root=_seed_output(
            tmp_path,
            pipeline=ImagePipeline(
                name="nan-feedback",
                filters={
                    "edge": EdgeCorrector(
                        on="Shape_Area",
                        groupby=["Metadata_Strain"],
                        pvalue=float("nan"),
                    )
                },
            ),
        )
    )
    recipe = app.server.config[CFG_RECIPE_STATE]
    save_calls = 0
    real_save = recipe.save
    before_feedback = recipe.path.read_bytes()

    def _counted_save() -> bool:
        nonlocal save_calls
        save_calls += 1
        return real_save()

    recipe.save = _counted_save  # type: ignore[method-assign]
    feedback = _post_pattern_param(
        app,
        kind="edge",
        name="pvalue",
        widget_type="param-num",
        value=float("nan"),
    )

    assert feedback.status_code == 204
    assert save_calls == 0
    assert recipe.path.read_bytes() == before_feedback


def test_matching_array_feedback_returns_no_update_without_save(
    tmp_path: Path,
) -> None:
    """Array/list feedback with paired NaNs is a transport-level no-op."""
    app = create_app(
        output_root=_seed_output(
            tmp_path,
            pipeline=ImagePipeline(
                name="array-feedback",
                model=LinearLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                ),
            ),
        )
    )
    recipe = app.server.config[CFG_RECIPE_STATE]
    model = recipe.pipeline.get_model()
    assert model is not None
    model.groupby = np.array(["Metadata_Strain"])
    save_calls = 0
    real_save = recipe.save
    before_feedback = recipe.path.read_bytes()

    def _counted_save() -> bool:
        nonlocal save_calls
        save_calls += 1
        return real_save()

    recipe.save = _counted_save  # type: ignore[method-assign]
    feedback = _post_pattern_param(
        app,
        kind="model",
        name="groupby",
        widget_type="param-column-multi",
        value=["Metadata_Strain"],
    )

    assert feedback.status_code == 204
    assert save_calls == 0
    assert recipe.path.read_bytes() == before_feedback


def test_stale_issued_revision_is_refused_before_render(
    tmp_path: Path,
) -> None:
    """Rev1 returns 204 when rev2 was issued before its render request."""
    app = create_app(output_root=_seed_output(tmp_path))
    rev1_response = _post_value_mutation(
        app,
        _ids.ANALYSIS_MODEL_DROPDOWN,
        "LogGrowthModel",
    )
    assert rev1_response.status_code == 200
    rev1 = _response_value(
        rev1_response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )

    rev2_response = _post_value_mutation(
        app,
        _ids.ANALYSIS_FILTER_ADD_DROPDOWN,
        "TukeyOutlierRemover",
    )
    assert rev2_response.status_code == 200
    rev2 = _response_value(
        rev2_response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    assert rev2["revision"] > rev1["revision"]
    assert json.loads(rev1["pipeline_json"])["filters"] == {}
    assert len(json.loads(rev2["pipeline_json"])["filters"]) == 1

    tampered_rev1 = {**rev1, "pipeline_json": "{}"}
    assert _post_reconciliation(app, tampered_rev1).status_code == 204
    assert _post_reconciliation(app, rev1).status_code == 204

    rendered_rev2 = _post_reconciliation(app, rev2)
    assert rendered_rev2.status_code == 200
    rev2_filters = _response_value(
        rendered_rev2,
        _ids.ANALYSIS_FILTER_STACK,
        "children",
    )
    assert len(rev2_filters) == 1
    assert "TukeyOutlierRemover" in json.dumps(rev2_filters)
    assert "1 filters" in json.dumps(
        _response_value(
            rendered_rev2,
            _ids.ANALYSIS_PIPELINE_HEADER,
            "children",
        )
    )


def test_blocked_render_and_concurrent_mutation_are_revision_coherent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older render is coherent and completes before the latest trigger."""
    app = create_app(
        output_root=_seed_output(
            tmp_path,
            pipeline=ImagePipeline(
                name="before",
                model=LinearLagModel(
                    on="Shape_Area",
                    groupby=["Metadata_Strain"],
                ),
            ),
        )
    )
    recipe = app.server.config[CFG_RECIPE_STATE]
    model_response = _post_value_mutation(
        app,
        _ids.ANALYSIS_MODEL_DROPDOWN,
        "LogGrowthModel",
    )
    assert model_response.status_code == 200
    old_trigger = _response_value(
        model_response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )

    render_entered = threading.Event()
    release_render = threading.Event()
    blocked_once = False
    blocked_once_lock = threading.Lock()
    real_build_stack = _callbacks.build_section_stack

    def _blocked_build_stack(*args: Any, **kwargs: Any) -> Any:
        nonlocal blocked_once
        rendered = real_build_stack(*args, **kwargs)
        with blocked_once_lock:
            should_block = not blocked_once
            if should_block:
                blocked_once = True
        if should_block:
            render_entered.set()
            assert release_render.wait(5.0)
        return rendered

    monkeypatch.setattr(
        _callbacks,
        "build_section_stack",
        _blocked_build_stack,
    )
    old_render_result: dict[str, Any] = {}

    def _render_old_revision() -> None:
        old_render_result["response"] = _post_reconciliation(
            app,
            old_trigger,
        )

    old_renderer = threading.Thread(target=_render_old_revision)
    old_renderer.start()
    assert render_entered.wait(5.0)

    mutation_finished = threading.Event()
    real_mutate_and_save = recipe.mutate_and_save

    def _observed_mutation(
        mutation: Callable[[ImagePipeline], bool | None],
    ) -> bool:
        result = real_mutate_and_save(mutation)
        mutation_finished.set()
        return result

    recipe.mutate_and_save = _observed_mutation  # type: ignore[method-assign]
    latest_mutation_result: dict[str, Any] = {}

    def _mutate_latest_revision() -> None:
        latest_mutation_result["response"] = _post_value_mutation(
            app,
            _ids.ANALYSIS_FILTER_ADD_DROPDOWN,
            "TukeyOutlierRemover",
        )

    latest_mutation = threading.Thread(target=_mutate_latest_revision)
    latest_mutation.start()
    assert mutation_finished.wait(5.0)
    assert latest_mutation.is_alive()

    release_render.set()
    old_renderer.join(timeout=5.0)
    latest_mutation.join(timeout=5.0)
    assert not old_renderer.is_alive()
    assert not latest_mutation.is_alive()

    old_render = old_render_result["response"]
    assert old_render.status_code == 200
    assert (
        _response_value(
            old_render,
            _ids.ANALYSIS_FILTER_STACK,
            "children",
        )
        == []
    )
    assert "0 filters" in json.dumps(
        _response_value(
            old_render,
            _ids.ANALYSIS_PIPELINE_HEADER,
            "children",
        )
    )
    assert "LogGrowthModel" in json.dumps(
        _response_value(
            old_render,
            _ids.ANALYSIS_MODEL_SECTION,
            "children",
        )
    )

    latest_mutation_response = latest_mutation_result["response"]
    assert latest_mutation_response.status_code == 200
    latest_trigger = _response_value(
        latest_mutation_response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    assert latest_trigger["revision"] > old_trigger["revision"]
    latest_render = _post_reconciliation(app, latest_trigger)
    assert latest_render.status_code == 200
    latest_filter_stack = _response_value(
        latest_render,
        _ids.ANALYSIS_FILTER_STACK,
        "children",
    )
    assert len(latest_filter_stack) == 1
    assert "TukeyOutlierRemover" in json.dumps(latest_filter_stack)
    assert "1 filters" in json.dumps(
        _response_value(
            latest_render,
            _ids.ANALYSIS_PIPELINE_HEADER,
            "children",
        )
    )
    assert "LogGrowthModel" in json.dumps(
        _response_value(
            latest_render,
            _ids.ANALYSIS_MODEL_SECTION,
            "children",
        )
    )


def test_rejected_model_callback_restores_authoritative_dropdown_and_state(
    tmp_path: Path,
) -> None:
    """A real Dash request cannot display or retain a rejected model edit."""
    app = create_app(output_root=_seed_output(tmp_path))
    recipe = app.server.config[CFG_RECIPE_STATE]
    original_bytes = recipe.path.read_bytes()
    recipe.publication_guard = lambda: False
    output_key = _callback_key_for_input(
        app,
        _ids.ANALYSIS_MODEL_DROPDOWN,
    )

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": [
                {
                    "id": _ids.ANALYSIS_MODEL_DROPDOWN,
                    "property": "value",
                    "value": "LogGrowthModel",
                }
            ],
            "state": [],
            "changedPropIds": [f"{_ids.ANALYSIS_MODEL_DROPDOWN}.value"],
        },
    )

    assert response.status_code == 200
    trigger = _response_value(
        response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    rendered = _post_reconciliation(app, trigger)
    assert rendered.status_code == 200
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_MODEL_DROPDOWN,
            "value",
        )
        == ""
    )
    assert json.loads(trigger["pipeline_json"]) == json.loads(original_bytes)
    assert recipe.pipeline.get_model() is None
    assert recipe.path.read_bytes() == original_bytes

    recipe.publication_guard = lambda: True
    assert (
        recipe.mutate_and_save(
            lambda pipeline: setattr(pipeline, "name", "unrelated")
        )
        is True
    )
    saved = json.loads(recipe.path.read_text(encoding="utf-8"))
    assert saved["name"] == "unrelated"
    assert saved["model"] is None


def test_external_recipe_cas_reconciles_every_model_callback_output(
    tmp_path: Path,
) -> None:
    """CAS refusal returns one coherent authoritative model rendering."""
    app = create_app(output_root=_seed_output(tmp_path))
    recipe = app.server.config[CFG_RECIPE_STATE]
    external = ImagePipeline(
        name="external",
        model=LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        ),
    )
    external_json = external.to_json() or ""
    recipe.path.write_text(external_json, encoding="utf-8")
    output_key = _callback_key_for_input(
        app,
        _ids.ANALYSIS_MODEL_DROPDOWN,
    )

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": [
                {
                    "id": _ids.ANALYSIS_MODEL_DROPDOWN,
                    "property": "value",
                    "value": "LogGrowthModel",
                }
            ],
            "state": [],
            "changedPropIds": [f"{_ids.ANALYSIS_MODEL_DROPDOWN}.value"],
        },
    )

    assert response.status_code == 200
    trigger = _response_value(
        response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    rendered = _post_reconciliation(app, trigger)
    assert rendered.status_code == 200
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_MODEL_DROPDOWN,
            "value",
        )
        == "LinearLagModel"
    )
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_RUN_BUTTON,
            "disabled",
        )
        is False
    )
    model_section = _response_value(
        rendered,
        _ids.ANALYSIS_MODEL_SECTION,
        "children",
    )
    header = _response_value(
        rendered,
        _ids.ANALYSIS_PIPELINE_HEADER,
        "children",
    )
    assert "LinearLagModel" in json.dumps(model_section)
    assert "external" in json.dumps(header)
    assert json.loads(trigger["pipeline_json"]) == json.loads(external_json)
    assert recipe.pipeline.name == "external"
    assert type(recipe.pipeline.get_model()).__name__ == "LinearLagModel"
    assert json.loads(recipe.path.read_text(encoding="utf-8")) == json.loads(
        external_json
    )

    assert (
        recipe.mutate_and_save(
            lambda pipeline: setattr(pipeline, "name", "external-unrelated")
        )
        is True
    )
    saved = json.loads(recipe.path.read_text(encoding="utf-8"))
    assert saved["name"] == "external-unrelated"
    assert saved["model"]["class"] == "LinearLagModel"


def test_external_recipe_cas_reconciles_add_stack_and_control(
    tmp_path: Path,
) -> None:
    """A rejected Add request returns the reloaded stack and clears control."""
    app = create_app(output_root=_seed_output(tmp_path))
    recipe = app.server.config[CFG_RECIPE_STATE]
    external = ImagePipeline(
        name="external-add",
        post=[
            AppendString(
                column="Metadata_Strain",
                value="_external",
            )
        ],
        filters={
            "external_filter": TukeyOutlierRemover(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
            ),
            "external_edge": EdgeCorrector(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
            ),
        },
        model=LinearLagModel(
            on="Shape_Area",
            groupby=["Metadata_Strain"],
        ),
    )
    external_json = external.to_json() or ""
    recipe.path.write_text(external_json, encoding="utf-8")
    output_key = _callback_key_for_input(
        app,
        _ids.ANALYSIS_FILTER_ADD_DROPDOWN,
    )

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": [
                {
                    "id": _ids.ANALYSIS_FILTER_ADD_DROPDOWN,
                    "property": "value",
                    "value": "TukeyOutlierRemover",
                }
            ],
            "state": [],
            "changedPropIds": [f"{_ids.ANALYSIS_FILTER_ADD_DROPDOWN}.value"],
        },
    )

    assert response.status_code == 200
    trigger = _response_value(
        response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    rendered = _post_reconciliation(app, trigger)
    assert rendered.status_code == 200
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_FILTER_ADD_DROPDOWN,
            "value",
        )
        is None
    )
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_POST_ADD_DROPDOWN,
            "value",
        )
        is None
    )
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_EDGE_ADD_DROPDOWN,
            "value",
        )
        is None
    )
    post_stack = _response_value(
        rendered,
        _ids.ANALYSIS_POST_STACK,
        "children",
    )
    filter_stack = _response_value(
        rendered,
        _ids.ANALYSIS_FILTER_STACK,
        "children",
    )
    edge_stack = _response_value(
        rendered,
        _ids.ANALYSIS_EDGE_STACK,
        "children",
    )
    model_section = _response_value(
        rendered,
        _ids.ANALYSIS_MODEL_SECTION,
        "children",
    )
    header = _response_value(
        rendered,
        _ids.ANALYSIS_PIPELINE_HEADER,
        "children",
    )
    assert len(post_stack) == 1
    assert "AppendString" in json.dumps(post_stack)
    assert len(filter_stack) == 1
    assert "external_filter" in json.dumps(filter_stack)
    assert len(edge_stack) == 1
    assert "external_edge" in json.dumps(edge_stack)
    assert "LinearLagModel" in json.dumps(model_section)
    assert "external-add" in json.dumps(header)
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_MODEL_DROPDOWN,
            "value",
        )
        == "LinearLagModel"
    )
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_RUN_BUTTON,
            "disabled",
        )
        is False
    )
    assert json.loads(trigger["pipeline_json"]) == json.loads(external_json)
    assert json.loads(recipe.path.read_text(encoding="utf-8")) == json.loads(
        external_json
    )


def test_external_recipe_cas_reconciles_all_remove_stacks(
    tmp_path: Path,
) -> None:
    """A rejected Remove request returns every reloaded analyzer stack."""
    original = ImagePipeline(
        name="before-remove",
        filters={
            "old_filter": TukeyOutlierRemover(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
            )
        },
    )
    app = create_app(output_root=_seed_output(tmp_path, pipeline=original))
    recipe = app.server.config[CFG_RECIPE_STATE]
    external = ImagePipeline(
        name="external-remove",
        filters={
            "new_filter": TukeyOutlierRemover(
                on="Shape_Area",
                groupby=["Metadata_Strain"],
            )
        },
    )
    external_json = external.to_json() or ""
    recipe.path.write_text(external_json, encoding="utf-8")
    output_key = next(
        key
        for key, callback in app.callback_map.items()
        if any(
            "analysis-section-remove" in callback_input["id"]
            for callback_input in callback["inputs"]
        )
    )
    concrete_id = {
        "index": 0,
        "kind": "filter",
        "type": "analysis-section-remove",
    }

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _single_output_from_key(output_key),
            "inputs": [
                {
                    "id": concrete_id,
                    "property": "n_clicks",
                    "value": 1,
                }
            ],
            "state": [],
            "changedPropIds": [
                f"{json.dumps(concrete_id, separators=(',', ':'))}.n_clicks"
            ],
        },
    )

    assert response.status_code == 200
    trigger = _response_value(
        response,
        _ids.ANALYSIS_PIPELINE_EVENT_STORE,
        "data",
    )
    rendered = _post_reconciliation(app, trigger)
    assert rendered.status_code == 200
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_POST_STACK,
            "children",
        )
        == []
    )
    assert (
        _response_value(
            rendered,
            _ids.ANALYSIS_EDGE_STACK,
            "children",
        )
        == []
    )
    filter_stack = _response_value(
        rendered,
        _ids.ANALYSIS_FILTER_STACK,
        "children",
    )
    assert len(filter_stack) == 1
    assert "new_filter" in json.dumps(filter_stack)
    assert "external-remove" in json.dumps(
        _response_value(
            rendered,
            _ids.ANALYSIS_PIPELINE_HEADER,
            "children",
        )
    )
    assert json.loads(trigger["pipeline_json"]) == json.loads(external_json)
    assert json.loads(recipe.path.read_text(encoding="utf-8")) == json.loads(
        external_json
    )
