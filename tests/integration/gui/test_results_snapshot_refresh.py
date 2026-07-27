"""Shared Results/Analysis snapshot binding and refresh integration tests."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.gui import analysis, results_viewer
from phenotypic.gui import analysis as analysis_module
from phenotypic.gui import results_viewer as results_viewer_module
from phenotypic.gui._binding_generation import (
    BINDING_GENERATION_PAYLOAD_KEY,
)
from phenotypic.gui._config import (
    CFG_ANALYSIS_SESSION,
    CFG_FILTERED_STATE,
    CFG_OUTPUT_ROOT,
    CFG_RECIPE_STATE,
    CFG_RESULTS_BINDING_COORDINATOR,
    CFG_RESULTS_BINDING_STATE,
)
from phenotypic.gui._shared._radial import RADIAL_RESTORE_SENTINEL
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer import _ids as results_ids
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._binding import BindingCoordinator
from phenotypic.gui.shell._session import ToolSession
from phenotypic.schema import METADATA
from phenotypic.sdk_ import atomic_write_json, gui_launch_owner_path
from tests._output_layout import seed_output_dir, write_complete_manifest


def _seed_output(parent: Path, name: str = "output") -> Path:
    """Create a minimal complete output accepted by both sub-apps."""
    output = parent / name
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
        pipeline=ImagePipeline(name="snapshot-refresh"),
    )
    (output / "results" / "dataset" / "measurements").mkdir(
        parents=True,
        exist_ok=True,
    )
    overlay = output / "deliverables" / "overlays" / "dataset" / "plate.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"overlay")
    write_complete_manifest(output, total_images=1)
    return output


def _source_tree(root: Path) -> dict[str, tuple[str, bytes | None]]:
    """Capture relative type and bytes for an immutable-tree assertion."""
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            snapshot[relative] = ("dir", None)
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _submit_binding(
    client: Any,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Submit one asynchronous binding request and await its terminal job."""
    response = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 202, response.get_json()
    return _poll_binding(client, response.get_json(), timeout=timeout)


def _poll_binding(
    client: Any,
    accepted: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Poll an already-accepted binding job to terminal state."""
    poll_path = accepted["poll_path"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        polled = client.get(poll_path)
        assert polled.status_code == 200, polled.get_json()
        terminal = polled.get_json()
        if terminal["job"]["terminal"]:
            return terminal
        threading.Event().wait(0.01)
    raise AssertionError(f"binding job did not become terminal: {accepted}")


def _bind(client: Any, relative: str) -> dict[str, Any]:
    """Bind one output and assert paired publication succeeded."""
    terminal = _submit_binding(client, {"path": relative})
    assert terminal["status"] == "succeeded", terminal
    return terminal


def _rewrite_mirror(output_root: OutputRoot, increment: float = 1.0) -> None:
    """Atomically publish a distinct measurements mirror revision."""
    mirror = output_root.layout.mirror_parquet
    replacement = mirror.with_name("measurements-replacement.parquet")
    (
        pl.read_parquet(mirror)
        .with_columns((pl.col("Shape_Area") + increment).alias("Shape_Area"))
        .write_parquet(replacement)
    )
    replacement.replace(mirror)


def _wait_for(predicate: Any, *, timeout: float = 5.0) -> None:
    """Wait until a deterministic concurrency predicate becomes true."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for concurrency state")
        threading.Event().wait(0.01)


def _outputs_from_key(output_key: str) -> list[dict[str, str]]:
    """Parse one Dash callback-map key into its response output records."""
    outputs: list[dict[str, str]] = []
    for segment in re.split(r"\.\.\.", output_key.strip(".")):
        component = segment.strip(".").split("@", 1)[0]
        component_id, prop = component.rsplit(".", 1)
        outputs.append({"id": component_id, "property": prop})
    return outputs


def _find_output_key(app: Any, *substrings: str) -> str:
    """Return the first callback key containing every substring."""
    for key, callback in app.callback_map.items():
        searchable = key
        for callback_input in callback.get("inputs", []):
            if isinstance(callback_input, dict):
                searchable += str(callback_input.get("id", ""))
            elif isinstance(callback_input, list):
                for item in callback_input:
                    if isinstance(item, dict):
                        searchable += str(item.get("id", ""))
        if all(substring in searchable for substring in substrings):
            return key
    raise KeyError(substrings)


def _find_component(component: Any, component_id: str) -> Any:
    """Find one Dash component recursively by its string id."""
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if child is None or isinstance(child, (str, int, float)):
            continue
        try:
            return _find_component(child, component_id)
        except KeyError:
            continue
    raise KeyError(component_id)


def _post_bound_colony_wedge(
    app: Any,
    *,
    generation: str,
    image_file: str,
    label: int,
    category: str,
) -> Any:
    """POST one generation-fenced colony curation action."""
    output_key = _find_output_key(
        app,
        f"{results_ids.STORE_REMOVED_KEYS}.data",
        "colony-cat-wedge",
    )
    triggered_id = {
        "type": "colony-cat-wedge",
        "image_file": image_file,
        "label": label,
        "category": category,
    }
    changed_prop = (
        '{"category":"%s","image_file":"%s","label":%d,'
        '"type":"colony-cat-wedge"}.n_clicks'
        % (category, image_file, label)
    )
    return app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output_key,
            "outputs": _outputs_from_key(output_key),
            "inputs": [
                [
                    {
                        "id": triggered_id,
                        "property": "n_clicks",
                        "value": 1,
                    }
                ]
            ],
            "state": [],
            "changedPropIds": [changed_prop],
            BINDING_GENERATION_PAYLOAD_KEY: generation,
        },
    )


def _threaded_post(
    app: Any,
    payload: dict[str, Any],
    result: dict[str, Any],
    key: str,
) -> None:
    """Submit/poll from a thread-local Flask client and retain terminal data."""
    terminal = _submit_binding(
        app.server.test_client(),
        payload,
    )
    result[key] = (terminal["status"], terminal)


def _callback_probe(
    app: Any,
    *,
    generation: str | None,
    component_id: str,
) -> Any:
    """Issue one mutation-shaped Dash callback payload."""
    payload: dict[str, Any] = {
        "output": f"{component_id}.data",
        "outputs": {"id": component_id, "property": "data"},
        "inputs": [],
        "state": [],
        "changedPropIds": [],
    }
    if generation is not None:
        payload[BINDING_GENERATION_PAYLOAD_KEY] = generation
    return app.server.test_client().post(
        "/_dash-update-component",
        json=payload,
    )


def test_bind_is_source_preserving_and_uses_external_cache(
    tmp_path: Path,
) -> None:
    """Binding reads the selected tree without migrating or caching in it."""
    output = _seed_output(tmp_path)
    legacy_sidecar = output / ".viewer_cache" / "qc_recipe.json"
    legacy_sidecar.parent.mkdir(parents=True)
    legacy_sidecar.write_text('{"checks": []}', encoding="utf-8")
    before = _source_tree(output)

    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    response = _bind(shell_app.server.test_client(), output.name)
    bound = viewer_session.get().server.config[CFG_OUTPUT_ROOT]

    assert response["snapshot"]["processing_fingerprint"]
    assert _source_tree(output) == before
    assert bound.cache_dir.is_relative_to(
        tmp_path / ".phenotypic-gui" / "viewer_cache"
    )
    assert not bound.cache_dir.is_relative_to(output)


def test_refresh_atomically_swaps_results_and_analysis_to_one_descriptor(
    tmp_path: Path,
) -> None:
    """A successful refresh publishes both consumers at one revision."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)

    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    assert old_root is old_analysis.server.config[CFG_OUTPUT_ROOT]

    _rewrite_mirror(old_root)
    response = _submit_binding(client, {"refresh": True})

    assert response["status"] == "succeeded", response
    new_viewer = viewer_session.get()
    new_analysis = analysis_session.get()
    new_root = new_viewer.server.config[CFG_OUTPUT_ROOT]
    assert new_viewer is not old_viewer
    assert new_analysis is not old_analysis
    assert new_root is new_analysis.server.config[CFG_OUTPUT_ROOT]
    assert new_root.snapshot != old_root.snapshot
    assert shell_app.server.config[CFG_RESULTS_BINDING_STATE]["snapshot"] is (
        new_root.snapshot
    )


def test_failed_refresh_keeps_both_live_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction failure rolls back without publishing either candidate."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_state = dict(shell_app.server.config[CFG_RESULTS_BINDING_STATE])
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    _rewrite_mirror(old_root)

    def _raise_analysis_failure(**_kwargs: Any) -> Any:
        raise RuntimeError("candidate analysis failed")

    monkeypatch.setattr(analysis, "create_app", _raise_analysis_failure)
    response = _submit_binding(client, {"refresh": True})

    assert response["status"] == "failed"
    assert response["job"]["error_kind"] == "unavailable"
    assert response["job"]["error"] == "candidate analysis failed"
    assert viewer_session.get() is old_viewer
    assert analysis_session.get() is old_analysis
    assert shell_app.server.config[CFG_RESULTS_BINDING_STATE] == old_state


def test_concurrent_source_change_is_stale_and_keeps_old_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source race after Results construction is refused before publish."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    real_results_factory = results_viewer.create_app

    def _build_then_change(**kwargs: Any) -> Any:
        candidate = real_results_factory(**kwargs)
        _rewrite_mirror(kwargs["output_root"])
        return candidate

    monkeypatch.setattr(results_viewer, "create_app", _build_then_change)
    response = _submit_binding(client, {"refresh": True})

    assert response["status"] == "failed"
    assert response["job"]["error_kind"] == "stale"
    assert viewer_session.get() is old_viewer
    assert analysis_session.get() is old_analysis
    assert viewer_session.get().server.config[CFG_OUTPUT_ROOT] is old_root


def test_cache_ownership_relocates_with_equivalent_sandbox(
    tmp_path: Path,
) -> None:
    """Equivalent copied outputs use the configured sandbox cache owner."""
    roots: list[OutputRoot] = []
    for sandbox_name in ("sandbox-a", "sandbox-b"):
        sandbox_path = tmp_path / sandbox_name
        sandbox_path.mkdir()
        output = _seed_output(sandbox_path)
        before = _source_tree(output)
        shell_app, viewer_session = compose_hub(
            SandboxRoot.from_path(sandbox_path),
            start_idle_thread=False,
        )
        _bind(shell_app.server.test_client(), output.name)
        bound = viewer_session.get().server.config[CFG_OUTPUT_ROOT]
        roots.append(bound)
        assert bound.cache_dir.is_relative_to(
            sandbox_path / ".phenotypic-gui" / "viewer_cache"
        )
        assert _source_tree(output) == before

    assert roots[0].cache_dir != roots[1].cache_dir
    assert roots[0].root != roots[1].root


def test_nonterminal_owner_marks_both_apps_as_active_snapshot(
    tmp_path: Path,
) -> None:
    """The shared descriptor records and displays active-run capture state."""
    output = _seed_output(tmp_path)
    owner = gui_launch_owner_path(output)
    owner.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "active",
            "generation": "generation-active",
            "status": "running",
        },
    )
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )

    response = _bind(shell_app.server.test_client(), output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    viewer_root = viewer_session.get().server.config[CFG_OUTPUT_ROOT]
    analysis_root = analysis_session.get().server.config[CFG_OUTPUT_ROOT]

    assert response["snapshot"]["active_run"] is True
    assert viewer_root.snapshot.active_run is True
    assert analysis_root.snapshot is viewer_root.snapshot
    assert "Active run snapshot" in str(viewer_session.get().layout)
    assert "Active run snapshot" in str(analysis_session.get().layout)
    assert not any(
        results_ids.STORE_REMOVED_KEYS in key
        or results_ids.STORE_QC_RECIPE_REVISION in key
        for key in viewer_session.get().callback_map
    )
    assert not any(
        analysis_ids.ANALYSIS_PIPELINE_STORE in key
        for key in analysis_session.get().callback_map
    )

    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "active",
            "generation": "generation-active",
            "status": "complete",
        },
    )
    refreshed = _submit_binding(
        shell_app.server.test_client(),
        {"refresh": True},
    )
    assert refreshed["status"] == "succeeded"
    assert any(
        results_ids.STORE_REMOVED_KEYS in key
        for key in viewer_session.get().callback_map
    )
    assert any(
        analysis_ids.ANALYSIS_PIPELINE_STORE in key
        for key in analysis_session.get().callback_map
    )


def test_final_publish_gap_change_returns_stale_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The locked commit rechecks after both candidate apps finish building."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    real_factory = analysis_module.create_app

    def _build_then_change(**kwargs: Any) -> Any:
        candidate = real_factory(**kwargs)
        _rewrite_mirror(kwargs["output_root"])
        return candidate

    monkeypatch.setattr(analysis_module, "create_app", _build_then_change)
    response = _submit_binding(client, {"refresh": True})

    assert response["status"] == "failed"
    assert response["job"]["error_kind"] == "stale"
    assert viewer_session.get() is old_viewer
    assert analysis_session.get() is old_analysis
    # Failed publication reopens the old fence rather than wedging the page.
    generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    current_page_probe = old_viewer.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": "unknown.children",
            "outputs": {"id": "unknown", "property": "children"},
            "inputs": [],
            "state": [],
            "changedPropIds": [],
            BINDING_GENERATION_PAYLOAD_KEY: generation,
        },
    )
    assert current_page_probe.status_code != 409
    assert old_root.refresh_state_is_current() is False


def test_newer_bind_supersedes_slow_older_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow A request cannot publish after the newer B request is issued."""
    output_a = _seed_output(tmp_path, "output-a")
    output_b = _seed_output(tmp_path, "output-b")
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    coordinator: BindingCoordinator = shell_app.server.config[
        CFG_RESULTS_BINDING_COORDINATOR
    ]
    entered = threading.Event()
    release = threading.Event()
    real_factory = results_viewer_module.create_app

    def _slow_a(**kwargs: Any) -> Any:
        output_root = kwargs.get("output_root")
        if output_root is not None and output_root.root == output_a:
            entered.set()
            assert release.wait(5.0)
        return real_factory(**kwargs)

    monkeypatch.setattr(results_viewer_module, "create_app", _slow_a)
    responses: dict[str, Any] = {}
    thread_a = threading.Thread(
        target=_threaded_post,
        args=(shell_app, {"path": output_a.name}, responses, "a"),
    )
    thread_a.start()
    assert entered.wait(5.0)
    ticket_a = coordinator.latest_request
    thread_b = threading.Thread(
        target=_threaded_post,
        args=(shell_app, {"path": output_b.name}, responses, "b"),
    )
    thread_b.start()
    _wait_for(lambda: coordinator.latest_request == ticket_a + 1)
    release.set()
    thread_a.join(10.0)
    thread_b.join(10.0)

    assert responses["a"][0] == "superseded"
    assert responses["b"][0] == "succeeded"
    state = shell_app.server.config[CFG_RESULTS_BINDING_STATE]
    assert state["bound_path"] == output_b


def test_duplicate_refresh_reuses_slow_active_refresh_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate Refresh reuses the same active job and publication ticket."""
    output = _seed_output(tmp_path)
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    _bind(shell_app.server.test_client(), output.name)
    coordinator: BindingCoordinator = shell_app.server.config[
        CFG_RESULTS_BINDING_COORDINATOR
    ]
    old_generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    entered = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    blocked = False
    real_factory = results_viewer_module.create_app

    def _slow_first_refresh(**kwargs: Any) -> Any:
        nonlocal blocked
        should_block = False
        with call_lock:
            if not blocked:
                blocked = True
                should_block = True
        if should_block:
            entered.set()
            assert release.wait(5.0)
        return real_factory(**kwargs)

    monkeypatch.setattr(
        results_viewer_module,
        "create_app",
        _slow_first_refresh,
    )
    responses: dict[str, Any] = {}
    older = threading.Thread(
        target=_threaded_post,
        args=(shell_app, {"refresh": True}, responses, "older"),
    )
    older.start()
    assert entered.wait(5.0)
    older_ticket = coordinator.latest_request
    newer_response = shell_app.server.test_client().post(
        "/sandbox/api/viewer/output-root",
        json={"refresh": True},
    )
    assert newer_response.status_code == 202
    newer_accepted = newer_response.get_json()
    assert newer_accepted["deduplicated"] is True
    assert coordinator.latest_request == older_ticket
    release.set()
    responses["newer"] = (
        "succeeded",
        _poll_binding(
            shell_app.server.test_client(),
            newer_accepted,
        ),
    )
    older.join(10.0)

    assert responses["older"][0] == "succeeded"
    assert responses["newer"][0] == "succeeded"
    assert responses["older"][1]["job_id"] == responses["newer"][1]["job_id"]
    state = shell_app.server.config[CFG_RESULTS_BINDING_STATE]
    assert state["binding_generation"] != old_generation
    assert state["snapshot"].captured_at.isoformat() == (
        responses["newer"][1]["snapshot"]["captured_at"]
    )


def test_stale_results_qc_and_analysis_posts_are_rejected_before_dispatch(
    tmp_path: Path,
) -> None:
    """Old page generations cannot mutate a newly bound output."""
    output_a = _seed_output(tmp_path, "output-a")
    output_b = _seed_output(tmp_path, "output-b")
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output_a.name)
    stale_generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    _bind(client, output_b.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    current_generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    for app in (viewer_session.get(), analysis_session.get()):
        assert BINDING_GENERATION_PAYLOAD_KEY in app.renderer
        assert current_generation in app.index_string
        assert current_generation in str(app.layout)
    before = _source_tree(output_b)

    probes = (
        (
            viewer_session.get(),
            results_ids.STORE_REMOVED_KEYS,
        ),
        (
            viewer_session.get(),
            results_ids.STORE_QC_RECIPE_REVISION,
        ),
        (
            analysis_session.get(),
            analysis_ids.ANALYSIS_PIPELINE_EVENT_STORE,
        ),
    )
    for app, component_id in probes:
        response = _callback_probe(
            app,
            generation=stale_generation,
            component_id=component_id,
        )
        assert response.status_code == 409
        assert response.get_json()["status"] == "stale_binding"
    assert _source_tree(output_b) == before


@pytest.mark.parametrize("owner_status", ["running", "unknown"])
def test_nonterminal_run_after_bind_blocks_current_page_mutations(
    tmp_path: Path,
    owner_status: str,
) -> None:
    """Every registry nonterminal state makes a bound page read-only."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    owner = gui_launch_owner_path(output)
    owner.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "new-active-run",
            "generation": "new-active-generation",
            "status": owner_status,
        },
    )
    before = _source_tree(output)

    for app, component_id in (
        (viewer_session.get(), results_ids.STORE_REMOVED_KEYS),
        (viewer_session.get(), results_ids.STORE_QC_RECIPE_REVISION),
        (
            analysis_session.get(),
            analysis_ids.ANALYSIS_PIPELINE_EVENT_STORE,
        ),
    ):
        response = _callback_probe(
            app,
            generation=generation,
            component_id=component_id,
        )
        assert response.status_code == 423
        assert response.get_json()["status"] == "bound_output_read_only"
    viewer_app = viewer_session.get()
    status_key = next(
        key
        for key in viewer_app.callback_map
        if results_ids.HEADER_SNAPSHOT_STATUS_ID in key
    )
    status_response = viewer_app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": status_key,
            "outputs": _outputs_from_key(status_key),
            "inputs": [
                {
                    "id": results_ids.SNAPSHOT_STATUS_INTERVAL_ID,
                    "property": "n_intervals",
                    "value": 1,
                }
            ],
            "state": [],
            "changedPropIds": [
                f"{results_ids.SNAPSHOT_STATUS_INTERVAL_ID}.n_intervals"
            ],
            BINDING_GENERATION_PAYLOAD_KEY: generation,
        },
    )
    assert status_response.status_code == 200
    assert "Active run detected" in status_response.get_data(as_text=True)
    assert _source_tree(output) == before


def test_standalone_apps_block_processing_stale_mutation_callbacks(
    tmp_path: Path,
) -> None:
    """Standalone Results and Analysis enforce the processing snapshot guard."""
    output = _seed_output(tmp_path)
    root = OutputRoot.discover(
        output,
        cache_root=tmp_path / "viewer-cache",
    )
    viewer_app = results_viewer.create_app(root)
    analysis_app = analysis.create_app(output_root=root)
    assert _find_component(
        viewer_app.layout,
        results_ids.BTN_REFRESH_SNAPSHOT,
    ).disabled is True
    assert _find_component(
        analysis_app.layout,
        analysis_ids.ANALYSIS_REFRESH_SNAPSHOT,
    ).disabled is True
    assert all(
        results_ids.HEADER_REFRESH_ERROR_ID not in key
        for key in viewer_app.callback_map
    )
    assert all(
        analysis_ids.ANALYSIS_REFRESH_ERROR not in key
        for key in analysis_app.callback_map
    )
    owner = gui_launch_owner_path(output)
    owner.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "standalone-active",
            "generation": "standalone-generation",
            "status": "running",
        },
    )
    for app, status_id, interval_id in (
        (
            viewer_app,
            results_ids.HEADER_SNAPSHOT_STATUS_ID,
            results_ids.SNAPSHOT_STATUS_INTERVAL_ID,
        ),
        (
            analysis_app,
            analysis_ids.ANALYSIS_SNAPSHOT_STATUS,
            analysis_ids.ANALYSIS_SNAPSHOT_INTERVAL,
        ),
    ):
        status_key = _find_output_key(app, status_id)
        response = app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": status_key,
                "outputs": _outputs_from_key(status_key),
                "inputs": [
                    {
                        "id": interval_id,
                        "property": "n_intervals",
                        "value": 1,
                    }
                ],
                "state": [],
                "changedPropIds": [f"{interval_id}.n_intervals"],
            },
        )
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "restart app after it finishes" in body
        assert '"disabled":true' in body
    active_root = OutputRoot.discover(
        output,
        cache_root=tmp_path / "active-viewer-cache",
    )
    assert active_root.snapshot.active_run is True
    active_viewer_app = results_viewer.create_app(active_root)
    active_analysis_app = analysis.create_app(output_root=active_root)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "standalone-active",
            "generation": "standalone-generation",
            "status": "complete",
        },
    )
    for app, status_id, interval_id in (
        (
            active_viewer_app,
            results_ids.HEADER_SNAPSHOT_STATUS_ID,
            results_ids.SNAPSHOT_STATUS_INTERVAL_ID,
        ),
        (
            active_analysis_app,
            analysis_ids.ANALYSIS_SNAPSHOT_STATUS,
            analysis_ids.ANALYSIS_SNAPSHOT_INTERVAL,
        ),
    ):
        status_key = _find_output_key(app, status_id)
        response = app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": status_key,
                "outputs": _outputs_from_key(status_key),
                "inputs": [
                    {
                        "id": interval_id,
                        "property": "n_intervals",
                        "value": 2,
                    }
                ],
                "state": [],
                "changedPropIds": [f"{interval_id}.n_intervals"],
            },
        )
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "Run finished" in body
        assert "restart standalone app" in body
        assert '"disabled":true' in body
    overlay = output / "deliverables" / "overlays" / "dataset" / "plate.png"
    overlay.write_bytes(b"externally-rewritten-overlay")

    for app, component_id in (
        (viewer_app, results_ids.STORE_REMOVED_KEYS),
        (viewer_app, results_ids.STORE_QC_RECIPE_REVISION),
        (analysis_app, analysis_ids.ANALYSIS_PIPELINE_EVENT_STORE),
    ):
        response = _callback_probe(
            app,
            generation=None,
            component_id=component_id,
        )
        assert response.status_code == 423
        assert response.get_json()["status"] == "bound_output_read_only"


def test_hub_allows_two_sequential_curation_writes(
    tmp_path: Path,
) -> None:
    """GUI-owned consumed-state changes do not make curation one-shot."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    viewer_app = viewer_session.get()
    labels: CurationLabels = viewer_app.server.config[CFG_FILTERED_STATE]

    marked = _post_bound_colony_wedge(
        viewer_app,
        generation=generation,
        image_file="plate",
        label=1,
        category="debris",
    )
    restored = _post_bound_colony_wedge(
        viewer_app,
        generation=generation,
        image_file="plate",
        label=1,
        category=RADIAL_RESTORE_SENTINEL,
    )

    assert marked.status_code == 200
    assert restored.status_code == 200
    assert labels.is_removed("plate", 1) is False


def test_external_consumed_write_is_refused_by_curation_cas(
    tmp_path: Path,
) -> None:
    """External mirror replacement reaches, then loses, the writer's CAS."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    viewer_app = viewer_session.get()
    bound_root = viewer_app.server.config[CFG_OUTPUT_ROOT]
    labels: CurationLabels = viewer_app.server.config[CFG_FILTERED_STATE]
    _rewrite_mirror(bound_root)
    externally_written = bound_root.layout.mirror_parquet.read_bytes()

    response = _post_bound_colony_wedge(
        viewer_app,
        generation=generation,
        image_file="plate",
        label=1,
        category="debris",
    )

    assert response.status_code == 200
    assert labels.stale is True
    assert bound_root.layout.mirror_parquet.read_bytes() == externally_written


def test_publish_waits_for_admitted_analysis_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication drains an old Analysis writer before swapping sessions."""
    output_a = _seed_output(tmp_path, "output-a")
    output_b = _seed_output(tmp_path, "output-b")
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output_a.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_analysis = analysis_session.get()
    old_recipe = old_analysis.server.config[CFG_RECIPE_STATE]
    generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "binding_generation"
    ]
    save_entered = threading.Event()
    release_save = threading.Event()
    real_save = RecipeState.save

    def _blocked_save(recipe: RecipeState) -> bool:
        if recipe is old_recipe:
            save_entered.set()
            assert release_save.wait(5.0)
        return real_save(recipe)

    monkeypatch.setattr(RecipeState, "save", _blocked_save)
    output_key = next(
        key
        for key, callback in old_analysis.callback_map.items()
        if any(
            callback_input["id"] == analysis_ids.ANALYSIS_MODEL_DROPDOWN
            for callback_input in callback["inputs"]
        )
    )
    writer_result: dict[str, Any] = {}

    def _post_writer() -> None:
        response = old_analysis.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": output_key,
                "outputs": _outputs_from_key(output_key)[0],
                "inputs": [
                    {
                        "id": analysis_ids.ANALYSIS_MODEL_DROPDOWN,
                        "property": "value",
                        "value": "LogGrowthModel",
                    }
                ],
                "state": [],
                "changedPropIds": [
                    f"{analysis_ids.ANALYSIS_MODEL_DROPDOWN}.value"
                ],
                BINDING_GENERATION_PAYLOAD_KEY: generation,
            },
        )
        writer_result["status"] = response.status_code

    writer = threading.Thread(target=_post_writer)
    writer.start()
    assert save_entered.wait(5.0)
    bind_result: dict[str, Any] = {}
    binder = threading.Thread(
        target=_threaded_post,
        args=(shell_app, {"path": output_b.name}, bind_result, "bind"),
    )
    binder.start()
    # Candidate build may finish, but publication must wait for the writer.
    threading.Event().wait(0.1)
    assert binder.is_alive()
    assert "bind" not in bind_result
    release_save.set()
    writer.join(10.0)
    binder.join(10.0)

    assert writer_result["status"] == 200
    assert bind_result["bind"][0] == "succeeded"
    assert shell_app.server.config[CFG_RESULTS_BINDING_STATE][
        "bound_path"
    ] == output_b


def test_stuck_callback_times_out_then_later_refresh_recovers(
    tmp_path: Path,
) -> None:
    """A stuck old callback cannot hold the binder lock indefinitely."""
    output_a = _seed_output(tmp_path, "output-a")
    output_b = _seed_output(tmp_path, "output-b")
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
        binding_drain_timeout_seconds=0.05,
    )
    client = shell_app.server.test_client()
    _bind(client, output_a.name)
    state = shell_app.server.config[CFG_RESULTS_BINDING_STATE]
    old_generation = state["binding_generation"]
    old_fence = state["binding_fence"]
    assert old_fence.try_enter() is True

    timed_out = _submit_binding(client, {"path": output_b.name})

    assert timed_out["status"] == "failed"
    assert timed_out["job"]["error_kind"] == "unavailable"
    assert state["bound_path"] == output_a
    assert state["binding_generation"] == old_generation
    # The rollback reopens the old binding even while the admitted request is
    # still represented, then releasing it allows a later Refresh to proceed.
    assert old_fence.try_enter() is True
    old_fence.leave()
    old_fence.leave()

    recovered = _submit_binding(client, {"path": output_b.name})
    assert recovered["status"] == "succeeded"
    assert state["bound_path"] == output_b
