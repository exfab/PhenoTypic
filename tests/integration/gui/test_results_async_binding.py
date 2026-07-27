"""Flask integration tests for asynchronous Results/Analysis binding."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.gui import results_viewer as results_viewer_module
from phenotypic.gui._config import (
    CFG_ANALYSIS_SESSION,
    CFG_OUTPUT_ROOT,
    CFG_RESULTS_BINDING_COORDINATOR,
    CFG_RESULTS_BINDING_JOBS,
    CFG_RESULTS_BINDING_STATE,
)
from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._binding import BindingCoordinator
from phenotypic.gui.shell._binding_jobs import ResultsBindJobManager
from phenotypic.gui.shell._binding_jobs import ResultsBindJobContext
from phenotypic.gui.shell._session import ToolSession
from phenotypic.schema import METADATA
from tests._output_layout import seed_output_dir


def _seed_output(parent: Path, name: str = "output") -> Path:
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
        pipeline=ImagePipeline(name="async-results"),
    )
    (output / "results" / "dataset" / "measurements").mkdir(
        parents=True,
        exist_ok=True,
    )
    return output


def _poll_terminal(client: Any, poll_path: str, *, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(poll_path)
        assert response.status_code == 200
        payload = response.get_json()
        if payload["job"]["terminal"]:
            return payload
        threading.Event().wait(0.01)
    raise AssertionError("binding job did not become terminal")


def test_post_returns_pollable_job_and_publishes_both_sessions(
    tmp_path: Path,
) -> None:
    """POST is proxy-safe while polling observes atomic paired publication."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    empty_viewer = viewer_session.get()
    empty_analysis = analysis_session.get()
    for empty_app in (empty_viewer, empty_analysis):
        assert any(
            "acceptedResponse.status === 202" in script
            and "selection.path" in script
            for script in empty_app._inline_scripts
        )
    client = shell_app.server.test_client()
    try:
        response = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": output.name},
        )
        assert response.status_code == 202
        accepted = response.get_json()
        assert accepted["abs_path"] == str(output)
        assert accepted["status"] in {"queued", "running"}
        assert response.headers["Location"] == accepted["poll_path"]

        complete = _poll_terminal(client, accepted["poll_path"])
        assert complete["status"] == "succeeded"
        assert complete["abs_path"] == str(output)
        assert complete["snapshot"]["processing_fingerprint"]
        state = shell_app.server.config[CFG_RESULTS_BINDING_STATE]
        assert state["bound_path"] == output
        assert viewer_session.is_built() is True
        assert analysis_session.is_built() is True
        assert viewer_session.get().server.config[CFG_OUTPUT_ROOT].root == output
        assert analysis_session.get().server.config[CFG_OUTPUT_ROOT].root == output
        for bound_app in (viewer_session.get(), analysis_session.get()):
            assert any(
                "acceptedResponse.status === 202" in script
                and "requestBody = {refresh: true}" in script
                for script in bound_app._inline_scripts
            )
    finally:
        manager.shutdown()


def test_duplicate_post_reuses_active_job(tmp_path: Path, monkeypatch: Any) -> None:
    """Two same-target POSTs share one job while discovery is active."""
    output = _seed_output(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    real_discover = results_viewer_module.OutputRoot.discover

    def _blocked_discover(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(5.0)
        return real_discover(*args, **kwargs)

    monkeypatch.setattr(
        results_viewer_module.OutputRoot,
        "discover",
        _blocked_discover,
    )
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    client = shell_app.server.test_client()
    try:
        first = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": output.name},
        ).get_json()
        assert entered.wait(5.0)
        duplicate = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": output.name},
        ).get_json()
        assert duplicate["deduplicated"] is True
        assert duplicate["job_id"] == first["job_id"]
        assert duplicate["job"]["ticket"] == first["job"]["ticket"]
    finally:
        release.set()
        manager.shutdown()


def test_failed_and_cancelled_jobs_preserve_prior_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-successful candidate never replaces the coherent live pair."""
    good = _seed_output(tmp_path, "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    client = shell_app.server.test_client()
    try:
        initial = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": good.name},
        ).get_json()
        assert _poll_terminal(client, initial["poll_path"])["status"] == "succeeded"
        analysis_session: ToolSession[Any] = shell_app.server.config[
            CFG_ANALYSIS_SESSION
        ]
        old_viewer = viewer_session.get()
        old_analysis = analysis_session.get()
        old_generation = shell_app.server.config[CFG_RESULTS_BINDING_STATE][
            "binding_generation"
        ]

        failed = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": bad.name},
        ).get_json()
        failure = _poll_terminal(client, failed["poll_path"])
        assert failure["status"] == "failed"
        assert failure["job"]["error_kind"] == "invalid"
        assert viewer_session.get() is old_viewer
        assert analysis_session.get() is old_analysis

        entered = threading.Event()
        release = threading.Event()
        real_discover = results_viewer_module.OutputRoot.discover

        def _blocked_discover(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert release.wait(5.0)
            return real_discover(*args, **kwargs)

        monkeypatch.setattr(
            results_viewer_module.OutputRoot,
            "discover",
            _blocked_discover,
        )
        cancelled = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": good.name},
        ).get_json()
        assert entered.wait(5.0)
        cancel_response = client.delete(cancelled["cancel_path"])
        assert cancel_response.status_code == 200
        assert cancel_response.get_json()["status"] == "cancelled"
        release.set()
        assert _poll_terminal(client, cancelled["poll_path"])["status"] == "cancelled"
        assert viewer_session.get() is old_viewer
        assert analysis_session.get() is old_analysis
        assert shell_app.server.config[CFG_RESULTS_BINDING_STATE][
            "binding_generation"
        ] == old_generation
    finally:
        manager.shutdown()


def test_candidate_construction_does_not_hold_publication_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery and candidate factories advance while publication is locked."""
    output = _seed_output(tmp_path)
    entered_factory = threading.Event()
    release_factory = threading.Event()
    real_factory = results_viewer_module.create_app

    def _blocked_factory(**kwargs: Any) -> Any:
        if kwargs.get("output_root") is not None:
            entered_factory.set()
            assert release_factory.wait(5.0)
        return real_factory(**kwargs)

    monkeypatch.setattr(results_viewer_module, "create_app", _blocked_factory)
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    coordinator: BindingCoordinator = shell_app.server.config[
        CFG_RESULTS_BINDING_COORDINATOR
    ]
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    client = shell_app.server.test_client()
    try:
        with coordinator.serialized():
            accepted = client.post(
                "/sandbox/api/viewer/output-root",
                json={"path": output.name},
            ).get_json()
            assert entered_factory.wait(5.0)
            # If discovery/factory work held the publication lock, the worker
            # could not reach this phase while this context owned the lock.
            status = client.get(accepted["poll_path"]).get_json()
            assert status["job"]["phase"] == "building_results"
            release_factory.set()
        complete = _poll_terminal(client, accepted["poll_path"])
        assert complete["status"] == "succeeded"
    finally:
        release_factory.set()
        manager.shutdown()


def test_cancel_after_final_check_before_commit_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation in the final-check-to-commit gap fences publication."""
    output = _seed_output(tmp_path)
    entered_commit = threading.Event()
    release_commit = threading.Event()
    real_commit = ResultsBindJobContext.commit_publication

    def _block_before_commit(
        self: ResultsBindJobContext,
        commit: Any,
        *,
        result: Any,
    ) -> None:
        entered_commit.set()
        assert release_commit.wait(5.0)
        real_commit(self, commit, result=result)

    monkeypatch.setattr(
        ResultsBindJobContext,
        "commit_publication",
        _block_before_commit,
    )
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    # Materialize the prior empty Results and Analysis pair.
    old_viewer = viewer_session.get()
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_analysis = analysis_session.get()
    client = shell_app.server.test_client()
    cancel_result: dict[str, Any] = {}
    try:
        accepted = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": output.name},
        ).get_json()
        assert entered_commit.wait(5.0)

        def _cancel() -> None:
            response = shell_app.server.test_client().delete(
                accepted["cancel_path"]
            )
            cancel_result["status_code"] = response.status_code
            cancel_result["payload"] = response.get_json()

        cancel_thread = threading.Thread(target=_cancel)
        cancel_thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            snapshot = manager.get(accepted["job_id"])
            if snapshot is not None and snapshot.status == "cancelled":
                break
            threading.Event().wait(0.01)
        else:
            raise AssertionError("cancel did not fence the publishing job")

        release_commit.set()
        cancel_thread.join(5.0)
        assert cancel_result["status_code"] == 200
        assert cancel_result["payload"]["status"] == "cancelled"
        assert viewer_session.get() is old_viewer
        assert analysis_session.get() is old_analysis
        assert shell_app.server.config[CFG_RESULTS_BINDING_STATE][
            "bound_path"
        ] is None
    finally:
        release_commit.set()
        manager.shutdown()


def test_cancel_after_pair_commit_reports_succeeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE cannot relabel an already committed pair as cancelled."""
    output = _seed_output(tmp_path)
    committed = threading.Event()
    release_worker = threading.Event()
    real_commit = ResultsBindJobContext.commit_publication

    def _commit_then_block(
        self: ResultsBindJobContext,
        commit: Any,
        *,
        result: Any,
    ) -> None:
        real_commit(self, commit, result=result)
        committed.set()
        assert release_worker.wait(5.0)

    monkeypatch.setattr(
        ResultsBindJobContext,
        "commit_publication",
        _commit_then_block,
    )
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    client = shell_app.server.test_client()
    try:
        accepted = client.post(
            "/sandbox/api/viewer/output-root",
            json={"path": output.name},
        ).get_json()
        assert committed.wait(5.0)

        cancelled = client.delete(accepted["cancel_path"])
        assert cancelled.status_code == 200
        assert cancelled.get_json()["status"] == "succeeded"
        assert shell_app.server.config[CFG_RESULTS_BINDING_STATE][
            "bound_path"
        ] == output
        assert viewer_session.is_built() is True

        release_worker.set()
        complete = _poll_terminal(client, accepted["poll_path"])
        assert complete["status"] == "succeeded"
    finally:
        release_worker.set()
        manager.shutdown()


@pytest.mark.parametrize("payload", [[], "output", True])
def test_post_rejects_non_mapping_json(
    tmp_path: Path,
    payload: Any,
) -> None:
    """Truthy JSON arrays, strings, and booleans are client errors."""
    shell_app, _viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]
    try:
        response = shell_app.server.test_client().post(
            "/sandbox/api/viewer/output-root",
            json=payload,
        )
        assert response.status_code == 400
        assert response.get_json() == {
            "status": "error",
            "error": "JSON body must be an object",
        }
        assert manager.tracked_job_count == 0
    finally:
        manager.shutdown()
