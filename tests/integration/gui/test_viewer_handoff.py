"""Integration tests for the sidebar -> viewer hand-off.

Covers ``POST /sandbox/api/viewer/output-root`` and the
``compose_hub`` plumbing that lets the endpoint swap the viewer's
``OutputRoot`` and trigger a rebuild on the next request to
``/results/``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from phenotypic.gui._config import CFG_RESULTS_BINDING_JOBS
from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._binding_jobs import ResultsBindJobManager

from tests._output_layout import write_master
from phenotypic.schema import IMAGE


def _make_minimal_output(root: Path, dataset: str = "d1") -> None:
    """Mirror ``tests/gui/results_viewer/test_output_root._make_minimal_output``.

    Kept inline here to avoid a cross-package test import. The master
    archive lands under ``root/deliverables/``; the per-image ``results/``
    tree stays at the output root.
    """
    (root / "results" / dataset / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = root / "deliverables" / "overlays" / dataset
    overlay_dir.mkdir(parents=True, exist_ok=True)
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": [dataset, dataset],
                str(IMAGE.IMAGE_NAME): ["a", "b"],
                "Metadata_Strain": ["s1", "s2"],
                "Object_Label": [1, 1],
                "Size_Area": [100.0, 200.0],
            }
        ),
    )
    for stem in ("a", "b"):
        (overlay_dir / f"{stem}.png").touch()


def _poll_terminal(
    client: Any,
    poll_path: str,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Poll an asynchronous viewer hand-off to a terminal job snapshot."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(poll_path)
        assert response.status_code == 200
        payload: dict[str, Any] = response.get_json()
        if payload["job"]["terminal"]:
            return payload
        threading.Event().wait(0.01)
    raise AssertionError("viewer hand-off did not become terminal")


@pytest.fixture()
def sandbox_with_output(tmp_path: Path) -> tuple[SandboxRoot, str]:
    out_dir = tmp_path / "output_synth"
    out_dir.mkdir()
    _make_minimal_output(out_dir)
    return SandboxRoot.from_path(tmp_path), "output_synth"


def test_post_swaps_output_root_and_rebuilds(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, rel = sandbox_with_output
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]

    try:
        # First /results/ hit builds the empty-state viewer. Dash returns the
        # bare index HTML; the actual layout is fetched via /_dash-layout.
        assert client.get("/results/").status_code == 200
        assert viewer_session.is_built() is True
        layout_before = client.get("/results/_dash-layout").get_json()
        assert "results-viewer-empty-state" in str(layout_before)

        # POST queues discovery and candidate construction. Only the terminal
        # success publishes the already-built Results/Analysis pair.
        post_resp = client.post(
            "/sandbox/api/viewer/output-root",
            data=json.dumps({"path": rel}),
            content_type="application/json",
        )
        assert post_resp.status_code == 202, post_resp.data
        accepted = post_resp.get_json()
        assert accepted["status"] in {"queued", "running"}
        assert accepted["abs_path"].endswith(rel)
        assert post_resp.headers["Location"] == accepted["poll_path"]

        payload = _poll_terminal(client, accepted["poll_path"])
        assert payload["status"] == "succeeded"
        assert payload["abs_path"].endswith(rel)
        assert viewer_session.is_built() is True

        # The next /results/ hit serves the atomically published loaded app.
        assert client.get("/results/").status_code == 200
        assert viewer_session.is_built() is True
        layout_after = client.get("/results/_dash-layout").get_json()
        layout_after_s = str(layout_after)
        assert "results-viewer-empty-state" not in layout_after_s
        assert "results-viewer-root" in layout_after_s
    finally:
        manager.shutdown()


def test_post_rejects_path_outside_sandbox(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, _ = sandbox_with_output
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": "../../etc"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "path escapes sandbox"


def test_post_rejects_invalid_layout(
    tmp_path: Path,
) -> None:
    # A directory inside the sandbox that is NOT a CLI output (no
    # master_measurements.parquet). OutputRoot.discover raises
    # FileNotFoundError; the endpoint surfaces it as 400.
    bad = tmp_path / "not_an_output"
    bad.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()
    manager: ResultsBindJobManager = shell_app.server.config[
        CFG_RESULTS_BINDING_JOBS
    ]

    try:
        resp = client.post(
            "/sandbox/api/viewer/output-root",
            data=json.dumps({"path": "not_an_output"}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        terminal = _poll_terminal(client, resp.get_json()["poll_path"])
        assert terminal["status"] == "failed"
        assert terminal["job"]["error_kind"] == "invalid"
        assert "master_measurements.parquet" in terminal["job"]["error"]
    finally:
        manager.shutdown()


def test_post_requires_path_in_body(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, _ = sandbox_with_output
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing 'path'"
