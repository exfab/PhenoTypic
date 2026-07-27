"""Lifecycle integration tests for the Phase 3 shell.

These exercise the seams that the chrome registers but don't require a real
browser to validate:

    * The RSS readout callback wires through to ``psutil`` without raising.
    * The Help-modal toggle inverts ``is_open`` on click.
    * The Sidebar Refresh callback flushes the classifier LRU and bumps the
      ``SHELL_CLASSIFIER_CACHE_STORE`` version.
    * ``viewer_session.touch()`` resets idle timing when the
      ``/sandbox/api/*`` and ``/runs/*`` blueprints answer requests.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot, ToolSession
from phenotypic.gui.shell._app import create_app
from phenotypic.gui.shell._classifier import (
    _classify_cached,
    classify,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def _flush_cache() -> None:
    invalidate_cache()


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


# ---------------------------------------------------------------------------
# RSS readout callback
# ---------------------------------------------------------------------------

def test_rss_readout_returns_string(sandbox: SandboxRoot) -> None:
    """Drive the RSS callback directly; assert format ``"RSS <N> MB"``."""
    app = create_app(sandbox)
    client = app.server.test_client()
    # The callback's output ID is "shell-rss-label.children". We POST a
    # fake interval tick to /_dash-update-component.
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": "shell-rss-label.children",
            "outputs": {"id": "shell-rss-label", "property": "children"},
            "inputs": [
                {
                    "id": "shell-rss-interval",
                    "property": "n_intervals",
                    "value": 1,
                }
            ],
            "state": [],
            "changedPropIds": ["shell-rss-interval.n_intervals"],
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    text = payload["response"]["shell-rss-label"]["children"]
    assert text.startswith("RSS ")
    assert text.endswith("MB")


# ---------------------------------------------------------------------------
# Refresh callback flushes classifier cache
# ---------------------------------------------------------------------------

def test_refresh_callback_flushes_cache(sandbox: SandboxRoot) -> None:
    """Refresh button bumps the version and clears the LRU cache."""
    # Prime the cache.
    classify(sandbox.root)
    info = _classify_cached.cache_info()
    assert info.currsize > 0

    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": "shell-classifier-cache-store.data",
            "outputs": {
                "id": "shell-classifier-cache-store", "property": "data",
            },
            "inputs": [
                {
                    "id": "shell-sidebar-refresh",
                    "property": "n_clicks",
                    "value": 1,
                }
            ],
            "state": [
                {
                    "id": "shell-classifier-cache-store",
                    "property": "data",
                    "value": 0,
                }
            ],
            "changedPropIds": ["shell-sidebar-refresh.n_clicks"],
        },
    )
    assert resp.status_code == 200
    bumped = resp.get_json()["response"]["shell-classifier-cache-store"]["data"]
    assert bumped == 1
    # Cache was flushed by the callback.
    assert _classify_cached.cache_info().currsize == 0


def test_explicit_sidebar_reselection_repairs_same_path_payloads(
    sandbox: SandboxRoot,
) -> None:
    """A same-path click upgrades V1 and repairs a mismatched V2 payload."""
    from phenotypic.gui.shell._source_context import sandbox_fingerprint

    plates = sandbox.root / "plates"
    plates.mkdir()
    (plates / "plate.tif").write_bytes(b"")
    app = create_app(sandbox)
    callback_id = next(
        callback_id
        for callback_id, metadata in app.callback_map.items()
        if callback_id.startswith("shell-source-image-root-store.data")
        and any(
            item["id"] == "shell-sidebar-selection-store"
            for item in metadata["inputs"]
        )
    )
    client = app.server.test_client()
    selection = {
        "path": "plates",
        "abs_path": str(plates.resolve()),
        "is_dir": True,
        "capabilities": {"is_image_dir": True},
    }
    stale_payloads = [
        {
            "version": 1,
            "abs_path": str(plates.resolve()),
            "rel_path": "plates",
            "label": "plates",
            "validated": True,
        },
        {
            "version": 2,
            "kind": "image_source",
            "relative_path": "plates",
            "absolute_path_at_selection": str(plates.resolve()),
            "sandbox_fingerprint": "different-sandbox",
            "validation": {"exists": True, "is_directory": True},
            "selected_at": "2026-07-23T00:00:00+00:00",
            "abs_path": str(plates.resolve()),
            "rel_path": "plates",
            "label": "plates",
            "validated": True,
        },
    ]

    for stale_payload in stale_payloads:
        response = client.post(
            "/_dash-update-component",
            json={
                "output": callback_id,
                "outputs": {
                    "id": "shell-source-image-root-store",
                    "property": "data",
                },
                "inputs": [
                    {
                        "id": "shell-sidebar-selection-store",
                        "property": "data",
                        "value": selection,
                    }
                ],
                "state": [
                    {
                        "id": "shell-source-image-root-store",
                        "property": "data",
                        "value": stale_payload,
                    }
                ],
                "changedPropIds": ["shell-sidebar-selection-store.data"],
            },
        )

        assert response.status_code == 200
        refreshed = response.get_json()["response"][
            "shell-source-image-root-store"
        ]["data"]
        assert refreshed["version"] == 2
        assert refreshed["abs_path"] == str(plates.resolve())
        assert refreshed["sandbox_fingerprint"] == sandbox_fingerprint(sandbox)


# ---------------------------------------------------------------------------
# Help-modal toggle
# ---------------------------------------------------------------------------

def test_help_modal_toggles_open(sandbox: SandboxRoot) -> None:
    """Click the ``?`` button → modal opens."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": "shell-help-modal.is_open",
            "outputs": {"id": "shell-help-modal", "property": "is_open"},
            "inputs": [
                {
                    "id": "shell-help-button",
                    "property": "n_clicks",
                    "value": 1,
                },
                {
                    "id": {"type": "shell-help-close", "scope": "modal"},
                    "property": "n_clicks",
                    "value": 0,
                },
            ],
            "state": [
                {"id": "shell-help-modal", "property": "is_open", "value": False},
            ],
            "changedPropIds": ["shell-help-button.n_clicks"],
        },
    )
    assert resp.status_code == 200
    is_open = resp.get_json()["response"]["shell-help-modal"]["is_open"]
    assert is_open is True


def test_help_modal_close_button_closes(sandbox: SandboxRoot) -> None:
    """Click the modal's Close button → modal closes (M1 regression).

    Previously the close button rendered but had no callback wiring; this
    test enforces the M1 fix.
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.post(
        "/_dash-update-component",
        json={
            "output": "shell-help-modal.is_open",
            "outputs": {"id": "shell-help-modal", "property": "is_open"},
            "inputs": [
                {
                    "id": "shell-help-button",
                    "property": "n_clicks",
                    "value": 0,
                },
                {
                    "id": {"type": "shell-help-close", "scope": "modal"},
                    "property": "n_clicks",
                    "value": 1,
                },
            ],
            "state": [
                {"id": "shell-help-modal", "property": "is_open", "value": True},
            ],
            "changedPropIds": [
                '{"scope":"modal","type":"shell-help-close"}.n_clicks'
            ],
        },
    )
    assert resp.status_code == 200
    is_open = resp.get_json()["response"]["shell-help-modal"]["is_open"]
    assert is_open is False


# ---------------------------------------------------------------------------
# Viewer session touched by Phase 2 blueprints registered on shell server
# ---------------------------------------------------------------------------

def test_sandbox_api_request_touches_viewer_session(
    tmp_path: Path,
) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    viewer = ToolSession[str]("viewer", build=lambda: "state")
    app = create_app(sandbox, viewer_session=viewer)  # type: ignore[arg-type]
    viewer.get()
    time.sleep(0.05)
    pre = viewer.idle_seconds()
    assert pre > 0.0

    client = app.server.test_client()
    resp = client.get("/sandbox/api/root")
    assert resp.status_code == 200

    post = viewer.idle_seconds()
    assert post < pre


def test_runs_request_touches_viewer_session(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "x.txt").write_text("hi")
    sandbox = SandboxRoot.from_path(sandbox_dir)
    viewer = ToolSession[str]("viewer", build=lambda: "state")
    app = create_app(sandbox, viewer_session=viewer)  # type: ignore[arg-type]
    viewer.get()
    time.sleep(0.05)
    pre = viewer.idle_seconds()
    assert pre > 0.0

    client = app.server.test_client()
    resp = client.get("/runs/x.txt")
    assert resp.status_code == 200

    post = viewer.idle_seconds()
    assert post < pre
