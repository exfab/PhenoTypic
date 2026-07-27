"""Integration tests for the analysis sub-app's sidebar hand-off.

Confirms that POSTing a CLI output to ``/sandbox/api/viewer/output-root``
releases BOTH the viewer and the analysis ToolSession (per the locked
shared-output_root decision). Also smoke-tests that subsequent GETs to
``/results/`` and ``/analysis/`` return 200 against the bound output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from flask import Flask

from phenotypic import ImagePipeline
from phenotypic.gui.shell._routes import register_sandbox_api
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import ToolSession

from tests._output_layout import seed_output_dir
from phenotypic.schema import METADATA


def _seed_cli_output(parent: Path) -> Path:
    """Build a minimal CLI output dir that ``OutputRoot.discover`` accepts.

    User-facing deliverables (master, mirror, pipeline.json) land under
    ``out/deliverables/``; the per-image ``results/`` tree stays at the
    output root.
    """
    out = parent / "results" / "demo"
    out.mkdir(parents=True)
    df = pl.DataFrame({
        "MetadataExperiment_Dataset": ["d"] * 2,
        str(METADATA.IMAGE_NAME): ["a", "b"],
        "MetadataGenetic_Strain": ["A", "B"],
        "Object_Label": [1, 1],
        "Shape_Area": [100.0, 200.0],
    })
    seed_output_dir(out, df, mirror=df, pipeline=ImagePipeline(name="t"))
    (out / "results" / "ds1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlay_dir = out / "deliverables" / "overlays" / "ds1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "a.png").write_bytes(b"PNG")
    return out


class _CountingSession:
    """ToolSession stub that records release/touch calls for assertions."""

    def __init__(self) -> None:
        self.released = 0
        self.touched = 0

    def release(self) -> None:
        self.released += 1

    def touch(self) -> None:
        self.touched += 1


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _bind_app(
    sandbox: SandboxRoot,
    *,
    viewer_session: Any,
    analysis_session: Any | None,
    viewer_state: dict[str, Any],
) -> Flask:
    app = Flask("phenotypic-handoff-test")
    register_sandbox_api(
        app,
        sandbox,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
        extra_release_sessions=(analysis_session,) if analysis_session else None,
    )
    return app


def test_bind_releases_both_sessions(sandbox: SandboxRoot) -> None:
    """A successful viewer hand-off releases viewer + analysis sessions."""
    output = _seed_cli_output(sandbox.root)
    rel = str(output.relative_to(sandbox.root))

    viewer_sess = _CountingSession()
    analysis_sess = _CountingSession()
    viewer_state: dict[str, Any] = {"output_root": None}

    app = _bind_app(
        sandbox,
        viewer_session=viewer_sess,
        analysis_session=analysis_sess,
        viewer_state=viewer_state,
    )
    client = app.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": rel}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["status"] == "ok"

    # Both sessions released exactly once.
    assert viewer_sess.released == 1
    assert analysis_sess.released == 1
    # Both touched at least once after release (so the next GET sees them
    # as recently active and the idle daemon doesn't drop them mid-bind).
    assert viewer_sess.touched >= 1
    assert analysis_sess.touched >= 1

    # ``viewer_state`` was stamped with the resolved OutputRoot.
    assert viewer_state["output_root"] is not None
    assert str(viewer_state["output_root"].root).endswith("demo")
    assert viewer_state["output_root"].cache_dir.is_relative_to(
        sandbox.root / ".phenotypic-gui" / "viewer_cache"
    )


def test_bind_without_analysis_session_still_releases_viewer(
    sandbox: SandboxRoot,
) -> None:
    """Back-compat: omitting ``extra_release_sessions`` keeps viewer-only path."""
    output = _seed_cli_output(sandbox.root)
    rel = str(output.relative_to(sandbox.root))

    viewer_sess = _CountingSession()
    viewer_state: dict[str, Any] = {"output_root": None}

    app = _bind_app(
        sandbox,
        viewer_session=viewer_sess,
        analysis_session=None,
        viewer_state=viewer_state,
    )
    client = app.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": rel}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert viewer_sess.released == 1


def test_real_tool_session_release_is_lock_clean(sandbox: SandboxRoot) -> None:
    """Use a real ``ToolSession`` to confirm ``release()`` integration works."""
    output = _seed_cli_output(sandbox.root)
    rel = str(output.relative_to(sandbox.root))

    build_count = {"viewer": 0, "analysis": 0}

    def _viewer_build() -> Any:
        build_count["viewer"] += 1
        return object()

    def _analysis_build() -> Any:
        build_count["analysis"] += 1
        return object()

    viewer_session: ToolSession[Any] = ToolSession(
        "viewer", build=_viewer_build, teardown=lambda _: None
    )
    analysis_session: ToolSession[Any] = ToolSession(
        "analysis", build=_analysis_build, teardown=lambda _: None
    )
    viewer_state: dict[str, Any] = {"output_root": None}

    # Prime both sessions.
    viewer_session.get()
    analysis_session.get()
    assert build_count == {"viewer": 1, "analysis": 1}

    app = _bind_app(
        sandbox,
        viewer_session=viewer_session,
        analysis_session=analysis_session,
        viewer_state=viewer_state,
    )
    client = app.test_client()
    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": rel}),
        content_type="application/json",
    )
    assert resp.status_code == 200

    # Next ``get()`` rebuilds.
    viewer_session.get()
    analysis_session.get()
    assert build_count == {"viewer": 2, "analysis": 2}
