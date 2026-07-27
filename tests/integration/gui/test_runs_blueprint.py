"""Integration tests for ``phenotypic.gui.shell._runs_blueprint``.

We mount the blueprint on a bare :class:`flask.Flask` (no Dash) and drive it
through ``app.test_client()`` — fast, deterministic, no port allocation.

Coverage:

    * ``dashboard.html`` and canonical hidden progress paths serve.
    * Legacy root-level ``progress/manifest.json`` paths remain readable.
    * Path-traversal URLs (``..``-style, absolute, symlink-escape) return 404.
    * Permission-denied targets return 403 (POSIX-only).
    * ``viewer_session.touch()`` is called on every successful request and
      NOT called on rejected requests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from flask import Flask

from phenotypic.gui.shell._runs_blueprint import register
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import ToolSession


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


@pytest.fixture()
def viewer_session() -> ToolSession[str]:
    return ToolSession[str]("viewer", build=lambda: "state")


def _make_app(
    sandbox: SandboxRoot,
    viewer_session: ToolSession[str] | None = None,
) -> Flask:
    app = Flask("phenotypic-test")
    register(app, sandbox, viewer_session=viewer_session)  # type: ignore[arg-type]
    return app


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_dashboard_html_served(sandbox: SandboxRoot) -> None:
    (sandbox.root / "plate" / "output").mkdir(parents=True)
    (sandbox.root / "plate" / "output" / "dashboard.html").write_text(
        "<html><body>OK</body></html>"
    )
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/plate/output/dashboard.html")
    assert resp.status_code == 200
    assert b"<body>OK</body>" in resp.data


def test_nested_progress_polls_served(sandbox: SandboxRoot) -> None:
    """The catch-all retains read compatibility for legacy progress polls.

    Dashboards generated before the hidden machine-state migration poll
    ``progress/manifest.json``. The generic route must continue serving those
    files even though current writers use ``.phenotypic/progress/``.
    """
    (sandbox.root / "plate" / "progress").mkdir(parents=True)
    (sandbox.root / "plate" / "progress" / "manifest.json").write_text(
        '{"chunks": []}'
    )
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/plate/progress/manifest.json")
    assert resp.status_code == 200
    assert b'"chunks"' in resp.data


def test_canonical_hidden_progress_polls_served(
    sandbox: SandboxRoot,
) -> None:
    """Current dashboard polls reach ``.phenotypic/progress/``."""
    progress = sandbox.root / "plate" / ".phenotypic" / "progress"
    progress.mkdir(parents=True)
    (progress / "manifest.json").write_text(
        '{"chunks": []}',
        encoding="utf-8",
    )
    app = _make_app(sandbox)
    client = app.test_client()

    resp = client.get("/runs/plate/.phenotypic/progress/manifest.json")

    assert resp.status_code == 200
    assert b'"chunks"' in resp.data


# ---------------------------------------------------------------------------
# Security: path traversal
# ---------------------------------------------------------------------------

def test_dotdot_traversal_returns_404(sandbox: SandboxRoot) -> None:
    """``/runs/../../etc/passwd`` must 404, not 200 with /etc/passwd."""
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/../../etc/passwd")
    # Flask's URL parser rejects literal ``..`` segments before our handler
    # ever sees them — Werkzeug returns 404 with a different code path. We
    # only care that the file is not served.
    assert resp.status_code in (404, 308)
    assert b"root:x:" not in resp.data
    assert b"/etc/passwd" not in resp.data


def test_absolute_path_returns_404(sandbox: SandboxRoot) -> None:
    """An attacker-supplied ``//etc/passwd`` must not leak ``/etc/passwd``.

    Werkzeug normalises double-slash URLs via a 308 redirect; following it
    lands on ``/runs/etc/passwd``, which our handler then sandbox-rejects.
    """
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs//etc/passwd", follow_redirects=True)
    assert resp.status_code == 404
    assert b"root:x:" not in resp.data


def test_symlink_escape_returns_404(
    tmp_path: Path,
) -> None:
    """A symlink inside the sandbox pointing OUT must not leak."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("LEAKED")
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "escape").symlink_to(outside)
    sandbox = SandboxRoot.from_path(sandbox_dir)

    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/escape/secret.txt")
    assert resp.status_code == 404
    assert b"LEAKED" not in resp.data


def test_directory_target_returns_404(sandbox: SandboxRoot) -> None:
    """We do not enumerate directories — only individual files."""
    (sandbox.root / "plate").mkdir()
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/plate")
    assert resp.status_code == 404


def test_missing_file_returns_404(sandbox: SandboxRoot) -> None:
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/runs/nope.html")
    assert resp.status_code == 404


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0,
    reason="chmod-based permission test is POSIX-specific and meaningless as root",
)
def test_permission_denied_returns_403(sandbox: SandboxRoot) -> None:
    f = sandbox.root / "locked.txt"
    f.write_text("secret")
    os.chmod(f, 0o000)
    try:
        app = _make_app(sandbox)
        client = app.test_client()
        resp = client.get("/runs/locked.txt")
        # Flask may translate PermissionError to 500 if it bubbles before
        # our handler catches it; we ensure the file is NOT served.
        assert resp.status_code in (403, 500)
        assert b"secret" not in resp.data
    finally:
        os.chmod(f, 0o644)


# ---------------------------------------------------------------------------
# Lifecycle: viewer_session.touch
# ---------------------------------------------------------------------------

def test_touch_called_on_successful_request(
    sandbox: SandboxRoot,
    viewer_session: ToolSession[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spy directly on `touch` instead of inferring from `idle_seconds()`
    # arithmetic. Clock-derived assertions are flaky on Windows, where
    # the scheduler tick (~15.6 ms) can swallow the entire request and
    # collapse pre/post to byte-identical floats.
    (sandbox.root / "x.txt").write_text("hi")
    viewer_session.get()

    calls: list[None] = []
    real_touch = viewer_session.touch

    def spy() -> None:
        calls.append(None)
        real_touch()

    monkeypatch.setattr(viewer_session, "touch", spy)

    app = _make_app(sandbox, viewer_session=viewer_session)
    client = app.test_client()
    resp = client.get("/runs/x.txt")
    assert resp.status_code == 200
    assert len(calls) == 1, "blueprint must call viewer_session.touch on 200"


def test_touch_not_called_on_rejected_request(
    sandbox: SandboxRoot, viewer_session: ToolSession[str]
) -> None:
    """Rejected requests must not bump the idle timer.

    Otherwise an attacker hammering ``/runs/../../...`` URLs would keep the
    viewer session alive indefinitely.
    """
    viewer_session.get()
    import time
    time.sleep(0.05)
    pre = viewer_session.idle_seconds()

    app = _make_app(sandbox, viewer_session=viewer_session)
    client = app.test_client()
    client.get("/runs/nope.html")  # 404
    client.get("/runs/../../etc/passwd")  # rejected

    post = viewer_session.idle_seconds()
    # Allow a tiny slack for the time spent processing the request.
    assert post >= pre - 0.005


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0,
    reason="chmod-based permission test is POSIX-specific and meaningless as root",
)
def test_touch_not_called_on_403(
    sandbox: SandboxRoot, viewer_session: ToolSession[str]
) -> None:
    """A 403 (PermissionError) must not bump the idle timer either.

    Regression for M1: previously the blueprint touched ``viewer_session``
    before ``send_from_directory``, so an in-sandbox-but-unreadable path
    would keep the viewer alive every time it was hit.
    """
    f = sandbox.root / "locked.txt"
    f.write_text("secret")
    os.chmod(f, 0o000)
    try:
        viewer_session.get()
        import time
        time.sleep(0.05)
        pre = viewer_session.idle_seconds()

        app = _make_app(sandbox, viewer_session=viewer_session)
        client = app.test_client()
        client.get("/runs/locked.txt")  # 403 or 500

        post = viewer_session.idle_seconds()
        assert post >= pre - 0.005
    finally:
        os.chmod(f, 0o644)


def test_touch_optional_works_without_session(sandbox: SandboxRoot) -> None:
    (sandbox.root / "x.txt").write_text("hi")
    app = _make_app(sandbox, viewer_session=None)
    client = app.test_client()
    resp = client.get("/runs/x.txt")
    assert resp.status_code == 200
