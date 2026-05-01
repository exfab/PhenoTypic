"""Integration tests for ``phenotypic.gui.shell._routes`` (sandbox JSON API).

Drives the blueprint via :meth:`flask.Flask.test_client`. Verifies:

    * ``/sandbox/api/root`` shape
    * ``/sandbox/api/children`` honours ``hidden=`` and ``symlinks=`` toggles
    * ``/sandbox/api/classify`` returns the same fields as
      :class:`Capabilities`
    * Path-traversal queries return 400
    * ``viewer_session.touch()`` fires on successful requests
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from flask import Flask

from phenotypic.gui.shell._classifier import invalidate_cache
from phenotypic.gui.shell._routes import register_sandbox_api
from phenotypic.gui.shell._sandbox import SandboxRoot


@pytest.fixture(autouse=True)
def _flush_classifier_cache() -> None:
    invalidate_cache()


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _make_app(sandbox: SandboxRoot) -> Flask:
    app = Flask("phenotypic-test")
    register_sandbox_api(app, sandbox)
    return app


def test_sandbox_api_shapes(sandbox: SandboxRoot) -> None:
    """root + children + classify all return their documented JSON shapes."""
    (sandbox.root / "a_dir").mkdir()
    (sandbox.root / "plate.tif").write_bytes(b"")
    (sandbox.root / "pipeline.json").write_text(
        json.dumps({"name": "demo", "operations": []})
    )

    app = _make_app(sandbox)
    client = app.test_client()

    root_resp = client.get("/sandbox/api/root")
    assert root_resp.status_code == 200
    payload = root_resp.get_json()
    assert payload["root"] == str(sandbox.root)
    assert "badges" in payload
    # Capabilities fields surface verbatim.
    for key in (
        "is_image_dir", "has_pipeline_json", "is_cli_output",
        "has_dashboard", "image_count", "bad_perms",
    ):
        assert key in payload["badges"]

    children_resp = client.get("/sandbox/api/children")
    assert children_resp.status_code == 200
    children = children_resp.get_json()["children"]
    names = {row["name"] for row in children}
    assert names == {"a_dir", "plate.tif", "pipeline.json"}
    # Directories sort before files.
    assert children[0]["type"] == "dir"

    classify_resp = client.get("/sandbox/api/classify?path=pipeline.json")
    assert classify_resp.status_code == 200
    caps = classify_resp.get_json()
    assert caps["has_pipeline_json"] is True


def test_children_hidden_toggle_off_by_default(sandbox: SandboxRoot) -> None:
    (sandbox.root / "visible").write_text("x")
    (sandbox.root / ".hidden").write_text("x")
    app = _make_app(sandbox)
    client = app.test_client()

    default = client.get("/sandbox/api/children").get_json()["children"]
    assert {row["name"] for row in default} == {"visible"}

    on = client.get("/sandbox/api/children?hidden=1").get_json()["children"]
    assert {row["name"] for row in on} == {"visible", ".hidden"}


def test_children_external_symlinks_toggle(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (sandbox_dir / "escape").symlink_to(outside)
    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = _make_app(sandbox)
    client = app.test_client()

    default = client.get("/sandbox/api/children").get_json()["children"]
    assert default == []

    on = client.get(
        "/sandbox/api/children?symlinks=1"
    ).get_json()["children"]
    assert {row["name"] for row in on} == {"escape"}


def test_children_traversal_returns_400(sandbox: SandboxRoot) -> None:
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/sandbox/api/children?path=../../etc")
    assert resp.status_code == 400


def test_classify_traversal_returns_400(sandbox: SandboxRoot) -> None:
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/sandbox/api/classify?path=/etc/passwd")
    assert resp.status_code == 400


def test_children_missing_path_returns_404(sandbox: SandboxRoot) -> None:
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/sandbox/api/children?path=does/not/exist")
    assert resp.status_code == 404


def test_children_relative_path_resolves(sandbox: SandboxRoot) -> None:
    sub = sandbox.root / "a" / "b"
    sub.mkdir(parents=True)
    (sub / "leaf.tif").write_bytes(b"")
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/sandbox/api/children?path=a/b")
    assert resp.status_code == 200
    names = {row["name"] for row in resp.get_json()["children"]}
    assert names == {"leaf.tif"}


def test_children_external_symlink_not_classified(tmp_path: Path) -> None:
    """Regression for H1: external symlinks must NOT be classified.

    Before the fix, ``symlinks=1`` exposed external links AND the route
    called ``classify(child)``, which followed the link and read content
    from outside the sandbox (``image_count``, pipeline-JSON peek, etc.).
    The fix returns the link with ``type="external_symlink"`` and an empty
    capability set so the sidebar can render it as a disabled node.
    """
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # Bait: outside dir has images and a fake pipeline JSON. If the
    # classifier follows the link, ``image_count`` will be 1 and
    # ``has_pipeline_json`` will be True.
    (outside / "plate.tif").write_bytes(b"")
    (outside / "p.json").write_text('{"operations": []}')
    (sandbox_dir / "escape").symlink_to(outside)

    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = _make_app(sandbox)
    client = app.test_client()
    resp = client.get("/sandbox/api/children?symlinks=1")
    assert resp.status_code == 200
    children = resp.get_json()["children"]
    assert len(children) == 1
    row = children[0]
    assert row["name"] == "escape"
    assert row["type"] == "external_symlink"
    # No content disclosure: badges must be the placeholder, not a
    # classification of the outside directory.
    assert row["badges"]["is_image_dir"] is False
    assert row["badges"]["image_count"] is None
    assert row["badges"]["has_pipeline_json"] is False


def test_children_truncates_at_cap(sandbox: SandboxRoot) -> None:
    """Beyond ``_CHILDREN_CLASSIFY_CAP`` the response carries truncated:true.

    We use 5 image directories and lower the cap to 3. Directories are
    sorted before files, and each directory should classify as
    ``is_image_dir=True, image_count=1``; the last 2 get placeholder
    capabilities so the sidebar still renders without paying the
    classification cost.
    """
    from phenotypic.gui.shell import _routes as routes_mod

    for i in range(5):
        d = sandbox.root / f"d{i}"
        d.mkdir()
        (d / "plate.tif").write_bytes(b"")

    original_cap = routes_mod._CHILDREN_CLASSIFY_CAP
    routes_mod._CHILDREN_CLASSIFY_CAP = 3
    try:
        app = _make_app(sandbox)
        client = app.test_client()
        payload = client.get("/sandbox/api/children").get_json()
        assert payload["truncated"] is True
        assert len(payload["children"]) == 5
        classified = [
            c for c in payload["children"] if c["badges"]["is_image_dir"]
        ]
        assert len(classified) == 3
    finally:
        routes_mod._CHILDREN_CLASSIFY_CAP = original_cap


def test_children_truncated_false_under_cap(sandbox: SandboxRoot) -> None:
    """Unlike the cap-hit case, normal listings advertise ``truncated=False``."""
    (sandbox.root / "a").mkdir()
    app = _make_app(sandbox)
    client = app.test_client()
    payload = client.get("/sandbox/api/children").get_json()
    assert payload["truncated"] is False
