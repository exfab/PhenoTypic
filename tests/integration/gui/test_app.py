"""Integration tests for the Phase 3 standalone shell Dash app.

We boot the app via :func:`phenotypic.gui.shell.create_app` and exercise it
two ways:

    * ``app.server.test_client()`` for HTTP endpoints (the index page,
      ``/_dash-layout``, ``/_dash-dependencies``, the Flask blueprints from
      Phase 2).
    * Direct ``app.layout`` introspection for component-tree content checks
      — Dash hydrates client-side, so the index HTML is a React loader,
      not a server-rendered page. We walk the layout tree ourselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.shell import SandboxRoot, create_app


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _walk(component: Any) -> Iterator[Any]:
    """Yield every Dash component in a layout tree."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif isinstance(children, str):
        return
    else:
        yield from _walk(children)


def _all_strings(component: Any) -> str:
    """Concatenate every string-typed children leaf in the layout."""
    out: list[str] = []
    for c in _walk(component):
        children = getattr(c, "children", None)
        if isinstance(children, str):
            out.append(children)
        elif isinstance(children, (list, tuple)):
            out.extend(x for x in children if isinstance(x, str))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Factory smoke + endpoints
# ---------------------------------------------------------------------------

def test_create_app_returns_dash_app(sandbox: SandboxRoot) -> None:
    """Smoke: factory returns a Dash app with chrome wrapped in."""
    import dash

    app = create_app(sandbox)
    assert isinstance(app, dash.Dash)
    assert app.layout is not None


def test_root_returns_200(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


def test_layout_endpoint_serves_chrome(sandbox: SandboxRoot) -> None:
    """``/_dash-layout`` returns the chrome component tree as JSON.

    This is what Dash's clientside fetches to hydrate the React UI. Asserting
    against this catches "factory built but layout never assigned" bugs.
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/_dash-layout")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    # Component IDs surface verbatim in the JSON.
    assert "shell-top-bar" in text
    assert "shell-sidebar" in text
    assert "shell-main-pane" in text
    assert "shell-rss-label" in text
    assert "shell-help-modal" in text


# ---------------------------------------------------------------------------
# Layout tree assertions (component-level)
# ---------------------------------------------------------------------------

def test_top_bar_has_active_home_tab(sandbox: SandboxRoot) -> None:
    """The top bar marks the home tab active and the others inactive."""
    app = create_app(sandbox)
    classes_by_id: dict[str, str] = {}
    for c in _walk(app.layout):
        cid = getattr(c, "id", None)
        if isinstance(cid, str) and cid.startswith("shell-tab-"):
            classes_by_id[cid] = getattr(c, "className", "")
    assert "shell-tab-home" in classes_by_id
    assert "shell-tab-active" in classes_by_id["shell-tab-home"]
    assert "shell-tab-active" not in classes_by_id["shell-tab-builder"]


def test_layout_contains_root_label(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    text = _all_strings(app.layout)
    assert str(sandbox.root) in text


def test_home_capability_summary_renders(tmp_path: Path) -> None:
    """Home page summary counts images / outputs / pipelines.

    Stubbed sandbox has 2 image dirs, 1 CLI output dir, 1 pipeline JSON.
    """
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    for name in ("plate1", "plate2"):
        d = sandbox_dir / name
        d.mkdir()
        (d / "img.tif").write_bytes(b"")
    out = sandbox_dir / "out"
    out.mkdir()
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir()
    (deliverables / "master_measurements.parquet").write_bytes(b"")
    (out / "results").mkdir()
    (sandbox_dir / "pipeline.json").write_text('{"operations": []}')

    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = create_app(sandbox)

    # Walk the layout pairing shell-summary-lbl labels with their
    # corresponding shell-summary-num values. Order-independent.
    pairs: dict[str, str] = {}
    for c in _walk(app.layout):
        cls = getattr(c, "className", "") or ""
        if "shell-summary-card" not in cls:
            continue
        label: str | None = None
        value: str | None = None
        for inner in _walk(c):
            inner_cls = getattr(inner, "className", "") or ""
            children = getattr(inner, "children", None)
            text = children if isinstance(children, str) else None
            if "shell-summary-num" in inner_cls and text is not None:
                value = text
            elif "shell-summary-lbl" in inner_cls and text is not None:
                label = text
        if label is not None and value is not None:
            pairs[label] = value
    assert pairs == {
        "Image dirs": "2",
        "CLI outputs": "1",
        "Pipeline files": "1",
    }


def test_sidebar_renders_root_listing(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "alpha").mkdir()
    (sandbox_dir / "beta.tif").write_bytes(b"")
    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = create_app(sandbox)
    text = _all_strings(app.layout)
    assert "alpha" in text
    assert "beta.tif" in text


# ---------------------------------------------------------------------------
# Phase 2 blueprints reachable through the shell server
# ---------------------------------------------------------------------------

def test_sandbox_api_root_reachable(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/sandbox/api/root")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["root"] == str(sandbox.root)


def test_runs_blueprint_reachable(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "out").mkdir()
    (sandbox_dir / "out" / "dashboard.html").write_text("<html/>")
    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/runs/out/dashboard.html")
    assert resp.status_code == 200
    assert b"<html/>" in resp.data


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def test_chrome_callbacks_registered(sandbox: SandboxRoot) -> None:
    """All three chrome callbacks (RSS, help, refresh) appear in deps."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/_dash-dependencies")
    assert resp.status_code == 200
    deps_text = json.dumps(resp.get_json())
    assert "shell-rss-label" in deps_text
    assert "shell-help-modal" in deps_text
    assert "shell-classifier-cache-store" in deps_text


def test_every_callback_id_is_in_layout(tmp_path: Path) -> None:
    """Hard invariant: every callback Output/Input/State must be in the layout.

    With ``suppress_callback_exceptions=True`` Dash registers callbacks
    even when their referenced IDs don't exist in the DOM. This test
    enforces the runtime invariant by parsing both the layout and the
    dependencies graph and asserting no callback references an ID absent
    from the layout. Catches the regression "removed a Store but forgot
    to update the callback" before it surfaces as a clientside
    ReferenceError.

    Plants one child in the sandbox so that pattern-matching callback
    types (e.g. ``shell-sidebar-entry``) have at least one concrete
    layout instance to match against — otherwise the empty-sandbox
    fixture would falsely report the pattern as unbound.
    """
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    (sandbox_dir / "plate1").mkdir()
    sandbox = SandboxRoot.from_path(sandbox_dir)
    app = create_app(sandbox)
    client = app.server.test_client()

    layout = client.get("/_dash-layout").get_json()

    def collect_ids(node: object, out: set[str]) -> None:
        if isinstance(node, dict):
            comp_id = node.get("props", {}).get("id")
            if isinstance(comp_id, str):
                out.add(comp_id)
            elif isinstance(comp_id, dict):
                # Pattern-matching IDs: serialise to JSON the way Dash
                # encodes them in dependencies (sorted keys).
                out.add(
                    json.dumps(comp_id, sort_keys=True, separators=(",", ":"))
                )
            children = node.get("props", {}).get("children")
            # ``children`` can be a single dict (single child), a list of
            # dicts, a string (text leaf), or absent. Recurse only into
            # actual sub-components.
            if isinstance(children, dict):
                collect_ids(children, out)
            elif isinstance(children, list):
                for c in children:
                    collect_ids(c, out)
        elif isinstance(node, list):
            for c in node:
                collect_ids(c, out)

    layout_ids: set[str] = set()
    collect_ids(layout, layout_ids)

    def _classify_dep_id(cid: object) -> tuple[str | None, str | None]:
        """Return ``(literal_id, pattern_type)`` for a dep-entry id.

        Dash serialises pattern-matching dict ids as JSON strings inside
        ``/_dash-dependencies`` (sometimes — Dash 4.x is inconsistent).
        Treat both forms uniformly: if the ID parses as a JSON dict
        whose values include a list (i.e. an ALL/MATCH/ALLSMALLER
        wildcard), classify as a pattern type; otherwise as a literal.
        """
        if isinstance(cid, dict):
            parsed_dict = cid
        elif isinstance(cid, str):
            try:
                parsed = json.loads(cid)
            except json.JSONDecodeError:
                return cid, None
            if not isinstance(parsed, dict):
                return cid, None
            parsed_dict = parsed
        else:
            return None, None

        if any(isinstance(v, list) for v in parsed_dict.values()):
            type_ = parsed_dict.get("type")
            return None, type_ if isinstance(type_, str) else None
        return (
            json.dumps(parsed_dict, sort_keys=True, separators=(",", ":")),
            None,
        )

    deps = client.get("/_dash-dependencies").get_json()
    referenced: set[str] = set()
    pattern_types_referenced: set[str] = set()
    for dep in deps:
        for slot in ("output", "inputs", "state"):
            entries = dep.get(slot)
            if entries is None:
                continue
            if not isinstance(entries, list):
                entries = [entries]
            for e in entries:
                cid = e.get("id") if isinstance(e, dict) else None
                literal, pattern_type = _classify_dep_id(cid)
                if literal is not None:
                    referenced.add(literal)
                if pattern_type is not None:
                    pattern_types_referenced.add(pattern_type)

    missing = referenced - layout_ids
    assert not missing, (
        f"Callback references IDs not in layout: {sorted(missing)}"
    )

    # Pattern-matching IDs: at least one concrete layout id with the
    # matching ``type`` must exist. Walk the layout ids again, parsing
    # the JSON-encoded dict ones, and check coverage.
    layout_pattern_types: set[str] = set()
    for lid in layout_ids:
        if lid.startswith("{") and lid.endswith("}"):
            try:
                parsed = json.loads(lid)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                type_ = parsed.get("type")
                if isinstance(type_, str):
                    layout_pattern_types.add(type_)
    missing_patterns = pattern_types_referenced - layout_pattern_types
    assert not missing_patterns, (
        f"Pattern-matching callback types with no concrete layout "
        f"instance: {sorted(missing_patterns)}"
    )
