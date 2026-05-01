"""Phase 5 ID-collision invariants.

Each Dash app the hub mounts has its own callback dispatch table — Dash
does not share IDs across separate ``dash.Dash`` instances. We rely on
that for the chrome wrap to add the same ``shell-*`` IDs to every app
without conflict.

Within a single app, however, duplicate IDs are fatal: Dash raises at
layout-build / callback-registration time. These tests confirm the
invariant holds across the four mounts in the hub.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot, create_app


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _collect_ids(node: object, out: set[str]) -> None:
    if isinstance(node, dict):
        comp_id = node.get("props", {}).get("id")
        if isinstance(comp_id, str):
            out.add(comp_id)
        elif isinstance(comp_id, dict):
            # Pattern-matching IDs: Dash JSON-encodes them with sorted keys.
            import json

            out.add(json.dumps(comp_id, sort_keys=True, separators=(",", ":")))
        children = node.get("props", {}).get("children")
        if isinstance(children, dict):
            _collect_ids(children, out)
        elif isinstance(children, list):
            for c in children:
                _collect_ids(c, out)
    elif isinstance(node, list):
        for c in node:
            _collect_ids(c, out)


def _layout_ids_for(client, layout_path: str) -> set[str]:
    resp = client.get(layout_path)
    assert resp.status_code == 200, layout_path
    layout = resp.get_json()
    ids: set[str] = set()
    _collect_ids(layout, ids)
    return ids


def test_shell_layout_no_duplicate_ids(sandbox: SandboxRoot) -> None:
    """The shell home + chrome layout has no duplicate IDs."""
    app = create_app(sandbox)
    client = app.server.test_client()
    # Walk the layout JSON; track ids via a counting collector to flag dupes.
    resp = client.get("/_dash-layout")
    layout = resp.get_json()
    counts: dict[str, int] = {}

    def _count(node: object) -> None:
        if isinstance(node, dict):
            comp_id = node.get("props", {}).get("id")
            if isinstance(comp_id, str):
                counts[comp_id] = counts.get(comp_id, 0) + 1
            children = node.get("props", {}).get("children")
            if isinstance(children, dict):
                _count(children)
            elif isinstance(children, list):
                for c in children:
                    _count(c)
        elif isinstance(node, list):
            for c in node:
                _count(c)

    _count(layout)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"Duplicate IDs in shell layout: {duplicates}"


def test_run_console_layout_no_duplicate_ids(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/run/_dash-layout")
    layout = resp.get_json()
    counts: dict[str, int] = {}

    def _count(node: object) -> None:
        if isinstance(node, dict):
            comp_id = node.get("props", {}).get("id")
            if isinstance(comp_id, str):
                counts[comp_id] = counts.get(comp_id, 0) + 1
            children = node.get("props", {}).get("children")
            if isinstance(children, dict):
                _count(children)
            elif isinstance(children, list):
                for c in children:
                    _count(c)
        elif isinstance(node, list):
            for c in node:
                _count(c)

    _count(layout)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, (
        f"Duplicate IDs in run-console layout: {duplicates}"
    )


def test_chrome_ids_appear_in_each_mount(sandbox: SandboxRoot) -> None:
    """Each mounted Dash app carries the chrome's ``shell-*`` IDs.

    Confirms ``wrap_in_chrome`` ran on every app — not just the shell.
    Same ID across different apps is legitimate (separate Dash instances
    have separate callback dispatch tables).
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    chrome_required = {
        "shell-top-bar",
        "shell-sidebar",
        "shell-rss-label",
        "shell-help-modal",
    }
    for path in (
        "/_dash-layout",
        "/builder/_dash-layout",
        "/results/_dash-layout",
        "/run/_dash-layout",
    ):
        layout_ids = _layout_ids_for(client, path)
        assert chrome_required.issubset(layout_ids), (
            f"Chrome IDs missing from {path}: "
            f"{chrome_required - layout_ids}"
        )
