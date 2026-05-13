"""Playwright E2E tests for clientside resilience (spec §8.3.10, §9, §4.10).

These tests block specific JS assets via Playwright route interception
and assert that the builder renders, the asset-status banner reports
the missing file, and existing state still surfaces.  This is the
canonical place for "what happens when ``wire_drawing.js`` /
``palette_dnd.js`` / ``viewport_ops.js`` fail to load" coverage.

Phase 4 ships only ``test_clientside_wire_drawing_js_fails_to_load`` per
the orchestrator brief.  Future phases extend this file with the
companion checks for ``viewport_ops.js`` failure modes and dagre
absence.

Run gates: ``PLAYWRIGHT=1`` + ``PHENOTYPIC_GUI_DAG=1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page

from tests.e2e.gui.conftest import _build_sandbox, _start_live_server


# ---------------------------------------------------------------------------
# Live-server override (mirrors test_palette_drag.py).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resilience_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped sandbox shared across resilience tests."""

    parent = tmp_path_factory.mktemp("e2e_clientside_resilience")
    return _build_sandbox(parent)


@pytest.fixture(scope="module")
def live_server(resilience_sandbox: Path) -> Iterator[str]:
    """Spawn ``phenotypic-gui`` with ``PHENOTYPIC_GUI_DAG=1``."""

    yield from _start_live_server(
        resilience_sandbox,
        env_overrides={"PHENOTYPIC_GUI_DAG": "1"},
    )


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    """String alias for ``live_server``."""

    return live_server


# ---------------------------------------------------------------------------
# 8.3.10 — Resilience / failure modes
# ---------------------------------------------------------------------------


def test_clientside_wire_drawing_js_fails_to_load(
    page: Page, hub_url: str
) -> None:
    """Block ``wire_drawing.js`` via route interception → banner + state intact.

    Drops a Playwright route override that returns 404 for the JS
    asset, navigates fresh, and asserts the asset-status banner shows
    "Wire drawing offline" while pre-existing wires still render on
    the canvas.
    """

    # Install a route block BEFORE navigation so the JS request is
    # intercepted on first paint.
    page.route(
        "**/assets/wire_drawing.js",
        lambda route: route.fulfill(status=404, body=""),
    )
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#canvas-cytoscape", timeout=15_000)
    # The asset-status banner mounts unconditionally; wait for the poll
    # cycle (~500ms) to flip wire_drawing → False.
    page.wait_for_function(
        "() => document.querySelector('#banner-asset-status') !== null",
        timeout=15_000,
    )
    # The banner should be visible (display != "none") once the
    # poller has flipped wire_drawing → False.
    page.wait_for_function(
        """() => {
            const el = document.querySelector('#banner-asset-status');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none';
        }""",
        timeout=15_000,
    )
    # The banner copy must include the wire-drawing-offline message
    # (spec §6: "Wire drawing offline").  Text-matching keeps the test
    # tolerant of additional banner rows for other missing assets.
    banner_text = page.evaluate(
        """() => {
            const el = document.querySelector('#banner-asset-status');
            return el ? el.innerText : '';
        }"""
    )
    assert "Wire drawing" in banner_text and "offline" in banner_text.lower(), (
        f"Expected 'Wire drawing offline' in banner; got {banner_text!r}"
    )

    # Verify the readiness sentinel is False (the IIFE never ran).
    sentinel = page.evaluate(
        "() => window.phenotypic_wire_drawing_ready === true"
    )
    assert not sentinel, (
        "wire_drawing.js IIFE should not have run when the asset is blocked"
    )

    # Canvas still mounts; cytoscape has at least the auto-seeded
    # InputImage node — existing wires (if any) survive the asset
    # failure because they're server-rendered from STORE_BUILDER_STATE,
    # not from the wire_drawing JS.
    node_count = page.evaluate(
        """() => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            return cy ? cy.nodes().length : 0;
        }"""
    )
    assert node_count >= 1, (
        "Cytoscape canvas should still render the auto-seeded InputImage "
        "block even when wire_drawing.js fails to load"
    )
