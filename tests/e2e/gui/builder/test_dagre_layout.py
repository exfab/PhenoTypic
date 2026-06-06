"""Smoke test for the DAG canvas + dagre layout (Phase 2 of redesign).

Boots ``phenotypic-gui --root <sandbox>``, navigates to ``/builder/``,
and asserts that:

* The cytoscape canvas mounts (the ``#canvas-cytoscape`` element exists).
* The ``Re-layout`` toolbar button is present in the DOM.

This is the bare smoke test for Phase 2 — full DAG interactions
(palette drag, wire drawing, drill-in, etc.) come in Phases 3-5 and
each phase adds its own E2E coverage.

The test is opt-in via ``PLAYWRIGHT=1`` (handled by the parent
conftest at ``tests/e2e/gui/conftest.py``).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page


pytestmark = pytest.mark.skip(
    reason=(
        "Retired Cytoscape/dagre canvas smoke; default builder renders "
        "#linear-map-container instead."
    )
)


def test_builder_canvas_mounts(page: Page, hub_url: str) -> None:
    """The cytoscape canvas mounts under ``/builder/`` (DOM presence smoke)."""

    page.goto(hub_url + "/builder/")
    # The canvas wrapper has a stable id (the cytoscape mount point).
    page.wait_for_selector("#canvas-cytoscape", timeout=15_000)
    # The cytoscape instance renders an internal ``<div>`` carrying the
    # ``cy`` class once cytoscape.js boots; smoke-check it appears.
    page.wait_for_selector("#canvas-cytoscape", state="visible", timeout=15_000)


def test_relayout_button_present(page: Page, hub_url: str) -> None:
    """The DAG ``Re-layout`` button renders in the toolbar (Phase 2).

    The button drives the dagre re-layout pass via ``viewport_ops.js``;
    Phase 2 ships the button + the asset-status disable wiring (the
    button is disabled when ``viewport_ops.js`` failed to register its
    readiness sentinel).

    With the feature flag off (the default), the button may not be
    visible — but the id must always resolve so the
    ``asset_status_disables`` callback's output never errors.
    """

    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#canvas-cytoscape", timeout=15_000)
    # The id must exist in the DOM regardless of the flag's state so the
    # callback contract is satisfied.  When the flag is off, the button
    # may be hidden but the id is still resolvable.
    btn = page.locator("#btn-relayout")
    pytest.importorskip("playwright")
    # ``count() >= 1`` is the smoke gate; visibility is flag-dependent.
    assert btn.count() >= 1, (
        "The Re-layout button (id=btn-relayout) is required by the "
        "asset_status_disables callback regardless of the feature flag"
    )
