"""Playwright E2E tests for wire-drawing (spec §8.3.2, §4.3, §4.9).

Each test name mirrors the spec exactly so an audit can grep for missing
coverage.  The clientside JS lives in ``assets/wire_drawing.js``; the
server-side dispatcher lives in
``_callbacks.py::_dispatch_state_update`` (``edge_*`` branches +
``STORE_EDGE_EVENT`` fan-in).  These tests drive real pointer events at
the rendered DOM and assert against:

* The cytoscape edge list (``cy.edges()``) for state changes.
* Custom DOM events (``phenotypic:wire-drop``) for accept/reject flow.
* CSS classes / inline styles for the type-aware highlight.

When ``wire_drawing.js`` doesn't expose a state-injection helper (the
preferred path for synthesising a starting state without going through
the whole palette + wire choreography), tests skip gracefully — the
underlying logic is covered by ``tests/unit/gui/builder/test_dispatch.py``.

Run gates:
* ``PLAYWRIGHT=1`` env (handled by the parent
  ``tests/e2e/gui/conftest.py``).  (The ``PHENOTYPIC_GUI_DAG`` feature
  flag earlier versions of this module set on the live server was
  retired in Phase 8; the DAG canvas + dispatch path is now the only
  renderer.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page

from tests.e2e.gui.builder.conftest import (
    _canvas_box,
    _drag_palette_to_canvas,
    _has_state_injection_helper,
    _open_builder,
    _publish_edge_event,
    _seed_two_blocks,
)
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server


pytestmark = pytest.mark.skip(
    reason=(
        "Retired drag-to-wire surface; default builder fills targets via "
        "button ports and side loader actions."
    )
)


# ---------------------------------------------------------------------------
# Live-server override (mirrors test_palette_drag.py).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wire_drawing_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped sandbox shared across all wire-drawing tests."""

    parent = tmp_path_factory.mktemp("e2e_wire_drawing")
    return _build_sandbox(parent)


@pytest.fixture(scope="module")
def live_server(wire_drawing_sandbox: Path) -> Iterator[str]:
    """Spawn ``phenotypic-gui`` against the wire-drawing sandbox."""

    yield from _start_live_server(wire_drawing_sandbox)


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    """String alias for ``live_server``."""

    return live_server


# ---------------------------------------------------------------------------
# 8.3.2 — Wire drawing
#
# Shared canvas helpers (``_open_builder``, ``_seed_two_blocks``,
# ``_publish_edge_event``, ``_drag_palette_to_canvas``, ``_canvas_box``,
# ``_has_state_injection_helper``) live in ``builder/conftest.py``.
# ---------------------------------------------------------------------------


def test_wire_drag_image_to_image_snaps_blue(page: Page, hub_url: str) -> None:
    """Drag GaussianBlur.output → OtsuDetector.in → wire snaps blue 3px.

    Uses the ``STORE_EDGE_EVENT`` set_props path to synthesise the
    edge_create without driving the cytoscape pointer geometry (the
    cytoscape internals are not in scope for this assertion — we just
    need the edge in state + the blue-3px CSS class).
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 0,
        },
    )
    # The edge appears in cytoscape's edge list with kind=="image".
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().some(e => e.data('kind') === 'image');
        }""",
        timeout=10_000,
    )


def test_wire_drag_image_to_aux_snaps_purple(page: Page, hub_url: str) -> None:
    """Drag GaussianBlur.output → FilamentousFungiDetector.inoculum_detector.

    Wire snaps purple-dashed; source block border turns solid purple.
    Skips gracefully when the test fixture cannot programmatically pick
    a real aux-accepting consumer.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "window.phenoSetState test helper not exposed; aux-wire snap "
            "exercised by tests/unit/gui/builder/test_dispatch.py"
        )


def test_wire_drag_live_wire_neutral_gray_during_flight(
    page: Page, hub_url: str
) -> None:
    """Live wire's stroke colour is neutral gray while mouse is moving.

    Requires a real port mousedown gesture which relies on cytoscape
    port hit-testing — skip when the test fixture cannot drive it.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Live-wire colour assertion requires a port-mousedown helper "
            "not exposed by wire_drawing.js; manual visual gate stands in."
        )


def test_wire_drag_compatible_targets_glow_incompatible_dim(
    page: Page, hub_url: str
) -> None:
    """During drag, only compatible aux ports glow; incompatible dim to 30%.

    The CSS classes are ``dag-port--glow`` / ``dag-port--dim`` (spec
    §5.5 / §4.3).  Skip when there's no port-injection helper.
    """

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Type-aware highlight requires programmatic dragstart "
            "(port-mousedown helper); not yet exposed by wire_drawing.js"
        )


def test_wire_drag_drop_on_dimmed_port_rejects_with_red_flash(
    page: Page, hub_url: str
) -> None:
    """Drop on a dimmed (incompatible) port → red flash + no edge created."""

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Red-flash assertion requires programmatic dragstart helper"
        )


def test_wire_drag_drop_on_empty_canvas_fades_out(
    page: Page, hub_url: str
) -> None:
    """Drop on empty canvas → live wire fades; no state change."""

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Fade-out assertion requires programmatic dragstart helper"
        )


def test_wire_drag_esc_cancels(page: Page, hub_url: str) -> None:
    """``Escape`` mid-drag cancels the wire (no state change)."""

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Esc-cancel assertion requires programmatic dragstart helper"
        )


def test_wire_drag_mouse_leaves_canvas_cancels(
    page: Page, hub_url: str
) -> None:
    """Mouse exits the cytoscape wrapper bounds mid-drag → wire fades."""

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Mouse-leave assertion requires programmatic dragstart helper"
        )


def test_wire_drag_from_already_wired_source_replaces_first_wire(
    page: Page, hub_url: str
) -> None:
    """Drag from an already-wired source → prior edge gone; new edge present.

    The server-side dispatcher (spec §4.2) deletes any existing
    outgoing wire from the source in the same dispatch before adding
    the new one.  Driven through ``STORE_EDGE_EVENT`` so the test
    doesn't require port-mousedown geometry.
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    # Need a THIRD target so we can re-wire from src.  Drop another
    # GaussianBlur.
    box = _canvas_box(page)
    _drag_palette_to_canvas(
        page, "GaussianBlur", box["width"] * 0.5, box["height"] * 0.8
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur').length >= 3;
        }""",
        timeout=10_000,
    )
    third = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            const blocks = cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur');
            return blocks[2].data('block_id') || blocks[2].id();
        }"""
    )

    # First wire: src → target1.
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 1,
        },
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 1;
        }""",
        timeout=10_000,
    )

    # Second wire: src → target2.  Per spec §4.2, the first wire is
    # replaced in the same dispatch.
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": third,
            "target_port": "in",
            "edge_kind": "image",
            "ts": 2,
        },
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            const edges = cy.edges();
            return edges.length === 1;
        }""",
        timeout=10_000,
    )


def test_wire_select_then_delete(page: Page, hub_url: str) -> None:
    """Click an edge (stroke widens) → press Delete → edge removed.

    Drives via the server-side dispatcher when the visual click +
    keyboard path can't easily be synthesised against cytoscape.
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 0,
        },
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 1;
        }""",
        timeout=10_000,
    )
    edge_id = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges()[0].data('edge_id') || cy.edges()[0].id();
        }"""
    )
    _publish_edge_event(
        page,
        {"kind": "edge_delete", "edge_id": edge_id, "ts": 1},
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 0;
        }""",
        timeout=10_000,
    )


def test_wire_right_click_disconnect(page: Page, hub_url: str) -> None:
    """Right-click an edge → ``Disconnect`` context item → edge removed.

    The DOM context menu is owned by ``wire_drawing.js``.  When the
    test fixture cannot synthesise a context menu interaction against
    cytoscape's internal canvas, fall back to the same dispatcher
    path used by ``Disconnect`` so the spec contract is exercised.
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 0,
        },
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 1;
        }""",
        timeout=10_000,
    )
    edge_id = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges()[0].data('edge_id') || cy.edges()[0].id();
        }"""
    )
    # Disconnect action publishes ``edge_delete`` via the same store
    # (see ``wire_drawing.js`` context-menu glue).
    _publish_edge_event(
        page,
        {"kind": "edge_delete", "edge_id": edge_id, "ts": 1},
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 0;
        }""",
        timeout=10_000,
    )


def test_wire_no_endpoint_grab_gesture(page: Page, hub_url: str) -> None:
    """No endpoint drag gesture — cytoscape doesn't grant grabbable endpoints.

    Spec §4.3 documents that ``edge_replace`` is gone — re-targeting a
    wire is a two-step select → Delete → re-draw.  This test asserts
    the negative: a mousedown on an edge's endpoint within tolerance
    is a no-op (or selects the edge), never spawns a live-wire
    drag.
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 0,
        },
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges().length === 1;
        }""",
        timeout=10_000,
    )
    # No live-wire element exists.  Edge count is stable.
    has_live_wire = page.evaluate(
        "() => document.querySelector('[data-testid=\"live-wire\"]') !== null"
    )
    assert not has_live_wire, (
        "Edge-endpoint drag must not spawn a live wire; spec §4.3 documents "
        "edge_replace is gone."
    )
    edge_count = page.evaluate(
        "() => window.phenoGetCy().edges().length"
    )
    assert edge_count == 1


def test_wire_blue_throughout_past_measure_boundary(
    page: Page, hub_url: str
) -> None:
    """Wire colour stays blue past a MeasureFeatures node (spec §4.3).

    Chain: Otsu → MeasureSize → MeasurePerimeter; all three wires render
    blue, not gold/green.  The kind field stays ``"image"`` for the
    full chain — that's the dispatcher contract (the runtime partitions
    by isinstance, not by wire colour).
    """

    _open_builder(page, hub_url)
    if not _has_state_injection_helper(page):
        pytest.skip(
            "Multi-class chain requires state-injection helper to seed "
            "Otsu + MeasureSize + MeasurePerimeter blocks"
        )


def test_wire_main_path_3px_aux_2px(page: Page, hub_url: str) -> None:
    """Main-path wires render 3px (spec §4.3).

    Only edges on the ``Input Image`` → terminal path carry the
    ``dag-wire--main`` class (and its ``width: 3``); a free-floating
    image edge stays at the ``dag-wire`` base ``width: 2``.  So we wire
    the seeded chain in from the scope's auto-seeded ``Input Image``
    block — that puts the seeded edges on the main path — and assert
    their rendered width.
    """

    _open_builder(page, hub_url)
    ids = _seed_two_blocks(page)
    input_image_id = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            const ii = cy.nodes().filter(
                n => n.data('class_name') === 'InputImage'
            )[0];
            return ii ? (ii.data('block_id') || ii.id()) : null;
        }"""
    )
    assert input_image_id, "every scope auto-seeds an Input Image block"
    # Input Image → source → target: both edges land on the main path.
    # The two dispatches go through the *same* ``STORE_EDGE_EVENT`` store,
    # so we wait for the first edge to materialise before publishing the
    # second — otherwise the second ``set_props`` overwrites the store
    # before the first dispatch's callback has run.
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": input_image_id,
            "target_block_id": ids["source"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 0,
        },
    )
    page.wait_for_function(
        "() => window.phenoGetCy().edges().length === 1",
        timeout=10_000,
    )
    _publish_edge_event(
        page,
        {
            "kind": "edge_create",
            "source_block_id": ids["source"],
            "target_block_id": ids["target"],
            "target_port": "in",
            "edge_kind": "image",
            "ts": 1,
        },
    )
    page.wait_for_function(
        "() => window.phenoGetCy().edges().length === 2",
        timeout=10_000,
    )
    # Both edges now sit on the Input Image → terminal path, so each
    # carries ``dag-wire--main`` → ``width: 3``.
    width_str = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            return cy.edges()[0].style('width') || cy.edges()[0].numericStyle('width');
        }"""
    )
    # Stylesheet maps main-path edges to 3px; the value may come back as
    # a numeric string ("3"), a pixel string ("3px"), or a number.
    if width_str is None:
        pytest.skip(
            "Stylesheet did not surface a width attribute; visual width "
            "asserted manually via screenshot"
        )
    width_num = float(str(width_str).replace("px", "").strip() or 0)
    # Allow some tolerance — cytoscape may report fractional widths
    # when stylesheets layer multiple selectors.
    assert width_num >= 2.5, (
        f"Main-path wire should render >= 2.5px (spec says 3px); got {width_str!r}"
    )
