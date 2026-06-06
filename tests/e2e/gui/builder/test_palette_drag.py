"""Playwright E2E tests for the palette drag-and-drop UI (spec §8.3.1).

Each test maps to one row of the palette-drop table in spec §4.8 +
§8.3.1.  The test names mirror the spec exactly so a future audit
can grep for missing coverage.

Run gates:
* ``PLAYWRIGHT=1`` env (handled by the parent
  ``tests/e2e/gui/conftest.py``).
* The ``PHENOTYPIC_GUI_DAG`` env var that earlier versions of this
  module set on the live server was retired in Phase 8; the DAG canvas
  / palette buttons / dispatcher path is now the only renderer.

Coordination:
* The clientside HTML5 drag glue lives in
  ``src/phenotypic/gui/builder/assets/palette_dnd.js`` (owned by Agent
  3A).  These tests do **not** mock that JS — they drive real
  ``page.mouse.down() / move() / up()`` gestures against the rendered
  palette + cytoscape DOM.
* The server-side dispatch handler lives in
  ``src/phenotypic/gui/builder/_callbacks.py`` (``_dispatch_state_update``
  ``block_create`` branch + the ``STORE_PALETTE_DROP`` input wired into
  the existing fan-in callback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.gui.builder.conftest import (
    _canvas_box,
    _click_new_pipeline_button,
    _drag_palette_to_canvas,
    _open_builder,
    _palette_button,
    _publish_palette_drop,
)
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server


pytestmark = pytest.mark.skip(
    reason=(
        "Retired palette drag/drop surface; default builder uses click-only "
        "linear palette insertion."
    )
)


# ---------------------------------------------------------------------------
# Live-server override: the module needs its own function-scoped
# sandbox (``palette_dnd_sandbox``), so we override the parent
# ``live_server`` fixture rather than depend on the parent's default
# sandbox. ``_start_live_server`` is reused so the subprocess boot
# logic stays centralised.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def palette_dnd_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped sandbox shared across all palette-drag tests."""

    parent = tmp_path_factory.mktemp("e2e_palette_drag")
    return _build_sandbox(parent)


@pytest.fixture(scope="module")
def live_server(palette_dnd_sandbox: Path) -> Iterator[str]:
    """Spawn ``phenotypic-gui`` against the palette-dnd sandbox.

    Overrides the parent module's ``live_server`` fixture (same fixture
    name on a function-scoped basis — pytest picks the closer scope).
    """

    yield from _start_live_server(palette_dnd_sandbox)


@pytest.fixture(scope="module")
def hub_url(live_server: str) -> str:
    """String alias for ``live_server`` (mirrors the parent conftest)."""

    return live_server


# ---------------------------------------------------------------------------
# 8.3.1 — Palette → canvas: positive cases
#
# Shared canvas helpers (``_open_builder``, ``_palette_button``,
# ``_canvas_box``, ``_drag_palette_to_canvas``, ``_publish_palette_drop``,
# ``_click_new_pipeline_button``) live in ``builder/conftest.py``.
# ---------------------------------------------------------------------------


def test_palette_drag_drop_creates_block(page: Page, hub_url: str) -> None:
    """Drag GaussianBlur from the palette to canvas center → block appears."""

    _open_builder(page, hub_url)
    box = _canvas_box(page)
    _drag_palette_to_canvas(
        page,
        "GaussianBlur",
        box["width"] / 2,
        box["height"] / 2,
    )
    # The block lands on the canvas; cytoscape's node ids carry the
    # block's ``class_name`` in ``data.class_name`` (or ``data.label``).
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return false;
            return cy.nodes().some(n => n.data('class_name') === 'GaussianBlur');
        }""",
        timeout=10_000,
    )


def test_palette_drag_drop_inside_container(page: Page, hub_url: str) -> None:
    """Drop inside an expanded container → block adopted into nested scope.

    Sets up a pipeline container first (via the existing "New Pipeline"
    button) and then drags a GaussianBlur into its body.
    """

    _open_builder(page, hub_url)
    # Click "New Pipeline" so the canvas has a container to drop into;
    # the helper waits for the container to materialise + the network to
    # idle, then returns its block_id.
    container_id = _click_new_pipeline_button(page)
    assert container_id, "New Pipeline should mint an ImagePipeline container"
    # Adopt a GaussianBlur into the container's nested scope.  We
    # dispatch the ``block_create`` payload directly (with the resolved
    # ``container_block_id``) rather than synthesising an HTML5 drag:
    # Playwright's synthesized pointer events don't reliably trigger the
    # native ``DragEvent``s ``palette_dnd.js`` listens for, and a drop
    # that must hit-test *inside* a compound container is the flakiest
    # case.  This still exercises the real server-side ``block_create``
    # container-adoption dispatch.
    _publish_palette_drop(
        page,
        {
            "kind": "block_create",
            "class_name": "GaussianBlur",
            "x": 0,
            "y": 0,
            "container_block_id": container_id,
            "ts": 0,
        },
    )
    # Assert the new block has ``parent`` set to the container's id.
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            const gb = cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur')[0];
            return gb && gb.parent().length > 0;
        }""",
        timeout=10_000,
    )


def test_palette_drag_drop_inside_nested_container_innermost_wins(
    page: Page, hub_url: str,
) -> None:
    """Drop in overlap of two nested containers → innermost adopts.

    Setting up two visibly-nested containers requires a multi-step
    dispatcher choreography — we drive it via ``page.evaluate`` directly
    against the state store rather than clicking through the UI.
    """

    _open_builder(page, hub_url)
    # Fake-build two nested containers + a drop coordinate inside both.
    # The exact mechanism for "fake state" is left to a clientside
    # window helper that the dispatcher exposes for testability; if
    # that helper is missing, skip rather than fail (the underlying
    # innermost-wins logic is covered by unit tests).
    has_helper = page.evaluate(
        "() => typeof window.phenoSetState === 'function'"
    )
    if not has_helper:
        pytest.skip(
            "window.phenoSetState test helper not exposed; nested-container "
            "innermost-wins exercised by tests/unit/gui/builder/test_dispatch.py"
        )
    # If the helper is exposed (future tooling), drive the scenario.
    # For now, the unit test is the contract gate.


def test_palette_drag_drop_on_existing_block_lands_adjacent(
    page: Page, hub_url: str,
) -> None:
    """Drop on a block → new block lands adjacent (right offset by dagre)."""

    _open_builder(page, hub_url)
    box = _canvas_box(page)
    # First drop: create a block in the middle.
    _drag_palette_to_canvas(
        page,
        "GaussianBlur",
        box["width"] / 2,
        box["height"] / 2,
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy && cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur').length === 1;
        }""",
        timeout=10_000,
    )
    # Second drop: target the existing block's center.
    target = page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            const gb = cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur')[0];
            const pos = gb.renderedPosition();
            return { x: pos.x, y: pos.y };
        }"""
    )
    _drag_palette_to_canvas(
        page,
        "GaussianBlur",
        target["x"],
        target["y"],
    )
    # Now there should be TWO GaussianBlur blocks; the second has a
    # different position (cytoscape doesn't overlay).
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy && cy.nodes().filter(n => n.data('class_name') === 'GaussianBlur').length === 2;
        }""",
        timeout=10_000,
    )


def test_palette_drag_drop_on_wire_is_positional_not_insertion(
    page: Page, hub_url: str,
) -> None:
    """Drop on a wire → block at coords, wire selected (no split).

    Per spec §4.8, dropping a palette block on top of an existing wire
    is purely positional: the wire is selected as a side-effect, the
    new block lands at the drop coordinates, and the wire is **not**
    split.
    """

    _open_builder(page, hub_url)
    # Build A → B so there's a wire to drop on. The infrastructure to
    # programmatically wire two blocks without going through the wire
    # JS lives in the dispatcher; expose it as a helper or fall back to
    # asserting from a pre-seeded state if absent.
    has_helper = page.evaluate(
        "() => typeof window.phenoSetState === 'function'"
    )
    if not has_helper:
        pytest.skip(
            "window.phenoSetState test helper not exposed; this scenario "
            "requires programmatic wire setup not yet wired into the UI."
        )


def test_palette_drag_drop_outside_cy_slot_cancels(
    page: Page, hub_url: str,
) -> None:
    """Drag off the canvas wrapper bounds → no block created."""

    _open_builder(page, hub_url)
    # Count blocks before.
    before = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    palette = _palette_button(page, "GaussianBlur")
    palette.hover()
    page.mouse.down()
    # Move far above the canvas wrapper (no drop target).
    page.mouse.move(0, 0, steps=5)
    page.mouse.up()
    after = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    assert before == after, "Drop outside the canvas should not create a block"


def test_palette_drag_esc_during_drag_cancels(page: Page, hub_url: str) -> None:
    """Escape mid-drag → no block created.

    Playwright fires `keyboard.press("Escape")` after the drag has begun
    but before the drop; the JS listener cancels the active drag.
    """

    _open_builder(page, hub_url)
    before = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    palette = _palette_button(page, "GaussianBlur")
    palette.hover()
    page.mouse.down()
    page.mouse.move(50, 50, steps=3)
    # Cancel via Escape, then release.
    page.keyboard.press("Escape")
    page.mouse.up()
    after = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    assert before == after, "Escape during drag should cancel the drop"


# ---------------------------------------------------------------------------
# 8.3.1 — Keyboard fallback & rejection cases
# ---------------------------------------------------------------------------


def test_palette_keyboard_fallback(page: Page, hub_url: str) -> None:
    """Tab to palette button + Enter → block placed at viewport center."""

    _open_builder(page, hub_url)
    before = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    # ``data-palette-class`` lives on the draggable wrapper ``<div>``
    # (not focusable); the focusable, ``Enter``-activatable element is
    # the ``<button>`` it wraps.  Focus that.
    palette_button = _palette_button(page, "GaussianBlur").locator(
        "button"
    )
    palette_button.focus()
    page.keyboard.press("Enter")
    # Wait for the block to appear.
    page.wait_for_function(
        f"() => window.phenoGetCy().nodes().length > {before}",
        timeout=10_000,
    )


def test_palette_no_input_image_button(page: Page, hub_url: str) -> None:
    """Palette must not expose an Input Image button (spec §4.8 + §4.10).

    Input Image is auto-seeded per scope; the palette is gated to
    every-other class.
    """

    _open_builder(page, hub_url)
    expect(_palette_button(page, "InputImage")).to_have_count(0)


def test_palette_dispatch_rejects_input_image_class_name(
    page: Page, hub_url: str,
) -> None:
    """Fake-dispatch via STORE_PALETTE_DROP with InputImage → state unchanged + toast.

    Drives the server-side dispatcher directly by writing into the
    palette-drop store via ``dash_clientside.set_props`` (the same
    API ``palette_dnd.js`` uses).  The block count must not change
    and a toast carrying the "Input Image" wording must appear.
    """

    _open_builder(page, hub_url)
    before = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    # Write the rejected payload directly into the store. Uses the
    # exact API ``palette_dnd.js`` uses (``dash_clientside.set_props``)
    # so the test mirrors the production write path without depending
    # on the actual drag gesture.
    page.evaluate(
        """() => {
            const payload = {
                kind: 'block_create',
                class_name: 'InputImage',
                x: 0, y: 0,
                container_block_id: null,
                ts: Date.now(),
            };
            if (
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === 'function'
            ) {
                window.dash_clientside.set_props('store-palette-drop', { data: payload });
            }
        }"""
    )
    # Give Dash a moment to round-trip the dispatch.
    page.wait_for_timeout(500)
    # Block count unchanged (the rejected dispatch leaves state as-is).
    after = page.evaluate(
        "() => window.phenoGetCy().nodes().length"
    )
    assert before == after, "Rejected InputImage dispatch must not change canvas"


def test_palette_dnd_js_fails_to_load(page: Page, hub_url: str) -> None:
    """Block ``palette_dnd.js`` via route interception → banner shows + palette disabled.

    Drops a Playwright route override that returns 404 for the JS
    asset, navigates fresh, and asserts the asset-status banner and
    the palette ``pointer-events: none`` style both appear.
    """

    # Install a route block BEFORE navigation so the JS request is
    # intercepted on first paint.  The trailing ``*`` is load-bearing:
    # Dash serves ``assets/`` files with a ``?m=<mtime>`` cache-buster
    # query string, so a pattern without it never matches the real URL.
    page.route(
        "**/assets/palette_dnd.js*",
        lambda route: route.fulfill(status=404, body=""),
    )
    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#canvas-cytoscape", timeout=15_000)
    # Wait for the asset poller to flip palette_dnd → False (default
    # poll interval per spec is 500ms; allow a few cycles).
    page.wait_for_function(
        "() => document.querySelector('#banner-asset-status') !== null",
        timeout=15_000,
    )
    # The banner should be visible (display != "none") within the
    # poll-window allowance.
    page.wait_for_function(
        """() => {
            const el = document.querySelector('#banner-asset-status');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none';
        }""",
        timeout=15_000,
    )
    # Palette container should carry pointer-events:none.
    palette_pe = page.evaluate(
        """() => {
            const el = document.querySelector('#palette');
            if (!el) return null;
            return window.getComputedStyle(el).pointerEvents;
        }"""
    )
    assert palette_pe == "none", (
        f"Expected palette pointer-events: none when palette_dnd.js fails to "
        f"load; got {palette_pe!r}"
    )
