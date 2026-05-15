"""Shared helpers for the builder-canvas Playwright E2E tests.

``test_wire_drawing.py``, ``test_palette_drag.py`` and ``test_containers.py``
all drive the same ``/builder/`` page through the same handful of
primitives — navigating to the canvas, locating palette buttons,
synthesising drags, and publishing ``set_props`` payloads into the
clientside stores. They previously each carried near-identical private
copies; this module is the single hardened home so a fix lands once.

It lives as ``conftest.py`` (rather than a plain ``_helpers.py``) to
mirror ``tests/e2e/gui/conftest.py``, which already hosts the shared
``expand_palette_accordions`` helper, and so pytest auto-discovers it for
the ``tests/e2e/gui/builder/`` package.

CI-hardening rationale — these helpers pass locally but flaked in CI's
slower headless runner, where a ``set_props`` publish could race the tail
of a prior Dash callback:

* :func:`_open_builder` waits for the *entire* clientside surface — both
  JS readiness sentinels, ``phenoGetCy`` returning a live cytoscape
  instance, and ``dash_clientside.set_props`` — before returning, so a
  test never publishes into a store before the layer that consumes it is
  bound.
* :func:`_publish_edge_event` / :func:`_publish_palette_drop` *throw* when
  ``set_props`` is missing instead of silently no-opping, turning a
  masked failure into an explicit error rather than a downstream
  ``wait_for_function`` timeout.
* :func:`_seed_two_blocks` and :func:`_click_new_pipeline_button` wait for
  the network to go idle after their multi-step setup, so the caller's
  next ``set_props`` publish doesn't race an in-flight Dash callback. The
  builder's asset-readiness poller is bounded (<=1500 ms, no ``fetch``),
  so ``networkidle`` always settles.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from playwright.sync_api import Page

from tests.e2e.gui.conftest import expand_palette_accordions


# ---------------------------------------------------------------------------
# TEMP DIAGNOSTIC (PR #95 cluster-2 E2E flake)
# ---------------------------------------------------------------------------
# The builder-canvas tests flake only in CI: a test's ``set_props`` publish
# into ``store-edge-event`` / ``store-palette-drop`` doesn't take effect and
# the post-publish ``wait_for_function`` times out. Static analysis can't pin
# the race (it passes locally), so we need CI-side evidence. On any builder-
# test failure this dumps (a) the browser console + page errors collected
# during the test and (b) the tail of the GUI subprocess log
# (``_start_live_server`` redirects its stdout+stderr there) — which carries
# the ``fan_in_state_mutation`` trigger log. Remove this whole block once the
# flake is root-caused and fixed.


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ANN001, ANN201
    """Stash each phase's report on the item so fixtures can read the outcome."""

    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"_rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _builder_failure_diagnostics(request, page: Page):  # noqa: ANN001, ANN201
    """On a builder-test failure, dump the browser console + GUI server log."""

    console: list[str] = []
    page.on(
        "console",
        lambda msg: console.append(f"[console:{msg.type}] {msg.text}"),
    )
    page.on("pageerror", lambda exc: console.append(f"[pageerror] {exc}"))
    yield
    rep = getattr(request.node, "_rep_call", None)
    if rep is None or not rep.failed:
        return
    print("\n========== builder-test failure diagnostics ==========")
    print(f"--- browser console ({len(console)} msgs, last 250) ---")
    for line in console[-250:]:
        print(line)
    if "hub_url" in request.fixturenames:
        hub_url = request.getfixturevalue("hub_url")
        port = hub_url.rsplit(":", 1)[-1]
        log_path = Path(tempfile.gettempdir()) / f"phenotypic-gui-e2e-{port}.log"
        print(f"--- GUI server log: {log_path} (last 150 lines) ---")
        if log_path.exists():
            lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            for line in lines[-150:]:
                print(line)
        else:
            print("(log file not found)")
    # Clientside snapshot: what does the live cytoscape instance actually
    # hold at timeout? Splits "server produced no edge" from "edge never
    # reached cytoscape".
    try:
        cy_state = page.evaluate(
            """() => {
                const cy = window.phenoGetCy && window.phenoGetCy();
                if (!cy) {
                    return { phenoGetCy: typeof window.phenoGetCy, cy: null };
                }
                return {
                    cy: true,
                    nodes: cy.nodes().length,
                    edges: cy.edges().length,
                    node_classes: cy.nodes().map(n => n.data('class_name')),
                    edge_endpoints: cy.edges().map(
                        e => e.data('source') + '->' + e.data('target')
                    ),
                };
            }"""
        )
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic
        cy_state = f"(clientside snapshot failed: {exc})"
    print(f"--- clientside cytoscape state: {cy_state}")
    print("======================================================")


def _open_builder(page: Page, hub_url: str) -> None:
    """Navigate to ``/builder/`` and wait for the full clientside surface.

    Waits for the canvas wrapper, *both* builder JS readiness sentinels
    (``palette_dnd`` + ``wire_drawing``), ``window.phenoGetCy`` returning
    a live cytoscape instance, and ``dash_clientside.set_props`` — every
    primitive the builder-canvas tests touch — then expands the palette
    accordions so buttons in any category are draggable.
    """

    page.goto(hub_url + "/builder/")
    page.wait_for_selector("#canvas-cytoscape", timeout=15_000)
    page.wait_for_function(
        """() => (
            window.phenotypic_palette_dnd_ready === true
            && window.phenotypic_wire_drawing_ready === true
            && typeof window.phenoGetCy === 'function'
            && window.phenoGetCy() != null
            && window.dash_clientside != null
            && typeof window.dash_clientside.set_props === 'function'
        )""",
        timeout=15_000,
    )
    # Palette categories past the first start collapsed; expand them so
    # buttons like ``GaussianBlur`` (Enhancer) are visible + draggable.
    expand_palette_accordions(page)


def _palette_button(page: Page, class_name: str):
    """Locate a palette button by its ``data-palette-class`` attribute.

    The attribute selector avoids depending on the pattern-matching Dash
    id encoding (a JSON-serialised dict that can shift between Dash
    versions).
    """

    return page.locator(f"[data-palette-class='{class_name}']")


def _canvas_box(page: Page) -> dict:
    """Return the cytoscape canvas wrapper's bounding box."""

    box = page.locator("#canvas-cytoscape").bounding_box()
    assert box is not None, "Canvas wrapper not on screen"
    return box


def _drag_palette_to_canvas(
    page: Page, class_name: str, canvas_x: float, canvas_y: float
) -> None:
    """Synthesize an HTML5 drag from a palette button to a canvas point.

    Playwright's high-level ``drag_to`` does not synthesise ``DragEvent``
    objects (which ``palette_dnd.js`` listens for), so this primes with a
    ``hover`` and drives ``mouse.down`` / ``mouse.move`` / ``mouse.up``
    manually. The two-step move guarantees a non-zero delta so the
    browser doesn't cancel the drag below its movement threshold.
    """

    palette = _palette_button(page, class_name)
    palette.hover()
    page.mouse.down()
    box = _canvas_box(page)
    target_x = box["x"] + canvas_x
    target_y = box["y"] + canvas_y
    page.mouse.move(target_x - 5, target_y - 5, steps=5)
    page.mouse.move(target_x, target_y, steps=5)
    page.mouse.up()


def _has_state_injection_helper(page: Page) -> bool:
    """Return True iff the JS layer exposes ``window.phenoSetState``.

    Scenarios that need a precise starting state (nested containers,
    inner edges) skip gracefully when the helper is absent — the
    underlying logic is covered by
    ``tests/unit/gui/builder/test_dispatch.py``.
    """

    return page.evaluate("() => typeof window.phenoSetState === 'function'")


def _publish_edge_event(page: Page, payload: dict) -> None:
    """Write a payload into ``STORE_EDGE_EVENT`` via ``set_props``.

    Mirrors the JS publish path so tests can synthesise wire creations /
    deletions without driving cytoscape port geometry. Throws when
    ``dash_clientside.set_props`` is unavailable — :func:`_open_builder`
    waits for it, so a missing ``set_props`` here is a real failure that
    must surface immediately rather than as a downstream timeout.
    """

    page.evaluate(
        """(payload) => {
            if (
                !window.dash_clientside
                || typeof window.dash_clientside.set_props !== 'function'
            ) {
                throw new Error(
                    'dash_clientside.set_props unavailable — cannot publish '
                    + 'STORE_EDGE_EVENT'
                );
            }
            // TEMP DIAGNOSTIC (PR #95 cluster-2 flake): mark the publish in
            // the browser console so the failure-diagnostics fixture can
            // correlate it against the server-side fan_in trigger log.
            console.log(
                '[e2e] publish store-edge-event ' + JSON.stringify(payload)
            );
            window.dash_clientside.set_props(
                'store-edge-event', { data: payload }
            );
        }""",
        payload,
    )


def _publish_palette_drop(page: Page, payload: dict) -> None:
    """Write a ``block_create`` payload into ``STORE_PALETTE_DROP``.

    Mirrors what ``palette_dnd.js`` emits on a real drop, bypassing the
    browser's native HTML5 drag machinery (Playwright's synthesised
    pointer events don't reliably fire ``DragEvent``s, especially for a
    drop that must hit-test inside a compound container). Throws when
    ``set_props`` is unavailable — see :func:`_publish_edge_event`.
    """

    page.evaluate(
        """(payload) => {
            if (
                !window.dash_clientside
                || typeof window.dash_clientside.set_props !== 'function'
            ) {
                throw new Error(
                    'dash_clientside.set_props unavailable — cannot publish '
                    + 'STORE_PALETTE_DROP'
                );
            }
            // TEMP DIAGNOSTIC (PR #95 cluster-2 flake): mark the publish in
            // the browser console so the failure-diagnostics fixture can
            // correlate it against the server-side fan_in trigger log.
            console.log(
                '[e2e] publish store-palette-drop ' + JSON.stringify(payload)
            );
            window.dash_clientside.set_props(
                'store-palette-drop', { data: payload }
            );
        }""",
        payload,
    )


def _seed_two_blocks(page: Page) -> dict:
    """Drop two ``GaussianBlur`` blocks on the canvas; return their ids.

    Returns ``{"source": <block_id>, "target": <block_id>}``. Waits for
    each block to materialise in cytoscape, then for the network to go
    idle — so the caller's next ``set_props`` publish doesn't race the
    tail of the second palette-drop's Dash callback.
    """

    box = _canvas_box(page)
    _drag_palette_to_canvas(
        page, "GaussianBlur", box["width"] * 0.3, box["height"] * 0.5
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return false;
            return cy.nodes().filter(
                n => n.data('class_name') === 'GaussianBlur'
            ).length >= 1;
        }""",
        timeout=10_000,
    )
    _drag_palette_to_canvas(
        page, "GaussianBlur", box["width"] * 0.7, box["height"] * 0.5
    )
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy();
            return cy.nodes().filter(
                n => n.data('class_name') === 'GaussianBlur'
            ).length >= 2;
        }""",
        timeout=10_000,
    )
    # Let the second palette-drop's Dash round-trip fully settle before
    # the caller publishes the next store event.
    page.wait_for_load_state("networkidle")
    return page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            const blocks = cy.nodes().filter(
                n => n.data('class_name') === 'GaussianBlur'
            );
            return {
                source: blocks[0].data('block_id') || blocks[0].id(),
                target: blocks[1].data('block_id') || blocks[1].id(),
            };
        }"""
    )


def _click_new_pipeline_button(page: Page) -> str:
    """Click ``+ New Pipeline`` and wait for the container to materialise.

    Spec §4.4: the button mints an ``ImagePipeline`` container in the
    current scope. The button is both draggable and clickable (keyboard
    fallback), so ``.click()`` is enough. Waits for the container node to
    appear in cytoscape and for the network to go idle, then returns the
    container's ``block_id`` so callers can target it for a nested-scope
    drop without a second ``page.evaluate`` round-trip.
    """

    page.locator("#btn-new-pipeline-node").click()
    page.wait_for_function(
        """() => {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (!cy) return false;
            return cy.nodes().some(
                n => n.data('class_name') === 'ImagePipeline'
            );
        }""",
        timeout=10_000,
    )
    page.wait_for_load_state("networkidle")
    return page.evaluate(
        """() => {
            const cy = window.phenoGetCy();
            const cont = cy.nodes().filter(
                n => n.data('class_name') === 'ImagePipeline'
            )[0];
            return cont ? (cont.data('block_id') || cont.id()) : null;
        }"""
    )
