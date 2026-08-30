"""One bounded toolbar camera drives every passive Colony view.

The assertions cover the fixed crop's fit zoom, linked zoom and pan commands,
and the absence of direct per-cell mouse controllers.

Both halves are asserted, and the second is the one that matters. Asserting
only ``len(set(zooms)) == 1`` is satisfied PERFECTLY by the bug this module
exists to catch: a ``View`` carries no ``target`` -- ``target`` and ``zoom``
both live in the viewState -- so "one View per cell over one shared
viewState", built literally, gives every cell the same target and renders
the same colony N times, with identical zooms.

These tests PAINT, so they need a real GL stack: ``channel="chromium"``
(Playwright's default launch uses ``chromium_headless_shell``, which ships
no GL at all) on an X display. The ``xvfb_display`` fixture is imported from
``test_viv_facade_renders``, which owns it.

Gated by ``PLAYWRIGHT=1`` via the module-level skip in ``conftest.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from phenotypic.gui.results_viewer._zarr_routes import (
    store_generation_token,
    zarr_store_url,
)
from phenotypic.sdk_ import zarr_store_path
from tests.e2e.gui.conftest import _start_live_server, bind_results_output
from tests.e2e.gui.test_viv_codec_reads_a_real_store import (
    DATASET,
    IMAGE_SHAPE,
    STEM,
    _OUTPUT_NAME,
    _build_viv_sandbox,
)

# Imported for its fixture, which that module owns. Re-exported into this
# module's namespace is how pytest resolves a fixture defined elsewhere
# without a conftest.
from tests.e2e.gui.test_viv_facade_renders import xvfb_display  # noqa: F401

CONTAINER = "colony-grid-camera"
PROBE_W, PROBE_H = 640, 480

#: Cell side length, in CSS px, and the gap between cells. Small enough that
#: every cell of the grid below is on screen: an offscreen view is culled,
#: and a culled view has no viewport to read a zoom off.
CELL_PX = 96
CELL_GAP = 4

#: A 3x3 block of colonies spread over the fixture store's extent. Nine is
#: enough to make "every target is distinct" a real claim and small enough
#: that the whole grid fits the probe.
GRID_ROWS, GRID_COLS = 3, 3


def _cells() -> list[dict[str, float | str]]:
    """Nine colony centroids, evenly spread over the store's extent."""
    rows, cols = IMAGE_SHAPE
    return [
        {
            "id": f"r{r}c{c}",
            "centroidRr": (r + 0.5) * rows / GRID_ROWS,
            "centroidCc": (c + 0.5) * cols / GRID_COLS,
            "size": CELL_PX,
        }
        for r in range(GRID_ROWS)
        for c in range(GRID_COLS)
    ]


@pytest.fixture(scope="module")
def camera_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_viv_sandbox(tmp_path_factory.mktemp("colony-camera"))


@pytest.fixture(scope="module")
def camera_hub(camera_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(camera_sandbox)


@pytest.fixture(scope="module")
def camera_store_url(camera_sandbox: Path) -> str:
    store = zarr_store_path(
        camera_sandbox / "results" / _OUTPUT_NAME, DATASET, STEM
    )
    return zarr_store_url(
        "/results/", DATASET, STEM, store_generation_token(store)
    )


@pytest.fixture(scope="module")
def camera_page(playwright, xvfb_display: str, camera_hub: str, camera_store_url: str):  # noqa: F811
    """A page in the FULL Chromium build with the colony grid mounted.

    Module-scoped and mounted once: each deck.gl instance holds a WebGL
    context, and re-mounting per test leaks one.
    """
    env = {**os.environ, "DISPLAY": xvfb_display}
    browser = playwright.chromium.launch(channel="chromium", env=env)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        assert page.evaluate(
            "() => document.createElement('canvas')"
            ".getContext('webgl2') !== null"
        ), (
            "no WebGL2 in this browser -- deck.gl would build no viewports "
            "and every assertion below would read an empty list"
        )
        bind_results_output(page, camera_hub, f"results/{_OUTPUT_NAME}")
        page.wait_for_function(
            "() => window.phenotypicViv !== undefined", timeout=30_000
        )
        page.evaluate(
            """([id, w, h]) => {
                const el = document.createElement('div');
                el.id = id;
                el.style.cssText = `position:fixed;left:0;top:0;`
                    + `width:${w}px;height:${h}px;background:#000;z-index:9999;`;
                document.body.appendChild(el);
            }""",
            [CONTAINER, PROBE_W, PROBE_H],
        )
        outcome = page.evaluate(
            """async ([id, url, cells, size, gap]) => {
                try {
                    await window.phenotypicViv.mount(id, {});
                    await window.phenotypicViv.setSource(id, {
                        storeUrl: url,
                        seriesPath: 'rgb',
                        labelPath: 'rgb/labels/objmap',
                    });
                    const n = await window.phenotypicViv.setGridViews(
                        id, cells, {zoomOffset: 0, offsetX: 0, offsetY: 0},
                        {cellSize: size, gap: gap, cropSize: 256});
                    return {ok: true, views: n};
                } catch (e) {
                    return {ok: false, err: String(e && e.message)};
                }
            }""",
            [CONTAINER, camera_store_url, _cells(), CELL_PX, CELL_GAP],
        )
        assert outcome["ok"], outcome
        assert outcome["views"] == GRID_ROWS * GRID_COLS, outcome
        page.wait_for_timeout(2_000)  # let deck.gl build the viewports
        yield page
    finally:
        browser.close()


def _view_states(page) -> list[dict]:
    """Read back what deck.gl actually rendered each cell with."""
    states = page.evaluate(
        "([id]) => window.phenotypicViv.__debugViewStates(id)", [CONTAINER]
    )
    assert states, (
        "no viewports -- deck.gl built none, so nothing below is testing "
        "the camera"
    )
    return states


def test_toolbar_zoom_and_pan_move_every_tile_together(camera_page) -> None:
    """One toolbar command changes every zoom and target by the same amount."""
    before = _view_states(camera_page)
    camera_page.evaluate(
        "([id]) => window.phenotypicViv.setGridCamera("
        "id, {action: 'zoom', delta: 0.5})",
        [CONTAINER],
    )
    camera_page.wait_for_timeout(1_000)
    zoomed = _view_states(camera_page)

    before_zooms = [s["zoom"] for s in before]
    zooms = [s["zoom"] for s in zoomed]
    assert len(set(before_zooms)) == 1
    assert len(set(zooms)) == 1
    assert zooms[0] == pytest.approx(before_zooms[0] + 0.5)

    camera_page.evaluate(
        "([id]) => window.phenotypicViv.setGridCamera("
        "id, {action: 'pan', dx: 1, dy: 0})",
        [CONTAINER],
    )
    camera_page.wait_for_timeout(1_000)
    panned = _view_states(camera_page)
    deltas = {
        (
            after["target"][0] - prior["target"][0],
            after["target"][1] - prior["target"][1],
        )
        for prior, after in zip(zoomed, panned, strict=True)
    }
    assert len(deltas) == 1, f"linked pan drifted: {sorted(deltas)}"
    dx, dy = next(iter(deltas))
    assert dx > 0
    assert dy == pytest.approx(0)

    targets = [tuple(s["target"][:2]) for s in panned]
    assert len(set(targets)) == len(targets), (
        f"every cell is showing the same region: {targets[:4]}"
    )

    expected = {
        (float(cell["centroidCc"]) + dx, float(cell["centroidRr"]))
        for cell in _cells()
    }
    assert sorted(targets) == pytest.approx(sorted(expected))


def test_a_wheel_gesture_cannot_start_a_per_cell_camera(camera_page) -> None:
    """Direct wheel input is inert because comparison tiles are passive."""
    before = {s["id"]: s["zoom"] for s in _view_states(camera_page)}
    assert len(set(before.values())) == 1, before

    # Over the middle of the first cell, which is at the container's origin.
    camera_page.mouse.move(CELL_PX / 2, CELL_PX / 2)
    camera_page.mouse.wheel(0, -400)
    camera_page.wait_for_timeout(1_500)

    after = {s["id"]: s["zoom"] for s in _view_states(camera_page)}
    assert set(after) == set(before), (sorted(after), sorted(before))
    assert after == before


def test_a_drag_gesture_cannot_move_one_cell_independently(camera_page) -> None:
    """Direct dragging cannot bypass the toolbar's bounded shared offset."""
    before = [tuple(s["target"][:2]) for s in _view_states(camera_page)]

    camera_page.mouse.move(CELL_PX / 2, CELL_PX / 2)
    camera_page.mouse.down()
    camera_page.mouse.move(CELL_PX / 2 + 60, CELL_PX / 2 + 40, steps=8)
    camera_page.mouse.up()
    camera_page.wait_for_timeout(1_500)

    after = [tuple(s["target"][:2]) for s in _view_states(camera_page)]
    assert after == before, (
        f"a drag moved the cells' targets: {before[:3]} -> {after[:3]}"
    )
