"""One shared viewState drives every colony view.

The assertion is that all views report the SAME zoom after one is changed --
not that they converge. A sync protocol would pass a convergence test and
still show tearing mid-gesture; a shared value cannot.

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
                        id, cells, {zoom: 0},
                        {cellSize: size, gap: gap});
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


def test_zooming_one_cell_moves_every_cell(camera_page) -> None:
    """One programmatic zoom lands on every view, and targets stay apart."""
    camera_page.evaluate(
        """([id]) => window.phenotypicViv.setViewState(
               id, {zoom: 3, target: [0, 0, 0]})""",
        [CONTAINER],
    )
    camera_page.wait_for_timeout(1_000)
    states = _view_states(camera_page)

    # `page.evaluate` marshals JS objects into Python DICTS -- attribute
    # access raises AttributeError and the test ERRORS rather than failing
    # informatively, which is the worst outcome for a test whose job is to
    # fail informatively.
    zooms = [s["zoom"] for s in states]
    targets = [tuple(s["target"][:2]) for s in states]

    assert len(zooms) > 1
    assert set(zooms) == {3}, f"zoom drifted apart: {sorted(set(zooms))}"

    # The complementary half, and the one that matters. A single shared
    # viewState gives every cell the same target, so the grid renders one
    # colony N times -- with identical zooms, passing the assertion above.
    assert len(set(targets)) == len(targets), (
        f"every cell is showing the same region: {targets[:4]}"
    )

    # And each target is the centroid it was handed, not merely distinct:
    # a layout bug that assigned targets in the wrong order would still
    # produce N distinct values. `target` is [x, y, z] = [cc, rr, 0].
    expected = {
        (float(cell["centroidCc"]), float(cell["centroidRr"]))
        for cell in _cells()
    }
    assert set(targets) == expected, (targets, sorted(expected))


def test_a_wheel_gesture_over_one_cell_moves_every_cell(camera_page) -> None:
    """The half a programmatic test cannot reach.

    The rejected keyed-viewState design fails HERE and nowhere else:
    ``onViewStateChange`` fires with the ONE view a gesture touched, so a
    keyed map keeps N zooms and only the gestured cell moves. Driving
    ``setViewState`` never exercises that path. This dispatches a real wheel
    event over the first cell and asserts every cell followed.
    """
    before = {s["id"]: s["zoom"] for s in _view_states(camera_page)}
    assert len(set(before.values())) == 1, before

    # Over the middle of the first cell, which is at the container's origin.
    camera_page.mouse.move(CELL_PX / 2, CELL_PX / 2)
    camera_page.mouse.wheel(0, -400)
    camera_page.wait_for_timeout(1_500)

    after = {s["id"]: s["zoom"] for s in _view_states(camera_page)}
    assert set(after) == set(before), (sorted(after), sorted(before))
    assert len(set(after.values())) == 1, (
        f"the wheel moved some cells and not others: {sorted(set(after.values()))}"
        " -- the camera is a per-view sync protocol, not one shared value"
    )
    moved = next(iter(after.values())) != next(iter(before.values()))
    assert moved, (
        f"zoom did not move at all ({before} -> {after}) -- the views carry "
        "no controller, so the 'Shared camera' lock constrains nothing"
    )


def test_a_pan_gesture_cannot_move_a_cell_off_its_colony(camera_page) -> None:
    """`target` is per-view and re-applied every render, so pan is discarded.

    Not a nicety: the colony grid's whole contract is that cell *i* shows
    colony *i*. A controller that let a drag write `target` back into the
    shared state would slide every cell off its colony together, which is
    worse than the per-cell drift the shared value prevents.
    """
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
