"""The facade actually paints a CLI-written store through deck.gl.

Separate from ``test_viv_codec_reads_a_real_store.py`` on purpose. Those
tests only decode, so they run under Playwright's default headless Chromium
and cost nothing. This one PAINTS, and painting needs a real GL stack:

* Playwright's default ``chromium`` launch uses ``chromium_headless_shell``,
  which ships no GL at all -- ``canvas.getContext('webgl2')`` returns
  ``null``, and no flag combination changes it (six were tried in the
  phase-0 spike, including ``--enable-unsafe-swiftshader``). The full
  ``chrome-linux64`` build beside it does have one, reached with
  ``channel="chromium"``.
* That build additionally needs an X display: under bare headless the GPU
  process still dies with ``BindToCurrentSequence failed``. This module
  starts its own ``Xvfb`` rather than requiring the caller to wrap pytest in
  ``xvfb-run``.

Without both, deck.gl reports ``Failed to create WebGL context`` and paints
zero pixels -- a red that looks like a rendering bug and is not one.

Gated by ``PLAYWRIGHT=1`` via the module-level skip in ``conftest.py``.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from tests.e2e.gui.test_viv_codec_reads_a_real_store import (
    DATASET,
    EXPECTED_LEVELS,
    IMAGE_SHAPE,
    STEM,
    _OUTPUT_NAME,
    _build_viv_sandbox,
)
from phenotypic.gui.results_viewer._zarr_routes import (
    store_generation_token,
    zarr_store_url,
)
from phenotypic.sdk_ import zarr_store_path
from tests.e2e.gui.conftest import _start_live_server, bind_results_output

#: Background painted behind the deck canvas. Magenta because the store is
#: uint8 noise: a random pixel lands on exactly this colour about once in
#: 16.8 million, so "not this colour" counts painted pixels honestly.
BACKDROP = (255, 0, 255)

CONTAINER = "viv-render-probe"
PROBE_W, PROBE_H = 640, 480


@pytest.fixture(scope="module")
def xvfb_display() -> Iterator[str]:
    """Run an ``Xvfb`` and yield its ``DISPLAY``.

    Uses ``-displayfd`` so the server picks a free display number and tells
    us which -- probing numbers ourselves would race another job on a shared
    compute node.
    """
    if shutil.which("Xvfb") is None:
        pytest.skip("Xvfb is required to give Chromium a GL stack")
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen(
        ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "1280x1024x24"],
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    try:
        with os.fdopen(read_fd, "r") as handle:
            number = handle.readline().strip()
        if not number:
            proc.kill()
            pytest.skip("Xvfb did not report a display number")
        yield f":{number}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def gl_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_viv_sandbox(tmp_path_factory.mktemp("viv-render"))


@pytest.fixture(scope="module")
def gl_hub(gl_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(gl_sandbox)


@pytest.fixture(scope="module")
def gl_store_url(gl_sandbox: Path) -> str:
    store = zarr_store_path(gl_sandbox / "results" / _OUTPUT_NAME, DATASET, STEM)
    return zarr_store_url(
        "/results/", DATASET, STEM, store_generation_token(store)
    )


@pytest.fixture(scope="module")
def gl_page(playwright, xvfb_display: str, gl_hub: str):
    """A page in the FULL Chromium build, on an X display, bound to the run.

    A separate BROWSER, but the session's existing ``playwright`` driver:
    pytest-playwright's ``browser`` fixture is session-scoped and already
    launched with the default channel by the time this module runs, and its
    launch args cannot be changed after the fact -- while opening a second
    ``sync_playwright()`` inside the same session raises "Sync API inside
    the asyncio loop" as soon as any other e2e module has started one.
    """
    env = {**os.environ, "DISPLAY": xvfb_display}
    browser = playwright.chromium.launch(channel="chromium", env=env)
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        assert page.evaluate(
            "() => document.createElement('canvas')"
            ".getContext('webgl2') !== null"
        ), (
            "no WebGL2 in this browser -- deck.gl would paint nothing and "
            "the failure would look like a rendering bug"
        )
        bind_results_output(page, gl_hub, f"results/{_OUTPUT_NAME}")
        page.wait_for_function(
            "() => window.phenotypicViv !== undefined", timeout=30_000
        )
        yield page
    finally:
        browser.close()


def _canvas_stats(page, selector: str) -> tuple[int, int]:
    """``(painted, distinct)`` for ``selector``'s screenshot.

    ``painted`` counts pixels that are not the backdrop; ``distinct`` counts
    how many different colours those pixels take. Both are needed: an
    all-zeros label layer paints an opaque black rectangle over the whole
    extent, so ``painted`` alone cannot tell "the noise image rendered" from
    "a flat rectangle rendered".
    """
    from PIL import Image as PILImage

    raw = page.locator(selector).screenshot()
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    colours = {px for px in img.getdata() if px != BACKDROP}
    painted = sum(1 for px in img.getdata() if px != BACKDROP)
    return painted, len(colours)


def _mount_and_source(page, store_url: str) -> dict:
    page.evaluate(
        """([id, w, h, backdrop]) => {
            const el = document.createElement('div');
            el.id = id;
            el.style.cssText = `position:fixed;left:0;top:0;width:${w}px;`
                + `height:${h}px;background:${backdrop};z-index:9999;`;
            document.body.appendChild(el);
        }""",
        [CONTAINER, PROBE_W, PROBE_H, f"rgb{BACKDROP}"],
    )
    rows, cols = IMAGE_SHAPE
    outcome = page.evaluate(
        """async ([id, url, cx, cy]) => {
            try {
                await window.phenotypicViv.mount(id, {
                    initialViewState: {target: [cx, cy, 0], zoom: -2},
                });
                const loaded = await window.phenotypicViv.setSource(id, {
                    storeUrl: url,
                    seriesPath: 'rgb',
                    // RESOLVED server-side from phenotypic.labels.objmap --
                    // never derived here as `${seriesPath}/labels/objmap`.
                    labelPath: 'rgb/labels/objmap',
                });
                return {ok: true, levels: loaded.image.data.length,
                        labelLevels: loaded.label
                            ? loaded.label.data.length : 0};
            } catch (e) { return {ok: false, err: String(e && e.message)}; }
        }""",
        [CONTAINER, store_url, cols / 2, rows / 2],
    )
    assert outcome["ok"], outcome
    return outcome


def test_the_facade_paints_a_cli_written_store(gl_page, gl_store_url: str) -> None:
    """mount + setSource put real pixels on the canvas.

    The store is 1200x900 uint8 noise across three recorded pyramid levels,
    drawn at ``zoom: -2`` so it covers 300x225 = 67,500 device-independent
    pixels of the 640x480 probe. Above ``stop_px`` on purpose: this is the
    MULTISCALE path (``MultiscaleImageLayer`` over a tiled, sharded read),
    which a small single-level store would never reach. The floor below is
    deliberately well under the extent -- this asserts "deck.gl painted the
    image", not a pixel-exact layout.
    """
    loaded = _mount_and_source(gl_page, gl_store_url)
    assert loaded["levels"] == EXPECTED_LEVELS, loaded
    assert loaded["labelLevels"] == EXPECTED_LEVELS, loaded
    gl_page.wait_for_function(
        """([id]) => {
            const c = document.querySelector('#' + id + ' canvas');
            return !!c && c.width > 0 && c.height > 0;
        }""",
        arg=[CONTAINER],
        timeout=30_000,
    )
    gl_page.wait_for_timeout(2_000)  # let the tile layer settle
    painted, distinct = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert painted > 20_000, (
        f"only {painted} painted pixels -- deck.gl rendered nothing, or the "
        "image layer never received a decoded tile"
    )
    assert distinct > 100, (
        f"{painted} pixels painted but only {distinct} distinct colours -- "
        "that is a flat rectangle, not decoded noise. A pixel source that "
        "returned fill_value zeros for every chunk would look exactly like "
        "this, which is why the colour count is asserted and not just the "
        "painted area."
    )


def test_layer_visibility_reaches_deck_gl(gl_page, gl_store_url: str) -> None:
    """`setLayerVisibility` drives deck.gl, in both directions.

    Runs against the instance the previous test mounted -- the module-scoped
    page is shared, and re-mounting would leak a second deck.gl context.

    BOTH layers are hidden, not just the image: this store's objmap is all
    zeros (a Stage-1 store), and an all-zeros label layer paints an opaque
    black rectangle over the whole extent. Hiding only the image leaves that
    rectangle and the painted-pixel count does not move -- which reads as
    "visibility is broken" when it is not.
    """
    painted_before, _ = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert painted_before > 20_000, "precondition: the image must be painted"

    gl_page.evaluate(
        """async ([id]) => {
            await window.phenotypicViv.setLayerVisibility(id, 'image', false);
            await window.phenotypicViv.setLayerVisibility(id, 'labels', false);
        }""",
        [CONTAINER],
    )
    gl_page.wait_for_timeout(1_500)
    painted_hidden, _ = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert painted_hidden < painted_before / 20, (
        f"{painted_hidden} pixels still painted after hiding both layers "
        f"(was {painted_before}) -- visibility never reached deck.gl"
    )

    gl_page.evaluate(
        """async ([id]) =>
            await window.phenotypicViv.setLayerVisibility(id, 'image', true)""",
        [CONTAINER],
    )
    gl_page.wait_for_timeout(1_500)
    painted_again, distinct_again = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert painted_again > 20_000, (
        f"{painted_again} pixels after re-showing the image layer -- hiding "
        "a layer discarded it rather than filtering it out of the render"
    )
    assert distinct_again > 100, distinct_again
