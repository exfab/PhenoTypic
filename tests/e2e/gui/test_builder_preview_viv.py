"""The builder's node-preview pane paints a preview store through Viv.

This is the render swap's end-to-end statement, and it needs a real GL stack
for the reason recorded on ``test_viv_facade_renders.py``: Playwright's
default ``chromium`` launch uses ``chromium_headless_shell``, which has no GL
at all, so deck.gl reports ``Failed to create WebGL context`` and paints zero
pixels -- a red that looks like a rendering bug and is not one. The full
``chrome-linux64`` build (``channel="chromium"``) has one and additionally
needs an X display, so this module starts its own ``Xvfb``.

Three things are asserted that no unit test can reach:

* ``window.phenotypicViv`` exists on the BUILDER page. That is the whole
  point of the shared asset mount -- the vendored bundle and facade are
  served out of the results viewer's package directory by a blueprint,
  because Dash takes one ``assets_folder`` per app and the builder's already
  holds eight files.
* ``mountViewer`` paints decoded pixels from the byte route.
* The layer radio's channels reach deck.gl: the ``objmap`` channel hides the
  image layer, and switching back to a pixel series restores it. The facade's
  viewer keeps its hidden-id set across a re-source, so this direction is the
  one a plausible implementation gets wrong.

The preview cache root is process-independent
(``tempfile.gettempdir()/phenotypic/pipeline-preview``) and the hub WIPES it
at builder-app construction, so the scope is seeded AFTER the server is up,
under a session id unique to this module.

Gated by ``PLAYWRIGHT=1`` via the module-level skip in ``conftest.py``.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_callbacks import build_channel_spec
from phenotypic.gui.builder._preview_zarr_routes import preview_zarr_url
from phenotypic.gui.results_viewer._zarr_routes import store_generation_token
from tests.e2e.gui.conftest import _start_live_server

#: Backdrop behind the probe. Magenta because the image is uint8 noise: a
#: random pixel lands on exactly this colour about once in 16.8 million.
BACKDROP = (255, 0, 255)

CONTAINER = "preview-viv-stage"
PROBE_W, PROBE_H = 640, 480

#: Large enough to have a real pyramid ladder, so the multiscale path is the
#: one exercised rather than the single-level one.
IMAGE_SHAPE = (900, 1200)

SESSION = "e2epreview" + uuid.uuid4().hex[:8]
BLOCK = "previewblock" + uuid.uuid4().hex[:8]
STORE_NAME = pc.BASE_STORE_NAME


@pytest.fixture(scope="module")
def xvfb_display() -> Iterator[str]:
    """Run an ``Xvfb`` and yield its ``DISPLAY``.

    ``-displayfd`` so the server picks a free display number and reports it;
    probing numbers ourselves would race another job on a shared node.
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
def preview_hub(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    yield from _start_live_server(tmp_path_factory.mktemp("builder-preview"))


@pytest.fixture(scope="module")
def seeded_scope(preview_hub: str) -> Path:
    """One preview scope, seeded after the hub wiped the cache root."""
    rows, cols = IMAGE_SHAPE
    rng = np.random.default_rng(7)
    image = Image(arr=rng.integers(0, 255, (rows, cols, 3), dtype=np.uint8))
    labels = np.zeros((rows, cols), dtype=np.int32)
    labels[100:200, 100:200] = 1
    labels[400:520, 700:820] = 2
    image.objmap[:] = labels

    scope_dir = pc.scope_dir(SESSION, [])
    store = image.save2zarr(scope_dir / STORE_NAME)
    pc.write_manifest(SESSION, [], {
        "version": pc.MANIFEST_VERSION, "fingerprint": "fp",
        "fingerprint_inputs": [], "scope_key": "", "error": None,
        "nodes": {BLOCK: {"store": STORE_NAME,
                          "layers": ["rgb", "gray", "detect_mat", "objmap"],
                          "shape": [rows, cols], "num_objects": 2}},
    })
    return store


def _spec_for(store: Path, channel: str) -> dict:
    """The spec Python would hand the client for one channel, right now."""
    base = preview_zarr_url(
        "/builder/", SESSION, pc.scope_hash([]), BLOCK,
        store_generation_token(store),
    )
    return build_channel_spec(store, base, channel)


@pytest.fixture(scope="module")
def gl_page(playwright, xvfb_display: str, preview_hub: str, seeded_scope):
    """A page in the FULL Chromium build, on an X display, on the BUILDER.

    A separate browser but the session's existing ``playwright`` driver:
    pytest-playwright's ``browser`` fixture is session-scoped and already
    launched by the time this module runs, while opening a second
    ``sync_playwright()`` in the same session raises "Sync API inside the
    asyncio loop".
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
        page.goto(f"{preview_hub}/builder/", wait_until="domcontentloaded")
        # The shared asset mount, asserted where it matters: the builder is
        # the app that does NOT own these files.
        page.wait_for_function(
            "() => window.phenotypicViv !== undefined", timeout=30_000
        )
        page.wait_for_function(
            "() => window.__phenotypicNodePreview !== undefined",
            timeout=30_000,
        )
        yield page
    finally:
        browser.close()


def _canvas_stats(page, selector: str) -> tuple[int, int]:
    """``(painted, distinct)`` for ``selector``'s screenshot.

    Both numbers are needed. A label layer whose values sit far below its
    dtype-wide contrast domain paints a near-flat rectangle over the whole
    extent, so ``painted`` alone cannot separate "the noise image rendered"
    from "a flat rectangle rendered".
    """
    from PIL import Image as PILImage

    raw = page.locator(selector).screenshot()
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    data = list(img.getdata())
    painted = sum(1 for px in data if px != BACKDROP)
    return painted, len({px for px in data if px != BACKDROP})


def _mount(page, spec: dict) -> None:
    """Create the probe host (once) and drive the real glue module."""
    rows, cols = IMAGE_SHAPE
    page.evaluate(
        """([id, w, h, backdrop]) => {
            if (document.getElementById(id)) return;
            const el = document.createElement('div');
            el.id = id;
            el.style.cssText = `position:fixed;left:0;top:0;width:${w}px;`
                + `height:${h}px;background:${backdrop};z-index:9999;`;
            document.body.appendChild(el);
        }""",
        [CONTAINER, PROBE_W, PROBE_H, f"rgb{BACKDROP}"],
    )
    outcome = page.evaluate(
        """async ([id, spec, cx, cy]) => {
            try {
                await window.__phenotypicNodePreview.mountViewer(id, spec);
                window.phenotypicViv.setViewState(
                    id, {target: [cx, cy, 0], zoom: -2}
                );
                return {ok: true};
            } catch (e) { return {ok: false, err: String(e && e.message)}; }
        }""",
        [CONTAINER, spec, cols / 2, rows / 2],
    )
    assert outcome["ok"], outcome
    page.wait_for_function(
        """([id]) => {
            const c = document.querySelector('#' + id + ' canvas');
            return !!c && c.width > 0 && c.height > 0;
        }""",
        arg=[CONTAINER],
        timeout=30_000,
    )
    page.wait_for_timeout(2_000)  # let the tile layer settle


def test_the_preview_pane_paints_a_node_store(gl_page, seeded_scope) -> None:
    """The render swap, end to end.

    The floor is deliberately well under the drawn extent: this asserts
    "deck.gl painted the store", not a pixel-exact layout. The distinct
    colour count is what separates a decoded read from a pixel source that
    returned ``fill_value`` zeros for every chunk, which would paint an
    equally large flat rectangle.
    """
    _mount(gl_page, _spec_for(seeded_scope, "rgb"))
    painted, distinct = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert painted > 20_000, (
        f"only {painted} painted pixels -- the byte route served nothing, or "
        "the image layer never received a decoded tile"
    )
    assert distinct > 100, (
        f"{painted} pixels painted but only {distinct} distinct colours -- "
        "a flat rectangle, not decoded noise"
    )


def test_the_objmap_channel_hides_the_image_layer(
    gl_page, seeded_scope
) -> None:
    """``imageVisible`` reaches deck.gl, and comes BACK.

    Runs against the instance the previous test mounted -- the module-scoped
    page is shared, and re-mounting would leak a second deck.gl context.

    The return direction is the one worth pinning: ``setSource`` rebuilds
    every layer but the facade's viewer keeps its own hidden-id set across a
    re-source, so an implementation that only ever hides leaves the image
    layer gone for the rest of the session.
    """
    _, distinct_image = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert distinct_image > 100, "precondition: the noise image must be shown"

    _mount(gl_page, _spec_for(seeded_scope, "objmap"))
    _, distinct_label_only = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert distinct_label_only < distinct_image / 10, (
        f"{distinct_label_only} distinct colours with the image hidden (was "
        f"{distinct_image}) -- the image layer is still painting"
    )

    _mount(gl_page, _spec_for(seeded_scope, "rgb"))
    _, distinct_again = _canvas_stats(gl_page, f"#{CONTAINER}")
    assert distinct_again > 100, (
        f"{distinct_again} distinct colours after switching back to rgb -- "
        "the hidden image layer was never re-shown"
    )


def test_the_overlay_channel_loads_both_layers(gl_page, seeded_scope) -> None:
    """``labelPath`` is READ from the store, never constructed.

    ``build_phenotypic_attributes`` records it under
    ``phenotypic.labels.objmap``, and an rgb-less store puts it under
    ``gray`` -- so deriving it as ``f"{seriesPath}/labels/objmap"`` would be
    wrong for exactly the stores that matter.
    """
    spec = _spec_for(seeded_scope, "overlay")
    assert spec["labelPath"], spec
    _mount(gl_page, spec)
    loaded = gl_page.evaluate(
        """async ([id, spec]) => {
            const l = await window.phenotypicViv.setSource(id, spec);
            return {image: l.image.data.length,
                    label: l.label ? l.label.data.length : 0};
        }""",
        [CONTAINER, spec],
    )
    assert loaded["image"] > 0 and loaded["label"] > 0, loaded
