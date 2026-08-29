"""The vendored Viv facade, against a store the real writer produced.

Spec section 5.1 makes the zstd registration a hard ORDERING rule: register
late and every read fails, so the assertion has to be on decoded pixel
values, not on the registry's contents. These four tests each pin one
property that a plausible implementation gets wrong:

* ``test_the_wasm_zstd_codec_decodes_a_cli_written_chunk`` -- the browser
  agrees with Python, byte for byte, on a chunk ``save2zarr`` wrote.
* ``test_a_read_without_the_codec_fails_rather_than_returning_zeros`` -- the
  negative control. Deleting the codec must break the READ. Zarr's data
  model fills an unreadable chunk with ``fill_value``, so the failure mode
  this guards against is a broken bundle rendering as an empty plate.
* ``test_the_facade_survives_loading_before_the_bundle`` -- Dash loads
  ``viv_viewer.js`` BEFORE ``viv/viv-bundle.min.js`` (root-level assets sort
  ahead of subdirectory ones). The facade must resolve ``window.__vivBundle``
  lazily; the test re-evaluates the facade with the global deleted, which is
  exactly the order the browser sees.
* ``test_a_stale_generation_token_raises_its_own_error`` -- a 409 must not be
  swallowed into ``undefined``. An absent chunk is ``fill_value``, so a
  swallowed 409 renders black tiles after every promote.

None of these paints anything, so none needs a GL stack: they run under
Playwright's default headless Chromium. The rendering test that does need
one lives in ``test_viv_facade_renders.py``.

Gated by ``PLAYWRIGHT=1`` via the module-level skip in ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import pytest
from playwright.sync_api import Page

from phenotypic import Image
from phenotypic.gui.results_viewer._zarr_routes import (
    store_generation_token,
    zarr_store_url,
)
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path
from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)

_OUTPUT_NAME = "CliOutputExample"
DATASET = "d1"
STEM = "img001"

#: Level-0 ``rgb``. Three path components because ``rgb`` is ``(c, y, x)``.
RGB_LEVEL_0 = "rgb/0"

#: Corner edge length compared against Python. Small on purpose: the read
#: still decodes the WHOLE chunk, so four columns prove the codec ran.
CORNER = 4


#: Image extent of the fixture store, (rows, cols).
IMAGE_SHAPE = (900, 1200)

#: What the writer's ladder produces at 1200 px against ``stop_px`` 512.
#: Asserted rather than assumed -- three levels is what puts the store on
#: the MULTISCALE code path, and a 96x96 store (one level) silently would
#: not.
EXPECTED_LEVELS = 3


def _noise_image(seed: int) -> Image:
    """A 1200x900 RGB image whose pixels are noise.

    Noise, not a flat fill: zarr omits any chunk equal to ``fill_value``, so
    a constant image writes no chunk files at all and every read here would
    404 while proving nothing about the codec.

    Sized above the pyramid's ``stop_px`` (512) on purpose. A store below it
    resolves to a single level, which loads through ``ImageLayer`` and never
    touches ``MultiscaleImageLayer``, tiling, or the sharded read path.
    """
    rng = np.random.default_rng(seed)
    rows, cols = IMAGE_SHAPE
    return Image(arr=rng.integers(0, 255, (rows, cols, 3), dtype=np.uint8))


def _build_viv_sandbox(parent_dir: Path) -> Path:
    """Standard E2E sandbox plus one real, store-backed image."""
    sandbox = _build_sandbox(parent_dir)
    output_dir = sandbox / "results" / _OUTPUT_NAME
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": [DATASET],
            str(IMAGE.IMAGE_NAME): [STEM],
            "Object_Label": [1],
            "Bbox_CenterRR": [16.0],
            "Bbox_CenterCC": [16.0],
            "Size_Area": [100.0],
        }
    )
    write_master(output_dir, frame)
    write_measurements_mirror(output_dir, frame)
    (output_dir / "results" / DATASET / "measurements").mkdir(parents=True)
    store = zarr_store_path(output_dir, DATASET, STEM)
    store.parent.mkdir(parents=True, exist_ok=True)
    _noise_image(0).save2zarr(store)
    publish_coherent_terminal_evidence(output_dir, total_images=1)
    return sandbox


@pytest.fixture(scope="module")
def viv_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One sandbox with a real store, built once. ``save2zarr`` dominates."""
    return _build_viv_sandbox(tmp_path_factory.mktemp("viv-facade"))


@pytest.fixture(scope="module")
def viv_hub(viv_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(viv_sandbox)


@pytest.fixture(scope="module")
def store_dir(viv_sandbox: Path) -> Path:
    return zarr_store_path(viv_sandbox / "results" / _OUTPUT_NAME, DATASET, STEM)


@pytest.fixture(scope="module")
def store_url(store_dir: Path) -> str:
    """The browser-visible base URL of the store's current generation."""
    return zarr_store_url(
        "/results/", DATASET, STEM, store_generation_token(store_dir)
    )


@pytest.fixture(scope="module")
def expected_corner(store_dir: Path) -> list[int]:
    """``rgb[0, :CORNER, :CORNER]``, flat, read by Python's zarr."""
    import zarr

    array = zarr.open_array(str(store_dir / RGB_LEVEL_0), mode="r")
    return [
        int(v) for v in np.asarray(array[0, :CORNER, :CORNER]).ravel()
    ]


def test_the_fixture_store_really_is_multi_level(store_dir: Path) -> None:
    """The premise every other test in this module rests on.

    ``phenotypic.pyramid.levels`` is the RECORDED ladder -- never
    recomputed client-side (plan Global Constraints; the ``floor``/``ceil``
    boundary has been got wrong once already). If the writer's ladder moved
    and this fixture dropped to one level, the browser tests would quietly
    stop exercising the multiscale and sharded read paths while still
    passing.
    """
    import json

    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON

    block = json.loads((store_dir / STORE_ROOT_JSON).read_text())[
        "attributes"
    ]["phenotypic"]
    assert block["pyramid"]["levels"] == EXPECTED_LEVELS, block["pyramid"]
    assert block["labels"]["objmap"] == "rgb/labels/objmap", block["labels"]


def _open_viewer(page: Page, viv_hub: str) -> None:
    """Bind the run and wait for the facade global to be installed."""
    bind_results_output(page, viv_hub, f"results/{_OUTPUT_NAME}")
    page.wait_for_function(
        "() => window.phenotypicViv !== undefined", timeout=30_000
    )


def test_the_wasm_zstd_codec_decodes_a_cli_written_chunk(
    page: Page, viv_hub: str, store_url: str, expected_corner: list[int]
) -> None:
    """The browser decodes a chunk ``save2zarr`` wrote, pixel for pixel.

    This is the assertion spec section 5.1 asks for. A test that only
    checked ``registry.has("zstd")`` would pass against a bundle whose
    codec is registered but broken, and against one registered too late
    for the loaders' own registry copy.
    """
    _open_viewer(page, viv_hub)
    decoded = page.evaluate(
        """async ([url, path, edge]) => {
            return await window.phenotypicViv.__debugReadChunk(url, path, edge);
        }""",
        [store_url, RGB_LEVEL_0, CORNER],
    )
    assert decoded == expected_corner


def test_a_read_without_the_codec_fails_rather_than_returning_zeros(
    page: Page, viv_hub: str, store_url: str
) -> None:
    """Deleting the codec must break the READ, not merely the registry.

    An earlier draft of this step asserted only that the registry entry
    was gone -- it never attempted a read, so it passed either way and
    proved nothing about ordering. ``registry.delete`` exists and the read
    throws ``Unknown codec: zstd`` (measured in the phase-0 spike), so the
    ``'no-delete'`` branch is a dead path kept as an honest guard. Do not
    weaken the assertion to accept it.
    """
    _open_viewer(page, viv_hub)
    outcome = page.evaluate(
        """async ([url, path]) => {
            // Warm the read first, so a failure below cannot be blamed on
            // the route, the token or the store layout.
            await window.phenotypicViv.__debugReadChunk(url, path, 2);
            try { window.__vivBundle.zarr.registry.delete('zstd'); }
            catch (e) { return 'no-delete'; }
            try {
                await window.phenotypicViv.__debugReadChunk(url, path, 2);
                return 'read-succeeded';
            } catch (e) { return 'threw: ' + String(e && e.message); }
        }""",
        [store_url, RGB_LEVEL_0],
    )
    assert outcome.startswith("threw"), (
        f"expected the read to fail without the zstd codec, got {outcome!r}; "
        "'read-succeeded' means a decode path bypasses the registry"
    )
    assert "zstd" in outcome


def test_the_facade_survives_loading_before_the_bundle(
    page: Page, viv_hub: str
) -> None:
    """The bundle global is resolved at await time, not at load time.

    Dash appends every root-level asset before any subdirectory asset, so
    ``viv_viewer.js`` executes BEFORE ``viv/viv-bundle.min.js``. This test
    reproduces that order exactly: it re-evaluates the facade source with
    ``window.__vivBundle`` deleted, restores the global afterwards, and then
    calls a method. A facade that snapshots the global at module scope
    captures ``undefined`` and fails here; the load order in production is a
    property of Dash's asset walk, not of either file, so nothing else
    catches a regression to the eager form.
    """
    _open_viewer(page, viv_hub)
    outcome = page.evaluate(
        """async () => {
            const src = await (await fetch(
                '/results/assets/viv_viewer.js'
            )).text();
            const saved = window.__vivBundle;
            delete window.__vivBundle;
            try {
                // The eager form throws right here, reading VERSION off
                // `undefined`.
                (0, eval)(src);
            } catch (e) {
                window.__vivBundle = saved;
                return 'eval-threw: ' + String(e && e.message);
            }
            window.__vivBundle = saved;   // the bundle asset now executes
            try {
                return 'version: ' + (await window.phenotypicViv.version());
            } catch (e) {
                return 'call-threw: ' + String(e && e.message);
            }
        }"""
    )
    assert outcome.startswith("version: viv "), outcome


def test_a_stale_generation_token_raises_its_own_error(
    page: Page, viv_hub: str, store_url: str
) -> None:
    """A 409 is its own error, never a silently absent chunk.

    ``FetchStore`` maps 404 onto ``undefined`` and every other non-2xx onto
    one opaque ``Unexpected response status`` -- which is why the facade
    carries its own store. If a stale token reached zarr as ``undefined``,
    the chunk would be filled with ``fill_value`` and a re-promoted plate
    would render BLACK rather than raising.
    """
    _open_viewer(page, viv_hub)
    stale = store_url.rsplit("/", 1)[0] + "/deadbeefdeadbeef-0"
    outcome = page.evaluate(
        """async ([url, path]) => {
            try {
                await window.phenotypicViv.__debugReadChunk(url, path, 2);
                return 'read-succeeded';
            } catch (e) {
                return (e && e.name) + ': ' + String(e && e.message);
            }
        }""",
        [stale, RGB_LEVEL_0],
    )
    assert outcome.startswith("StaleGenerationError"), outcome
