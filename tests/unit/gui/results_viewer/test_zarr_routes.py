"""The OME-Zarr byte route: Range, per-store readable roots, a generation token.

Everything here is observable at the Flask boundary, which is where the
token, the readable-root restriction, and Range negotiation all live. A
browser adds no evidence for any of it.

Three of these tests pin properties that a plausible implementation gets
wrong:

* ``test_honours_a_range_request`` -- without Range negotiation the response
  is a 200 carrying the whole shard. Measured in the phase-0 spike: a cold
  sharded tile costs 1,049,381 B with Range and 72,090,062 B without.
  Mutation-checked against ``conditional=False`` and against a hand-built
  ``Response`` over ``read_bytes()``; **not** against dropping the
  ``conditional=True`` keyword, which is Flask 3.1's default and changes
  nothing.
* ``test_zarr_v2_metadata_probes_are_404_not_400`` -- a zarr client probes
  ``.zattrs`` / ``.zgroup`` / ``.zarray`` beside every ``zarr.json``, and its
  fetch store returns ``undefined`` on 404 but **throws** on any other
  non-2xx. The leading-dot rule would make these 400s and abort every open.
* ``test_a_rewritten_nested_chunk_is_served_fresh`` -- the token keys on the
  root ``zarr.json`` only, so an in-place chunk rewrite must remain visible
  *without* invalidating the URL.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import dash
import numpy as np
import polars as pl
import pytest

from phenotypic import Image
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._zarr_routes import (
    readable_roots_for,
    register_zarr_routes,
    store_generation_token,
    zarr_store_url,
)
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

from tests._output_layout import write_master

DATASET = "d1"
STEM = "img001"

#: A level-0 ``rgb`` chunk key. One path segment per chunk, ``.`` separated,
#: three indices because ``rgb`` is ``(c, y, x)`` -- confirmed against the
#: real writer in the phase-0 spike.
RGB_CHUNK = "rgb/0/c.0.0.0"


def _noise_image(seed: int) -> Image:
    """A 96x96 image whose pixels are noise.

    Noise, not zeros: zarr omits any chunk whose contents equal
    ``fill_value``, so an all-zeros image writes **no** chunk files at all
    and a byte-route test against it would 404 on every read while proving
    nothing.
    """
    rng = np.random.default_rng(seed)
    return Image(arr=rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def run_template(tmp_path_factory) -> Path:
    """One discoverable run with a store-backed image, built once.

    ``save2zarr`` dominates this module's runtime, so it happens once and
    each test copies the tree. Copying is what lets a test re-promote or
    corrupt a store without leaking into the next.
    """
    root = tmp_path_factory.mktemp("zarr-route-run")
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": [DATASET],
                str(IMAGE.IMAGE_NAME): [STEM],
                "Object_Label": [1],
                "Bbox_CenterRR": [16.0],
                "Bbox_CenterCC": [16.0],
            }
        ),
    )
    (root / "results" / DATASET / "measurements").mkdir(parents=True)
    store = zarr_store_path(root, DATASET, STEM)
    store.parent.mkdir(parents=True, exist_ok=True)
    _noise_image(0).save2zarr(store)
    return root


class RouteFixture:
    """A registered byte route over a real, republishable run."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.store = zarr_store_path(root, DATASET, STEM)
        self.output_root = OutputRoot.discover(
            root, cache_root=root.parent / ".test-phenotypic-viewer-cache"
        )
        app = dash.Dash(f"zarr-routes-{id(self)}")
        # Dash 4.x validates the layout in a before_request hook; a trivial
        # layout keeps that from 500-ing before the request reaches the
        # blueprint.
        app.layout = dash.html.Div()
        register_zarr_routes(app, self.output_root)
        self.client = app.server.test_client()

    @property
    def token(self) -> str:
        """The store's current generation token."""
        return store_generation_token(self.store)

    def url(self, tail: str, *, token: str | None = None) -> str:
        """Build a byte-route URL for one store-relative path."""
        return (
            f"/zarr/{DATASET}/{STEM}.ome.zarr/"
            f"{token if token is not None else self.token}/{tail}"
        )

    def get(self, tail: str, **kwargs):
        """GET one store-relative path at the current token."""
        return self.client.get(self.url(tail), **kwargs)

    def repromote(self, seed: int) -> None:
        """Republish the store with different pixels.

        A promote, not an in-place write: nothing in the design opens a
        promoted store for writing, so a promote is the only way a store's
        contents ever change.
        """
        _noise_image(seed).save2zarr(self.store)


@pytest.fixture
def route(run_template: Path, tmp_path: Path) -> RouteFixture:
    """A byte route over a private copy of the template run."""
    root = tmp_path / "run"
    shutil.copytree(run_template, root, symlinks=True)
    return RouteFixture(root)


# ---------------------------------------------------------------------------
# Serving bytes
# ---------------------------------------------------------------------------


def test_serves_the_root_zarr_json(route: RouteFixture) -> None:
    """The client bootstraps from it, and no allow-list entry names it."""
    resp = route.get(STORE_ROOT_JSON)
    assert resp.status_code == 200
    assert json.loads(resp.data)["attributes"][PhenotypicAttr.ROOT]


def test_serves_a_level_zero_chunk(route: RouteFixture) -> None:
    resp = route.get(RGB_CHUNK)
    assert resp.status_code == 200
    assert resp.data == (route.store / RGB_CHUNK).read_bytes()


def test_honours_a_range_request(route: RouteFixture) -> None:
    """The property this phase exists to get right.

    Sharding makes a tile read two ranged GETs -- a suffix read of the shard
    index, then the inner chunk. Without Range negotiation both return the
    whole shard.
    """
    resp = route.get(RGB_CHUNK, headers={"Range": "bytes=0-15"})
    assert resp.status_code == 206
    assert len(resp.data) == 16
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert resp.data == (route.store / RGB_CHUNK).read_bytes()[:16]


def test_honours_a_suffix_range_request(route: RouteFixture) -> None:
    """The shard-index read's shape: the last N bytes of the shard."""
    whole = (route.store / RGB_CHUNK).read_bytes()
    resp = route.get(RGB_CHUNK, headers={"Range": "bytes=-32"})
    assert resp.status_code == 206
    assert resp.data == whole[-32:]


def test_a_ranged_read_is_smaller_than_the_whole_file(
    route: RouteFixture,
) -> None:
    """Asserted on bytes, not on a status code.

    A 206 whose body is the whole file would satisfy a status-only check and
    would be exactly the regression this test exists to catch.
    """
    whole = len((route.store / RGB_CHUNK).read_bytes())
    ranged = route.get(RGB_CHUNK, headers={"Range": "bytes=0-15"})
    assert len(ranged.data) < whole


def test_a_head_request_reports_content_length(route: RouteFixture) -> None:
    """zarrita's suffix-range path issues HEAD first when it cannot use one.

    It reads ``Content-Length`` off the HEAD response and converts the
    suffix request into an explicit byte range. A HEAD without a length
    breaks every sharded read.
    """
    resp = route.client.head(route.url(RGB_CHUNK))
    assert resp.status_code == 200
    assert int(resp.headers["Content-Length"]) == len(
        (route.store / RGB_CHUNK).read_bytes()
    )


def test_the_label_group_metadata_is_readable(route: RouteFixture) -> None:
    """Resolved through ``phenotypic.labels``, never a hard-coded path.

    Only the FIRST resolved component is restricted, so ``labels`` and
    ``objmap`` -- which appear nowhere in the allow-list -- must resolve.
    """
    resp = route.get("rgb/labels/objmap/zarr.json")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# The readable-root restriction
# ---------------------------------------------------------------------------


def test_readable_roots_are_derived_from_the_store(route: RouteFixture) -> None:
    roots = readable_roots_for(route.store)
    assert {"rgb", "gray", "detect_mat", "OME"} <= roots
    assert "tables" not in roots


def test_an_original_series_is_readable(
    run_template: Path, tmp_path: Path
) -> None:
    """A store carrying ``original`` must not 404 on it.

    ``_write_store_part`` appends ``"original"`` to ``series_names`` when the
    image has one, so a hard-coded ``{rgb, gray, detect_mat}`` breaks a
    legitimate store -- and makes the Layers panel offer a series the route
    refuses.
    """
    root = tmp_path / "run"
    shutil.copytree(run_template, root, symlinks=True)
    image = _noise_image(1)
    image._original = np.random.default_rng(2).integers(
        0, 255, (96, 96, 3), dtype=np.uint8
    )
    image.save2zarr(zarr_store_path(root, DATASET, STEM))

    fixture = RouteFixture(root)
    assert "original" in readable_roots_for(fixture.store)
    assert fixture.get("original/0/c.0.0.0").status_code == 200


def test_the_measurements_table_is_never_served(route: RouteFixture) -> None:
    """The authoritative per-object measurements now live INSIDE the store."""
    table = route.store / "tables" / "measurements" / "table.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"secret")
    assert route.get("tables/measurements/table.parquet").status_code == 404


def test_a_symlink_from_a_series_into_the_table_is_not_served(
    route: RouteFixture,
) -> None:
    """The route-level form of the escape a head-only check misses."""
    table = route.store / "tables" / "measurements" / "table.parquet"
    table.parent.mkdir(parents=True)
    table.write_bytes(b"secret")
    (route.store / "rgb" / "sneak").symlink_to(table)
    resp = route.get("rgb/sneak")
    assert resp.status_code == 404
    assert b"secret" not in resp.data


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/./../../zarr.json",
    ],
)
def test_traversal_is_rejected(route: RouteFixture, tail: str) -> None:
    """Werkzeug 3.1.x does not normalise dot segments before routing.

    The handler really does receive these, so the guard is what rejects
    them rather than the router doing it on the guard's behalf.
    """
    assert route.get(tail).status_code in (400, 404)


# ---------------------------------------------------------------------------
# The Zarr v2 probe contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail",
    [
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "rgb/.zattrs",
        "rgb/0/.zarray",
    ],
)
def test_zarr_v2_metadata_probes_are_404_not_400(
    route: RouteFixture, tail: str
) -> None:
    """A v3 store holds none of these, but every zarr client asks for them.

    zarrita's fetch store returns ``undefined`` on 404 and **throws** on any
    other non-2xx status. These names start with a dot, so the shared
    resolver's leading-dot rule would answer 400 and abort the store open
    before a single chunk was read.
    """
    assert route.get(tail).status_code == 404


# ---------------------------------------------------------------------------
# The generation token
# ---------------------------------------------------------------------------


def test_a_stale_token_is_409_not_404(route: RouteFixture) -> None:
    """A re-promote must not be served as a missing chunk.

    404 reads as "this chunk does not exist" and the client retries; 409
    tells it to re-read the source spec, which is the actual remedy. 410 is
    heuristically cacheable under RFC 9110, so a caching proxy could pin
    "gone" onto a chunk URL that is merely superseded.
    """
    url = route.url(RGB_CHUNK)
    assert route.client.get(url).status_code in (200, 206)
    route.repromote(seed=7)
    assert route.client.get(url).status_code == 409


def test_the_token_moves_on_a_repromote(route: RouteFixture) -> None:
    before = route.token
    route.repromote(seed=7)
    assert route.token != before


def test_a_rewritten_nested_chunk_is_served_fresh(route: RouteFixture) -> None:
    """A nested-chunk rewrite must be visible WITHOUT changing the token.

    A store directory's ``st_mtime_ns`` does not move when a nested chunk is
    rewritten, and neither does the root ``zarr.json`` -- so the token is
    unchanged and the URL stays valid. The route holds no cache, so the new
    bytes are served. This passes by construction today; it is a forward
    guard against a cache being added between the file and the response, and
    it is spec section 8's "staleness" check.
    """
    url = route.url(RGB_CHUNK)
    before = route.client.get(url).data

    chunk = route.store / RGB_CHUNK
    dir_mtime_before = route.store.stat().st_mtime_ns
    chunk.write_bytes(before[:-1] + bytes([before[-1] ^ 0xFF]))
    assert route.store.stat().st_mtime_ns == dir_mtime_before, (
        "premise broken: the store directory mtime moved, so this test no "
        "longer proves what it claims"
    )

    assert route.client.get(url).status_code != 409, (
        "the token moved on a nested-chunk rewrite; it must key on the root "
        "zarr.json only"
    )
    assert route.client.get(url).data != before


def test_zarr_store_url_carries_the_token_as_a_path_segment(
    route: RouteFixture,
) -> None:
    """A query parameter would let one client mix two generations.

    As a path segment, a new promote yields a new BASE url, so every
    relative key the client resolves against it belongs to one generation by
    construction.
    """
    token = route.token
    built = zarr_store_url("/", DATASET, STEM, token)
    assert built == f"/zarr/{DATASET}/{STEM}.ome.zarr/{token}"
    assert route.client.get(f"{built}/{RGB_CHUNK}").status_code == 200


def test_zarr_store_url_respects_a_reverse_proxy_prefix() -> None:
    assert (
        zarr_store_url("/node/hz01/30099/", DATASET, STEM, "tok")
        == f"/node/hz01/30099/zarr/{DATASET}/{STEM}.ome.zarr/tok"
    )


# ---------------------------------------------------------------------------
# The error contract
# ---------------------------------------------------------------------------


def test_an_unknown_chunk_is_404(route: RouteFixture) -> None:
    """Sparse storage is normal: zarr omits a chunk equal to ``fill_value``."""
    assert route.get("rgb/0/c.9.9.9").status_code == 404


def test_an_unknown_image_is_404(route: RouteFixture) -> None:
    resp = route.client.get(
        f"/zarr/{DATASET}/nosuch.ome.zarr/{route.token}/{STORE_ROOT_JSON}"
    )
    assert resp.status_code == 404


def test_a_vanished_store_is_404_not_500(route: RouteFixture) -> None:
    """``promote_store`` renames the whole store directory.

    A pan straddling a promote is the routine path, not an exotic race.
    """
    url = route.url(RGB_CHUNK)
    shutil.rmtree(route.store)
    assert route.client.get(url).status_code == 404


def test_a_store_with_no_phenotypic_block_is_404_not_500(
    route: RouteFixture,
) -> None:
    """``require_readable_store`` raises ``KeyError``, which is not an OSError."""
    root_json = route.store / STORE_ROOT_JSON
    payload = json.loads(root_json.read_text(encoding="utf-8"))
    payload["attributes"].pop(PhenotypicAttr.ROOT)
    root_json.write_text(json.dumps(payload), encoding="utf-8")
    assert route.get(RGB_CHUNK).status_code == 404


def test_an_undecodable_store_is_422_not_404(route: RouteFixture) -> None:
    """Matching what ``crop_colony`` already does, so the surfaces agree.

    404 would tell the user "no such image", which is false and hides a
    run-wide, actionable condition. The store's own message -- which names
    both schema versions and the remedy -- is passed through.
    """
    root_json = route.store / STORE_ROOT_JSON
    payload = json.loads(root_json.read_text(encoding="utf-8"))
    payload["attributes"][PhenotypicAttr.ROOT][
        PhenotypicAttr.STORE_SCHEMA_VERSION
    ] = 999
    root_json.write_text(json.dumps(payload), encoding="utf-8")

    resp = route.get(RGB_CHUNK)
    assert resp.status_code == 422
    assert b"store_schema_version" in resp.data
