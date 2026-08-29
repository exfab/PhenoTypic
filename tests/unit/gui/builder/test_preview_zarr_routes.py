"""The builder preview byte route: Range, a generation token, scope shape.

Observable at the Flask boundary, which is where the token, the readable-root
restriction and Range negotiation all live. No browser is needed for any of
it.

**The path-escape guard is NOT re-tested here.** It is
``gui/_shared/tiles.resolve_within_root``, exercised exhaustively in
``tests/unit/gui/shared/test_resolve_within_root.py``; the plan's Global
Constraints put it in one place precisely so two routes cannot test two
copies. What this file pins is that the route *routes through* it.

**On session scoping, read what is actually asserted.** ``_validate_scope`` is
a SHAPE check. Nothing binds a request to a session, and spec section 7
records that as an accepted capability-URL risk (user ruling). So the property
tested is the one that holds -- an unissued session id is not served -- and
NOT "session A cannot present session B's id", which it can, exactly as the
``/preview-tiles/`` route it replaced allowed.
"""

from __future__ import annotations

from pathlib import Path

import dash
import numpy as np
import pytest

from phenotypic import Image
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_zarr_routes import (
    PREVIEW_ZARR_PREFIX,
    preview_zarr_url,
    register_preview_zarr_routes,
)
from phenotypic.gui.results_viewer._zarr_routes import store_generation_token
from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON

SESSION = "previewsess0001"
BLOCK = "b" * 32
STORE_NAME = "base_00.ome.zarr"

#: A level-0 ``gray`` chunk key. TWO indices, not three: ``ngff_`` sets the
#: v3 ``chunk_key_encoding`` separator to ``"."`` and a default v3 key is
#: ``"c" + sep + indices``, so a 2-D array's level-0 chunk is ``c.0.0`` while
#: the 3-D ``rgb`` series' is ``c.0.0.0``. ``gray`` is used because
#: ``save2zarr`` writes it for every image, including a 2-D one.
GRAY_CHUNK = "gray/0/c.0.0"
RGB_CHUNK = "rgb/0/c.0.0.0"


def _preview_image(fill: int) -> Image:
    """A small image whose pixels are a function of *fill*."""
    rng = np.random.default_rng(fill)
    return Image(arr=rng.integers(0, 255, (64, 96, 3), dtype=np.uint8))


class RouteFixture:
    """A registered preview byte route over a real, recomputable scope."""

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.scope_hash = pc.scope_hash([])
        self.scope_dir = pc.scope_dir(SESSION, [])
        self.store = self.scope_dir / STORE_NAME
        self.publish(fill=1)
        self.write_manifest()
        app = dash.Dash(f"preview-zarr-{id(self)}")
        # Dash 4.x validates the layout in a before_request hook; a trivial
        # layout keeps that from 500-ing before the request reaches the
        # blueprint.
        app.layout = dash.html.Div()
        register_preview_zarr_routes(app)
        self.client = app.server.test_client()

    def publish(self, *, fill: int) -> Path:
        """(Re)write the node store -- what a scope recompute does."""
        return _preview_image(fill).save2zarr(self.store)

    def write_manifest(self, **node_overrides) -> None:
        node = {
            "store": STORE_NAME,
            "layers": ["rgb", "gray", "detect_mat", "objmap"],
            "shape": [64, 96],
            "num_objects": 0,
        }
        node.update(node_overrides)
        pc.write_manifest(SESSION, [], {
            "version": pc.MANIFEST_VERSION,
            "fingerprint": "fp",
            "fingerprint_inputs": [],
            "scope_key": "",
            "nodes": {BLOCK: node},
            "error": None,
        })

    @property
    def token(self) -> str:
        return store_generation_token(self.store)

    def url(self, tail: str, *, token: str | None = None) -> str:
        base = preview_zarr_url(
            "/", SESSION, self.scope_hash, BLOCK,
            self.token if token is None else token,
        )
        return f"{base}/{tail}"

    def get(self, tail: str, **kwargs):
        return self.client.get(self.url(tail), **kwargs)


@pytest.fixture
def route(tmp_path: Path, monkeypatch) -> RouteFixture:
    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    return RouteFixture(tmp_path / "cache")


# ---------------------------------------------------------------------------
# Serving bytes
# ---------------------------------------------------------------------------


def test_serves_the_root_zarr_json(route: RouteFixture) -> None:
    """The pixel client bootstraps from it, and no allow-list entry names it."""
    assert route.get(STORE_ROOT_JSON).status_code == 200


@pytest.mark.parametrize("chunk", [GRAY_CHUNK, RGB_CHUNK])
def test_serves_a_level_zero_chunk(route: RouteFixture, chunk: str) -> None:
    resp = route.get(chunk)
    assert resp.status_code == 200
    assert resp.data


def test_honours_a_range_request(route: RouteFixture) -> None:
    """Sharding makes one cold tile two ranged GETs, so this is not a nicety.

    Without Range negotiation the response is a 200 carrying the whole shard.
    """
    resp = route.get(GRAY_CHUNK, headers={"Range": "bytes=0-15"})
    assert resp.status_code == 206
    assert len(resp.data) == 16


def test_honours_a_suffix_range_request(route: RouteFixture) -> None:
    """zarrita reads a shard's index as a SUFFIX range before the inner chunk."""
    resp = route.get(GRAY_CHUNK, headers={"Range": "bytes=-8"})
    assert resp.status_code == 206
    assert len(resp.data) == 8


def test_a_ranged_read_is_smaller_than_the_whole_file(
    route: RouteFixture,
) -> None:
    whole = route.get(GRAY_CHUNK)
    ranged = route.get(GRAY_CHUNK, headers={"Range": "bytes=0-15"})
    assert len(ranged.data) < len(whole.data)


# ---------------------------------------------------------------------------
# The readable-root restriction, through the SHARED resolver
# ---------------------------------------------------------------------------


def test_the_measurements_table_is_never_served(route: RouteFixture) -> None:
    """``tables/`` is not in the store's own series or labels, so it is not readable."""
    table = route.store / "tables" / "measurements" / "table.parquet"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(b"secret")
    resp = route.get("tables/measurements/table.parquet")
    assert resp.status_code == 404
    assert b"secret" not in resp.data


def test_the_route_resolves_through_the_shared_guard(
    route: RouteFixture,
) -> None:
    """One escape case only.

    The guard itself is ``resolve_within_root`` and is tested exhaustively in
    ``tests/unit/gui/shared/test_resolve_within_root.py``. Duplicating that
    block here would give the two routes two suites over one implementation,
    which is what the shared-guard constraint exists to prevent. What is
    route-specific is that the tail reaches the guard at all.
    """
    assert route.get("rgb/../../../../etc/passwd").status_code in (400, 404)


def test_the_label_group_is_readable(route: RouteFixture) -> None:
    """``labels`` resolves through ``phenotypic.labels``, never as a literal.

    Preview stores are written by ``save2zarr`` (``apply_with_intermediates``
    passes ``full_layers=True``), so unlike a delta store they DO carry a
    label group -- under whichever series is primary.
    """
    assert route.get("rgb/labels/objmap/zarr.json").status_code == 200


# ---------------------------------------------------------------------------
# The generation token
# ---------------------------------------------------------------------------


def test_a_stale_token_is_409_not_404(route: RouteFixture) -> None:
    """409, never 410.

    404 reads as "chunk missing" to a zarr client and is retried forever; 410
    is heuristically cacheable under RFC 9110, and a cacheable "gone" for a
    chunk URL behind the documented OOD proxy would be poison.
    """
    stale = route.token
    route.publish(fill=2)
    assert route.token != stale
    assert route.get(STORE_ROOT_JSON).status_code == 200
    resp = route.client.get(route.url(STORE_ROOT_JSON, token=stale))
    assert resp.status_code == 409


def test_the_token_moves_when_a_node_is_recomputed(
    route: RouteFixture,
) -> None:
    """The case this phase exists to serve.

    A preview store is rewritten IN PLACE under the same ``scope_hash`` when a
    node's parameters change, so the token moves on an edit. This is why the
    source spec is rebuilt on every request instead of being cached in a
    ``dcc.Store``: a held token 409s permanently after the first edit.
    """
    before = route.token
    route.publish(fill=3)
    assert route.token != before


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------


def test_an_unissued_session_id_is_not_served(route: RouteFixture) -> None:
    """The property that actually holds: no cached scope, no bytes.

    NOT "session A cannot present session B's id" -- it can, and the route
    serves it, exactly as the ``/preview-tiles/`` route did. Asserting the
    stronger property would be asserting a binding that does not exist, and
    spec section 7 records its absence as accepted.
    """
    url = (
        f"{PREVIEW_ZARR_PREFIX}/{'0' * 32}/{'a' * 40}/blk/tok/{STORE_ROOT_JSON}"
    )
    assert route.client.get(url).status_code == 404


@pytest.mark.parametrize(
    "session_id, scope_hash, block_id",
    [
        ("..", "a" * 40, BLOCK),          # unsafe session component
        (SESSION, "not-a-sha1", BLOCK),   # not a 40-hex digest
        (SESSION, "a" * 40, ".."),        # unsafe block component
        (SESSION, "A" * 40, BLOCK),       # uppercase hex is not the digest form
    ],
)
def test_a_malformed_triple_is_404(
    route: RouteFixture, session_id: str, scope_hash: str, block_id: str
) -> None:
    url = (
        f"{PREVIEW_ZARR_PREFIX}/{session_id}/{scope_hash}/{block_id}"
        f"/tok/{STORE_ROOT_JSON}"
    )
    assert route.client.get(url, follow_redirects=True).status_code == 404


def test_an_unknown_block_is_404(route: RouteFixture) -> None:
    """``nodes`` is a DICT keyed by block_id -- there is no ``blocks`` list."""
    url = (
        f"{PREVIEW_ZARR_PREFIX}/{SESSION}/{route.scope_hash}/{'z' * 32}"
        f"/{route.token}/{STORE_ROOT_JSON}"
    )
    assert route.client.get(url).status_code == 404


def test_an_unknown_chunk_is_404(route: RouteFixture) -> None:
    """Sparse stores are legal, and a zarr client reads 404 as "absent"."""
    assert route.get("gray/0/c.9.9").status_code == 404


def test_a_rootless_store_is_404_not_500(route: RouteFixture) -> None:
    """An interrupted write leaves no root, and reads as ABSENT, not partial."""
    partial = route.scope_dir / "partial.ome.zarr"
    (partial / "gray" / "0").mkdir(parents=True)
    route.write_manifest(store="partial.ome.zarr")
    assert route.get(STORE_ROOT_JSON).status_code == 404


def test_a_vanished_store_is_404_not_500(route: RouteFixture) -> None:
    """A scope recompute renames the whole directory out from under a read."""
    token = route.token
    (route.store / STORE_ROOT_JSON).unlink()
    resp = route.client.get(route.url(STORE_ROOT_JSON, token=token))
    assert resp.status_code == 404


def test_zarr_v2_metadata_probes_are_404_not_400(route: RouteFixture) -> None:
    """A zarr client probes all four beside every ``zarr.json``.

    Its fetch store returns ``undefined`` on 404 and THROWS on any other
    non-2xx, and the leading-dot rule would otherwise make these 400s and
    abort the open.
    """
    for name in (".zarray", ".zattrs", ".zgroup", ".zmetadata"):
        assert route.get(f"gray/0/{name}").status_code == 404


def test_an_undecodable_store_is_422_not_404(route: RouteFixture) -> None:
    """A store written by a newer build is a run-wide, actionable condition.

    404 would say "no such node", which is false, and would hide it.
    """
    import json

    root = route.store / STORE_ROOT_JSON
    doc = json.loads(root.read_text())
    block = doc["attributes"]["phenotypic"]
    block["store_schema_version"] = 999
    root.write_text(json.dumps(doc))
    assert route.get(STORE_ROOT_JSON).status_code == 422


# ---------------------------------------------------------------------------
# URL construction + mounting
# ---------------------------------------------------------------------------


def test_preview_zarr_url_carries_the_token_as_a_path_segment() -> None:
    """A path segment, not a query parameter.

    Every key the client resolves relative to the base URL therefore belongs
    to one publish by construction, so a torn read across a recompute cannot
    be assembled.
    """
    url = preview_zarr_url("/", "sess", "a" * 40, "blk", "tok123")
    assert url == f"{PREVIEW_ZARR_PREFIX}/sess/{'a' * 40}/blk/tok123"


def test_preview_zarr_url_respects_a_hub_mount_prefix() -> None:
    url = preview_zarr_url("/builder/", "sess", "a" * 40, "blk", "tok")
    assert url.startswith(f"/builder{PREVIEW_ZARR_PREFIX}/")


def test_create_app_mounts_the_preview_byte_route(
    tmp_path: Path, monkeypatch
) -> None:
    """The blueprint is reachable on a real builder app, not just in isolation.

    ``create_app`` runs ``init_preview_cache()``, which WIPES the cache root,
    so the scope is seeded afterwards.
    """
    from phenotypic.gui.builder._app import create_app

    monkeypatch.setattr(pc, "preview_cache_root", lambda: tmp_path / "cache")
    app = create_app(image_root=tmp_path)
    scope_dir = pc.scope_dir(SESSION, [])
    store = _preview_image(1).save2zarr(scope_dir / STORE_NAME)
    pc.write_manifest(SESSION, [], {
        "version": pc.MANIFEST_VERSION, "fingerprint": "fp",
        "fingerprint_inputs": [], "scope_key": "", "error": None,
        "nodes": {BLOCK: {"store": STORE_NAME, "layers": ["gray"],
                          "shape": [64, 96], "num_objects": 0}},
    })
    url = preview_zarr_url(
        "/", SESSION, pc.scope_hash([]), BLOCK, store_generation_token(store),
    )
    resp = app.server.test_client().get(f"{url}/{STORE_ROOT_JSON}")
    assert resp.status_code == 200
