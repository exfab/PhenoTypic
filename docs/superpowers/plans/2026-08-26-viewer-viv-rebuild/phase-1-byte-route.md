# Phase 1 — The byte route and the shared resolver

**Spec:** §4, §4.0, §4.1. **Depends on:** phase 0. **Blocks:** phases 2-4, 6.

**Deliverable:** one `resolve_within_root` in `gui/_shared/tiles.py`, and
`GET /zarr/<dataset>/<stem>.ome.zarr/<token>/<path...>` on the results viewer's blueprint —
raw store bytes with **HTTP Range**, a per-store readable-root restriction enforced on the
**resolved** path, and a generation token that makes a torn read across a promote
structurally impossible.

> **Why Range is load-bearing, not a nicety.** A sharded read is a shard-index fetch
> followed by a byte-range fetch into the shard. Without `conditional=True` the client
> pulls whole shards — up to 96 MB — for every 1024² tile. This is the single flag the
> phase exists to get right.

> **Write the resolver once.** Phase 6 needs the same logic for the builder preview. A
> path-escape guard is a security primitive and spec §9.1 makes correctness binding; two
> copies drift **silently**, because each phase would test only its own copy. The two
> *routes* stay separate — they have genuinely different resolution and guard regimes
> (`OutputRoot.store_path` here, a session shape-check there) — but those regimes live
> outside the resolver.

---

### Task 1.1: The shared resolver

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py`
- Test: `tests/unit/gui/shared/test_resolve_within_root.py` (create)

**Interfaces:**
- Consumes: `is_safe_path_component` (`tiles.py:755`).
- Produces: `resolve_within_root(root: Path, tail: str, *, allowed_roots: frozenset[str]) -> Path`,
  raising `werkzeug.exceptions.BadRequest` / `NotFound`. Phase 6 task 6.1 calls it too.

- [ ] **Step 1: The guard is already verified — do not re-derive it**

`is_safe_path_component` was run against this route's inputs during plan refinement:

```text
'..' -> False   '.' -> False    '...' -> False   '.hidden' -> False
'a/b' -> False  'a\b' -> False  '%2e%2e' -> False  '' -> False
'rgb' -> True   'gray' -> True  'detect_mat' -> True  'OME' -> True
'zarr.json' -> True  'c.0.0.0' -> True  '0.0' -> True
'labels' -> True     'objmap' -> True   'tables' -> True
```

It rejects empty, any leading dot, `/`, `\`, literal `..`, then requires
`^[A-Za-z0-9._-]+$` (`_NAME_RE`, `:752`). Werkzeug 3.1.x does **not** normalise dot-segments
before routing, so the handler really does receive `../../../../etc/passwd` and the
traversal tests exercise the guard rather than passing on Werkzeug's behalf.

**Note `'tables' -> True`.** The guard does not keep the in-store measurements parquet off
the wire. The `allowed_roots` restriction does — and only if enforced correctly, which is
step 3.

- [ ] **Step 2: Write the failing tests**

```python
"""One resolver for every client-controlled path tail inside a store.

The third test is the one that matters and the one an earlier draft failed:
the restriction must bind the RESOLVED path, not the URL segments. Checking
the unresolved head lets a symlink inside a readable root escape it while
still passing containment.
"""

import pytest
from werkzeug.exceptions import BadRequest, NotFound

from phenotypic.gui._shared.tiles import resolve_within_root

ROOTS = frozenset({"OME", "rgb", "gray", "detect_mat"})


def test_resolves_a_file_inside_an_allowed_root(tmp_store):
    got = resolve_within_root(tmp_store, "rgb/0/c.0.0.0", allowed_roots=ROOTS)
    assert got == (tmp_store / "rgb" / "0" / "c.0.0.0").resolve()


def test_rejects_a_disallowed_root(tmp_store):
    with pytest.raises(NotFound):
        resolve_within_root(
            tmp_store, "tables/measurements/table.parquet", allowed_roots=ROOTS
        )


def test_a_symlink_into_a_disallowed_root_is_rejected(tmp_store):
    """The escape an unresolved head check misses.

    `rgb/sneak` passes a head check on segments[0] and resolves to a path
    still INSIDE the store, so containment passes too. Only testing the
    resolved path's first component catches it.
    """
    target = tmp_store / "tables" / "measurements" / "table.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"secret")
    (tmp_store / "rgb" / "sneak").symlink_to(target)

    with pytest.raises((NotFound, BadRequest)):
        resolve_within_root(tmp_store, "rgb/sneak", allowed_roots=ROOTS)


def test_label_group_under_a_series_resolves(tmp_store):
    """Only the FIRST resolved component is restricted, by design.

    Restricting every component would block `labels`, `objmap` and every
    level index, killing the label layer.
    """
    p = tmp_store / "rgb" / "labels" / "objmap" / "0"
    p.mkdir(parents=True, exist_ok=True)
    (p / "c.0.0").write_bytes(b"x")
    assert resolve_within_root(
        tmp_store, "rgb/labels/objmap/0/c.0.0", allowed_roots=ROOTS
    )


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "rgb/./../../zarr.json",
        "",
    ],
)
def test_rejects_traversal_in_any_segment(tmp_store, tail):
    with pytest.raises((BadRequest, NotFound)):
        resolve_within_root(tmp_store, tail, allowed_roots=ROOTS)


def test_an_empty_allow_list_rejects_everything(tmp_store):
    """Fail-closed, and the reason `allowed_roots` has no permissive value."""
    with pytest.raises(NotFound):
        resolve_within_root(tmp_store, "rgb/0/c.0.0.0", allowed_roots=frozenset())


def test_a_vanished_root_is_404_not_500(tmp_store):
    """A promote mid-request renames the whole store directory.

    That is the routine path -- it is the event the generation token exists
    to handle -- so it must not surface as an unhandled exception.
    """
    missing = tmp_store.parent / "gone.ome.zarr"
    with pytest.raises(NotFound):
        resolve_within_root(missing, "rgb/0/c.0.0.0", allowed_roots=ROOTS)
```

- [ ] **Step 3: Implement — restriction AFTER resolution**

```python
def resolve_within_root(
    root: Path,
    tail: str,
    *,
    allowed_roots: frozenset[str],
) -> Path:
    """Resolve a client-controlled ``tail`` to a file inside ``root``.

    The single path-escape guard for every route that serves bytes out of a
    store directory. Two properties are load-bearing and easy to get wrong:

    * Segments are validated INDIVIDUALLY. The traversal surface here is
      wider than a two-component route's because the tail is arbitrary depth.
    * ``allowed_roots`` is tested on the RESOLVED path, not on the URL
      segments. Testing the unresolved head lets a symlink inside a readable
      root (``<root>/rgb/x -> ../tables/measurements/table.parquet``) satisfy
      both the head check and containment, and the file is served.

    Only the FIRST resolved component is restricted. Restricting every
    component would reject ``labels``, ``objmap`` and every level index,
    which would kill the label layer.

    Args:
        root: Directory the result must live inside.
        tail: Client-controlled path, ``/``-separated.
        allowed_roots: First-component allow-list. **Required, and there is
            no permissive value.** A security primitive whose default is "no
            restriction" is one forgotten keyword from serving
            ``tables/measurements/table.parquet``, and the omission would read
            as ordinary code at review. An empty ``frozenset()`` rejects
            everything, which is the correct fail-closed shape.

    Returns:
        The resolved file path.

    Raises:
        BadRequest: A segment is unsafe, or the resolved path escapes ``root``.
        NotFound: The path does not exist, is not a file, or its first
            resolved component is not in ``allowed_roots``.
    """
    segments = [s for s in tail.split("/") if s]
    if not segments:
        raise NotFound()
    for segment in segments:
        if not is_safe_path_component(segment):
            raise BadRequest()

    # BOTH resolves inside the try. `root` itself can vanish mid-request:
    # `promote_store` republishes by renaming the whole store directory
    # (`sdk_/ngff_.py:1235-1300`), so this is the routine path, not an exotic
    # race -- it is the very event the generation token exists to handle.
    # Left outside, a promote during a pan raises FileNotFoundError and the
    # client gets a 500 where 404 is meant.
    try:
        root_resolved = root.resolve(strict=True)
        resolved = root.joinpath(*segments).resolve(strict=True)
    except (OSError, RuntimeError):
        raise NotFound() from None
    if not resolved.is_relative_to(root_resolved):
        raise BadRequest()
    if not resolved.is_file():
        raise NotFound()

    rel = resolved.relative_to(root_resolved)
    head = rel.parts[0]
    if head not in allowed_roots and not (
        len(rel.parts) == 1 and head == ngff_.STORE_ROOT_JSON
    ):
        raise NotFound()
    return resolved
```

- [ ] **Step 4: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/shared/test_resolve_within_root.py -v
uv run ruff check --fix src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_resolve_within_root.py
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_resolve_within_root.py
git commit -m "feat(gui): add one path-escape resolver for store byte routes"
```

---

### Task 1.2: Per-store readable roots and the generation token

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_zarr_routes.py`
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `resolve_within_root`, `OutputRoot.store_path` (`_output_root.py:495`, returns
  `Path | None`), `paths_fingerprint`, `ngff_.STORE_ROOT_JSON`.
- Produces: `register_zarr_routes(app, output_root)`,
  `zarr_store_url(url_prefix, dataset, stem, token)`,
  `readable_roots_for(store) -> frozenset[str]`, `store_generation_token(store) -> str`.
  Phase 6 calls the last two.

- [ ] **Step 1: Derive the readable set from the store, never a literal**

A fixed `{rgb, gray, detect_mat}` is wrong: `_write_store_part` appends `"original"` to
`series_names` whenever the image carries one (`_image_io_handler.py:1012-1014`), and that
list lands in `attributes.phenotypic.series`. A literal set makes the Layers panel list a
series the route 404s — the same hard-coding spec §1's label-path rule forbids, one layer
down.

```python
def readable_roots_for(store: Path) -> frozenset[str]:
    """First-path-components a pixel client may read from this store.

    Derived from the store's own ``attributes.phenotypic`` block, so a
    series the writer legitimately added (``original``) is readable without
    editing this function -- and ``tables/``, which holds the per-object
    measurement parquet, never is.
    """
    block = _readable_block(store)          # raises StoreUnreadable
    roots = set(block.get(PhenotypicAttr.SERIES, {}).keys())
    for label_path in block.get(PhenotypicAttr.LABELS, {}).values():
        roots.add(PurePosixPath(label_path).parts[0])
    roots.add("OME")
    return frozenset(roots)
```

Reuse the landed `_readable_block` (`_shared/tiles.py:304`) rather than a raw `json.loads`
— it raises `StoreUnreadable` on a schema-version mismatch, which is what keeps Plate and
Colony agreeing about a store this build cannot decode.

- [ ] **Step 2: The generation token**

`promote_store` (`sdk_/ngff_.py:1235-1300`) republishes by renaming the whole store
directory. The route resolves fresh per request and holds no handle, so without a token a
client can read `zarr.json` from promote *N* and chunks from *N+1*. Reuse the landed token
rather than inventing one:

```python
def store_generation_token(store: Path) -> str:
    """Short opaque token identifying one promote of ``store``.

    Same construction as ``_tile_routes._store_content_token``: the root
    ``zarr.json``'s content fingerprint AND its mtime. Neither alone is
    enough -- a rewrite can reproduce byte-identical metadata while the
    pixels underneath differ.
    """
    root_json = store / ngff_.STORE_ROOT_JSON
    digest = paths_fingerprint([root_json]).removeprefix("sha256:")[:16]
    return f"{digest}-{os.stat(root_json).st_mtime_ns}"
```

**Memoize it.** Both this and `readable_roots_for` run per request, and this one digests the
root `zarr.json` — at Viv tile rates that is thousands of re-reads and re-digests per pan.
Cache on `(path, st_mtime_ns)`; the mtime is already in the token, so the key costs one
`stat`. Spec §9.1 makes interactivity a target rather than a gate, so this is not a
correctness item — but it is nearly free.

The token is a **path segment**, so a new promote yields a new base URL and mixing is
structurally impossible rather than merely unlikely. A stale token returns **409**, which
tells the client to re-read the source spec — a 404 would read as "chunk missing" and be
retried forever.

- [ ] **Step 3: Write the failing tests**

```python
def test_serves_the_root_zarr_json(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(f"/zarr/ds/plate.ome.zarr/{token}/zarr.json")
    assert resp.status_code == 200


def test_honours_a_range_request(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(
        f"/zarr/ds/plate.ome.zarr/{token}/rgb/0/c.0.0.0",
        headers={"Range": "bytes=0-15"},
    )
    assert resp.status_code == 206
    assert len(resp.data) == 16
    assert resp.headers["Accept-Ranges"] == "bytes"


def test_a_stale_token_is_409_not_404(zarr_route_client, spike_store, token):
    """A re-promote must not be served as a missing chunk.

    404 reads as 'this chunk does not exist' and the client retries; 409
    tells it to re-read the source spec, which is the actual remedy.
    """
    url = f"/zarr/ds/plate.ome.zarr/{token}/rgb/0/c.0.0.0"
    assert zarr_route_client.get(url).status_code in (200, 206)
    _repromote(spike_store)
    assert zarr_route_client.get(url).status_code == 409


def test_an_original_series_is_readable(zarr_route_client, store_with_original):
    """A store carrying `original` must not 404 on it.

    `_write_store_part` appends "original" to series_names when the image
    has one, so a hard-coded readable set breaks a legitimate store.
    """
    tok = store_generation_token(store_with_original)
    resp = zarr_route_client.get(
        f"/zarr/ds/orig.ome.zarr/{tok}/original/0/c.0.0.0"
    )
    assert resp.status_code in (200, 206)


def test_the_measurements_table_is_never_served(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(
        f"/zarr/ds/plate.ome.zarr/{token}/tables/measurements/table.parquet"
    )
    assert resp.status_code == 404
```

- [ ] **Step 4: Implement the route on the shared resolver**

```python
def register_zarr_routes(app: dash.Dash, output_root) -> None:
    bp = Blueprint("zarr_bytes", __name__, url_prefix="/zarr")

    @bp.route("/<dataset>/<stem>.ome.zarr/<token>/<path:tail>")
    def store_bytes(dataset: str, stem: str, token: str, tail: str):
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            abort(400)
        store = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)
        # Both calls read the root `zarr.json`, which a concurrent promote can
        # rename away between the `is_dir()` above and here. `_readable_block`
        # additionally raises `StoreUnreadable` on a schema mismatch. Unguarded,
        # the routine promote path yields a 500 where 404 is meant -- and with
        # `--debug` plus the documented `--host 0.0.0.0`, an unhandled
        # exception is the Werkzeug interactive debugger.
        try:
            expected = store_generation_token(store)
            roots = readable_roots_for(store)
        except (OSError, KeyError):
            # Root gone (promote in flight) or carrying no `phenotypic`
            # block. `require_readable_store` raises FileNotFoundError,
            # KeyError AND ValueError (`ngff_.py:646-649`) -- KeyError is
            # NOT an OSError, so it must be named.
            abort(404)
        except StoreUnreadable:
            # 422, NOT 404 -- matching what Colony already does. `crop_colony`
            # deliberately does not catch this (`tiles.py:685-688`) because a
            # store this build cannot decode is a run-wide, actionable
            # condition; 404 would tell the user "no such image", which is
            # false and hides it. The two surfaces must agree.
            abort(422, description=str(sys.exc_info()[1]))
        if token != expected:
            abort(409)
        return send_file(
            resolve_within_root(store, tail, allowed_roots=roots),
            conditional=True,
        )

    app.server.register_blueprint(bp)
```

- [ ] **Step 4b: Restore spec §8's nested-chunk staleness check**

> **This test was deleted in round 1 and nothing replaced it.** Round-0 task 1.2 was
> "Prove the route respects the landed staleness key"; the rewrite replaced the whole task
> and dropped its test, while four documents still promised it (phase 5's §8 checklist,
> DRIFT.md twice, and the plan README's definition of done). **The generation token does not
> cover it** — the token keys on the root `zarr.json`, which an in-place nested-chunk rewrite
> does not touch, so the URL stays valid. That is *correct*, and it is exactly the property
> worth pinning.

```python
def test_a_rewritten_nested_chunk_is_served_fresh(zarr_route_client, spike_store, token):
    """A nested-chunk rewrite must be visible WITHOUT changing the token.

    A store directory's ``st_mtime_ns`` does not move when a nested chunk is
    rewritten, and neither does the root ``zarr.json`` -- so the token is
    unchanged and the URL stays valid. The route holds no cache, so the new
    bytes are served. This passes by construction today; it is a forward
    guard against a cache being added between the file and the response, and
    it is spec section 8's "staleness" check.
    """
    url = f"/zarr/ds/plate.ome.zarr/{token}/rgb/0/c.0.0.0"
    before = zarr_route_client.get(url).data

    chunk = spike_store / "rgb" / "0" / "c.0.0.0"
    dir_mtime_before = spike_store.stat().st_mtime_ns
    chunk.write_bytes(before[:-1] + bytes([before[-1] ^ 0xFF]))
    assert spike_store.stat().st_mtime_ns == dir_mtime_before, (
        "premise broken: the store directory mtime moved, so this test no "
        "longer proves what it claims"
    )

    assert zarr_route_client.get(url).status_code != 409, (
        "the token moved on a nested-chunk rewrite; it must key on the root "
        "zarr.json only"
    )
    assert zarr_route_client.get(url).data != before
```

The mid-test assertion on `dir_mtime_before` is deliberate: it fails loudly if the
platform's directory-mtime behaviour differs, rather than letting the test pass while
proving nothing.

- [ ] **Step 5: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer/test_zarr_routes.py -v
uv run ruff check --fix src/phenotypic/gui/results_viewer/_zarr_routes.py \
                        src/phenotypic/gui/results_viewer/_app.py \
                        tests/unit/gui/results_viewer/test_zarr_routes.py
git add src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer
git commit -m "feat(gui): serve OME-Zarr bytes with Range, a per-store root set and a generation token"
```

---

### Note — what the route still exposes

> Not a task: it produces no artifact and takes no commit. Fold this paragraph into task
> 1.2's commit body. Spec §4.0 already records the narrowing; this states the *residual*
> exposure as a fact rather than leaving it implied by "pixels only".

The root `zarr.json` is **mandatory** — the client bootstraps from it — and carries
`attributes.phenotypic.metadata`: the `protected`, `public` and `imported` sections plus
`work_id` (`sdk_/ngff_.py:559-583`). `OME/METADATA.ome.xml` carries the same `Metadata_*`
sections. The narrowing keeps `tables/measurements/table.parquet` off the wire; it does
**not** make the route metadata-free.

Combined with the no-authentication assumption in spec §9, that means: on the documented
Open OnDemand recipe (`--host 0.0.0.0`, `gui_hub.md:116, :124`), anything that can reach
the node's port can read a run's image metadata. This is **not new** — the existing DZI and
crop routes already serve pixels the same way — but it is now written down.

This rides on the spec §4.0 sign-off already recorded.
