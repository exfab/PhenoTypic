# Phase 1 — The byte route

**Spec:** §4, §4.1. **Depends on:** phase 0. **Blocks:** phases 2-4.

**Deliverable:** `GET /zarr/<dataset>/<stem>.ome.zarr/<path...>` on the results viewer's
blueprint, serving raw store bytes with **HTTP Range**, guarded per path segment, and
restricted to the pixel groups the client needs. The server does no decode; per-request
memory is a sendfile buffer.

> **Why Range is load-bearing, not a nicety.** A sharded read is a shard-index fetch
> followed by a byte-range fetch into the shard. Without `conditional=True` the client
> pulls whole shards — up to 96 MB for `rgb` — for every 1024² tile. This is the single
> flag the phase exists to get right.

---

### Task 1.1: Serve bytes with Range

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_zarr_routes.py`
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (register the blueprint)
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `OutputRoot.store_path(dataset, stem)` (already exists — confirm its exact name
  with `uv run grep -n "def store_path" src/phenotypic/gui/results_viewer/_output_root.py`),
  `phenotypic.gui._shared.tiles.is_safe_path_component`.
- Produces: `register_zarr_routes(app, output_root) -> None`, and the URL builder
  `zarr_store_url(url_prefix, dataset, stem) -> str` that phase 2's façade calls.

- [ ] **Step 1: Write the failing tests**

```python
"""The zarr byte route serves store bytes with HTTP Range.

Three properties, in descending order of how quietly they fail:
Range (silently pulls whole shards), traversal (silently serves the tree),
and 404 (loudly wrong, therefore least dangerous).
"""

import pytest


def test_serves_the_root_zarr_json(zarr_route_client, spike_store):
    resp = zarr_route_client.get("/zarr/ds/plate.ome.zarr/zarr.json")
    assert resp.status_code == 200
    assert resp.json["attributes"]["phenotypic"]["store_schema_version"]


def test_honours_a_range_request(zarr_route_client, spike_store):
    resp = zarr_route_client.get(
        "/zarr/ds/plate.ome.zarr/rgb/0/c.0.0.0",
        headers={"Range": "bytes=0-15"},
    )
    assert resp.status_code == 206
    assert len(resp.data) == 16
    assert resp.headers["Accept-Ranges"] == "bytes"


def test_missing_chunk_is_404_not_500(zarr_route_client, spike_store):
    resp = zarr_route_client.get("/zarr/ds/plate.ome.zarr/rgb/0/c.99.99.99")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "rgb/./../../zarr.json",
    ],
)
def test_rejects_traversal_in_any_segment(zarr_route_client, spike_store, tail):
    resp = zarr_route_client.get(f"/zarr/ds/plate.ome.zarr/{tail}")
    assert resp.status_code in (400, 404)
```

Build `spike_store` by calling the real writer, as in phase 0 task 0.1 — a hand-built
fixture would not exercise the `"."` chunk keys.

- [ ] **Step 2: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py -v
```
Expected: all fail with 404 (no route registered).

- [ ] **Step 3: Implement the route**

```python
"""Serve raw OME-Zarr store bytes to the browser, with HTTP Range.

The client (Viv/zarrita) reads chunks directly, so this route decodes
nothing: it resolves a path inside one per-image store and hands the file to
``send_file(..., conditional=True)``. ``conditional=True`` is what provides
Range, which sharding requires -- a sharded read is a shard-index fetch
followed by a byte-range fetch into the shard, so without it every tile pulls
a whole shard.
"""

from __future__ import annotations

from pathlib import Path

import dash
from flask import Blueprint, abort, send_file

from phenotypic.gui._shared.tiles import is_safe_path_component

#: Groups inside a store the browser may read. Everything else -- notably the
#: embedded ``tables/measurements/table.parquet`` -- is out of scope for a
#: pixel route and is not exposed.
_READABLE_ROOTS: frozenset[str] = frozenset({"OME", "rgb", "gray", "detect_mat"})

#: Files at the store root the client needs to bootstrap.
_READABLE_ROOT_FILES: frozenset[str] = frozenset({"zarr.json"})


def _resolve_within_store(store: Path, tail: str) -> Path:
    """Resolve ``tail`` inside ``store``, or abort.

    Guards **per segment**, not once over the whole tail: the traversal
    surface here is wider than the DZI route's because the tail is arbitrary
    depth inside a store.
    """
    segments = [s for s in tail.split("/") if s]
    if not segments:
        abort(404)
    for segment in segments:
        if not is_safe_path_component(segment):
            abort(400)
    head = segments[0]
    if head not in _READABLE_ROOTS and not (
        len(segments) == 1 and head in _READABLE_ROOT_FILES
    ):
        abort(404)

    candidate = store.joinpath(*segments)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        abort(404)
    # Belt and braces: even with per-segment guards, a symlink inside the
    # store could escape it. Resolve both sides and compare.
    if not resolved.is_relative_to(store.resolve(strict=True)):
        abort(400)
    if not resolved.is_file():
        abort(404)
    return resolved
```

Then the blueprint:

```python
def register_zarr_routes(app: dash.Dash, output_root) -> None:
    """Mount ``/zarr/<dataset>/<stem>.ome.zarr/<path...>`` on ``app``."""
    bp = Blueprint("zarr_bytes", __name__, url_prefix="/zarr")

    @bp.route("/<dataset>/<stem>.ome.zarr/<path:tail>")
    def store_bytes(dataset: str, stem: str, tail: str):
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            abort(400)
        store = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)
        return send_file(
            _resolve_within_store(Path(store), tail),
            conditional=True,
        )

    app.server.register_blueprint(bp)


def zarr_store_url(url_prefix: str, dataset: str, stem: str) -> str:
    """Base URL a zarr client opens for one per-image store."""
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return f"{base}zarr/{dataset}/{stem}.ome.zarr"
```

**On `_READABLE_ROOTS`:** the spec sketches the route as an unrestricted tail. Since the
spec was written, measurements moved *inside* the store at
`tables/measurements/table.parquet` (see [DRIFT.md](DRIFT.md) D-6), so an unrestricted tail
serves the measurement table to any browser that asks. The allow-list is a deliberate
narrowing. **Flag it for sign-off** — it is a divergence from the spec, not an
implementation detail.

Two properties of the allow-list, both verified during plan refinement, that are easy to
get wrong when editing it:

- **It gates only `segments[0]`, and that is deliberate.** The label group is
  `<primary>/labels/objmap`, whose head is `rgb` or `gray` — already allow-listed. Gating
  every segment against this set instead would block `labels`, `objmap` and every level
  index, breaking the label layer entirely.
- **It is the only thing blocking `tables/`.** `is_safe_path_component('tables')` returns
  `True`, so the per-segment guard passes it. Delete the allow-list and the measurement
  table is served.

- [ ] **Step 4: Register it and run the tests**

Add `register_zarr_routes(app, output_root)` in `_app.py` beside the existing tile-route
registration. Then:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py -v
```
Expected: all PASS. If `test_honours_a_range_request` returns 200, `conditional=True` is
missing or a proxy is stripping the header.

- [ ] **Step 5: The guard is already verified — do not re-derive it**

`is_safe_path_component` (`_shared/tiles.py:755`) was run against this route's inputs during
plan refinement. Verified results:

```text
'..' -> False   '.' -> False    '...' -> False   '.hidden' -> False
'a/b' -> False  'a\b' -> False  '%2e%2e' -> False  '' -> False
'rgb' -> True   'gray' -> True  'detect_mat' -> True  'OME' -> True
'zarr.json' -> True  'c.0.0.0' -> True  '0.0' -> True
'labels' -> True     'objmap' -> True   'tables' -> True
```

It rejects empty, any leading dot, `/`, `\`, and literal `..`, then requires
`^[A-Za-z0-9._-]+$` (`_NAME_RE`, `:752`). `%2e%2e` fails the regex directly, and after
Werkzeug decoding it becomes `..` and fails the explicit check — so the traversal tests do
**not** pass merely because Werkzeug normalized first. Zarr's `"."`-separated chunk keys
pass, which is what makes a per-segment guard usable here at all.

**Note `'tables' -> True`.** The guard does not block
`tables/measurements/table.parquet`; `_READABLE_ROOTS` does. That allow-list is
load-bearing, not belt-and-braces — see step 3.

If you change the guard, re-run the table above and update it here.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_zarr_routes.py \
                        src/phenotypic/gui/results_viewer/_app.py \
                        tests/unit/gui/results_viewer/test_zarr_routes.py
git add src/phenotypic/gui/results_viewer/_zarr_routes.py \
        src/phenotypic/gui/results_viewer/_app.py \
        tests/unit/gui/results_viewer/test_zarr_routes.py
git commit -m "feat(gui): serve OME-Zarr store bytes with HTTP Range"
```

---

### Task 1.2: Prove the route respects the landed staleness key

**Files:**
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (extend)

**Interfaces:**
- Consumes: the root-`zarr.json` staleness key already landed on the store branch
  ([DRIFT.md](DRIFT.md) D-1).
- Produces: spec §8's "staleness" check.

> The traps are already fixed; this is the regression test that keeps them fixed **for the
> new route**. Spec §8 requires a test that "must fail if the check is keyed on the store
> directory" — so the test rewrites a nested chunk, which is exactly the mutation that does
> **not** move the directory's `st_mtime_ns`.

- [ ] **Step 1: Write the test**

```python
def test_a_rewritten_nested_chunk_is_served_fresh(zarr_route_client, spike_store):
    """A nested chunk rewrite must be visible through the route.

    A store directory's ``st_mtime_ns`` does NOT move when a nested chunk is
    rewritten. A route or cache keyed on the directory would serve the stale
    bytes and this test would fail -- which is the whole point of writing it
    against a nested chunk rather than against ``zarr.json``.
    """
    url = "/zarr/ds/plate.ome.zarr/rgb/0/c.0.0.0"
    before = zarr_route_client.get(url).data

    chunk = spike_store / "rgb" / "0" / "c.0.0.0"
    dir_mtime_before = spike_store.stat().st_mtime_ns
    chunk.write_bytes(before[:-1] + bytes([before[-1] ^ 0xFF]))
    assert spike_store.stat().st_mtime_ns == dir_mtime_before, (
        "premise broken: the store directory mtime moved, so this test no "
        "longer proves what it claims"
    )

    after = zarr_route_client.get(url).data
    assert after != before
```

The mid-test assertion on `dir_mtime_before` is deliberate: it fails loudly if the
platform's directory-mtime behaviour differs, rather than letting the test pass while
proving nothing.

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py::test_a_rewritten_nested_chunk_is_served_fresh -v
```
Expected: PASS (the route sends files directly and holds no cache). If it **fails**, a
cache has been introduced between the file and the response — key it on the root
`zarr.json` via `paths_fingerprint`, matching `_tile_routes.py:527`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/results_viewer/test_zarr_routes.py
git commit -m "test(gui): pin the zarr route against nested-chunk staleness"
```
