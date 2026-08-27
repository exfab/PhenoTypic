# Phase 1 — The preview byte route

**Spec:** §2, §6. **Depends on:** viv-rebuild phase 1. **Blocks:** phase 2.

**Deliverable:** `GET /preview-zarr/<session_id>/<scope_hash>/<block_id>/<path...>` on the
builder's blueprint, serving preview-store bytes with HTTP Range, guarded by the existing
`_validate` for the first three segments and per-segment `is_safe_path_component` on the
tail.

> **Why a second route rather than the results one.** The results viewer's `/zarr/...`
> route resolves through `OutputRoot.store_path`, which is a **run output root**. Preview
> stores live under `preview_cache_root()/<session>/<scope_hash>/<NN>_<op>.ome.zarr` and
> have no `OutputRoot` at all. Bending the results route to accept both would mean giving
> it two resolution modes and two guard regimes — the coupling costs more than the
> duplication.

---

### Task 1.1: Serve preview-store bytes with Range

**Files:**
- Create: `src/phenotypic/gui/builder/_preview_zarr_routes.py`
- Modify: `src/phenotypic/gui/builder/_app.py` (register the blueprint)
- Test: `tests/unit/gui/builder/test_preview_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `phenotypic.gui.builder._preview_cache.scope_dir`,
  `..._preview_tiles._validate`, `phenotypic.gui._shared.tiles.is_safe_path_component`.
- Produces: `register_preview_zarr_routes(app) -> None` and
  `preview_zarr_url(url_prefix, session_id, scope_hash, block_id) -> str` — phase 2's
  clientside callback calls the latter.

- [ ] **Step 1: Read the route this one mirrors**

```bash
uv run sed -n '101,175p' src/phenotypic/gui/builder/_preview_tiles.py
```
Note `preview_dzi_url`'s shape (`:101`), `_validate`'s signature and what it returns on
failure (`:107`), and how `register_node_preview_routes` builds its blueprint (`:118`).
The new route mirrors all three; matching them is what keeps the guard regime uniform.

- [ ] **Step 2: Confirm the on-disk store naming**

```bash
uv run grep -n "BASE_STORE_NAME\|ome.zarr" src/phenotypic/gui/builder/_preview_cache.py
```
Expected: `BASE_STORE_NAME = "base_00.ome.zarr"` at `:48` and the per-node
`f"{i:02d}_{op_key}.ome.zarr"` at `:255`. The route resolves a **block id** to one of those
names, so read how `_describe`/`_build_manifest` map block ids to store names (`:225-265`)
before writing the resolver.

- [ ] **Step 3: Write the failing tests**

```python
"""The preview byte route serves scratch-store bytes, session-scoped.

Three properties, in descending order of how quietly they fail: Range
(silently pulls whole shards), session isolation (silently serves another
session's sandbox), traversal (silently serves the tree).
"""

import pytest


def test_serves_a_preview_store_root(preview_client, preview_scope):
    session_id, scope_hash, block_id = preview_scope
    resp = preview_client.get(
        f"/preview-zarr/{session_id}/{scope_hash}/{block_id}/zarr.json"
    )
    assert resp.status_code == 200
    assert resp.json["attributes"]["phenotypic"]


def test_honours_a_range_request(preview_client, preview_scope):
    session_id, scope_hash, block_id = preview_scope
    resp = preview_client.get(
        f"/preview-zarr/{session_id}/{scope_hash}/{block_id}/gray/0/0.0",
        headers={"Range": "bytes=0-15"},
    )
    assert resp.status_code == 206
    assert len(resp.data) == 16


def test_one_session_cannot_reach_anothers_sandbox(
    preview_client, preview_scope, other_session_scope
):
    session_id, scope_hash, block_id = preview_scope
    other_session, other_hash, other_block = other_session_scope
    assert session_id != other_session

    resp = preview_client.get(
        f"/preview-zarr/{session_id}/{other_hash}/{other_block}/zarr.json"
    )
    assert resp.status_code in (400, 403, 404)


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "gray/../../../zarr.json",
        "gray/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "gray/./../../../base_00.ome.zarr/zarr.json",
    ],
)
def test_rejects_traversal_in_any_segment(preview_client, preview_scope, tail):
    session_id, scope_hash, block_id = preview_scope
    resp = preview_client.get(
        f"/preview-zarr/{session_id}/{scope_hash}/{block_id}/{tail}"
    )
    assert resp.status_code in (400, 404)
```

`preview_scope` must build a **real** scope by running the preview machinery, so the stores
carry real chunk keys. `other_session_scope` builds a second one under a different session
id — spec §6 requires session isolation asserted "against a real second sandbox, not a
crafted path", because a crafted path tests the guard while a real sandbox tests the
resolution.

The chunk key `gray/0/0.0` assumes the `"."` separator and a 2-D `gray` array. Confirm
against the real store before trusting it:

```bash
find <a-preview-store>/gray -maxdepth 2 | head
```

- [ ] **Step 4: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_zarr_routes.py -v
```
Expected: all fail with 404 — no route registered.

- [ ] **Step 5: Implement**

```python
"""Serve builder preview-store bytes to the browser, with HTTP Range.

The preview pane's zarr client reads chunks directly, so this route decodes
nothing. It mirrors ``_preview_tiles.register_node_preview_routes``: same
session/scope/block validation, same blueprint shape, ``conditional=True``
for Range because sharding needs it.

Session scoping is a SECURITY property, not a URL convention -- ``_validate``
is what keeps one browser session out of another's sandbox, and it is reused
here rather than reimplemented.
"""

from __future__ import annotations

from pathlib import Path

import dash
from flask import Blueprint, abort, send_file

from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui.builder import _preview_cache as pc
from phenotypic.gui.builder._preview_tiles import _validate


def _store_for_block(session_id: str, scope_hash: str, block_id: str) -> Path | None:
    """Resolve a block id to its ``.ome.zarr`` inside the scope sandbox."""
    manifest = pc.read_manifest_by_hash(session_id, scope_hash)
    if not manifest:
        return None
    for entry in manifest.get("blocks", []):
        if entry.get("block_id") == block_id:
            return pc.scope_dir_by_hash(session_id, scope_hash) / entry["store"]
    return None


def register_preview_zarr_routes(app: dash.Dash) -> None:
    """Mount ``/preview-zarr/<session>/<scope>/<block>/<path...>``."""
    bp = Blueprint("preview_zarr", __name__, url_prefix="/preview-zarr")

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<path:tail>")
    def preview_store_bytes(session_id, scope_hash, block_id, tail):
        invalid = _validate(session_id, scope_hash, block_id, "gray")
        if invalid is not None:
            return invalid

        store = _store_for_block(session_id, scope_hash, block_id)
        if store is None or not store.is_dir():
            abort(404)

        segments = [s for s in tail.split("/") if s]
        if not segments:
            abort(404)
        for segment in segments:
            if not is_safe_path_component(segment):
                abort(400)

        candidate = store.joinpath(*segments)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            abort(404)
        if not resolved.is_relative_to(store.resolve(strict=True)):
            abort(400)
        if not resolved.is_file():
            abort(404)

        return send_file(resolved, conditional=True)

    app.server.register_blueprint(bp)


def preview_zarr_url(
    url_prefix: str, session_id: str, scope_hash: str, block_id: str
) -> str:
    """Base URL a zarr client opens for one preview node's store."""
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return f"{base}preview-zarr/{session_id}/{scope_hash}/{block_id}"
```

`read_manifest_by_hash` / `scope_dir_by_hash` may not exist — `_preview_cache.py:95`
`read_manifest` and `:83` `scope_dir` take a `scope_path`, not a hash. Check first, and add
thin hash-keyed wrappers beside them if needed rather than duplicating the path logic:

```bash
uv run sed -n '71,116p' src/phenotypic/gui/builder/_preview_cache.py
```

- [ ] **Step 6: Register and run**

Add `register_preview_zarr_routes(app)` in `builder/_app.py` beside the existing
`register_node_preview_routes` call.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_zarr_routes.py -v
```
Expected: all PASS. A 200 on the Range test means `conditional=True` is missing.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/_preview_zarr_routes.py \
                        src/phenotypic/gui/builder/_app.py \
                        src/phenotypic/gui/builder/_preview_cache.py \
                        tests/unit/gui/builder/test_preview_zarr_routes.py
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "feat(gui): serve builder preview store bytes with HTTP Range"
```

---

### Task 1.2: Prove freshness survives on a nested-chunk rewrite

**Files:**
- Test: `tests/unit/gui/builder/test_preview_zarr_routes.py` (extend)

**Interfaces:**
- Consumes: the root-`zarr.json` freshness key at `_preview_tiles.py:86`.
- Produces: spec §6's "freshness survives the swap" check.

- [ ] **Step 1: Write the test**

```python
def test_a_rewritten_nested_chunk_is_served_fresh(preview_client, preview_scope):
    """Re-running a node with changed parameters must change what renders.

    Written against a NESTED CHUNK deliberately: a store directory's
    ``st_mtime_ns`` does not move when a nested chunk is rewritten, so a
    freshness check keyed on the directory would pass this test's setup and
    fail its assertion. The existing key is the root ``zarr.json``
    (``_preview_tiles.py:86``); this pins it for the new route.
    """
    session_id, scope_hash, block_id = preview_scope
    url = f"/preview-zarr/{session_id}/{scope_hash}/{block_id}/gray/0/0.0"
    before = preview_client.get(url).data

    store = _store_for_block(session_id, scope_hash, block_id)
    chunk = store / "gray" / "0" / "0.0"
    dir_mtime_before = store.stat().st_mtime_ns
    chunk.write_bytes(before[:-1] + bytes([before[-1] ^ 0xFF]))
    assert store.stat().st_mtime_ns == dir_mtime_before, (
        "premise broken: the store directory mtime moved, so this test no "
        "longer proves what it claims"
    )

    assert preview_client.get(url).data != before
```

- [ ] **Step 2: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_zarr_routes.py -v
git add tests/unit/gui/builder/test_preview_zarr_routes.py
git commit -m "test(gui): pin preview-route freshness to nested-chunk rewrites"
```
