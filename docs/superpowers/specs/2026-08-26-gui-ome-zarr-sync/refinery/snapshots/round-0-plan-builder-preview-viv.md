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
# Phase 2 — Bundle reuse and the render swap

**Spec:** §1, §4. **Depends on:** phase 1, viv-rebuild phases 2-3. **Blocks:** phase 3.

**Deliverable:** the builder app serves the *same* vendored Viv artifact as the results
viewer, its preview pane renders through the shared façade, and `_dzi_tiler` is off the
preview path (`_preview_tiles.py:30, :144`). The point picker is untouched.

---

### Task 2.1: Mount the shared bundle in the builder

**Files:**
- Modify: `src/phenotypic/gui/builder/_app.py`
- Test: `tests/unit/gui/builder/test_shared_viv_asset.py` (create)

**Interfaces:**
- Consumes: `results_viewer/_assets/viv/viv-bundle.min.js` and `_assets/viv_viewer.js` from
  viv-rebuild phase 2.
- Produces: `window.phenotypicViv` available on the builder page.

> **One artifact, two mounts.** Committing a second ~1 MB copy into `builder/assets/` would
> put two artifacts under one build recipe, drifting independently — and nothing would fail
> when they did, because there is no npm in CI to rebuild and compare. Spec §1.

- [ ] **Step 1: Establish which mechanism actually works**

Spec §7 open question 1 records this as unverified. Determine it before writing code:

```bash
uv run grep -n "assets_folder\|assets_url_path\|serve_locally" \
  src/phenotypic/gui/builder/_app.py \
  src/phenotypic/gui/results_viewer/_app.py \
  src/phenotypic/gui/shell/_app.py
```

Two candidates, in preference order:
1. Dash's `assets_folder` / `assets_url_path` on the builder app pointed at the results
   viewer's `_assets/viv/` — no file duplication, no symlink.
2. A small Flask route on the builder serving the two files from the results-viewer package
   directory.

Pick whichever the existing app construction supports without restructuring, and **record
which and why** in the commit body. Either satisfies decision A.

- [ ] **Step 2: Write the test**

```python
"""The builder serves the results viewer's Viv artifact, not a copy of it.

The second assertion is the one that matters: two artifacts under one build
recipe drift independently, and with no npm in CI nothing would fail when they
did.
"""

from pathlib import Path

import phenotypic.gui.builder as builder_pkg
import phenotypic.gui.results_viewer as rv_pkg

RV_BUNDLE = Path(rv_pkg.__file__).parent / "_assets" / "viv" / "viv-bundle.min.js"


def test_builder_page_exposes_the_facade(builder_client):
    resp = builder_client.get("/assets/viv/viv-bundle.min.js")
    assert resp.status_code == 200
    assert len(resp.data) == RV_BUNDLE.stat().st_size


def test_builder_does_not_carry_its_own_bundle_copy():
    stray = list((Path(builder_pkg.__file__).parent / "assets").rglob("viv-bundle*.js"))
    assert not stray, f"builder has its own bundle copy: {stray}"
```

Adjust the URL in the first test to whatever mechanism step 1 chose.

- [ ] **Step 3: Implement, run, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_shared_viv_asset.py -v
git add src/phenotypic/gui/builder/_app.py \
        tests/unit/gui/builder/test_shared_viv_asset.py
git commit -m "feat(gui): serve one shared Viv artifact to the builder"
```

---

### Task 2.2: Swap the preview pane's renderer

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` — remove `:30` (import) and `:144` (`_dzi_tiler.tile`)
- Modify: `src/phenotypic/gui/builder/_preview_callbacks.py` — clientside wiring
- Modify: `src/phenotypic/gui/builder/assets/preview.js`
- Test: `tests/e2e/gui/test_builder_preview_viv.py` (create)

**Interfaces:**
- Consumes: `preview_zarr_url` from phase 1, `window.phenotypicViv` from task 2.1.
- Produces: a preview pane with no DZI behind it.

- [ ] **Step 1: Write the failing e2e test**

```python
"""The builder preview renders through Viv, reading store chunks directly.

The network assertion is the substance: a preview that renders correctly while
still fetching ``.dzi`` has not been swapped, only decorated.
"""

import pytest


@pytest.mark.e2e
def test_preview_reads_zarr_chunks_not_dzi(page, live_builder_url, seeded_pipeline):
    requests: list[str] = []
    page.on("request", lambda r: requests.append(r.url))

    page.goto(live_builder_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    page.click("[data-testid='preview-node-0']")
    page.wait_for_selector("[data-testid='preview-viv-stage']")

    assert any("/preview-zarr/" in url for url in requests)
    assert not any(url.endswith(".dzi") for url in requests)
```

- [ ] **Step 2: Run and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_builder_preview_viv.py -v
```

- [ ] **Step 3: Build the source spec for a preview node**

The preview's store carries the same `attributes.phenotypic` block a run store does, so
reuse the results viewer's resolver rather than writing a second one:

```bash
uv run grep -n "def build_source_spec" src/phenotypic/gui/results_viewer/_store_source.py
```

It takes an `output_root`; the preview has none. Refactor it to take a **store path and a
base URL** and have the results viewer's caller supply those — one resolver, two callers.
That refactor belongs here rather than in viv-rebuild phase 3, because this is the phase
that discovers the second caller.

**The label-path rule carries over unchanged:** resolve through `phenotypic.labels.objmap`,
never construct `f"{series}/labels/objmap"`. A preview node's store may well be
`gray`-primary, since `save_intermediate_zarr` writes only the requested layers.

- [ ] **Step 4: Rewrite `preview.js`'s render path**

Replace the OpenSeadragon mount with `window.phenotypicViv.mount` / `.setSource`. Talk only
to the façade:

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/builder/
```
Expected: no hits.

- [ ] **Step 5: Unhook `_dzi_tiler`**

Remove `_preview_tiles.py:30` (`from phenotypic.gui.results_viewer import _dzi_tiler`) and
`:144` (`_dzi_tiler.tile(png_path, sdir / "dzi")`). Then decide the fate of
`stage_channel_png` (`:74`) and `_channel_to_rgb_uint8` (`:52`): with Viv reading the store
directly, the intermediate PNG has no consumer for the preview pane.

```bash
uv run grep -rn "stage_channel_png\|_channel_to_rgb_uint8\|preview_dzi_url" src/ tests/
```
Remove them if nothing else calls them; keep them if the point picker or a test does. Say
which in the commit body.

**Do not delete `_dzi_tiler`.** After this phase it still has four consumers —
`browse/_app.py:40`, `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`,
`builder/_point_picker.py:417`.

- [ ] **Step 6: Prove the point picker is untouched**

```bash
git diff --stat src/phenotypic/gui/builder/_point_picker.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -k "point_picker" -q
```
Expected: **empty diff**, tests PASS. Spec §4 makes this the executable statement that the
picker stays on DZI; a diff here means the phase overreached.

- [ ] **Step 7: Run the builder suites**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
```

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/ \
                        src/phenotypic/gui/results_viewer/_store_source.py
git add -A src/phenotypic/gui/builder src/phenotypic/gui/results_viewer/_store_source.py \
           tests/unit/gui/builder tests/e2e/gui
git commit -m "feat(gui): render builder node previews through Viv, off the DZI path"
```
# Phase 3 — Scratch store lifecycle

**Spec:** §3, §6. **Depends on:** phase 2. **Blocks:** phase 4.

**Deliverable:** a **measured** retention cap on scope revisions per session, enforced
oldest-first and never evicting the focused scope, plus a session-exit sweep at builder
startup.

> **Why this is a phase and not a cleanup.** Viv-rebuild decision E named "a scratch dir to
> garbage-collect" as an accepted cost but set no policy. `wipe_cache` (`:56`) and
> `wipe_scope` (`:90`) exist, and `wipe_scope` runs when a scope's fingerprint changes — but
> nothing bounds accumulation across *revisions* within a session, and nothing reclaims a
> dead session's tree. With one store per node per scope revision, a long authoring session
> grows without limit.

---

### Task 3.1: Measure before capping

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py`

**Interfaces:**
- Produces: `PREVIEW_SCOPE_RETENTION` — the number task 3.2 enforces.

- [ ] **Step 1: Measure one real scope's on-disk cost**

```bash
uv run du -sh <preview_cache_root>/<session>/<scope_hash>
uv run find <preview_cache_root>/<session>/<scope_hash> -type f | wc -l
```
Record bytes and inode count per scope revision at a realistic node count. Inodes matter as
much as bytes here — the backend spec's §1.4 records ~40 files per pyramided store, and a
preview store is single-level (16 files), so a 10-node pipeline is ~160 files **per
revision**.

- [ ] **Step 2: Write the budget script**

Per root `CLAUDE.md`: stdlib + numpy/scipy only, never imports `phenotypic`, exits non-zero
on failure.

```python
"""Re-derive the builder preview scratch budget.

Claim under test (builder-preview-viv spec section 3): one store per node per
scope revision accumulates without bound in a long authoring session, so the
retention cap must be chosen against a measured per-revision cost rather than
picked.

Fill MEASURED_* from task 3.1 step 1, then this script derives the cap.
Exits non-zero while the measurement is missing.
"""

import sys

#: Bytes for one scope revision, measured on a real session (task 3.1 step 1).
MEASURED_BYTES_PER_REVISION: int | None = None
#: Files for one scope revision, measured the same way.
MEASURED_FILES_PER_REVISION: int | None = None

#: Budgets the cap must fit inside.
BYTE_BUDGET = 2 * 1024**3
FILE_BUDGET = 20_000


def cap_from(bytes_per: int, files_per: int) -> int:
    """Largest revision count fitting both budgets."""
    return max(1, min(BYTE_BUDGET // bytes_per, FILE_BUDGET // files_per))


def main() -> int:
    if MEASURED_BYTES_PER_REVISION is None or MEASURED_FILES_PER_REVISION is None:
        print("NO MEASUREMENT: fill MEASURED_* from task 3.1 step 1")
        return 1
    cap = cap_from(MEASURED_BYTES_PER_REVISION, MEASURED_FILES_PER_REVISION)
    print(
        f"per revision: {MEASURED_BYTES_PER_REVISION / 1e6:.1f} MB, "
        f"{MEASURED_FILES_PER_REVISION} files"
    )
    print(f"PREVIEW_SCOPE_RETENTION = {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it — failing while unmeasured is the point**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py
```
Expected: `NO MEASUREMENT`, exit **1**. Fill the constants from step 1, re-run, and record
the derived cap.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/
git commit -m "docs(gui): measure the builder preview scratch budget"
```

---

### Task 3.2: Enforce the cap oldest-first

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_cache.py`
- Test: `tests/unit/gui/builder/test_preview_retention.py` (create)

**Interfaces:**
- Consumes: `PREVIEW_SCOPE_RETENTION` from task 3.1.
- Produces: `enforce_scope_retention(session_id, *, focused_scope_hash) -> list[str]`
  returning the scope hashes evicted.

- [ ] **Step 1: Write the failing tests**

```python
"""Scope revisions are capped oldest-first, sparing the focused scope.

The second test is the one with teeth: evicting the scope the user is looking
at would be a cache policy that produces a blank pane, which reads as a bug in
the renderer rather than in the cache.
"""

from phenotypic.gui.builder._preview_cache import (
    PREVIEW_SCOPE_RETENTION,
    enforce_scope_retention,
)


def test_evicts_oldest_first_down_to_the_cap(session_with_many_scopes):
    session_id, hashes_oldest_first = session_with_many_scopes
    evicted = enforce_scope_retention(
        session_id, focused_scope_hash=hashes_oldest_first[-1]
    )
    survivors = [h for h in hashes_oldest_first if h not in evicted]
    assert len(survivors) <= PREVIEW_SCOPE_RETENTION
    assert evicted == hashes_oldest_first[: len(evicted)]


def test_never_evicts_the_focused_scope(session_with_many_scopes):
    session_id, hashes_oldest_first = session_with_many_scopes
    focused = hashes_oldest_first[0]
    evicted = enforce_scope_retention(session_id, focused_scope_hash=focused)
    assert focused not in evicted
```

Note the deliberate tension: `test_never_evicts_the_focused_scope` focuses the **oldest**
scope, which is exactly the case a naive oldest-first sweep gets wrong.

- [ ] **Step 2: Run, implement, run**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_retention.py -v
```
Expected: FAIL (no `enforce_scope_retention`), then PASS after implementing.

Order revisions by the manifest's own recorded time rather than by directory mtime — a
store directory's mtime does not move when a nested chunk is rewritten, which is the same
trap the freshness checks already avoid.

- [ ] **Step 3: Call it where a scope is promoted**

`_preview_cache.py:158` `_promote_scope_state` is the natural hook. Confirm:

```bash
uv run sed -n '151,175p' src/phenotypic/gui/builder/_preview_cache.py
```

- [ ] **Step 4: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py \
                        tests/unit/gui/builder/test_preview_retention.py
git add src/phenotypic/gui/builder/_preview_cache.py \
        tests/unit/gui/builder/test_preview_retention.py
git commit -m "feat(gui): cap retained preview scope revisions oldest-first"
```

---

### Task 3.3: Sweep dead sessions at startup

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (`init_cache`, `:61`)
- Test: `tests/unit/gui/builder/test_preview_retention.py` (extend)

- [ ] **Step 1: Write the test**

```python
def test_startup_sweeps_sessions_with_no_live_dash_session(stale_session_tree):
    """A session id with no live Dash session is reclaimable at startup.

    Startup is the only safe moment: mid-run there is no way to distinguish a
    dead session from one whose browser tab is merely backgrounded.
    """
    from phenotypic.gui.builder._preview_cache import init_cache, preview_cache_root

    stale_id, live_id = stale_session_tree
    init_cache(live_session_ids={live_id})

    assert not (preview_cache_root() / stale_id).exists()
    assert (preview_cache_root() / live_id).exists()
```

- [ ] **Step 2: Implement**

Give `init_cache` an optional `live_session_ids` parameter defaulting to `None`, meaning
"sweep nothing" — so an existing caller that does not pass it keeps today's behaviour
exactly. The builder app passes the live set at startup.

**Startup is the only safe moment** to sweep: mid-run there is no way to distinguish a dead
session from a backgrounded browser tab, and a wrong guess deletes the stores under a
working pane.

- [ ] **Step 3: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_retention.py -v
uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py \
                        tests/unit/gui/builder/test_preview_retention.py
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "feat(gui): sweep dead-session preview sandboxes at startup"
```
# Phase 4 — Verification, ledgers and docs

**Spec:** §6. **Depends on:** phases 1-3.

**Deliverable:** all six of spec §6's checks green, the three `gui-checks` gates passing,
and the ledgers, tutorial and `gui/CLAUDE.md` describing the Viv-backed preview.

---

### Task 4.1: Close out spec §6's six checks

- [ ] **Step 1: Walk the checklist**

| Spec §6 check | Where it lives |
|---|---|
| Range on the preview route — `206`, not `200` | phase 1 task 1.1 |
| Session isolation — against a **real** second sandbox, not a crafted path | phase 1 task 1.1 |
| Traversal — per-segment guard rejects `..` in any position | phase 1 task 1.1 |
| Freshness survives the swap — via a **nested chunk** rewrite | phase 1 task 1.2 |
| Point picker unaffected — its tests pass **unmodified** | phase 2 task 2.2 step 6 |
| Scratch cap — oldest-first, focused scope never evicted | phase 3 task 3.2 |

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_zarr_routes.py \
  tests/unit/gui/builder/test_preview_retention.py \
  tests/unit/gui/builder/test_shared_viv_asset.py \
  -n 4 -v
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -k point_picker -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
```
Expected: all PASS.

- [ ] **Step 2: Re-prove the point picker diff is empty across the whole plan**

```bash
git diff --stat <plan-baseline-sha> -- src/phenotypic/gui/builder/_point_picker.py
```
Expected: **empty**. Spec §4 makes this the executable statement that the picker stays on
DZI. A non-empty diff is a stop-and-escalate.

- [ ] **Step 3: Confirm both measurements were actually taken**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py
```
Expected: exit 0 with the derived cap printed. A non-zero exit means phase 3's constants
were never filled in and the cap in the code is an invented number.

---

### Task 4.2: Confirm `_dzi_tiler`'s real consumer set

**Files:**
- Test: `tests/unit/gui/test_dzi_tiler_consumers.py` (create)

**Interfaces:**
- Produces: a guard against a future "cleanup" deleting a module four surfaces still need.

> Three separate specs describe `_dzi_tiler` as being "removed" from *a* path. Read
> together they invite the conclusion that the module is dead. It is not: Browse keeps
> libvips → DZI → `BrowseCache` → OSD as its **only** pixel path, and the point picker has
> no store to read.

- [ ] **Step 1: Write the test**

```python
"""``_dzi_tiler`` keeps four consumers after the Viv migrations.

Recorded as a test because three specs each say the tiler is 'removed from
this path', and read together they suggest a module that can be deleted. It
cannot: Browse has no store behind its source images, and the point picker
picks points before any pipeline node has run.
"""

from pathlib import Path

import phenotypic.gui as gui_pkg

EXPECTED_CONSUMERS = {
    "browse/_app.py",
    "browse/_preparation.py",
    "browse/_preparation_routes.py",
    "builder/_point_picker.py",
}


def test_dzi_tiler_consumer_set_is_exactly_what_the_specs_expect():
    root = Path(gui_pkg.__file__).parent
    found = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*.py")
        if path.name != "_dzi_tiler.py" and "_dzi_tiler" in path.read_text("utf-8")
    }
    assert found == EXPECTED_CONSUMERS, (
        "the _dzi_tiler consumer set changed; update this test AND the specs "
        f"that enumerate it.\n  unexpected: {sorted(found - EXPECTED_CONSUMERS)}"
        f"\n  missing:    {sorted(EXPECTED_CONSUMERS - found)}"
    )
```

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/test_dzi_tiler_consumers.py -v
```
Expected: PASS. A failure naming `results_viewer/_tile_routes.py` means viv-rebuild phase 3
step 4 was not completed; naming `builder/_preview_tiles.py` means phase 2 step 5 was not.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/test_dzi_tiler_consumers.py
git commit -m "test(gui): pin the _dzi_tiler consumer set against stray cleanup"
```

---

### Task 4.3: Ledgers, tutorial and CLAUDE.md

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Modify: `docs/source/tutorials/gui/03_build_pipeline.md`
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Update the preview rows**

```bash
uv run grep -n "preview\|node preview\|DZI" src/phenotypic/gui/FEATURES.md | head -20
```
Update the node-preview rows to describe a Viv-backed pane over `/preview-zarr/...`, and
add a row for the retention policy — a cache that silently evicts is a user-visible
behaviour and belongs in the ledger.

- [ ] **Step 2: Refresh the builder tutorial and its capture**

`03_build_pipeline.md` shows the preview pane. Update prose and re-capture per the
**`gui-tutorial-capture`** skill, keeping the WORKFLOWS.md ↔ capture-function ↔
tutorial-page round trip closed:

```bash
uv run grep -n "build_pipeline\|_capture_build" \
  src/phenotypic/gui/WORKFLOWS.md scripts/capture_gui_tutorial_screenshots.py
```

- [ ] **Step 3: Update `gui/CLAUDE.md`**

Record: the builder preview reads its scratch `.ome.zarr` through `/preview-zarr/...` and
renders with the shared Viv façade; the point picker stays on `_dzi_tiler`; scratch scopes
are capped and swept at startup; the Viv artifact is served once, from the results-viewer
package.

- [ ] **Step 4: Run the three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all exit 0.

- [ ] **Step 5: Full suite as a Slurm job**

```bash
sbatch docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch
```
Report as "green except the known baseline failure", re-confirming it is still that test
failing for that reason.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           src/phenotypic/gui/CLAUDE.md scripts/capture_gui_tutorial_screenshots.py \
           docs/source
git commit -m "docs(gui): record the Viv-backed builder preview and its cache policy"
```

---

### Task 4.4: Resolve or re-file the spec's open questions

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-builder-preview-viv/design.md`

- [ ] **Step 1: Close OQ1 with the mechanism actually used**

Spec §7 OQ1 asks whether `DispatcherMiddleware` can serve one `_assets/viv/` to both
sub-apps. Phase 2 task 2.1 step 1 determined it. Record the answer and the evidence.

- [ ] **Step 2: Close OQ2 with the measured cap**

Record `PREVIEW_SCOPE_RETENTION` and the per-revision measurement behind it.

- [ ] **Step 3: Leave OQ3 open, deliberately**

Whether preview stores should pyramid at all stays open — they are single-level today,
which is right for a preview pane, and it only changes if the pane grows a deep-zoom
gesture. Recorded so the decision is not made by accident later.

- [ ] **Step 4: Present the spec edits to the user; do not self-approve**

Closing an open question is a spec change. Draft it, report it, and wait.
# Builder node preview on Viv: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The builder's per-node preview pane renders its layer by reading zarr chunks in
the browser through Viv, over a session-scoped byte route — with no server-side DZI
pyramid built for it, and with the scratch stores garbage-collected under a measured
retention policy.

**Architecture:** Reuse, not rebuild. The preview already writes and reads `.ome.zarr`
stores; only the render path is still DZI. So this plan adds a second byte blueprint over
the builder sandbox, mounts the results viewer's *existing* Viv bundle and façade in the
builder app, swaps the preview pane's renderer, and gives the scratch stores a lifecycle
they currently lack.

**Tech Stack:** Viv + deck.gl (the vendored bundle from the Viv rebuild), zarrita.js,
Flask `send_file(conditional=True)`, Dash clientside callbacks, Python 3.11+, `uv`.

**Spec:** [`docs/superpowers/specs/2026-08-26-builder-preview-viv/design.md`](../../specs/2026-08-26-builder-preview-viv/design.md)

**Blocked on:** [`2026-08-26-viewer-viv-rebuild`](../2026-08-26-viewer-viv-rebuild/README.md)
phases 1-3 — the byte route pattern, the bundle, and the façade. **Not** blocked on its
phase 4.

**Baseline:** branch `feat/gui-ome-zarr-sync`, restacked onto
`worktree-ome-zarr-image-store` head `bf0d01a1`.

---

## Global Constraints

Everything in the removals plan's Global Constraints applies — `uv` only,
`QT_QPA_PLATFORM=offscreen`, never `-n auto`, explicit ruff paths, the known-failing
baseline test, the three `gui-checks` CI gates. Additionally:

- **The data half is already landed. Do not rebuild it.** Verified in this worktree:
  `_preview_cache.py:48` (`BASE_STORE_NAME = "base_00.ome.zarr"`), `:255`
  (`f"{i:02d}_{op_key}.ome.zarr"`), `_preview_tiles.py:52-65` (`Image.load_layer_zarr`),
  `:78-87` (freshness on `ngff_.STORE_ROOT_JSON`). Viv-rebuild spec §7's decision E is
  **implemented**; this plan is the render swap only.
- **`_dzi_tiler` keeps four consumers after this plan** — `browse/_app.py:40`,
  `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`, and
  `builder/_point_picker.py:417`. Only `_preview_tiles.py:30, :144` come off. The module is
  not a deletion candidate.
- **The point picker is out of scope** (spec §4). Its tests must pass **unmodified**; a
  diff in `tests/` touching the point picker means the plan overreached.
- **Session isolation is a security property.** `_preview_tiles.py:107` `_validate` is what
  keeps one browser session out of another's sandbox. The new route **reuses** it; it does
  not reimplement it.
- **One bundle, not two.** Committing a second ~1 MB copy of the Viv artifact into
  `builder/assets/` would put two artifacts under one build recipe, drifting
  independently. See spec §1 and phase 2.

---

## Phases

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 1 | Preview byte route | `/preview-zarr/...` with Range, session-scoped, traversal-guarded | [phase-1](phase-1-preview-byte-route.md) |
| 2 | Bundle reuse + render swap | Builder mounts the shared façade; preview pane renders through Viv; `_dzi_tiler` off the preview path | [phase-2](phase-2-render-swap.md) |
| 3 | Scratch lifecycle | Measured retention cap, session-exit sweep | [phase-3](phase-3-scratch-lifecycle.md) |
| 4 | Verification & ledgers | Spec §6's six checks, FEATURES/WORKFLOWS, tutorial | [phase-4](phase-4-verification.md) |

## Definition of done

1. A ranged chunk request to the preview route returns `206`.
2. A request carrying session A's id cannot reach session B's sandbox.
3. Re-running a node with changed parameters changes what the browser renders — proven by
   rewriting a **nested chunk**, which does not move the store directory's `st_mtime_ns`.
4. The point picker's tests pass **unmodified**.
5. The retention cap is enforced oldest-first and never evicts the focused scope.
6. `uv run pytest tests/unit/gui -n 4` green (minus the known baseline failure); the three
   `gui-checks` gates exit 0.
