# Phase 6 — Builder node preview on Viv

**Spec:** §7. **Depends on:** phases 1-3. **Not** blocked on phase 4.

**Deliverable:** the builder's per-node preview pane renders through the shared Viv façade,
over a session-scoped byte route, with `_dzi_tiler` off the preview path
(`_preview_tiles.py:30, :144`). The point picker is untouched.

> **Folded in from a retired cycle-3 spec** (`specs/2026-08-26-builder-preview-viv/`,
> withdrawn 2026-08-26). The split bought no parallelism — this work is blocked on phases
> 1-3 and reuses all three — while forcing phase 3 to write `build_source_spec` at a
> signature it would then refactor, parking phase 2's own asset-mount question in a separate
> document, and running the CI-gate ledger pass twice.

> **The data half already landed.** `_preview_cache.py:48, :255` write per-node
> `.ome.zarr` stores; `_preview_tiles.py:52-65` reads them via `Image.load_layer_zarr`;
> `:78-87` keys freshness on the root `zarr.json`. Decision E is implemented. **Do not
> rebuild it.** This phase is the client-side render swap only.

> **No scratch garbage-collection work.** `init_cache()` already calls `wipe_cache()` on the
> whole cache root at startup plus an `atexit` wipe (`_preview_cache.py:61-68`), and
> `wipe_scope` reclaims on fingerprint change. That is the stated policy (spec §7). An
> earlier draft specified a measured retention cap, oldest-first eviction and a startup
> sweep; **withdrawn** — the sweep duplicated `wipe_cache`, and the eviction policy
> generated the blank-pane bug its own second test existed to guard.

---

### Task 6.1: The preview byte route

**Files:**
- Create: `src/phenotypic/gui/builder/_preview_zarr_routes.py`
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (hash-keyed helpers)
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (split `_validate`)
- Modify: `src/phenotypic/gui/builder/_app.py`
- Test: `tests/unit/gui/builder/test_preview_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `resolve_within_root` from phase 1 task 1.1, `_validate_scope` (this task).
- Produces: `register_preview_zarr_routes(app)` and
  `preview_zarr_url(url_prefix, session_id, scope_hash, block_id, token)`.

- [ ] **Step 1: Split `_validate` into a channel-free core**

`_validate` (`_preview_tiles.py:107-116`) takes a channel this route does not have. Calling
it with a fabricated `"gray"` works only because `"gray" in _VALID_CHANNELS` makes that
clause constant-true — narrowing `_VALID_CHANNELS` would silently change an unrelated
route's guard. Extract:

```python
def _validate_scope(session_id, scope_hash, block_id) -> Optional[Response]:
    """Shape-validate the session/scope/block triple.

    NOT authentication. Nothing here binds the request to a session -- see
    the capability-URL note in step 2.
    """
    if (
        is_safe_path_component(session_id)
        and bool(_HASH_RE.match(scope_hash))
        and is_safe_path_component(block_id)
    ):
        return None
    return _json_error("invalid preview request", 404)
```

`_validate` becomes `_validate_scope(...) or (channel check)`.

- [ ] **Step 1b: Make the preview cache tree private (`mode=0o700`)**

Two lines, and they are **not** the machinery that was dropped. What the scratch-lifecycle
phase lost was a retention cap, oldest-first eviction and a startup sweep; a directory mode
is none of those and should not have travelled with them.

Why it matters *more* now, not less: spec §7 as amended makes session-id secrecy the
**recorded mitigation** — "The id is a secret; treat it as one." But
`preview_cache_root()` is `tempfile.gettempdir()/phenotypic/pipeline-preview`
(`_preview_cache.py:51-53`), created by `mkdir(parents=True)` at the default umask, and the
per-session directories are **named by that secret**. Where `$TMPDIR` is unset or shared,
any local user can `ls` the tree, read every session id, and at umask 022 read the preview
stores directly without ever touching the route. That is the accepted-risk ruling being
undercut by a filesystem mode.

```python
# init_cache
preview_cache_root().mkdir(parents=True, exist_ok=True, mode=0o700)
# _scope_path_by_hash / scope_dir
d.mkdir(parents=True, exist_ok=True, mode=0o700)
```

**Scope honestly:** on an HPCC batch job `$TMPDIR` is typically the per-job
`/scratch/<user>/<jobid>`, already private — so the exposure is real on login and
interactive sessions and absent inside a job. `mode=` applies only on creation; an existing
world-readable tree keeps its mode, which is fine because `init_cache` wipes and recreates.

- [ ] **Step 2: Add the hash-keyed cache helpers**

`read_manifest` (`_preview_cache.py:95`) and `scope_dir` (`:83`) key by `scope_path`, a
list; the route receives a hash. No reverse index is needed — `_scope_path` (`:78`) already
names the directory by the hash:

```python
def _scope_path_by_hash(session_id: str, scope_hash_hex: str) -> Path:
    return preview_cache_root() / session_id / scope_hash_hex


def read_manifest_by_hash(session_id: str, scope_hash_hex: str) -> Optional[dict]:
    path = _scope_path_by_hash(session_id, scope_hash_hex) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
```

Refactor `_scope_path` to delegate, so there is one join.

- [ ] **Step 3: Resolve a block id through the REAL manifest shape**

The manifest is **not** a list of blocks. `_build_manifest` (`:257-263`) returns:

```python
{"version": …, "fingerprint": …, "fingerprint_inputs": …,
 "scope_key": …, "nodes": {block_id: {"store", "layers", "shape", "num_objects"}},
 "error": None}
```

`nodes` is a **dict keyed by block_id**. There is no `"blocks"` list and no `"block_id"`
field — an earlier draft of this plan read `manifest.get("blocks", [])` and would have
404'd every request. Mirror the landed DZI route, which is already correct
(`_preview_tiles.py:127-140`):

```python
def _store_for_block(session_id: str, scope_hash_hex: str, block_id: str) -> Path | None:
    manifest = pc.read_manifest_by_hash(session_id, scope_hash_hex)
    if not manifest:
        return None
    node = manifest.get("nodes", {}).get(block_id)
    if node is None:
        return None
    store = pc._scope_path_by_hash(session_id, scope_hash_hex) / node["store"]
    # The ROOT, not the directory: the promote writes it last, so an
    # interrupted write reads as ABSENT rather than partial.
    return store if (store / ngff_.STORE_ROOT_JSON).is_file() else None
```

- [ ] **Step 4: Write the route on the shared resolver**

```python
def register_preview_zarr_routes(app: dash.Dash) -> None:
    bp = Blueprint("preview_zarr", __name__, url_prefix="/preview-zarr")

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<token>/<path:tail>")
    def preview_store_bytes(session_id, scope_hash, block_id, token, tail):
        invalid = _validate_scope(session_id, scope_hash, block_id)
        if invalid is not None:
            return invalid
        store = _store_for_block(session_id, scope_hash, block_id)
        if store is None:
            abort(404)
        # Guarded for the same reason as the results route: a promote renames
        # the store directory, so both of these can raise on the ROUTINE path.
        try:
            expected = store_generation_token(store)
            roots = readable_roots_for(store)
        except (OSError, StoreUnreadable):
            abort(404)
        if token != expected:
            abort(409)          # stale generation -- see step 5
        return send_file(
            resolve_within_root(store, tail, allowed_roots=roots),
            conditional=True,
        )

    app.server.register_blueprint(bp)
```

**On session scoping — write the tests against what is actually true.** `_validate_scope`
is a *shape* check. Nothing binds the request to a session. Isolation rests on `session_id`
being `uuid.uuid4().hex` (`builder/_callbacks.py:3662, :4083`) — 122 bits, unguessable —
carried in the URL path, where it reaches access logs, the OOD proxy's logs, browser
history and `Referer`. Spec §7 records this as an **accepted** capability-URL risk (user
ruling). So:

```python
def test_an_unissued_session_id_is_not_served(preview_client):
    """The property that actually holds: no sandbox, no bytes.

    NOT 'session A cannot present session B's id' -- it can, and the route
    serves it, exactly as the existing /preview-tiles/ route does. Asserting
    the stronger property would be asserting a binding that does not exist.
    """
    resp = preview_client.get(
        "/preview-zarr/" + "0" * 32 + "/" + "a" * 40 + "/blk/tok/zarr.json"
    )
    assert resp.status_code == 404
```

An earlier draft asserted session A's id against session B's *hash*, which 404s on a
manifest miss rather than on isolation — it would have passed with every isolation
mechanism deleted.

- [ ] **Step 5: Range, traversal and generation tests**

Reuse phase 1's shared parametrized traversal block over `resolve_within_root`; do not
duplicate it. Add the Range test and a generation test. **The chunk key is `c.0.0`, not
`0.0`** — `ngff_.py:409-412` sets `chunk_key_encoding` separator `"."` and a Zarr-v3
default key is `"c" + sep + indices`, so a 2-D `gray` level-0 chunk is `gray/0/c.0.0`
(phase 1's `rgb/0/c.0.0.0` is the 3-D form). With sharding, a small preview array is one
file on disk regardless.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder/test_preview_zarr_routes.py -v
```

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/ tests/unit/gui/builder/
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "feat(gui): serve builder preview store bytes with Range and a generation token"
```

---

### Task 6.2: Mount the shared bundle and swap the renderer

**Files:**
- Modify: `src/phenotypic/gui/builder/_app.py`
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (remove `:30`, `:144`)
- Modify: `src/phenotypic/gui/builder/_preview_callbacks.py`, `assets/preview.js`
- Test: `tests/unit/gui/builder/test_shared_viv_asset.py`, `tests/e2e/gui/test_builder_preview_viv.py`

- [ ] **Step 1: Settle the asset-mount mechanism (spec §10 OQ4)**

```bash
uv run grep -n "assets_folder\|assets_url_path\|serve_locally" \
  src/phenotypic/gui/builder/_app.py src/phenotypic/gui/results_viewer/_app.py \
  src/phenotypic/gui/shell/_app.py
```
**Use a small Flask route** serving the two files from the results-viewer package
directory. This closes OQ4 on evidence rather than on a discovery an executor pays for.

> An earlier draft preferred Dash's `assets_folder` / `assets_url_path` pointed at the
> results viewer's `_assets/viv/`. **That is not viable:** Dash takes **one**
> `assets_folder` per app, and `src/phenotypic/gui/builder/assets/` already holds
> `builder.js`, `preview.js`, `builder.css`, `cytoscape-dagre.min.js`, `point_picker.js`,
> `palette_dnd.js`, `viewport_ops.js` and `wire_drawing.js`. Repointing it drops all eight.

- [ ] **Step 2: Assert one artifact, not two**

```python
def test_builder_does_not_carry_its_own_bundle_copy():
    stray = list((Path(builder_pkg.__file__).parent / "assets").rglob("viv-bundle*.js"))
    assert not stray, f"builder has its own bundle copy: {stray}"
```

- [ ] **Step 3: Build the preview source spec through the shared resolver**

Reuse phase 3 task 3.1's `build_source_spec`, which phase 3 already wrote at its
store-path-plus-base-URL signature **because this caller was in view when it was written**.

**Recompute the source spec on every scope recompute — never cache it across one.**
Preview stores are rewritten in place under the same `scope_hash` when a node's parameters
change, so the **token moves on every re-run** — which is precisely the case this phase
exists to serve. If the spec is computed once and held in a `dcc.Store`, the pane 409s
permanently after the first parameter edit. Rebuild `preview_zarr_url(..., token)` from a
fresh `build_source_spec` each time the manifest updates, exactly where
`_preview_callbacks.py` builds its `dzi_url` today.

**`labelPath` is optional here and that is the common case.**
`build_phenotypic_attributes` omits the `labels` key entirely when `has_labels=False`
(`ngff_.py:576-581`), and `save_intermediate_zarr` sets `write_objmap = "objmap" in layers`
(`_image_io_handler.py:1244`) — so most preview stores have **no** `labels` key at all.
Phase 3 task 3.1 already makes `labelPath` optional; this is the caller that proves it.

- [ ] **Step 4: Swap `preview.js` onto the façade**

Replace the OpenSeadragon mount with `window.phenotypicViv.mount` / `.setSource`.

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/builder/
```
Expected: no hits — the façade is the only thing that touches the bundle.

- [ ] **Step 5: Unhook `_dzi_tiler` from the preview path**

Remove `_preview_tiles.py:30` and `:144`. Then decide the fate of `stage_channel_png`
(`:74`) and `_channel_to_rgb_uint8` (`:52`): with Viv reading the store directly the
intermediate PNG has no consumer, **and neither does the root-`zarr.json` freshness key at
`:78-87`** — the property the retired spec named as its own test target. Retire the key
with the staging code, or say what still consumes it.

```bash
uv run grep -rn "stage_channel_png\|_channel_to_rgb_uint8\|preview_dzi_url" src/ tests/
```

**Do not delete `_dzi_tiler`, and do not delete `_tile_routes.py`.** The tiler keeps four
consumers; and `_preview_tiles.py:31` imports `_TILE_NAME_RE` and `_json_error` **from
`results_viewer/_tile_routes`**, which `_validate_scope` now returns through — so deleting
that module breaks the builder from a different sub-app. See the plan README's Global
Constraints.

- [ ] **Step 6: Prove the point picker is untouched**

```bash
git diff --stat src/phenotypic/gui/builder/_point_picker.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/builder -k point_picker -q
```
Expected: **empty diff**, tests PASS. Spec §7 makes this the executable statement that the
picker stays on DZI.

- [ ] **Step 7: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
uv run ruff check --fix src/phenotypic/gui/builder/
git add -A src/phenotypic/gui/builder tests/unit/gui/builder tests/e2e/gui
git commit -m "feat(gui): render builder node previews through Viv, off the DZI path"
```

---

### Task 6.3: Record the tiler's real consumer set where a deleter will see it

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_dzi_tiler.py` (module docstring)

> An earlier draft added a test pinning the consumer set by text scan. Dropped: it is a
> tripwire, not a check — any legitimate future consumer fails it, and its own failure
> message says the remedy is to edit the test. A docstring reaches the person about to
> delete the module at the moment they open it, which a test in another directory does not.

- [ ] **Step 1: Add the consumer list to the module docstring**

```python
"""Deep-Zoom pyramid generation for the GUI's OpenSeadragon surfaces.

NOT DEAD CODE. Two specs each say the tiler is "removed from this path" --
the results Plate path, and the builder preview path. Read together they
suggest a module that can be deleted. It cannot. Live consumers:

    browse/_app.py:40                  DZI_BACKEND_INFO
    browse/_preparation.py:711         tile()
    browse/_preparation_routes.py:95   DZI_BACKEND_INFO
    builder/_point_picker.py:417       tile()

Browse keeps libvips -> DZI -> BrowseCache -> OSD as its ONLY pixel path
(viewer-viv-rebuild spec section 9), because it reads arbitrary source
images with no run behind them and so has no store to read. The point
picker picks points on a source image before any pipeline node has run.
"""
```

- [ ] **Step 2: Commit with the `gui/CLAUDE.md` edit in phase 5 task 5.3**

```bash
git add src/phenotypic/gui/results_viewer/_dzi_tiler.py
git commit -m "docs(gui): record _dzi_tiler's four remaining consumers on the module"
```
