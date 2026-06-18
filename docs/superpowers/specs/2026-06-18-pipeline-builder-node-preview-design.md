# Pipeline Builder — Per-Node Zoomable Layer Preview

- **Date:** 2026-06-18
- **Status:** Design approved; ready for implementation plan
- **Worktree / branch:** `pipeline-builder-preview` / `worktree-pipeline-builder-preview`
- **Surface:** `src/phenotypic/gui/builder/`

## Summary

Add a **preview button** (inline-SVG Material "image" icon) to every
image-producing node card in the pipeline builder. Clicking it opens a
**blocking modal** containing a **zoomable OpenSeadragon (OSD) viewer** of that
node's output, with a **layer toggle** over the layers that exist at that node
(`rgb`, `gray`, `detect_mat`, `objmap`, and the `objmap`-over-`detect_mat`
`overlay`).

Previews are computed **faithfully at full resolution** (identical to a real
single-image run) and cached as **full-resolution HDF5 files on disk**, one per
node, under a per-session **per-scope** temp directory (the scope unit is what
makes this compose with nested sub-pipelines — see **Embedded / nested
pipelines**). Zoom is served by the **existing DZI tiler** via a per-`(node,
channel)` staging step, so steady-state RAM is near zero. Cache freshness is
decided per scope by comparing the **stored (scope) pipeline** plus the
source-image identity against the current one; a mismatch wipes and recomputes
just that scope.

This feature is overwhelmingly an **extension of existing infrastructure**. The
builder's **point picker** is already an OSD-in-modal viewer with a channel
radio backed by a per-session disk PNG→DZI cache; this feature generalizes that
pattern to all layers, per node, sourced from on-disk HDFs.

## Goals

- One-click, per-node visual inspection of the pipeline's output at that stage.
- Show **all available layers** for the node, switchable without closing.
- **Zoom/pan** into full-resolution detail.
- **Faithful** previews: detection runs at the original image size; what you see
  matches a real run (only the rendered thumbnail is downsampled for display by
  the tiler/zoom, never the computation).
- **Low RAM**: never hold N full-resolution intermediates resident; data lives on
  disk and is read on demand.
- Reuse existing machinery (point picker pattern, DZI tiler, `_image_renderer`,
  `apply_with_intermediates`) rather than inventing parallel systems.

## Non-Goals (v1)

- Previews for **measurement / post** nodes (they produce DataFrames, not image
  layers; the inspector already renders their tables). No button on those cards.
- **Unifying** the modal's disk-HDF cache with the existing inspector PNG cache
  (`run_preview` → `IntermediatesCache`). They stay independent in v1 (see
  Background). Unification is a possible follow-up.
- A new `WORKFLOWS.md` tutorial row (a single button + modal does not warrant a
  tutorial page). FEATURES.md rows only.
- In-place OSD `world` layer cross-fade. v1 uses the existing destroy+remount
  model on layer switch, matching every current OSD surface.
- A memory-budget / auto-downscale policy. Rejected: previews must be faithful
  full-res; the RAM cost equals a real single-image run, which is already
  accepted for production runs.

## Background — what already exists (and is reused)

| Capability | Location | Reuse |
|---|---|---|
| OSD-in-modal + channel radio + per-session disk cache + DZI-per-channel URL + destroy/remount on switch | `builder/_point_picker.py`, `builder/assets/point_picker.js` | **Primary pattern** to mirror |
| DZI tile generator (`tile(png_path, output_dir, tile_size=254, overlap=1)`) — file-in/files-out; **pyvips sequential streaming** (low RAM) with Pillow fallback | `results_viewer/_dzi_tiler.py:69` | Tiler reused unchanged |
| Channel→PNG rendering: `to_png_bytes`, `to_overlay_png_bytes`, `bytes_to_data_uri`, `render_node_preview`, `_normalize_to_uint8`, `_read_channel` | `builder/_image_renderer.py` | Staging reuses normalizer + overlay |
| Per-node intermediate capture with optional disk persistence (`output_dir`) | `_core/.../_image_pipeline_core.py:884` (`apply_with_intermediates`) | **Extended** with `full_layers` flag |
| v2 HDF schema: `/layers/{rgb,gray,detect_mat,objmap}` as separate gzip datasets; `save_intermediate_layers(filename, layers)`, `load_hdf5` | `_core/_image_parts/_image_io_handler.py` | Per-layer reads + per-node writes |
| Per-session id + server-side cache | `STORE_SESSION_ID`, `builder/_session.py` (`IntermediatesCache`) | Session keying |
| Node-card render + pattern-matching action ids (`linear_node_action_id(action, scope_path, block_id, surface)`, type `LINEAR_NODE_ACTION`) | `builder/_linear_layout.py:493` (`_block_card`), `builder/_ids.py` | Button id + card hook |
| DAG→pipeline conversion + topo bake ordering | `builder/_conversion_dag.py`, `_callbacks.py:6946` (`_bake_preview_cache_dag`) | block_id↔op_key mapping |
| Scope promotion (drill into a sub-pipeline scope as a temp root) | `_callbacks.py:1706` (`_linear_prefix_state_for_preview`), `:5470` (`run_linear_prefix_preview`), `_linear_model.py:146` (`scope_at_path`) | Per-scope compute for nested nodes |
| Source-image resolution (loaded path or synthetic) | `_callbacks.py:7022` (`_load_preview_image`), `SYNTHETIC_SENTINEL` | Compute input |
| Sandbox-safe path components / token charset guards | `_shared/tiles.py` (`is_safe_path_component`) | Route validation |
| Ephemeral temp cache wiped on launch + `atexit` | `browse/_source_render.py` (`init_cache`/`wipe_cache`) | Lifecycle precedent |

**Why the modal needs its own full-res compute.** The existing inspector preview
(`run_preview`) computes intermediates in memory and bakes 512 px PNGs, then
**discards the `Image` objects** — no full-resolution data is retained. Zoom
therefore *requires* a dedicated full-resolution compute; it cannot reuse the
inspector's results. Keeping the two caches independent in v1 also bounds blast
radius on the existing, working inspector feature.

## Architecture & data flow

```
[node card preview button click]
        │  (block_id AND scope_path via ctx.triggered_id)
        ▼
[open callback] ── opens dbc.Modal (backdrop="static") + dcc.Loading
        │
        ▼
[compute/stage callback]    # cache unit = (session, scope), not whole pipeline
   1. scope_key = join(scope_path); scope_dir = preview_cache.ensure(session, scope_key)
   2. current_fp = fingerprint(scope_pipeline_json, source_image_identity)
   3. if manifest.fingerprint != current_fp:   # stale or missing
          wipe(scope_dir)
          scope        = scope_at_path(state.root, scope_path)
          prefix_state = temp_state_rooted_at(scope)   # reuse Preview-here machinery
          image        = _load_preview_image(...)       # full-res source
          pipe         = to_pipeline_dag(prefix_state)  # scope promoted to root
          pipe.apply_with_intermediates(image,
                     output_dir=scope_dir, full_layers=True)   # CORE FLAG
          write manifest.json {fingerprint, nodes:{block_id:{hdf, layers,
                     shape, num_objects}}, error?}        # all nodes in this scope
   4. choose default channel for this node (render_node_preview rule)
   5. stage (scope, node, channel) → DZI  (lazy; see tile route)
   6. write PICKER-style DZI URL store → triggers clientside OSD mount
        │
        ▼
[clientside OSD callback]  __phenotypicPreview.mountViewer(dziUrl)

[layer radio change] → stage (scope, node, new_channel) → DZI url store → remount

[GET /tiles/preview/<session>/<scope_hash>/<block_id>/<channel>.dzi (+ _files/...)]
   → resolve scope manifest → hdf path
   → stage PNG if absent: load_layer_hdf5(hdf, channel) → uint8/colorized PNG
   → _dzi_tiler.tile(png, dzi_dir)   # idempotent, pyvips-streamed
   → send_from_directory(tile)
```

> A **container (sub-pipeline) node** is one top-level op of its parent scope, so
> its preview HDF (the whole sub-pipeline's *final* output) lives in the **parent
> scope's** dir. Previewing nodes *inside* a sub-pipeline requires drilling into
> it (the builder renders one scope's cards at a time); those cards' buttons carry
> the inner `scope_path`, so they compute/read that inner scope's dir. See
> **Embedded / nested pipelines** below.

## Component design

### 1. Core change — `apply_with_intermediates(full_layers=...)`

**File:** `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`

Add a keyword-only `full_layers: bool = False` parameter. It only has effect when
`output_dir is not None`. Today the disk path writes **delta** layers via
`_layers_modified_by(operation)` (plus periodic full "base" snapshots) to
minimize disk. With `full_layers=True`, the inner `_capture` callback instead
writes **every available layer** for each non-read-only operation to its own
`{i:02d}_{key}.h5`:

```python
def _capture(i, key, current, operation):
    if output_dir is not None:
        layers = _layers_modified_by(operation)
        if layers is None:                      # read-only (Measure, GridFinder)
            intermediates[key] = None
        elif full_layers:                       # NEW: faithful full snapshot
            current.copy().save_intermediate_layers(
                output_dir / f"{i:02d}_{key}.h5", layers=_ALL_LAYERS,
            )
            intermediates[key] = None
        elif len(layers) == 4:                  # existing delta/base logic …
            …
```

- `_ALL_LAYERS = ("rgb", "gray", "detect_mat", "objmap")`. `save_intermediate_layers`
  already skips `rgb` when empty, so "available" falls out automatically.
- Read-only ops (`MeasureFeatures`, `GridFinder`) keep emitting no file
  (`intermediates[key] = None`); they get no preview button anyway.
- **Peak RAM unchanged** vs. the existing disk path: one working image + one
  transient `.copy()` per node, dropped after write. Never N intermediates
  resident.
- The initial `base_00.h5` (pre-pipeline, all layers) is still written, giving
  the input node a preview source.
- Backwards compatible: default `False` preserves today's delta behavior; the
  napari viewer and any `output_dir` callers are unaffected.

**File:** `src/phenotypic/_core/_image_parts/_image_io_handler.py`

Add an efficient single-layer reader used by staging (keeps `h5py` in `_core`):

```python
@classmethod
def load_layer_hdf5(cls, filename, layer: str) -> np.ndarray:
    """Read one layer dataset from a v2 (or legacy-flat) intermediate HDF."""
```

It opens the file read-only and reads just `layers/<layer>` (v2) or `<layer>`
(legacy flat), avoiding reconstruction of the whole `Image`. Hyperslab
sub-region reads are a possible future optimization but **not required** for v1
(staging reads the whole layer once, then releases it).

### 2. Preview disk cache — `builder/_preview_cache.py` (new)

**Cache unit = one scope** (root or a drilled-into sub-pipeline scope), under a
per-session directory, under a wiped-on-launch root. This is the integration
decision: the modal keeps its **own** disk cache, **independent** of the
inspector's in-memory PNG cache, but keyed by scope so it composes with the
embedded-pipeline model (see **Embedded / nested pipelines**).

- **Root:** `Path(tempfile.gettempdir()) / "phenotypic" / "pipeline-preview"`.
  `init_cache()` wipes it on import/app start; `atexit` wipes it on shutdown
  (mirror `browse/_source_render.py`).
- **Session dir:** `ensure(session_id)` → `tempfile.mkdtemp(prefix=session_id + "-", dir=root)`
  remembered in a process-global `{session_id: dir}` map. `mkdtemp` handles
  uniqueness + 0700 perms (multi-user HPCC safety).
- **Scope dir:** `<session_dir>/<scope_hash>/` where `scope_hash = sha1(scope_key)`
  (`scope_key` is the joined `scope_path`, i.e. the breadcrumb of container
  `block_id`s; the root scope's key is the empty string). Each scope's previews
  live in their own subdir, so **drilling between scopes never clobbers** another
  scope's cache (strictly better than the inspector's flat in-memory cache, which
  holds only the last-converted scope).
- **Layout inside a scope dir:**
  - `base_00.h5`, `{NN}_{key}.h5` — full-res per-node intermediates for that
    scope's blocks (written by the core flag against the scope-promoted pipeline).
  - `manifest.json` — `{ "fingerprint": str, "scope_key": str, "nodes": {
    block_id: { "hdf": filename, "layers": [available channel names], "shape":
    [H, W], "num_objects": int } }, "error": str | null }`. The scope's input
    node maps to `base_00.h5`; each operation node maps to its `{NN}_{key}.h5`
    via the same op_key↔block_id topological zip `_bake_preview_cache_dag`
    already uses **for a single converted scope** (valid precisely because that
    scope was promoted to root, so its blocks are the top-level ops). `block_id`s
    are globally unique UUID4s, so a flat per-scope node map is unambiguous.
  - `tiles_src/<block_id>__<channel>.png` — staged source PNGs (lazy).
  - `dzi/<block_id>__<channel>.dzi` (+ `_files/`) — tile pyramids (lazy).
- **Fingerprint (per scope):** `sha1(canonical_scope_pipeline_json + "\x00" + source_image_identity)`
  where `scope_pipeline_json` is the serialized **scope-promoted** pipeline (just
  that scope's blocks/edges) and `source_image_identity` = resolved image path
  (or `SYNTHETIC_SENTINEL`) + grid usage + `nrows`/`ncols`. Whole-scope compare —
  editing any node in a scope invalidates that scope's dir; one recompute pass
  refreshes every node in it. Other scopes' dirs are untouched.
- **API:** `ensure(session_id)`, `scope_dir(session_id, scope_key)`,
  `is_fresh(session_id, scope_key, fingerprint)`, `wipe_scope(session_id, scope_key)`,
  `manifest_path(...)`, `node_hdf_path(session_id, scope_key, block_id)`,
  `staged_png_path(...)`, `dzi_dir(...)`.

### 3. HDF→PNG→DZI staging + tile route — `builder/_preview_tiles.py` (new)

Registered on the Flask server in `builder/_app.py` (alongside the point-picker
route registration).

- **Route:** `GET /tiles/preview/<session_id>/<scope_hash>/<block_id>/<channel>.dzi`
  and the matching `…/<channel>_files/<level>/<col>_<row>.png`. Mirrors the point
  picker's `/tiles/<session_id>/<source>.dzi`, with `scope_hash` selecting the
  scope subdir.
- **Validation:** `is_safe_path_component` on `session_id`/`scope_hash`/`block_id`;
  `channel ∈ {rgb, gray, detect_mat, objmap, overlay}`. Reject → 404.
- **Resolution:** read the scope's `manifest.json` → node entry → `hdf` path. If
  the scope, node, or channel is absent → 404.
- **Staging (lazy, idempotent):** if `tiles_src/<block>__<channel>.png` is
  missing/stale, build it:
  - `rgb` → `load_layer_hdf5(hdf, "rgb")` (already uint8) → PNG.
  - `gray` / `detect_mat` → read → `_image_renderer._normalize_to_uint8` → PNG.
  - `objmap` → read → `label2rgb` colorization (reuse `_image_renderer`'s
    `_label_map_to_rgb`) → PNG.
  - `overlay` → read `detect_mat` + `objmap` → reuse `to_overlay_rgb_array`
    logic from arrays → PNG.
  Then `_dzi_tiler.tile(png_path, dzi_dir)` (idempotent mtime check inside the
  tiler). Serve via `send_from_directory`.
- **RAM:** one layer array + one PNG transiently per first view of a channel,
  released after tiling. With pyvips, `dzsave` streams sequentially. After
  tiling, OSD streams tiles from disk (≈0 resident). Prefer pyvips; Pillow
  fallback works but holds the channel's full image + pyramid transiently during
  the one-shot tiling.

### 4. Node-card button — `builder/_linear_layout.py`

In `_block_card` (`:493`), add an `html.Button` in the node header alongside the
existing help/title controls, **only for image-producing nodes** (input,
`ImageEnhancer`, `ObjectDetector`, `ObjectRefiner`, `ImageCorrector`, sub-
`ImagePipeline`). Skip `MeasureFeatures`/post nodes.

- `id = linear_node_action_id(action="preview", scope_path=..., block_id=block.block_id, surface="map")`.
- Child: inline `html.Svg` of the Material Design "image" icon (rounded frame +
  mountains + sun), `fill="currentColor"`, sized via design tokens, themeable.
  Follows the hand-rolled settings-gear SVG precedent (no icon font added).
- `title`/`aria-label`: "Preview this node's output".

### 5. Blocking modal + OSD + layer toggle — `builder/_layout.py` (+ `assets/preview.js`)

Add `build_node_preview_modal()` and mount it once in the boot-time `modals` Div
(`_layout.py:~4511`). Mirror `_point_picker.py` structure.

- `dbc.Modal(id=MODAL_NODE_PREVIEW, size="xl", is_open=False, backdrop="static",
  keyboard=False)` — **blocking**.
- Header: `dbc.ModalTitle` bound to a store-driven node-name string.
- Body:
  - **Layer radio** `dbc.RadioItems(id=PREVIEW_LAYER_RADIO)` populated per-node
    from the manifest's available layers; default = stage-appropriate channel
    (`render_node_preview` rule: Enhancer→`detect_mat`, Detector/Refiner→
    `overlay`, else `rgb`).
  - `dcc.Loading` wrapping `html.Div(id=PREVIEW_OSD_DIV)` (OSD canvas, ~70vh) —
    the spinner covers the full-res compute + first tiling.
  - Caption `html.Div(id=PREVIEW_CAPTION)` — `"{W}×{H} · {channel}"`; shows the
    error string when compute failed.
- Footer: Close button.
- **Client JS** `builder/assets/preview.js` → `window.__phenotypicPreview` with
  `mountViewer(dziUrl)` / `disposeViewer()`, CDN-first OSD 5.x with vendored
  fallback, destroy+remount when the DZI URL changes (copy point_picker.js).
  Add a scoped `.gitattributes` (`* binary`) for any vendored OSD assets if a
  local copy is added (autocrlf gotcha; browse/point-picker precedent).

### 6. Callbacks — `builder/_callbacks.py`

- **Open** (`ALL`-matched on the preview action):
  `Input({"type": LINEAR_NODE_ACTION, "action": "preview", ...: ALL}, "n_clicks")`
  → set `MODAL_NODE_PREVIEW.is_open = True`, write **both** `block_id` and
  `scope_path` from `ctx.triggered_id` to `STORE_PREVIEW_TARGET`. Guard the
  all-zero/initial fire (Dash `allow_duplicate` single-output → wrap return in a
  1-tuple per the gui CLAUDE.md gotcha).
- **Compute + stage** (triggered by `STORE_PREVIEW_TARGET` / modal open), with
  `State` on builder state, session id, image path, grid params:
  1. `scope = scope_at_path(state.root, scope_path)`; build the scope-promoted
     temp state (reuse `_linear_prefix_state_for_preview`'s scope-promotion, but
     keep the **whole** scope, not just the prefix, so one run caches every node
     in it); compute the scope fingerprint; if stale, `wipe_scope` and recompute
     via the core flag — wrapped in try/except → on `MemoryError`/op error, write
     `manifest.error` and surface it in the caption, leave OSD empty.
  2. resolve this node's available layers + default channel from the scope
     manifest; populate `PREVIEW_LAYER_RADIO.options/value`, `ModalTitle`,
     `PREVIEW_CAPTION`.
  3. stage default channel; write `PREVIEW_DZI_URL_STORE`
     (`/tiles/preview/<session>/<scope_hash>/<block>/<channel>.dzi`, prefixed by
     the app's `requests_pathname_prefix`).
- **Layer toggle:** `Input(PREVIEW_LAYER_RADIO, "value")` → stage that channel in
  the same scope → update caption + `PREVIEW_DZI_URL_STORE`.
- **Clientside OSD:** `Input(PREVIEW_DZI_URL_STORE, "data")` →
  `__phenotypicPreview.mountViewer`.

### 7. IDs / stores — `builder/_ids.py`

Add: `MODAL_NODE_PREVIEW`, `PREVIEW_OSD_DIV`, `PREVIEW_LAYER_RADIO`,
`PREVIEW_CAPTION`, `PREVIEW_DZI_URL_STORE`, `STORE_PREVIEW_TARGET` (holds
`{block_id, scope_path}`), `MODAL_NODE_PREVIEW_TITLE`. Reuse `LINEAR_NODE_ACTION`
with `action="preview"` via the existing `linear_node_action_id` factory (no new
type constant; `scope_path` already travels in the id).

### 8. Embedded / nested pipelines

The builder nests sub-pipelines (a container `BlockNode` with
`class_name == "ImagePipeline"` whose `nested` field holds a child scope;
unbounded depth). The preview design composes with this **without new pipeline
machinery**, by mirroring how the inspector already handles scopes:

- **Conversion nests, not flattens.** `to_pipeline_dag` makes a sub-pipeline one
  top-level op (a nested `ImagePipeline`); `apply_with_intermediates` captures
  only top-level op outputs and does **not** recurse into a nested pipeline
  (`_image_pipeline_core.py:779`, `:858`). So a single top-level run can only ever
  produce previews for the **current scope's** blocks.
- **One scope at a time.** The builder renders exactly one scope's cards per the
  breadcrumb, and each node-action button is stamped with its `scope_path`
  (`linear_node_action_id`). A container card previews the **whole sub-pipeline's
  final output** (its single top-level intermediate, cached in the parent scope's
  dir). To preview nodes *inside* it, the user drills in; those cards carry the
  inner `scope_path` and compute/read the inner scope's dir.
- **Scope promotion = the compute primitive.** For any clicked node we promote its
  scope to a temporary root (the exact mechanism `run_linear_prefix_preview` /
  `_linear_prefix_state_for_preview` already use, `_callbacks.py:1706`,`:5470`),
  so that scope's inner blocks become top-level ops and the same op_key↔block_id
  topological zip from `_bake_preview_cache_dag` maps them correctly.
- **Per-scope dirs avoid the inspector's clobbering.** The inspector's flat
  in-memory cache only ever holds the last-converted scope; drilling in/out
  overwrites it. The modal's disk cache is keyed by `scope_hash`, so every scope's
  previews coexist and survive drilling — a deliberate improvement enabled by
  globally-unique `block_id`s.
- **Fidelity caveat (matches existing behavior):** a promoted inner scope is fed
  by the resolved sample image at the inner scope's own `InputImage`, not by the
  parent pipeline's output up to the container (because intermediate capture does
  not recurse into nested pipelines). This is exactly what "Preview here" does
  today; threading the container's true input — or recursing capture into nested
  pipelines — is a future enhancement, noted below.

## RAM analysis (the binding constraint)

| Phase | Resident RAM | Notes |
|---|---|---|
| Compute (full-res run) | ~1 working image + 1 transient copy | = a real single-image `apply()`; accepted |
| Storage | 0 (disk) | full-res HDFs per node |
| Staging (per first-viewed channel) | 1 layer array + 1 PNG, transient | released after tiling; pyvips `dzsave` streams |
| Serving tiles | ~0 | `send_from_directory` of pre-written tiles |
| Modal steady state | ~0 added | no per-node `Image`/PNG retained in the server cache |

Net: the feature adds **no steady-state resident RAM**; the only spikes are the
compute pass (equal to a real run) and transient per-channel staging.

## Error handling

- **Compute failure** (`MemoryError`, op raises): caught; `manifest.error`
  recorded; modal caption shows "Preview couldn't complete: …"; OSD stays empty.
  GUI server does not crash.
- **Missing layer / bad token:** route returns 404; caption shows a friendly
  "layer unavailable" message.
- **pyvips absent:** Pillow fallback tiles correctly (higher transient RAM during
  the one-shot tiling); log an info note recommending pyvips for large plates.
- **objmap/overlay before detection:** not offered (manifest `layers` excludes
  them when `num_objects == 0`), so they can't be requested.

## Files changed / added

**Changed**
- `_core/_pipeline_parts/_image_pipeline_core.py` — `full_layers` flag in
  `apply_with_intermediates` / `_capture`.
- `_core/_image_parts/_image_io_handler.py` — `load_layer_hdf5` reader.
- `gui/builder/_linear_layout.py` — SVG preview button in `_block_card`.
- `gui/builder/_layout.py` — `build_node_preview_modal()`, mount in `modals`.
- `gui/builder/_ids.py` — new ids/stores.
- `gui/builder/_callbacks.py` — open / compute+stage / layer-toggle / clientside;
  the compute path reuses `scope_at_path` + `_linear_prefix_state_for_preview`'s
  scope-promotion (factor a shared "promote scope to temp root" helper if not
  already cleanly callable).
- `gui/builder/_app.py` — register the preview tile route.
- `gui/FEATURES.md` — new rows (CI-gated).

**Added**
- `gui/builder/_preview_cache.py` — session/scope dirs, per-scope manifest,
  fingerprint, scope wipe.
- `gui/builder/_preview_tiles.py` — staging + DZI tile route.
- `gui/builder/assets/preview.js` — `__phenotypicPreview` OSD glue.
- Tests (see below).

## Testing

Reuse `load_synth_yeast_plate()`; microbiology framing in docstrings.

- **Unit — core flag:** `apply_with_intermediates(output_dir=tmp, full_layers=True)`
  writes one HDF per non-read-only op, each containing all available layers;
  read back with `load_layer_hdf5`; assert read-only ops write nothing; assert
  delta behavior unchanged when `full_layers=False`.
- **Unit — staging:** each channel (`rgb/gray/detect_mat/objmap/overlay`) stages
  a valid PNG from a node HDF; `objmap`/`overlay` only when objects exist.
- **Unit — cache/staleness:** fingerprint stable across no-op reloads; changes
  when a param/op/source changes; `ensure`/`wipe` lifecycle; root wiped on
  `init_cache`.
- **Integration — route:** `/tiles/preview/<s>/<scope>/<b>/<c>.dzi` returns a DZI
  manifest and a tile PNG; path-traversal/invalid channel/scope → 404.
- **Integration — open callback:** clicking a node's preview action opens the
  modal, populates the layer radio with available layers, sets the DZI URL store;
  measurement nodes have no button.
- **Integration — nested pipelines:** build a pipeline containing a sub-pipeline;
  assert (a) the container card previews the sub-pipeline's final output from the
  parent scope dir; (b) drilling in and previewing an inner node computes/reads
  the inner scope dir (its own `scope_hash`); (c) the two scopes' dirs coexist —
  previewing the inner node does **not** wipe the parent scope's cache.
- **Error path:** a node that raises during apply records `manifest.error` and the
  caption reflects it without a server crash.
- **e2e (manual / Playwright, FEATURES.md `manual`):** open modal, zoom/pan,
  toggle layers in a live browser (callback wiring + clientside OSD only fire on
  `/_dash-update-component`; verify in a real browser, not just unit tests).

## FEATURES.md additions (CI-gated)

Rows under the builder section, columns
`| Feature | Element | Expected behaviour | Status | Test layer | Test ref |`:

- Node preview button (`linear-node-action`/`preview` SVG icon button).
- Node preview modal (`MODAL_NODE_PREVIEW`, blocking OSD viewer).
- Preview layer toggle (`PREVIEW_LAYER_RADIO`, available layers only).
- Preview tile route (`/tiles/preview/<session>/<scope>/<block>/<channel>.dzi`).
- Preview disk cache + per-scope staleness (`_preview_cache`, scope-keyed).

Start as 🚧/🧪 with concrete `Test ref`s; flip to ✅ when the referenced tests
land (pre-commit validates `Test ref` on ✅ rows).

## Future follow-ups (out of scope)

- Unify the inspector PNG cache with the disk-HDF cache (single compute feeds
  both).
- True streaming HDF-hyperslab tile route (tile-on-demand without materializing a
  full PNG) for very large plates.
- In-place OSD `world` opacity cross-fade between layers (smoother than
  destroy+remount).
- Per-node (sub-pipeline) fingerprints to avoid recomputing unaffected nodes
  after a downstream edit.
- **Faithful inner-node input:** recurse intermediate capture into nested
  pipelines (or thread the container's real input image into a promoted inner
  scope) so inner-node previews reflect the parent pipeline's upstream output
  rather than the raw sample image. v1 matches the existing "Preview here"
  behavior instead.
- Preview button on measurement/post nodes (render the DataFrame/table).
