# Source Image **Browse** Tab — Design Spec

**Status:** Ready for implementation-plan
**Branch target:** `worktree-gui-source-image-viewer`
**Author:** Design brainstorm (Claude + Alex)
**Date:** 2026-06-11

---

## Problem Statement

The GUI hub can build pipelines (Builder), tune them (Tune), execute them
(Run), and inspect **post-pipeline output** (Viewer / Analysis). There is
**no way to browse the raw _input_ images** of a source directory before
or during processing. A user on a remote cluster, connected by SSH port
forward, currently has to `scp` plate scans down to eyeball them, or load
them in a separate tool.

The hub already tracks a **"source image root"** as a first-class concept
(`shell/_source_context.py`: a top-bar picker, a `localStorage` store
`SHELL_SOURCE_IMAGE_ROOT_STORE`, and `resolve_source_image_root()`), but
nothing _consumes_ it for viewing.

We want a new top-level **Browse** tab that lists every image under the
selected source root and renders any one of them in a deep-zoom viewport,
with a previous/next stepper and a small metadata panel — all working
behind a port forward on a possibly-offline cluster. (The tab is named
**Browse**, not "View", to avoid collision with the existing Results
**Viewer** tab.)

## Goals

- A new **Browse** leaf tab (own `/browse/` WSGI mount), placed right after
  **Home** — it inspects _inputs_, conceptually upstream of the pipeline.
- **Deep-zoom** pan/zoom of large plate scans (gigapixel-capable), served
  tile-by-tile so the browser never downloads a full 100 MP image.
- **Works over an SSH tunnel on an offline cluster** — no CDN dependency
  (OpenSeadragon is already vendored in-package).
- A **two-dropdown picker**: a **dataset** dropdown listing every
  image-containing subfolder (by path relative to the source root, so
  arbitrary nesting collapses to one flat list) and an **image** dropdown
  listing that folder's images, with **‹ previous / next ›** stepper buttons
  on the image dropdown. The dataset dropdown is **hidden when the source
  root is flat** (images directly under it).
- **Mixed input formats:** standard (`.png/.tif/.tiff/.jpg/.jpeg`) tile
  directly; camera **RAW** (`.raw/.nef/.cr2/.arw/.dng`) is decoded through
  `phenotypic.Image` (rawpy) first.
- **Faithful rendering** — the image is shown as decoded (no auto-contrast
  / enhancement); what you see matches the true capture.
- A **metadata panel**: pixel dimensions, file size, and EXIF (capture
  time, camera) read from the original file.
- DZI tile + normalized-PNG cache is **ephemeral and out-of-tree**: written
  under a temp dir (`tempfile.gettempdir() / "phenotypic" / "browse"`), keyed
  per image by a base64url token, **wiped on launch and on shutdown**, and
  regenerated each session. Never writes into the user's data tree — no
  read-only-source problem, no litter to clean up by hand.

## Non-Goals (v1)

- **No multi-pane / side-by-side cards.** Single viewport for v1. (The
  card layout makes multi-pane a trivial later add; the Results Viewer
  already does multi-card comparison of _outputs_.)
- **No annotation / curation / flagging.** Pure viewing + metadata. No
  durable per-image state.
- **No editing, no processing preview, no overlays, no auto-contrast.**
  That is the Builder's job; Browse shows the raw input as decoded.
- **No multi-user / cloud deployment** (consistent with the hub's v1
  single-user, frozen-at-launch posture).
- **No thumbnail wall / contact sheet.** The picker is a dropdown, not a
  gallery (YAGNI; revisit if requested).

## Design Decisions (resolved in brainstorm)

| # | Decision | Rationale |
|---|----------|-----------|
| **F1** | **OpenSeadragon + DZI tiles** for the viewport | Only option delivering true deep-zoom on large scans; vendored OSD is offline-safe over a tunnel; the picker+‹›+card UI already exists in the Results Viewer. |
| **B1** | **Normalize → PNG, then reuse `_dzi_tiler` unchanged** | The tested tiler only ever sees PNG, so _all_ formats collapse to one uniform path. A new `_source_render.py` owns "any file → cached 8-bit RGB PNG". |
| **A1** | **Stateless, sandbox-relative tile URLs**; Browse sub-app is **eager** (no `ToolSession`) | The frozen `sandbox.root` is the security boundary; the source root is just a sub-path within it. No heavy parquet → no lazy session. The `localStorage` source store only drives _picker options_, never security or rebuilds. |
| **C1** | **Ephemeral temp cache** at `tempfile.gettempdir() / "phenotypic" / "browse" / <token>.{png,dzi}` (token = base64url of the image's sandbox-relative path) | Keeps the user's data tree clean; sidesteps read-only source mounts entirely; on HPC `$TMPDIR` points at fast node-local scratch. Token is globally unique within the sandbox → no collisions. Tiles regenerate each session. |
| **C2** | **Wipe-on-start + `atexit` wipe** (best-effort) | Wipe-on-start clears stale tiles from a crashed prior run (guarantees fresh tiles); `atexit` handles graceful shutdown. Cleanup never raises. |
| **U1** | **Two cascading dropdowns** (dataset folder → image; dataset hidden when flat); **single pane**; **view + metadata** | Per user choice. Two dropdowns sidestep Dash's missing `<optgroup>` and read cleanly on nested trees. |
| **R1** | **Faithful rendering** — no auto-contrast/enhancement; ‹/› **stops at dataset bounds** | Browse inspects the true capture; enhancement is out of scope. Stepping stays within the selected dataset (cross-dataset rollover deferred). |

## Proposed Design

A new self-contained sub-app package `src/phenotypic/gui/browse/`,
composed into the hub exactly like Builder / Run / Tune (eager Dash app
behind a `DispatcherMiddleware` mount). It reuses the Results Viewer's
deep-zoom machinery and the shell's source-root context; the only genuinely
new logic is "arbitrary file → tiles" and the two-dropdown source picker.

### Package layout (`gui/browse/`)

| File | Responsibility |
|------|----------------|
| `__init__.py` | Exports `create_app`. |
| `_app.py` | Dash factory: mount tile route, build layout, register callbacks, inject `window.__phenotypicAppPrefix` (same index-string trick as `results_viewer/_app.py`). |
| `_ids.py` | All component IDs (dataset dropdown, image dropdown, prev/next, OSD div, metadata chips, current-image + source-root mirror stores). |
| `_layout.py` | Single-pane layout: dataset dropdown + image picker group (‹ dropdown ›) + OSD canvas div + metadata panel. |
| `_callbacks.py` | Populate the dataset dropdown from the resolved source root; cascade dataset→image options (default first image); step prev/next; bounds-disable; drive the metadata panel; emit the JS-consumed "current image" payload. |
| `_source_lister.py` | Walk the source root → ordered `{dataset_rel: [filename, …]}` map (each distinct image-containing subfolder, sorted). Drives the two dropdowns. Reuses `IMAGE_EXTS`. |
| `_source_render.py` | **New core piece.** Normalize any source file → cached 8-bit RGB PNG (standard via PIL/pyvips or `Image`; RAW via `Image.imread`). Owns the original→PNG mtime check + the temp-cache path resolution + the wipe-on-start / `atexit` lifecycle. |
| `_tile_routes.py` | Flask blueprint: `GET /<BROWSE_TILES_PREFIX>/<token>.dzi` (+ `<token>_files/…` tiles). Validates + base64url-decodes the token, `sandbox.resolve`s the image, renders via `_source_render`, tiles via the shared `_dzi_tiler`. |
| `_metadata.py` | Pure: original file → `{width, height, bytes, exif:{...}}`. |
| `_assets/browse.js` | OSD mount/dispose (adapted from `results_viewer.js`). |
| `_assets/browse.css` | Minimal viewport + metadata styling (design tokens only). |
| `_assets/openseadragon/` | The **same vendored OSD** — symlink/copy, or load from the Results Viewer's assets via a shared static route (see Reuse). |

### Hub composition (`shell/_app.py`, `shell/_layout.py`, `shell/_ids.py`)

- **`shell/_ids.py`:** add `SHELL_TAB_BROWSE = "shell-tab-browse"`.
- **`shell/_layout.py`:** add `SHELL_TAB_BROWSE → MOUNT_BROWSE` to `_TAB_HREFS`,
  `"Browse"` to `_TAB_LABELS`, and insert `SHELL_TAB_BROWSE` into `NAV_MODEL`
  as a **leaf** immediately after `SHELL_TAB_HOME`.
- **`shell/_app.py` `compose_hub`:** build `browse.create_app(...)`
  **eagerly** (lightweight), `wrap_in_chrome(active_tab=SHELL_TAB_BROWSE)`,
  and add `MOUNT_BROWSE.rstrip("/"): browse_app.server` to the
  `DispatcherMiddleware` map. No `ToolSession`.
- The Browse app needs the **`sandbox`** (for path resolution) at build time —
  passed straight into `create_app` like the run console receives it.

### Source resolution + tile route (A1, stateless) — opaque token

Browse's image paths are **nested** (`plates/batch7/day3/A1_scan.nef`), and
Werkzeug percent-decodes `%2F` back to `/` before routing, so the Results
Viewer's two-segment `<dataset>/<stem>` scheme can't carry a nested path in
one URL segment. Instead the image's **sandbox-relative path** is encoded as
a single **slash-free base64url token**, which mirrors the proven
`<stem>.dzi` single-segment route exactly:

```
/browse{BROWSE_TILES_PREFIX}/<token>.dzi
/browse{BROWSE_TILES_PREFIX}/<token>_files/<level>/<col>_<row>.png
   token = base64url("<image path relative to sandbox.root>"), "=" stripped
   e.g.  token = "cGxhdGVzL2JhdGNoNy9kYXkzL0ExX3NjYW4ubmVm"
```

- The token is **slash-free** (`[A-Za-z0-9_-]+`), so both routes are simple
  single-segment captures — structurally identical to `_tile_routes.py`'s
  `/<stem>.dzi` and `/<stem>_files/<int:level>/<filename>`.
- **Decode + guard:** the blueprint validates the token against
  `^[A-Za-z0-9_-]+$`, base64url-decodes it to a relative path string, then
  calls `sandbox.resolve(rel)` — which raises `ValueError` on any `..` /
  symlink escape (→ 404). The frozen sandbox stays the sole security
  boundary; the source root is **not** in the URL at all (it only drives the
  picker's listing).
- OSD derives the `_files/...` tile URLs from the `.dzi` URL by string
  substitution, so reusing the same token in both routes keeps tile requests
  hitting the same cache entry.

### Cache layout + lifecycle (C1/C2)

The cache is **ephemeral and lives outside the data tree**, keyed by the
same `<token>` the tile URL uses (globally unique within the sandbox, so no
per-source digest or nesting is needed). For a process temp base
`T = tempfile.gettempdir()` (respects `$TMPDIR`; node-local scratch on HPC):

```
T/phenotypic/browse/<token>.png        # normalized 8-bit RGB (B1)
T/phenotypic/browse/<token>.dzi        # manifest (existing tiler)
T/phenotypic/browse/<token>_files/…    # tile pyramid (existing tiler)
```

- The normalized PNG is named `<token>.png`, so `_dzi_tiler.tile()` (which
  names its outputs after `png_path.stem`) emits `<token>.dzi` +
  `<token>_files/` — matching the tile URLs OSD derives. The tile route
  serves `T/phenotypic/browse/<token>_files/<level>/<file>` directly.
- **Lifecycle (C2):** `T/phenotypic/browse` is **wiped on launch**
  (clears stale tiles from a crashed prior run → guarantees fresh tiles per
  the "remake each time" requirement) and **wiped at shutdown** via
  `atexit`. Both wipes are best-effort (`shutil.rmtree(..., ignore_errors=True)`)
  and never raise. The wipe is wired once, in the Browse `create_app` (build
  → clear + recreate the dir; register the `atexit` cleanup).
- **In-session caching still applies** — one image is 1 manifest + N tile
  GETs, so the manifest endpoint must tile **once** and the tile GETs stream
  from disk. The existing mtime idempotency in `_dzi_tiler` provides this
  within a session; "remake each time" refers to **across launches**, which
  the wipe-on-start delivers. `_source_render` likewise mtime-checks the
  original→PNG step so re-selecting an image in the same session is free.
- **No read-only-source concern** — nothing is ever written under `S`.
  `/tmp`-class temp dirs are writable by definition; if even `T` is
  unwritable the viewer surfaces an inline error rather than corrupting the
  data tree.

### `_source_render.py` — the one new core algorithm

`phenotypic.Image.imread` already loads **every** supported format (standard
*and* RAW), and `skimage.util.img_as_ubyte` is the project's faithful
full-range 8-bit downcast (the same primitive the accessor IO handler uses
when saving). So the render is uniform — only RAW gets a graceful-degrade
guard:

```
normalize_to_png(original: Path, cache_png: Path) -> Path
    if cache_png exists and cache_png.mtime >= original.mtime: return cache_png
    try:
        rgb = Image.imread(str(original)).rgb[:]          # all formats; rawpy for RAW
    except Exception as e:
        if original.suffix.lower() in _RAW_EXTS:          # Windows: rawpy excluded
            raise SourceRenderUnavailable("cannot decode RAW on this platform") from e
        raise
    rgb8 = skimage.util.img_as_ubyte(rgb)                 # faithful full-range downcast
    cache_png.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(rgb8).save(cache_png, format="PNG")
    return cache_png
```

- `_RAW_EXTS = {".raw", ".nef", ".cr2", ".arw", ".dng"}` (subset of the
  shared `IMAGE_EXTS`).
- **Faithful (R1):** `img_as_ubyte` is a fixed full-range scale (uint16 →
  ÷257, float[0,1] → ×255), no per-image auto-contrast/percentile stretch;
  `Image.rgb` already applied the capture's gamma/illuminant, so the
  displayed image matches the true capture. Auto-contrast is a deferred
  future toggle, not v1. (The builder's `_image_renderer._normalize_to_uint8`
  / `_encode_png` are precedent; consider lifting to `gui/_shared` instead of
  re-rolling — decide in planning.)
- A `SourceRenderUnavailable` is caught in the tile route → JSON 415/422 +
  the JS shows an inline "cannot render this file on this platform" notice
  instead of a broken viewport.
- Memory: render is **lazy** (only on first view of an image) and bounded
  by one image at a time; the normalized PNG is the only full-res
  intermediate and it is written straight to disk.

### Picker — two cascading dropdowns (`_source_lister.py` + `_callbacks.py`)

- `_source_lister.list_datasets(source_root)` walks the resolved source root
  (bounded, no symlink escapes — mirror `_directory_browser`'s safe-walk
  posture), collecting files whose suffix is in `IMAGE_EXTS`, and returns an
  ordered `{dataset_rel: [filename, …]}` map. `dataset_rel` is the image's
  parent directory **relative to the source root** (`"."` for the root
  itself); arbitrary nesting collapses to one flat set of dataset keys.
- **Dropdown 1 (dataset):** options = the sorted `dataset_rel` keys; `value`
  = `dataset_rel`. Built once per resolved source root.
- **Dropdown 2 (image):** options = images of the selected dataset; `value`
  = `filename`. A cascade callback (Input: dataset value → Output: image
  options + default-select the first image) repopulates it on dataset change.
- **Flat-source rule:** when the only dataset key is `"."` (no
  image-containing subfolders), the dataset dropdown is hidden
  (`style={"display": "none"}`) and the image dropdown lists the root images
  directly.
- **Tile token:** the callback joins `src_root_rel` (source root relative to
  `sandbox.root`) + `dataset_rel` + `filename` into the image's
  **sandbox-relative** POSIX path, then `encode_token(...)` → the
  current-image `dcc.Store` payload `{token, label}` the JS reads.
- **Prev/next reuse:** ‹/› step **Dropdown 2** via `step_picker_value` +
  `picker_button_disabled_states` (within the selected dataset; **disabled
  at the dataset's first/last image** — cross-dataset rollover is a deferred
  nicety, R1). These helpers are pure and surface-agnostic — **lift to
  `gui/_shared/`** so `browse` doesn't import `results_viewer` (see Reuse).

### Single-pane layout + metadata panel

- One header row: **dataset dropdown** (hidden when flat) · ‹ button ·
  **image dropdown** · › button (the ‹ dropdown › cluster reuses the
  verbatim shape from `_viewer_card.py`'s `picker_group`).
- One `osd-canvas` div (the JS mounts OSD here on image change).
- A **metadata panel** beneath: chips/rows for `width×height`, file size
  (humanized), and EXIF capture-time + camera when present (omit rows that
  are absent). Driven by a callback that calls `_metadata.read(original)`.
- No details DataTable, no remove button, no Status column (those are
  Results-Viewer curation concerns, out of scope here).

### Metadata (`_metadata.py`)

- `width/height`: from the **normalized PNG** (already decoded) or a cheap
  PIL `.size` probe of the original.
- `bytes`: `original.stat().st_size`.
- `exif`: read from the freshly-loaded original via `Image.imread`'s
  imported metadata. **Note (memory):** EXIF lives in
  `image._metadata.imported` and is **dropped by `Image.copy()`**, so it
  must be read from the original load, never a copy. Surfaced fields:
  `DateTimeOriginal` (capture time) + `Make`/`Model` (camera). Degrade
  gracefully (empty dict) when EXIF is absent (PNG/synthetic) or unreadable.

### JS (`_assets/browse.js`)

- Adapt `results_viewer.js`'s OSD mount/dispose: read
  `window.__phenotypicAppPrefix`, build `…/tiles/<token>.dzi` from the
  current-image payload (a `dcc.Store` written by the picker callback), mount
  OSD on the single `osd-canvas` div, dispose the prior instance on change.
- **Loads OSD from the vendored copy** (`assets/openseadragon/...`,
  prefixed by `__phenotypicAppPrefix`) — no CDN round-trip, so an offline
  cluster gets no failed request + delay. (The Results Viewer is CDN-first
  with a vendored fallback; Browse goes vendored-only for tunnel
  reliability.) Constrained zoom; `showNavigator: false`.
- Show a transient "rendering…" state while the manifest request is in
  flight so a multi-second first paint (large RAW) doesn't look hung.

### Reuse vs. new (inventory)

**Reused unchanged:** `_dzi_tiler.tile()`; `is_safe_path_component`;
vendored `openseadragon/`; `IMAGE_EXTS`; `Image.imread` / `Image.rgb`;
`resolve_source_image_root` / `SourcePayload`; `wrap_in_chrome`; the
`DispatcherMiddleware` composition pattern; the index-string prefix-inject
trick.

**Reused after a small lift to `gui/_shared/`:**
`step_picker_value` / `picker_button_disabled_states` /
`enabled_picker_values` (move from `results_viewer/_picker_navigation.py`
to `gui/_shared/_picker_navigation.py`; re-export from the old path for
back-compat). Likewise consider lifting `IMAGE_EXTS` from
`builder/_directory_browser` to `gui/_config.py` and re-exporting, so
neither `browse` nor the classifier imports the builder package.

**New:** the `browse/` package files listed above (notably
`_source_render.py`, `_source_lister.py`, `_tile_routes.py`).

### New constants

In `gui/_config.py` (Python-identifier constants) unless noted:

| Constant | Value | Notes |
|----------|-------|-------|
| `MOUNT_BROWSE` | `"/browse/"` | New WSGI mount. |
| `BROWSE_TILES_PREFIX` | `"/tiles"` | Per-server, mirrors `VIEWER_TILES_PREFIX`. |
| `BROWSE_CACHE_TMP_SUBPATH` | `("phenotypic", "browse")` | Joined under `tempfile.gettempdir()`. Single-sourced; never hand-join. |
| `TITLE_BROWSE` | `"PhenoTypic Source Browser"` | Dash title. |
| `SHELL_TAB_BROWSE` | `"shell-tab-browse"` | In `shell/_ids.py`. |

Path/token helpers live in `browse/_source_render.py` so the temp join +
encoding exist in exactly one place:

- `browse_cache_base()` → `Path(tempfile.gettempdir()).joinpath(*BROWSE_CACHE_TMP_SUBPATH)`
  — the wipe target.
- `encode_token(sandbox_rel: str)` / `decode_token(token: str)` → base64url
  round-trip of the image's sandbox-relative POSIX path (`=` stripped/re-padded).
- `cache_png_path(token)` → `browse_cache_base() / f"{token}.png"` — the
  per-image normalized PNG the tiler consumes.

## Request lifecycle (end to end)

1. User selects a source root in the top bar → `SHELL_SOURCE_IMAGE_ROOT_STORE`
   (localStorage) updates with a validated `SourcePayload`.
2. Browse tab callback reads the store, `resolve_source_image_root()` → `Path`,
   `_source_lister.list_datasets()` → dataset-dropdown options (dataset
   dropdown hidden if the only key is `"."`).
3. User picks a dataset → cascade callback fills the image dropdown (default:
   first image). User picks an image (or clicks ‹/›) → the callback computes
   the image's **sandbox-relative path** → `encode_token(...)` → a `dcc.Store`
   "current image" payload `{token, label}` updates.
4. JS sees the payload change → builds `…/tiles/<token>.dzi` → mounts OSD.
5. OSD requests the manifest → tile route validates + decodes `<token>` →
   `sandbox.resolve(rel)`, `_source_render.normalize_to_png(...)` (lazy,
   cached in the temp dir), `_dzi_tiler.tile(<token>.png, browse_cache_base())`,
   returns the manifest; subsequent `<token>_files/…` GETs stream from cache.
6. In parallel, a metadata callback reads `_metadata.read(original)` →
   fills the metadata panel.

## Phasing

### Phase 1 — Backend core (P0, pure, unit-testable)
- `_source_render.normalize_to_png` (standard + RAW + mtime + faithful 8-bit
  downcast + `SourceRenderUnavailable`) + temp-cache path helpers +
  wipe-on-start / `atexit` lifecycle.
- `_source_lister` (recursive `{dataset_rel: […]}` listing, safe walk).
- `_metadata.read`.
- Cache-path helper + new constants.
- Lift `_picker_navigation` (+ optionally `IMAGE_EXTS`) to `gui/_shared` /
  `gui/_config` with back-compat re-exports.

### Phase 2 — Flask tile route (P0)
- `_tile_routes.register(app, sandbox)` with two-segment sandbox-relative
  resolution, reusing `is_safe_path_component` + `_dzi_tiler`.

### Phase 3 — Dash app + layout + callbacks (P0)
- `_app.create_app(sandbox, url_prefix)`, `_layout`, `_ids`, `_callbacks`
  (dataset/image dropdowns, cascade, prev/next, bounds-disable,
  current-image store, metadata panel), index-string prefix inject.

### Phase 4 — Frontend JS/CSS (P0)
- `browse.js` OSD mount/dispose; `browse.css`; vendored OSD wiring.

### Phase 5 — Hub composition + nav (P0)
- `MOUNT_BROWSE`, `SHELL_TAB_BROWSE`, `NAV_MODEL` leaf after Home, eager
  mount in `compose_hub`.

### Phase 6 — Ledgers, docs, screenshots (P0, CI-gated)
- `FEATURES.md` rows; `WORKFLOWS.md` row + `_capture_browse_*` +
  `docs/source/tutorials/gui/` page; regenerate screenshots.

## Testing

- **`_source_render` / cache lifecycle:** standard PNG/TIFF/JPEG → cached
  PNG under the temp base; RAW path (skip/xfail where `rawpy` is
  unavailable, e.g. Windows) → either a PNG or `SourceRenderUnavailable`;
  faithful downcast (no stretch) on a 16-bit fixture; mtime-staleness
  re-render; distinct images get distinct `<token>.png` cache files;
  wipe-on-start clears a pre-seeded stale dir; `atexit` cleanup removes the
  base (best-effort, swallows errors). Point the temp base at a `tmp_path`
  in tests — never touch the real `gettempdir()`.
- **`_source_lister`:** recursion into `{dataset_rel: [filename,…]}`, `"."`
  key for root images, flat-vs-nested detection, suffix filtering,
  symlink-escape refusal, empty dir. Plus the cascade + flat-hide callback
  logic (dataset change → image options + default selection).
- **`_metadata`:** dims/bytes; EXIF present vs absent; reads from original
  (not a copy).
- **`_tile_routes`:** token round-trip (`encode`∘`decode == identity`);
  malformed token → 404; a token decoding to a `..`/escape path → 404 (via
  `sandbox.resolve`); valid request → manifest + a tile; RAW-unavailable →
  the chosen 4xx; uses `app.server.test_client()` through `wsgi_app`.
- **Picker navigation:** the lifted `_shared/_picker_navigation` keeps its
  existing unit tests; add Browse-specific option-shape tests.
- **Composition smoke:** hub builds with the `/browse/` mount; `GET /browse/`
  returns the chrome-wrapped layout; nav shows the Browse tab as active.
- **Live callback check (per memory `gui_review_verify_with_browser`):**
  drive the picker + ‹/› + a zoom with Playwright MCP against a running
  hub, tail the log — callback-wiring/`500` bugs only surface on
  `/_dash-update-component`, not unit tests. Extract callback bodies into
  module-level helpers to keep them unit-testable.

## CI / docs gates (touches `src/phenotypic/gui/`)

- **`FEATURES.md`** — every new affordance (Browse tab anchor, dataset +
  image dropdowns, prev/next, OSD viewport, metadata panel, tile route,
  current-image store, each callback) gets a row; the `features-md-gate`
  job rejects the PR otherwise. `✅ shipping` rows need a `Test ref`.
- **`WORKFLOWS.md`** — the end-to-end "browse source images" flow gets a
  row, a `_capture_browse_*` function in
  `scripts/capture_gui_tutorial_screenshots.py`, and a walkthrough page
  under `docs/source/tutorials/gui/`; `workflows-md-gate` enforces the
  round-trip.
- Regenerate the **full** screenshot set on a workstation and **commit all**
  refreshed PNGs (font-noise churn is expected and accepted).

## Risks & Open Questions

- **Cross-dataset stepping (R1, decided).** v1 ‹/› stays within the selected
  dataset (disabled at its first/last image). Rollover into the
  next/previous dataset is a deferred nicety; recorded so the bounds-disable
  behaviour reads as a deliberate choice, not an oversight.
- **Very large RAW decode latency.** First view of a big RAW pays a rawpy
  decode + PNG write + tiling cost. Acceptable (lazy + cached); the JS
  "rendering…" state covers the multi-second first paint.
- **`IMAGE_EXTS` location.** Importing it from `builder/_directory_browser`
  couples `browse` → `builder`. Prefer lifting to `gui/_config.py` with a
  back-compat re-export; flagged as a Phase-1 task, not a blocker.
- **Vendored OSD sharing.** Cleanest is one shared static route serving the
  single vendored copy to every sub-app rather than duplicating
  `openseadragon/` bytes per package; decide in planning
  (`register_shared_static` already exists in `gui/_shared`).
- **Sandbox vs. source root containment.** `resolve_source_image_root`
  already guarantees the source root is inside the sandbox; the tile route
  re-checks. If a future deployment allows source roots _outside_ the
  sandbox, the A1 URL scheme needs revisiting — out of scope for v1.
- **Ephemeral cleanup edge cases.** `atexit` does not run on `SIGKILL` /
  hard `SIGTERM` (e.g. SLURM job cancel), so tiles can survive a kill;
  wipe-on-start of the next launch covers it, and the OS clears temp on
  reboot. Two concurrent hub processes would share `T/phenotypic/browse`, so
  one's exit-wipe could delete the other's in-session tiles — out of scope
  for single-user v1, but if it ever matters, namespace the base by PID
  (`…/browse/<pid>/<token>…`) and wipe only the own-PID subtree.
