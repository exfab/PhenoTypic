# Source Timeline View — Design Spec

**Date:** 2026-06-18
**Status:** Draft for review
**Surfaces:** Browse (`/browse/`) + Results viewer (`/results/`)
**Worktree/branch:** `source-timeline-view`

---

## 1. Purpose

Add a **two-dimensional, scrollable matrix view** of a full image set:

- **Horizontal axis = time** (the time-ordered series of a plate). Lets the user
  scan a plate's time-course to **find the ideal starting time for image
  processing** and **identify traits as they emerge**.
- **Vertical axis = a selectable grouping** (dataset / plate / media / replicate),
  so a column-of-plates can be compared across media or replicates.

Two surfaces render the same engine over different sources:

- **Browse** → **source images** (pre-processing; raw folders).
- **Results** → **overlay images** (post-processing; measured plates).

The matrix must scale to large sets without loading every image into the browser:
images are **downscaled to thumbnails**, the visible window is **virtualized**, and
the full set is **pre-cached in the background** after the page opens. Full-
resolution / deep analysis is **on demand** via a single-image pop-out and a
bounded, viewport-**synced "Compare" strip**.

---

## 2. Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Axis model | **X fixed = time; Y selectable** (dataset/plate/media/replicate). |
| D2 | Browse axis source | **Metadata CSV drives axes** when loaded; **folder/EXIF filmstrip fallback** + a dismissible "add CSV" nudge when not. |
| D3 | Phasing | **Both surfaces** in one spec, over a shared engine. |
| D4 | Interaction model | **Hybrid (C):** static downscaled matrix + single-image pop-out + bounded synced "Compare" strip. **(Scroll model superseded by §16 — focus-and-navigate.)** |
| D5 | Sync scope | **Both** row-header (one plate's time-course) **and** arbitrary multi-select cells. |
| D6 | CLI metadata copy | **Yes**, low priority, non-blocking: copy `--metadata` CSV → `deliverables/metadata.csv`. |
| D7 | Browse no-CSV behavior | **Folder/EXIF filmstrip + CSV nudge** (not a hard empty state). |
| D8 | Plate identity (no-CSV) | **Plate-identity pattern** — `{plate}`/`{time}` placeholder syntax (primary) + raw-regex advanced toggle, with live preview. |
| D9 | Results no-time-metadata | **Guided empty state** prompting `--metadata` / a time field. |

### Interaction-model rationale (the cost analysis)

Fully-synced **live deep-zoom tiles across the whole matrix is not viable**:

- Each OpenSeadragon viewer carries its own canvas/WebGL context; browsers cap
  live WebGL contexts (~16 in Chrome) — beyond that the oldest are dropped.
- Syncing N viewports is **O(N) work per animation frame** during a pan/zoom.
- Each deep-zoom tile needs its own DZI pyramid (the multi-second normalize+tile
  cost Browse already pays once per image) — a full-matrix live view front-loads
  hundreds of pyramids.

Therefore the synced-zoom capability is a **bounded, opt-in mode over a small
selection (≤ ~12 live viewers)**, while the default matrix is cheap static
`<img>` thumbnails with a single-image pop-out for full-res work.

---

## 3. Architecture overview

```
                       ┌────────────────────────────────────────────┐
                       │           gui/_shared/timeline/              │
                       │  _matrix.py   pure: records → ordered grid   │
                       │  _thumbnail.py downscale+cache+route factory │
                       │  _grid.py     pure: virtualized CSS-grid DOM │
                       │  timeline.js  windowing + warm + synced OSD  │
                       └───────────────┬───────────────┬─────────────┘
                                       │               │
                  resolver(records)    │               │  resolver(records)
                                       │               │
        ┌──────────────────────────────┘               └───────────────────────────┐
        │  Browse surface                                  Results surface           │
        │  browse/_timeline_*                              results_viewer/timeline_view/
        │   - source: files on disk                         - source: OutputRoot.master_df
        │   - axes: CSV cols, else folder/EXIF + pattern    - axes: X=time-col, Y=selectable_axis_columns
        │   - thumb route: normalize_to_png → downscale     - thumb route: overlay PNG → downscale
        │   - deep-zoom pop-out: browse DZI route           - deep-zoom pop-out: viewer DZI route
        └────────────────────────────────────────────────────────────────────────────┘
```

**Isolation principle:** the engine is source-agnostic. Each surface supplies a
**resolver** that maps `(row_value, time_value) → a concrete image source` and a
**thumbnail route** that turns that source into a cached downscaled PNG. The
matrix model, virtualization, background warm, and synced-strip controller are
written once and reused.

---

## 4. Shared engine — `gui/_shared/timeline/`

Mirrors how `gui/_shared/tiles.py` is shared across the colony-view and QC tabs.

### 4.1 `_matrix.py` (pure)

```
build_matrix(records, *, time_key, row_key) -> TimelineMatrix
```

- `records`: an iterable of dict-like rows, each carrying at least
  `row_value`, `time_value`, and an opaque `cell_ref` (the surface's identity:
  a Browse token, or a `(dataset, stem)` pair).
- Produces:
  - `columns`: sorted-unique `time_value`s in **chronological order** via
    **value coercion at sort time** (try numeric → try datetime → else lexical).
    This is mandatory because the stored dtype is unreliable: `join_metadata`
    casts join-key columns to `pl.String` (`_cli_output_manager.py:113`), so a
    conceptually-numeric `Metadata_Time` of `"1","2","10"` would otherwise sort
    `1<10<2`. See §15.3.
  - `rows`: sorted-unique `row_value`s.
  - `cells`: `(row_value, time_value) -> [cell_ref, ...]` with a deterministic
    **representative** (first by a stable sort) and a member count.
- Parallels the colony grid's `_representative_per_cell` (representative + `N=k`).
- Empty `(row, time)` combinations are valid and render as size-matched
  placeholders downstream.

### 4.2 `_thumbnail.py` (route factory + cache)

```
register_thumbnail_route(app, *, segment, resolve_source, cache_base) -> None
downscale_to_thumb(src_png: Path, size: int) -> bytes   # pure
thumb_cache_path(cache_base, identity, size) -> Path
```

- `GET /<segment>/<identity...>?size=<bucket>` → PNG bytes of the source
  downscaled to `size` (longest edge), preserving aspect ratio.
- `resolve_source(identity) -> Path | ThumbUnavailable`: surface-supplied; maps
  the URL identity to an on-disk source PNG (Browse: normalized source PNG;
  Results: overlay PNG). Returns a sentinel when unavailable (e.g. RAW on
  Windows) → route answers **422 + a fixed client message**, mirroring the DZI
  route's `SourceRenderUnavailable` handling.
- **Size buckets**: a small fixed set (e.g. `64,96,128,192,256`) so the cache key
  space is bounded; the requested display size snaps **up** to the nearest bucket
  (never upscale a thumbnail).
- **Disk cache** keyed by `(identity, size_bucket, source_mtime)`; idempotent;
  per-source `threading.Lock` (reuse the `_dzi_tiler._get_lock` LRU pattern) so
  concurrent requests for the same thumbnail don't double-render.
- Cache location:
  - Browse → ephemeral temp (extend `BROWSE_CACHE_TMP_SUBPATH`; wiped on launch +
    atexit, same as the DZI cache).
  - Results → under the output root's `VIEWER_CACHE_DIRNAME` (persists with the
    run, like the viewer's other caches).

### 4.3 `_grid.py` (pure render)

```
build_timeline_grid(matrix, *, url_builder, display_size, surface, ...) -> Component
```

- Renders a CSS grid sized to the **full** matrix (top row = time headers, left
  column = row headers, corner empty) so scrollbars are correct.
- **Every data cell is emitted as a size-matched placeholder `<div>` carrying
  `data-src` (the thumbnail URL), `data-cell-ref`, `data-row`, `data-col`** — and
  per-cell chrome: a pop-out button, a multi-select checkbox (reuse the
  `data-key` convention from `build_tile_cell`), and an `N=k` badge for
  multi-member cells. The `<img>` is **not** in the initial DOM.
- Reuses axis-label rendering conventions from the colony grid
  (`_build_axis_label`: mono font, wrap, width-capped).

### 4.4 `timeline.js` (asset)

> **Revised (§16):** the scroll-based `IntersectionObserver` virtualization below is
> **superseded by the focus-and-navigate controller in §16**. The tile-size stepper,
> size buckets, background-warm, and synced-strip controller carry over; the
> mount/unmount **trigger** changes from viewport-intersection to **focus-distance**
> (mount the focused cell's neighborhood + a margin ring; offload beyond).

- **Virtualization:** an `IntersectionObserver` over the placeholder cells with a
  root margin of ~1 screen. On enter → set `img.src = data-src` (mount). When a
  cell leaves by more than the margin → drop the `<img>` (unmount) to bound
  decoded-image memory. This is the "N images in each direction" window.
- **Memory bound (the answer to "how many tiles at once"):** mounted `<img>`
  count is bounded by **viewport + margin, independent of total matrix size** —
  `mounted ≈ visible × (1 + 2·margin_screens)²`. On a 1600×900 viewport at the
  150px default that is ~66 visible / ~264–594 mounted; decoded memory
  ≈ `tiles × tile_px² × 4 B`, which stays in the **tens of MB** across tile sizes
  (smaller tiles are more numerous but cheaper, so the product self-balances). A
  **hard LRU mount cap** (default ~400 `<img>`, configurable; evict
  least-recently-visible past the cap) is the absolute ceiling regardless of
  viewport/margin — worst case ≈ 400 × 256 KB ≈ ~100 MB at the largest tiles.
  Off-window cells stay empty placeholder `<div>`s carrying only a `data-src`
  string (zero image bytes), so a 100-image and a 100 000-image matrix have the
  same browser footprint.
- **Tile size (`like the colony view`):** a `−`/`+` stepper + px readout reusing
  the colony pattern (`step_colony_tile_size` semantics: default 150, step 16,
  range [64, 400]). The stepper sets the CSS **display size**; the fetched
  thumbnail snaps **up** to the nearest §4.2 bucket so per-tile bytes track the
  display size (unlike the colony crop, which ships one full-res crop and only
  CSS-scales — a whole-plate thumbnail is too large for that). Within a bucket,
  stepping is pure CSS scaling (no re-fetch); crossing a bucket re-fetches a
  genuinely-downscaled PNG.
- **Background warm:** after first paint, a throttled, bounded-concurrency loop
  `fetch()`es every cell's thumbnail URL (low priority; body discarded) so the
  server cache is warm before the user scrolls there. Generalizes
  `browse.js`'s neighbor-prefetch. Concurrency + on/off configurable.
- **Synced "Compare" strip controller:** given ≤ ~12 `cell_ref`s, mount that many
  OSD viewers bound to a **shared viewport** (one viewer's `viewport` change is
  propagated to the others, guarded against feedback loops). Tear down on close.
- Loaded per-surface from each surface's vendored OSD copy (no CDN).

---

## 5. Surface: Browse

### 5.1 Placement & layout

- New header control: a **`Single | Timeline` segmented toggle**. `Single` is
  today's OSD pane (unchanged). `Timeline` swaps the body to the matrix. Both
  bodies stay mounted (CSS visibility) so toggling is instant.
- Timeline body: Y-axis source control, tile-size stepper (colony-style `−`/`+`),
  the plate-identity
  pattern controls (no-CSV mode), the CSV nudge banner (no-CSV mode), and the
  shared grid.

### 5.2 Axis derivation — per-axis source picker (refines D2; see §15.4)

File layouts vary (encoded-filename / time-foldered / plate-foldered), so there
is **no single correct key** and a hardcoded path column is unrealistic (paths
are session-variable). Instead each axis names its own source:

- **Row (plate) source:** `Dataset folder` | `Filename pattern {plate}` |
  `CSV column`.
- **Time (column) source:** `EXIF capture time` | `Filename pattern {time}` |
  `CSV column`.

Rules:

- **Default** (and the no-CSV path): row = `Dataset folder`, time = `EXIF →
  filename`. Always works with zero setup. A dismissible banner when no CSV is
  loaded: *"Add a metadata CSV for richer time × group axes."*
- **Pattern source** reveals the §5.3 plate-identity pattern controls. The row
  key is always **folder-scoped**: `(dataset_folder, {plate})` — identical
  `{plate}` strings in different folders are **separate rows** (D-15.5).
- **CSV column source** reveals a CSV-column dropdown for that axis. The CSV is
  joined to files by **image name, scoped within the dataset folder** — because
  `(folder, filename)` is unique (filenames are unique within a folder), this
  needs **no path column** and cannot collide. A designated CSV image-name column
  (default: the existing `METADATA.IMAGE_NAME` convention) supplies the match;
  non-unique matches surface a warning. Reuses
  `read_metadata_row_for_image_stem` conventions, folder-scoped.
- Mixed sources are allowed (e.g. row = CSV `media` column, time = EXIF).

### 5.3 Plate-identity pattern (D8)

- **Pure helper** `parse_plate_identity(stems, pattern, *, advanced=False) ->
  list[PlateMatch]` where `PlateMatch = (stem, plate | None, time | None)`.
- **Placeholder syntax (primary):** `{plate}` (required), optional `{time}`, `*`
  wildcard, literal text. Compiled to an **anchored, non-greedy** regex:
  `{plate}→(?P<plate>.+?)`, `{time}→(?P<time>.+?)`, `*→.*?`, literals escaped,
  wrapped `^…$`. Example: `{plate}_t{time}` on `Exp1_PlateA_t03` → plate
  `Exp1_PlateA`, time `03`.
- **Advanced toggle:** raw regex with named groups `plate` (required) / `time`
  (optional). Invalid regex → inline error, grid unchanged.
- **Time ordering:** `{time}` captures numeric-sorted when all-numeric, else
  lexical; absent → EXIF → filename.
- **Unmatched** stems → one `"unmatched"` row with a count (never silently
  dropped).
- **Live preview:** a sample table of the current folder's files → extracted
  `(plate, time)` with matched/unmatched counts, updating as the pattern changes
  (thin Dash callback over the pure helper).

### 5.4 Thumbnail + pop-out

- Thumbnail route on the browse server (e.g. `/thumb/<token>?size=`),
  token-keyed + sandbox-resolved exactly like the DZI route; `resolve_source`
  reuses `_source_render.normalize_to_png` (RAW-aware) then downscales.
- Pop-out reuses Browse's existing DZI route + OSD lifecycle.

---

## 6. Surface: Results viewer

### 6.1 Placement

- New **6th tab `Timeline`** (`TAB_TIMELINE_ID`) alongside Plate/Colony/QC/
  Heatmap/Error — kept mounted (CSS switch), consistent with the others.
- New package `results_viewer/timeline_view/` paralleling `colony_view/`:
  `_layout.py`, `_callbacks.py`, `_grid` adapter, `_thumb_routes.py`, `_ids.py`.

### 6.2 Axes

- Source = `OutputRoot.master_df` (the post-applied mirror, which already carries
  joined `Metadata_*` columns — see §8).
- **Y dropdown** = `selectable_axis_columns(master_df, output_root.column_value_sets)`
  — the existing helper takes the precomputed value-set mapping as its **second
  positional arg** (not a one-arg call); `max_cardinality` defaults to 50, which
  is fine for the row axis. See §15.1.
- **X = time** uses a **dedicated, cardinality-uncapped** time-column predicate
  (`selectable_time_columns`): name matches `Metadata_Time`-like OR numeric/
  datetime dtype, **no 50-value cap** (a long time-course is the whole point).
  A separate `is_large_time_axis(n, threshold≈100)` predicate fires the
  bucketing-warning banner. See §15.1–§15.2.
- Honors the active filter sidebar (same `master_df` slice as the other tabs).

### 6.3 No-time-metadata empty state (D9)

- When no eligible time column exists (run had no `--metadata` and no
  time-producing post step), the tab renders a guided empty state:
  *"The Timeline needs a time field. Re-run with `--metadata <csv>` (or add a
  post step like `ExpandMetadata`) so a column such as `Metadata_Time` is
  available."*

### 6.4 Thumbnail + pop-out + crops

- Thumbnail route downscales the overlay PNG
  (`results/<dataset>/overlays/<stem>.png`), reusing the `_load_overlay_rgb` LRU.
- Pop-out reuses the viewer's existing overlay DZI route
  (`results_viewer/_tile_routes.py`) + OSD.

---

## 7. Synced "Compare" strip (D4, D5)

- **Triggers:** (a) **row-header click** → strip of that row's full time-course;
  (b) **multi-select cells** (shift/ctrl-click) → arbitrary set. Both surfaces.
- **Bounded:** hard cap ~12 live OSD viewers (under the ~16 WebGL ceiling). If a
  selection exceeds the cap, show *"Showing first 12 of N — narrow the
  selection"*; never silently truncate without a notice.
- **Shared viewport:** pan/zoom in any viewer propagates to the rest, guarded
  against feedback loops.
- Hosted in a dedicated panel/modal; tear down all viewers on close.

---

## 8. CLI: `deliverables/metadata.csv` (D6 — low priority, non-blocking)

### 8.1 Current behavior (verified)

- The `--metadata` CSV path is recorded only in `progress/job_metadata.json`
  (`JobMetadataKey.METADATA_CSV`) — an absolute path on the run machine.
- `finalize_post_master_outputs` → `join_metadata()` **inner-joins** the CSV
  columns onto the working frame; `_seed_measurements` writes the result to
  `deliverables/measurements.parquet` (the mirror).
- `OutputRoot.master_df` loads that mirror → **all CSV `Metadata_*` columns are
  already available to the Results Timeline**. The core Results axes need **no**
  new copy.

### 8.2 Why still add the copy

- The join is **inner** (metadata on the left): measurement rows with no matching
  metadata key are **dropped from the mirror**; the full original mapping
  survives only in the user's file.
- `job_metadata.json` holds an absolute path useless once results move off the
  cluster — no portable, co-located metadata artifact exists.

### 8.3 Change

- In `finalize_post_master_outputs`, a **best-effort** `shutil.copy` of the
  `--metadata` CSV → `deliverables/metadata.csv`, guarded like the other finalize
  side effects (failure logged, never raises).
- New `phenotypic.sdk_._io_constants` name + path helper
  (`DELIVERABLES_METADATA_CSV`, `metadata_csv_deliverable_path(output_dir)`).
- **Not** added to the mid-run chunk writer (`_aggregate_chunks_locked`) — matches
  the existing finalize-only convention for the post pipeline / per-feature
  splits / analysis chain.

---

## 9. Constants & design tokens

- `_config.py` (new): thumbnail URL segments, thumbnail size buckets, virtualization
  window margin, **LRU mount cap**, **tile-size stepper default/step/min/max**
  (mirroring `COLONY_TILE_SIZE_*` — shared helper if practical),
  background-warm concurrency, Compare-strip cap. Plus
  `TAB_TIMELINE_ID` lives in `results_viewer/_ids.py`; Browse view-mode + pattern
  ids in `browse/_ids.py`.
- `_design.py` (new): default tile sizes, grid gaps. No re-spelled literals; per the
  GUI module guide a literal used in ≥2 files moves to `_config.py`/`_design.py`.
- CLI filename → `phenotypic.sdk_._io_constants` (`DELIVERABLES_METADATA_CSV`),
  re-exported through `gui/_config.py` per the existing convention.

---

## 10. Testing

- **Unit (pure):**
  - `_matrix.build_matrix` — chronological ordering (numeric/datetime/lexical),
    representative selection, multi-member counts, empty cells.
  - `_thumbnail.downscale_to_thumb` + cache-key derivation; 422 sentinel path.
  - `parse_plate_identity` — placeholder→regex compile, named groups, numeric vs
    lexical time, unmatched bucket, invalid-pattern error.
  - Browse per-axis source resolution (folder / pattern / CSV-column for each of
    row & time; folder-scoped image-name join + collision warning); Results
    time-column eligibility + empty-state predicate.
- **Live browser (Playwright)** — required because Dash callback wiring is only
  trustworthy when driven live (per project memory): virtualization mount/unmount
  on scroll, background warm, synced-strip viewport binding, pop-out, pattern
  live preview, view-mode toggle.
- **CLI:** `deliverables/metadata.csv` presence on a metadata run; best-effort
  failure isolation when the source CSV is unreadable.

---

## 11. CI gates (mandatory)

- **`gui/FEATURES.md`** — a row per affordance: Browse view-mode toggle, Timeline
  tab, Browse **row-source selector** + **time-source selector** + CSV
  image-name/column dropdowns, Results Y dropdown + time-column selector,
  tile-size stepper, plate-identity pattern input, advanced-regex toggle, pattern
  preview, CSV nudge banner, pop-out button, Compare strip, `N=k` badge, empty
  states. `features-md-gate` rejects a `gui/`
  PR that doesn't touch FEATURES.md; `✅ shipping` rows need a resolvable
  `path::test`.
- **`gui/WORKFLOWS.md`** + `scripts/capture_gui_tutorial_screenshots.py`
  `_capture_<id>` + `docs/source/tutorials/gui/` pages for the end-to-end flows:
  (a) Browse — *find the ideal starting time* (incl. the plate-identity pattern);
  (b) Results — *trait emergence over time*. `workflows-md-gate` enforces the
  round-trip. Regenerate + commit the **full** screenshot set (don't cherry-pick).

---

## 12. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| WebGL context ceiling (~16) | Default grid uses plain `<img>`, not OSD; Compare strip hard-capped ≤ ~12 with a notice. |
| Thumbnail-generation storm on open | Bounded-concurrency warm + size-bucket disk cache + per-source lock. |
| Unique-timestamp column explosion | Virtualization bounds the DOM; warn + offer time-bucketing when columns are very numerous. |
| Browse with no CSV | Folder/EXIF filmstrip + plate-identity pattern + nudge — never a dead end. |
| Ambiguous/invalid pattern | Live preview + matched/unmatched counts + inline regex error; grid unchanged on error. |
| RAW on Windows | Thumbnail route returns the same 422 + inline notice the DZI route already does. |
| Inner-join drops unmatched rows from mirror | `deliverables/metadata.csv` copy preserves the full original mapping (D6). |
| Per-scroll Dash callbacks (latency) | Structure rendered once server-side; all windowing/warm/sync in `timeline.js` — zero per-scroll round-trips. |

---

## 13. Phasing (single spec, sequential build)

1. **Shared engine** — `_shared/timeline/` (`_matrix`, `_thumbnail` route factory,
   `_grid`, `timeline.js` virtualization + warm). Unit tests for pure parts.
2. **Browse surface** — view-mode toggle, resolver, per-axis source picker
   (folder / pattern / CSV-column) + folder-scoped CSV join + plate-identity
   pattern + preview + nudge, thumbnail route.
3. **Results surface** — Timeline tab, resolver, Y dropdown, time-column gating +
   empty state, thumbnail route.
4. **Synced Compare strip** — both surfaces (row + multi-select).
5. **CLI** — `deliverables/metadata.csv` copy + sdk_ helper.
6. **Docs/CI** — FEATURES.md, WORKFLOWS.md, tutorials, screenshot capture; final
   code-simplifier pass + regression run on touched areas.

---

## 14. Out of scope (YAGNI)

- Full-resolution rendering of *every* tile inline (this is the non-viable
  Approach B; full res is on-demand only).
- Editing/curation from the Timeline (it is a viewing/scan surface; curation stays
  on the colony/QC tabs).
- Time-bucketing UI beyond a warning + simple control if column counts explode
  (can be a fast-follow).
- A standalone Timeline mount (it lives inside Browse and Results).

---

## 15. Resolutions from spec review (2026-06-18)

A `plan-reviewer` pass verified the spec against the code and surfaced errors +
underspecified contracts. Resolutions below are binding for the implementation
plan. (References like `file:line` are at review time.)

### 15.1 `selectable_axis_columns` arity (was a factual error)
Real signature: `selectable_axis_columns(df, column_value_sets, max_cardinality=50)`
(`colony_view/_grid.py:115`). The Y dropdown calls it with
`output_root.column_value_sets` as the 2nd positional arg. A one-arg call (as the
draft implied) raises `TypeError`.

> **Superseded by §16.5:** the **50-value cardinality cap is dropped for the
> timeline Y axis.** Focus-windowing (§16.3) bounds mounts regardless of axis size,
> so the timeline calls an **uncapped** row-axis predicate (`max_cardinality=None`
> or a large bound), making high-cardinality groupings like `Metadata_PlateNum`
> (74 values in the UCR_029 reference set) selectable. The colony-view caller keeps
> the 50 default; only the timeline's call changes.

### 15.2 Time axis is cardinality-uncapped + name/dtype-gated
Add `selectable_time_columns(df, column_value_sets)` (new helper, likely in the
shared engine or `timeline_view/_grid`): eligible = name matches a
`Metadata_Time`-like pattern **or** numeric/datetime dtype; **no cardinality
cap** (the 50-cap in `selectable_axis_columns` would silently hide any
≥51-timepoint course). `is_large_time_axis(n_values, threshold≈100)` gates the
bucketing-warning banner. Heatmap's hardcoded `"Metadata_Time"`
(`_heatmap_tab/_callbacks.py:367`) is the name-pattern seed, not the mechanism.

### 15.3 Time sort = coerce-at-sort, not by stored dtype
`build_matrix` sorts `time_value`s by **trying numeric, then datetime, then
lexical** — never by raw stored dtype. Rationale: `join_metadata` casts join-key
columns to `pl.String` (`_cli_output_manager.py:113`), and CSV-brought / EXIF /
post-derived columns each arrive with different dtypes, so the stored type can't
be trusted. Document the coercion in `build_matrix`'s docstring. Pure-module
change, no UI impact.

### 15.4 Browse axes = per-axis source picker; CSV joins folder-scoped by image name (no path column)
**Supersedes the draft's "require a sandbox-relative path column" — withdrawn.**
Paths are session-variable and metadata CSVs are authored against plate/sample
IDs, not file locations, so a path column is unrealistic. The correct key is also
**convention-dependent** (encoded-filename / time-foldered / plate-foldered), so
no single hardcoded key works. Resolution (chosen): a **per-axis source picker**
(§5.2). Each axis draws from `Dataset folder` | `{plate}`/`{time}` pattern |
`CSV column`. When a CSV column is the source, the CSV joins to files by **image
name scoped within the dataset folder**; `(folder, filename)` is unique
(filenames are unique within a folder), so this is unambiguous and needs **no
path column**. A non-unique image-name match surfaces a warning. Scope: this join
is **Browse-with-CSV only** — Results joins at CLI finalize (master_df rows are
already per-image), and Browse-no-CSV derives axes straight from folder/pattern/
EXIF.

### 15.5 Plate-identity pattern compilation rules (D8)
Lock these in `parse_plate_identity`:
- `{plate}` is **required**; absence → inline error (placeholder path, not only
  the advanced-regex path).
- **Duplicate** `{plate}`/`{time}` tokens → inline compile error (Python `re`
  rejects duplicate group names anyway).
- Pattern matches against `Path(filename).stem` (no directory, no extension).
- `{plate}→(?P<plate>.+?)`, `{time}→(?P<time>.+?)`, `*→.*?`, literals escaped,
  wrapped `^…$` (non-greedy + anchored). Wildcard/group adjacency ambiguity is
  accepted and **mitigated by the live preview**, not by clever quantifier rules.
- **Folder-scoped rows (D-15.5, user-confirmed):** when a pattern is active the
  row key is `(dataset_folder, {plate})` — the folder is **kept**, not ignored.
  Identical `{plate}` captures in different folders are **separate rows** (no
  cross-folder merge).

### 15.6 Thumbnail cache = self-invalidating filename + atomic write
Cache filename embeds source mtime: `<identity>_<bucket>_<mtime_ns>.png`
(self-invalidating; no stat-then-compare, immune to second-granularity mtime
truncation). Writes are **atomic** (`tempfile` + `os.replace`, matching the
staged-GPU sidecar) so the `_get_lock` LRU-eviction race can't surface a partial
PNG. The thumbnail route does **not** lean on the 8-entry `_load_overlay_rgb` LRU
(`tiles.py:63`) for the warm sweep (too small for 100+ distinct images); it
decodes the source and relies on the disk cache.

### 15.7 Tab re-mount handling (was the highest-impact risk)
`dbc.Tabs` here does **not** keep inactive tab content stably mounted; the
existing `results_viewer.js` (≈ lines 405-413, 523) polls for its container and
**re-attaches on re-mount (tab switch / re-render)**. The timeline JS follows the
same idiom for its `IntersectionObserver`. `display:none → visible` transitions
re-fire IntersectionObserver, so lazy mount works on tab activation. Because the
thumbnail cache is **server-side on disk**, re-entry is warm regardless of the JS
observer lifecycle — **no `dcc.Store` warm-state rehydration is needed** (simpler
than the reviewer's suggestion).

### 15.8 Background-warm lifecycle = generation-ID guard
Each warm run carries a monotonic generation id; the loop breaks when
`generationId !== currentGenerationId` (matrix rebuild on Y/time/pattern change,
or navigation away). Prevents an old matrix's warm fetches racing a new matrix.
Browse warm defaults to **concurrency 2** (RAW `normalize_to_png` is heavy);
first-open latency scales with the count of unique source images — documented,
not hidden.

### 15.9 Dash id / `data-*` namespacing
All timeline pattern-matched ids use `type` strings prefixed `timeline-`
(`timeline-cell`, `timeline-popout`, …) so they never cross-fire with colony/QC
pattern callbacks. Timeline JS event listeners are scoped to a surface-specific
container id (not delegated globally), so the colony view's `data-key` shift-click
handler and the timeline's selection handler stay isolated even though both use a
`data-key` attribute.

### 15.10 Synced Compare strip DZI spike (accepted for v1)
Opening the strip on ≤12 distinct cells fires ≤12 concurrent DZI pyramid builds
(the `_dzi_tiler._get_lock` per-image lock only serialises *duplicate* requests,
not distinct ones). Accepted as a known v1 CPU spike; subsequent opens are cached
and instant. Optional fast-follow: warm the selected cells' DZI on selection
before the strip opens.

### 15.11 Lightweight EXIF for the no-CSV time axis
The no-CSV filmstrip's EXIF time sort must **not** route through
`browse/_metadata.read` → `Image.imread(original).rgb[:]` (`_metadata.py:73`),
which fully decodes every image. Add an EXIF-only helper reading
`DateTimeOriginal` via `exifread` directly, with a per-`(path, mtime_ns)` cache,
so building the time axis over a 500-image RAW folder doesn't decode 500 images.

---

## 16. Interaction model v2 — focus-and-navigate (supersedes the D4 scroll model)

Locked during the 2026-06-18 review (user-directed). Replaces the scrollable,
`IntersectionObserver`-virtualized matrix with a **focus-and-navigate** model on
**both** surfaces. Rationale: a scrollable matrix mounts an unbounded, hard-to-reason
set of tiles as the user scrolls a large set, and the §15.1 cardinality cap was only a
band-aid for that. Anchoring the view to a single **focused cell** and mounting a
bounded neighborhood removes both problems and makes the two core motions — *scan one
plate across time* (←/→) and *compare plates at one time* (↑/↓) — first-class.

### 16.1 The model
- The matrix is **not scrollable.** The viewport renders a **centered window** of
  equal-size tiles that fit at the current tile size, with the **focused cell** at the
  center, visibly highlighted. (Chosen layout: the "centered window, fit-to-viewport"
  option.)
- **Exactly one cell is focused at all times.** It identifies a
  `(row_value, time_value)` — e.g. `(plate, ImageNumber)`. Focus starts at the
  first populated cell (top-left of the ordered matrix).
- **Navigation:** `←/→` move focus along the time axis; `↑/↓` move it along the row
  axis. Equivalent **on-edge directional buttons** (◀ ▶ ▲ ▼) render at the four
  viewport edges. Focus **clamps** at matrix bounds (no wrap); each edge button
  disables when focus is at that bound. Keyboard handlers are **scoped to the timeline
  container** and **ignored while a text input/dropdown holds focus** (so typing a
  pattern never moves focus).
- A compact **position readout** shows the focused cell's axis values + index
  (e.g. `plate 5 / 74 · image 9 / 24`).
- Both surfaces (Browse + Results) use this identical controller.

### 16.2 Tile sizing (retained — per user note)
- The colony-style `−`/`+` tile-size stepper (default 150, step 16, range [64, 400])
  is retained. Tile size sets the thumbnail display px **and** therefore how many cells
  fit in the centered window (smaller tiles → more context cells visible at once). The
  fetched thumbnail snaps to the §4.2 buckets exactly as before; within a bucket,
  stepping is pure CSS scaling.

### 16.3 Mounting & offload — the bounded window (incl. off-screen pre-mount)
- **Mounted set = the visible window + a margin ring of `TIMELINE_FOCUS_MARGIN`
  cells in every direction** beyond the visible edge (default **2**). The ring is the
  explicit, user-requested **off-screen pre-mount**: tiles just outside the viewport are
  mounted *ahead of time* so stepping (or a quick multi-step) into them is instant — a
  smoother UX than mounting only on entry.
- Cells farther than the margin are **offloaded** (their `<img>` removed; the
  placeholder `<div>` + `data-src` string stays, costing zero image bytes). They
  **re-mount** as focus approaches within the margin.
- `mounted ≈ (visible_cols + 2·margin) × (visible_rows + 2·margin)` — bounded by
  viewport + margin, **independent of total matrix size**. `TIMELINE_MOUNT_CAP`
  (§4.4) stays as the absolute LRU safety ceiling.
- **Background warm** is retained but **re-prioritized to neighborhood-first**: warm
  outward in expanding rings from the focus (the cells the user is most likely to reach
  next) rather than row-major, still generation-guarded (§15.8) and
  bounded-concurrency.

### 16.4 Full-resolution access (per user answer)
- Pressing **Enter** (or **Space**) opens the deep-zoom pop-out for the **focused**
  cell (the existing OSD modal + DZI route).
- Every tile additionally carries a **⤢ pop-out button revealed on hover**
  (CSS `:hover`), so any *visible* tile is one click from deep-zoom without first
  navigating focus to it.

### 16.5 Axis cardinality (supersedes §15.1's Y cap)
- Because mounts are bounded by the focus window, **the Results Y axis no longer needs
  the 50-value cap.** The timeline calls an **uncapped** row-axis predicate
  (`selectable_axis_columns(df, value_sets, max_cardinality=None)` or a large bound),
  mirroring the uncapped time predicate (§15.2). High-cardinality groupings such as
  `Metadata_PlateNum` (74) become selectable. Measurement-prefixed and per-object-id
  columns stay excluded. The colony view's own call is unchanged.

### 16.6 Time axis = user-pickable column (no composite-key machinery in v1)
- The X axis is **any user-selected** numeric / datetime / `Metadata_Time`-like column
  (Results: the X dropdown; Browse: the `{time}` pattern or CSV-column time source).
  **Composite date+time ordering is out of scope for v1**: where no single chronological
  column exists, the user picks an appropriate one. For the UCR_029 reference set this is
  **`Metadata_ImageNumber`** (Int64 1..24, strictly monotonic with capture date+time per
  plate, and exactly one image per `(PlateNum, ImageNumber)` cell). The
  no-eligible-time empty state (D9) names this guidance. (A composite-key builder —
  e.g. `Metadata_Date` + `Metadata_Time` → one ordered key — is a possible fast-follow.)

### 16.7 Constants delta (`_config.py`)
- **Add** `TIMELINE_FOCUS_MARGIN: int = 2` — mount-ring distance in cells (the
  off-screen pre-mount radius).
- **Retire** `TIMELINE_WINDOW_MARGIN_SCREENS` (a scroll-era concept); the focus margin
  replaces it. Keep `TIMELINE_MOUNT_CAP`, the tile-size constants, size buckets, and
  warm-concurrency unchanged.

### 16.8 Plan impact (which tasks change vs. the committed Phase 1/2 plans)
- **Phase 1 — `_grid.build_timeline_grid`:** each cell additionally carries
  `data-row-index` / `data-col-index` (0-based positions in `matrix.rows` /
  `matrix.columns`) so the JS can address cells by grid coordinate for focus math.
  Corner/headers + `grid_order` unchanged. The constants task adds
  `TIMELINE_FOCUS_MARGIN` and drops `TIMELINE_WINDOW_MARGIN_SCREENS`.
- **Phase 1 — `_grid` cell chrome:** the ⤢ button becomes hover-revealed (CSS class
  only; no DOM change).
- **Phase 2 — `timeline.js`:** **replace** the IntersectionObserver scroll
  virtualization with the **focus-navigate controller** — focus state, keyboard +
  edge-button handlers, centered-window layout, margin-ring mount/offload,
  neighborhood-first warm, Enter→pop-out.
- **Phase 2 — `_layout.py`:** add the four edge buttons, the focus/position readout,
  and the no-scroll centered-window container; the tile-size stepper stays.
- **Phase 3 (Results) + Phase 4 (Compare strip):** consume the same engine + controller
  unchanged in shape (Compare strip remains the bounded synced-deep-zoom feature; focus
  navigation does not replace it).

### 16.9 Testing — full e2e Playwright + live MCP verification (user-directed)
- **Every interactive surface carries pytest-Playwright e2e coverage**, not only unit
  tests: Browse focus-navigate (arrow keys + edge buttons move focus and mount the new
  neighborhood; far cells stay unmounted; margin-ring pre-mount present; Enter opens
  pop-out; hover reveals ⤢), Results focus-navigate + tab activation re-attach (§15.7),
  and the Compare strip (viewport-sync, ≤12 cap notice). These run under
  `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/…`.
- **Live MCP-driven verification** (the Playwright MCP) is part of the verification
  gate at each surface's completion: drive the *running* `phenotypic-gui` against the
  **real reference data** — Browse over `…/data/processed/` and Results over
  `…/data/results/2026-06-16/` (mirror reads `deliverables/measurements.parquet`,
  X=`Metadata_ImageNumber`, Y=`Metadata_PlateNum`) — navigating with arrows/buttons,
  opening a pop-out, and capturing screenshots to confirm real-image rendering that a
  fixture-only e2e cannot. The MCP pass is a manual orchestrator gate (it needs a live
  server + real data), documented in each phase's review step; the committed pytest
  e2e is the CI-enforced guard.
