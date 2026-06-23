# Source Timeline View — Phase 3: Results Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **6th `Timeline` tab** to the Results viewer that renders the SAME Phase 1 shared engine (`build_matrix` → `build_timeline_grid` → `register_thumbnail_route`) and the SAME Phase 2 focus-and-navigate `timeline.js` controller, but over **overlay images** (`results/<dataset>/overlays/<stem>.png`) with axes derived from `OutputRoot.master_df` (the post-applied mirror, which already carries joined `Metadata_*`). Per spec §6 + §16.

**Architecture:** A new package `results_viewer/timeline_view/` parallels `colony_view/`: `_ids.py`, `_layout.py` (toolbar + no-scroll focus-window viewport + stores + pop-out modal), `_grid.py` (a thin adapter that builds matrix records from a filtered `master_df` slice + the new `selectable_time_columns` predicate), `_thumb_routes.py` (overlay → cached downscaled PNG via the Phase 1 factory under `VIEWER_THUMB_URL_SEGMENT`), and `_callbacks.py` (render grid, populate X/Y dropdowns, empty state, attach the controller). The Y dropdown calls the existing `selectable_axis_columns` with `max_cardinality=None` (a small SHARED-FILE change to that helper, spec §16.5); the X (time) dropdown uses a new `selectable_time_columns` (spec §15.2). Pop-out reuses the viewer's existing overlay DZI route (`/tiles/<dataset>/<stem>.dzi`) + OSD. `timeline.js` is **copied into `results_viewer/_assets/`** (the viewer pins its own `assets_folder`, distinct from Browse's — see "Asset sharing" below).

**Tech Stack:** Dash + dash-bootstrap-components, polars (`OutputRoot.master_df`), Pillow (via Phase 1), the vendored OpenSeadragon already in `results_viewer/_assets/openseadragon/`, pytest + Playwright (`tests/e2e/gui`).

---

## Decision record — verified against the real code (read before implementing)

These are the load-bearing facts this plan was written against. Each was checked
with Read/Grep on the worktree. Discrepancies with the spec are flagged inline.

1. **Separate Dash apps, NOT co-mounted + the controller is surface-agnostic.**
   Browse (`MOUNT_BROWSE = "/browse/"`) and Results (`MOUNT_VIEWER = "/results/"`)
   are **separate `dash.Dash` instances on separate Flask servers**, mounted at
   distinct URL prefixes by the hub's `werkzeug.middleware.dispatcher.DispatcherMiddleware`
   (`shell/_app.py` `compose_hub` — each `*.create_app(...)` returns its own app;
   the dispatcher routes by prefix). Each `create_app` pins its OWN `assets_folder`
   (Browse → `browse/_assets`, Results → `results_viewer/_assets`, verified at
   `results_viewer/_app.py:157`). The two pages never share a JS `window`.
   **The Phase 2 `timeline.js` is fully surface-agnostic** (confirmed by the
   coordinator): it finds every sibling control **by CLASS scoped to `.timeline-body`**
   — `.timeline-nav-{up,down,left,right}`, `.timeline-position`,
   `.timeline-popout-bridge`, `.timeline-viewport`, `.timeline-grid-container` — never
   by any `browse-tl-*` id; `startReattachObserver` discovers the grid by
   `.timeline-grid-container` and re-attaches via *that element's own id*. So the SAME
   controller runs on both surfaces **unchanged** — Phase 3 must NOT edit or
   parameterize `timeline.js`. The ONLY surface-specific input is the `attach()`
   argument (the grid's element id). **Phase 3's whole obligation is therefore:**
   (a) vendor `timeline.js` **byte-for-byte** into `results_viewer/_assets/` (Task 1;
   a byte-equality CI guard enforces no drift), (b) put the surface-agnostic
   `timeline-*` **classes** on the Results layout elements (Task 6 / C2), and
   (c) call `window.__phenotypicTimeline.attach("timeline-grid")` from a Results
   clientside callback after each render. No per-container state keying is needed
   (separate Dash apps → no shared `window`; resolved at default, see Open Questions).

2. **`selectable_axis_columns(df, column_value_sets, max_cardinality=50)`** lives at
   `colony_view/_grid.py:115`. Body (verified `:164`): `if cardinality < 2 or
   cardinality > max_cardinality: continue`. With `max_cardinality=None`,
   `cardinality > None` raises `TypeError` on Python 3. **Spec §16.5 / §15.1 are
   correct** that the timeline needs the cap removed. The fix (Task 2) makes the
   guard `max_cardinality is None or cardinality <= max_cardinality`. The colony
   caller (`colony_view/_callbacks.py:237`, `selectable_axis_columns(filtered_df,
   column_value_sets)`) keeps the default 50 — unchanged. For the UCR_029 set this
   makes `Metadata_PlateNum` (74) selectable on the Y axis.

3. **`OutputRoot`** (`results_viewer/_output_root.py`): `master_df: pl.DataFrame`
   (the post-applied mirror), `column_value_sets: Mapping[str, list[str]]` (a
   **lazy** `_LazyColumnValueSets` — eager only for `Metadata_*`, computes others
   on first `__getitem__`, raises `KeyError` for columns not in `df`),
   `overlay_path(dataset, stem) -> Path` (`<root>/results/<dataset>/overlays/<stem>.png`,
   not existence-checked), `has_overlay(dataset, stem) -> bool` (O(1) frozenset
   snapshot `overlay_index`), and `image_pairs(df) -> list[tuple[str, str]]`
   (unique sorted `(Metadata_Dataset, Metadata_ImageFile)` pairs). `Metadata_Dataset`
   / `Metadata_ImageFile` are guaranteed present (backfilled by
   `_ensure_required_columns`). **Note:** `Metadata_ImageFile` IS the stem
   convention used by `image_pairs` and `overlay_path` (the overlay scan keys on
   `entry.stem`); but real masters may carry the filename WITH extension. The
   adapter (Task 3) must derive the overlay stem as `Path(image_file).stem` and
   prefer `has_overlay` to decide membership (mirroring how the colony crop route
   resolves overlays).

4. **Overlay DZI route** (`results_viewer/_tile_routes.py`, `register`): mounts
   `GET /tiles/<dataset>/<stem>.dzi` + the tile sub-route under
   `VIEWER_TILES_PREFIX = "/tiles"`, lazily tiling `output_root.overlay_path(...)`.
   The pop-out reuses this unchanged — same route Plate/Colony already point OSD at.

5. **`VIEWER_CACHE_DIRNAME = ".viewer_cache"`** (`_config.py:341`); `OutputRoot.cache_dir`
   is `<root>/.viewer_cache/dzi` (the DZI cache). The timeline thumbnail cache must
   live in a SIBLING subdir under the cache root, e.g.
   `output_root.root / VIEWER_CACHE_DIRNAME / "timeline_thumbs"`, so it persists
   with the run and never collides with the DZI pyramids.

6. **Tab declaration** (`results_viewer/_layout.py` `build_app_layout`,
   `:464`–`:494`): a `dbc.Tabs(... id=ids.TABS_ID, active_tab=ids.TAB_PLATE_ID)`
   holding five `dbc.Tab(body, label=..., tab_id=ids.TAB_*)`. All tab bodies stay
   mounted (`dbc.Tabs` renders the active body's subtree; bodies are built eagerly
   and switching is CSS-only per the docstring). `TAB_*` ids + `__all__` live in
   `results_viewer/_ids.py` (`:431`–`:449`, plus the `__all__` list `:749`–`:754`).
   We add `TAB_TIMELINE_ID = "tab-timeline"` there and a 6th `dbc.Tab`.

7. **`dbc.Tabs` does NOT keep inactive content stably mounted across re-renders**
   (spec §15.7): `results_viewer.js` polls-until-present (`setInterval(..., 100)`)
   AND runs a `<body>` `MutationObserver` to **re-attach** its delegated listeners
   when Dash replaces a container (verified ≈`results_viewer.js:405`–`:436` and the
   QC-splitter block `:523`+). `timeline.js`'s `attach` + `startReattachObserver`
   (Phase 2) follow the same idiom; `display:none → visible` on tab switch re-fires
   the controller. Because the thumbnail cache is server-side on disk, tab re-entry
   is warm regardless of the JS observer — **no `dcc.Store` warm-state rehydration**.

8. **Filter-sidebar df flow** (mirror exactly): the colony render/dropdown
   callbacks take `df = output_root.master_df` + `column_value_sets =
   output_root.column_value_sets` by closure, react to `Input(ids.STORE_FILTER_SPEC,
   "data")`, and slice via `FilterSpec.from_store(payload).apply_to(df)`
   (`colony_view/_callbacks.py:105`–`106`, `:228`–`:237`). The Timeline tab uses
   the identical slice so it honors the active filter offcanvas.

9. **Crop route is a DIFFERENT factory.** The colony/QC crops use
   `register_crop_route(app, output_root, segment)` from `gui/_shared/tiles.py`
   (centroid-cropping). The Timeline thumb route uses the **Phase 1**
   `register_thumbnail_route(app, *, segment, resolve_source, cache_base)` from
   `gui/_shared/timeline` (whole-image downscale). Do not conflate them.

10. **e2e fixtures (verified).** `tests/e2e/gui/conftest.py` ships `fake_sandbox`
    (module-scoped; its `_build_sandbox` writes a **stub** `master_measurements.parquet`
    = `b""`, so `OutputRoot.discover` would FAIL on it), `live_server`/`hub_url`
    (boots `python -m phenotypic.gui --root <sandbox>`), and `page`. There is NO
    ready-made loaded-viewer fixture. The PROVEN pattern for a real loaded viewer
    is `tests/e2e/gui/test_heatmap_tab.py`: a **function-scoped** `fake_sandbox`
    override that calls `_build_sandbox(tmp_path)` then `_seed_master_df_in_output`
    (which uses `write_master` + `write_measurements_mirror` from
    `tests._output_layout` and writes overlay PNGs into
    `results/<ds>/overlays/<stem>.png`), a function-scoped `live_server` override,
    and a `_hand_off_viewer(page, hub_url, output_rel)` helper that POSTs the output
    dir to `/sandbox/api/viewer/output-root` then navigates `/results/`. Phase 3
    reuses this idiom verbatim, seeding `Metadata_ImageNumber` (Int64 monotonic) +
    `Metadata_PlateNum` columns.

11. **Tile-size stepper helpers.** `_config.py` exposes
    `step_colony_tile_size(current, direction)` + `stepped_colony_tile_size_from_trigger(...)`
    (`:566`, `:583`) and `COLONY_TILE_SIZE_*`. Phase 1 added `TIMELINE_TILE_SIZE_*`
    + `snap_thumb_bucket`. **If Phase 2 added `step_timeline_tile_size`** (its Task 8
    review note W6 says it adds it to `_config.__all__`), reuse it; **if not present**,
    Task 4 adds a `step_timeline_tile_size` mirroring `step_colony_tile_size` over
    the `TIMELINE_TILE_SIZE_*` constants (and to `__all__`). Check first; do not
    duplicate.

12. **`window.__phenotypicAppPrefix`** is injected into the viewer's `index_string`
    (`_app.py:107`, `_index_string_with_prefix`) so `timeline.js` can build
    mount-aware URLs exactly as `results_viewer.js` does. The thumbnail `url_builder`
    (server-rendered into `data-src`) must likewise prepend the prefix.

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Phase 1 + Phase 2 must be merged/available.** This plan consumes
  `phenotypic.gui._shared.timeline` (`build_matrix`, `TimelineMatrix`,
  `build_timeline_grid` with `ref_builder` + `data-row-index`/`data-col-index`,
  `register_thumbnail_route`, `ThumbUnavailable`), `phenotypic.gui._config`
  (`THUMB_SIZE_BUCKETS`, `snap_thumb_bucket`, `TIMELINE_TILE_SIZE_DEFAULT/STEP/MIN/MAX`,
  `TIMELINE_FOCUS_MARGIN`, `TIMELINE_MOUNT_CAP`, `TIMELINE_WARM_CONCURRENCY`,
  `VIEWER_THUMB_URL_SEGMENT`, `VIEWER_CACHE_DIRNAME`), and the Phase 2
  `timeline.js` focus-and-navigate controller (copied in Task 1). The scroll-era
  `TIMELINE_WINDOW_MARGIN_SCREENS` was removed by Phase 1 (spec §16.7) — do not
  reference it.
- **Focus-and-navigate model (spec §16 — binding).** The Timeline tab is **not
  scrollable**: a no-scroll viewport renders a centered window around one focused
  cell; ←/→/↑/↓ + four on-edge ◀▶▲▼ buttons move focus (clamped, no wrap, ignored
  while a text input/dropdown holds focus); the focused neighborhood + a
  `TIMELINE_FOCUS_MARGIN` ring mounts `<img>` and everything beyond offloads;
  Enter/Space (or the hover-revealed ⤢) opens the deep-zoom pop-out for the focused
  (or hovered) cell. This is the SAME controller as Browse.
- **No 50-cap on either timeline axis (spec §16.5).** Y calls
  `selectable_axis_columns(df, value_sets, max_cardinality=None)`; X calls the new
  `selectable_time_columns(df, value_sets)` (uncapped, name/dtype-gated). The
  colony view's `selectable_axis_columns` call is unchanged.
- **Single-source constants** in `_config.py` / `_design.py`; new viewer component
  ids in `results_viewer/_ids.py` (`TAB_TIMELINE_ID`) and the timeline subpackage's
  own `timeline_view/_ids.py` (every other id). Don't re-spell literals; don't
  import `dash` from `_config.py`/`_design.py`.
- **FEATURES.md + WORKFLOWS.md gates:** any `src/phenotypic/gui/` change must modify
  `FEATURES.md`; the Timeline is a tutorial-worthy flow so it also needs a
  `WORKFLOWS.md` row + a `_capture_<id>` in `scripts/capture_gui_tutorial_screenshots.py`
  + a tutorial page. `✅ shipping` rows need a resolvable `path::test`; never leave a
  row `🚧 in progress` (merge gate rejects it).
- **Verify Dash wiring in a live browser (project rule + spec §16.9):** the controller
  + tab activation carry Playwright e2e tests, not only unit tests. Mark `ci_flaky`
  where DOM-poll-budgeted (per `tests/CLAUDE.md`).
- **Test collection:** `tests/gui` is already in `pyproject.toml` `testpaths`
  (Phase 1 Task 0). Unit tests land under `tests/gui/results_viewer/timeline_view/`;
  e2e under `tests/e2e/gui/`.

### Asset sharing — `timeline.js` for the Results app (decision)

The viewer pins `assets_folder = results_viewer/_assets` (verified). Dash auto-loads
every `*.js` in that folder as a `<script>`; `timeline.js` is an IIFE that
self-registers `window.__phenotypicTimeline`, so it must be in the viewer's assets
folder to load on `/results/`. Because Browse and Results are **separate apps with
separate assets folders** (Decision #1), the file genuinely needs to be present in
both. The repo's existing convention for "same JS in two apps" is **per-app
vendoring** (e.g. `openseadragon.min.js` is vendored under BOTH `browse/_assets/` and
`results_viewer/_assets/`). **Chosen approach: copy `timeline.js` into
`results_viewer/_assets/timeline.js`** (Task 1), matching the OSD vendoring
precedent and keeping each app self-contained. A single canonical source served via
`register_shared_static` (the `/_shared/<file>` blueprint) was considered but
rejected: that blueprint serves on-demand files (the logo) referenced by explicit
URL in layouts, whereas Dash assets are auto-discovered + auto-`<script>`-tagged
from `assets_folder` only — wiring a shared JS as a Dash-managed script would mean a
manual `external_scripts` entry per app anyway, no simpler than a vendored copy. The
**byte-equality CI guard (Task 1) asserts the two copies are byte-identical**, so the
vendored copy never silently drifts from Browse's controller — and because the
controller is surface-agnostic (Decision #1), the byte-identical copy "just works" on
Results once its layout carries the `timeline-*` classes (Task 6 / C2) and a
clientside callback calls `attach("timeline-grid")` (Task 6). This sharing decision
is resolved at default in the Open Questions.

---

### Task 1: Add `TAB_TIMELINE_ID` + vendored `timeline.js` + timeline_view package skeleton

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_ids.py` (add `TAB_TIMELINE_ID` + `__all__`)
- Create: `src/phenotypic/gui/results_viewer/_assets/timeline.js` (copy of Browse's)
- Create: `src/phenotypic/gui/results_viewer/timeline_view/__init__.py` (empty for now)
- Create: `tests/gui/results_viewer/__init__.py` (empty, if missing)
- Create: `tests/gui/results_viewer/timeline_view/__init__.py` (empty)
- Test: `tests/gui/results_viewer/timeline_view/test_assets.py`

**Interfaces:**
- Consumes: Phase 2 `src/phenotypic/gui/browse/_assets/timeline.js`.
- Produces: `TAB_TIMELINE_ID: str`; a viewer-served `timeline.js` byte-identical to
  Browse's; the empty `timeline_view` package.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/results_viewer/timeline_view/__init__.py` (empty) and
`tests/gui/results_viewer/__init__.py` (empty, if it does not already exist), then
`tests/gui/results_viewer/timeline_view/test_assets.py`:

```python
"""TAB_TIMELINE_ID presence + the vendored timeline.js stays in sync with Browse."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.results_viewer import _ids

# NB: resolve the asset files via the OWNING PACKAGE's __file__ + "/_assets/…".
# Do NOT `import phenotypic.gui.browse._assets` — asset directories carry no
# __init__.py, so importing one raises ModuleNotFoundError (S1).


def _browse_timeline_js() -> Path:
    import phenotypic.gui.browse as browse

    return Path(browse.__file__).parent / "_assets" / "timeline.js"


def _viewer_timeline_js() -> Path:
    import phenotypic.gui.results_viewer as rv

    return Path(rv.__file__).parent / "_assets" / "timeline.js"


def test_tab_timeline_id_present_and_unique() -> None:
    assert isinstance(_ids.TAB_TIMELINE_ID, str) and _ids.TAB_TIMELINE_ID
    tab_ids = {
        _ids.TAB_PLATE_ID,
        _ids.TAB_COLONY_ID,
        _ids.TAB_QC_ID,
        _ids.TAB_HEATMAP_ID,
        _ids.TAB_ERROR_ID,
        _ids.TAB_TIMELINE_ID,
    }
    assert len(tab_ids) == 6  # all six tab ids distinct
    assert "TAB_TIMELINE_ID" in _ids.__all__


def test_viewer_timeline_js_is_byte_identical_to_browse() -> None:
    # The viewer vendors its own copy (separate assets_folder); the CI guard
    # keeps it from drifting from the Browse-authored controller (Decision #1).
    assert _viewer_timeline_js().read_bytes() == _browse_timeline_js().read_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_assets.py -v`
Expected: FAIL (`AttributeError: ... TAB_TIMELINE_ID` and/or the viewer copy missing).

- [ ] **Step 3: Write minimal implementation**

In `src/phenotypic/gui/results_viewer/_ids.py`, add the constant in the **Tabs**
block (after `TAB_ERROR_ID`, `:449`) and to `__all__` (after `"TAB_ERROR_ID"`):

```python
#: ``dbc.Tab`` value for the Timeline view (the 6th tab). The body is a
#: focus-and-navigate matrix of overlay thumbnails over a (row × time)
#: axis pair drawn from ``OutputRoot.master_df`` (spec §6/§16). Kept
#: mounted alongside the others; switching is CSS-only.
TAB_TIMELINE_ID = "tab-timeline"
```

Copy the controller verbatim:

```bash
cp src/phenotypic/gui/browse/_assets/timeline.js \
   src/phenotypic/gui/results_viewer/_assets/timeline.js
```

Create the empty package inits:

```python
# src/phenotypic/gui/results_viewer/timeline_view/__init__.py
"""Results-viewer Timeline tab — overlay matrix over OutputRoot.master_df axes."""
```

> **Controller is surface-agnostic — copy it verbatim, do NOT edit it.** Phase 2's
> `timeline.js` finds all sibling controls **by class scoped to `.timeline-body`**
> and its `startReattachObserver` discovers the grid by `.timeline-grid-container`
> (re-attaching via that element's own id), so the byte-identical copy runs on
> `/results/` unchanged once the Results layout carries the `timeline-*` classes
> (Task 6 / C2). Phase 3 must **not** parameterize or otherwise modify the controller
> (that would break this byte-equality guard and contradict the Phase 2 contract).
> The only Results-side wiring is a clientside callback that calls
> `window.__phenotypicTimeline.attach("timeline-grid")` after each grid render
> (Task 6); `attach` is idempotent, so the controller's own observer re-firing is
> harmless.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_assets.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_ids.py \
  src/phenotypic/gui/results_viewer/_assets/timeline.js \
  src/phenotypic/gui/results_viewer/timeline_view/__init__.py \
  tests/gui/results_viewer/__init__.py \
  tests/gui/results_viewer/timeline_view/__init__.py \
  tests/gui/results_viewer/timeline_view/test_assets.py
git commit -m "feat(gui-timeline): TAB_TIMELINE_ID + vendor timeline.js into viewer assets"
```

---

### Task 2: Uncapped `selectable_axis_columns` (shared-file change, spec §16.5)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_grid.py` (`selectable_axis_columns`)
- Test: `tests/gui/results_viewer/colony_view/test_grid_axis_columns.py` (create or append — check first)

**Interfaces:**
- Consumes: nothing new.
- Produces: `selectable_axis_columns(df, column_value_sets, max_cardinality: int | None = 50)`
  where `max_cardinality=None` means "no upper cap". Default 50 preserved → the
  colony caller (`colony_view/_callbacks.py:237`) is unaffected.

- [ ] **Step 1: Write the failing test**

First check whether a colony-grid axis test already exists:
`uv run pytest --collect-only -q tests/gui/results_viewer/colony_view 2>/dev/null | grep -i axis` — if a file exists, append; else create
`tests/gui/results_viewer/colony_view/test_grid_axis_columns.py` (add an
`__init__.py` under `tests/gui/results_viewer/colony_view/` if missing):

```python
"""selectable_axis_columns: the uncapped (max_cardinality=None) path (spec §16.5)."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns


def _frame_with_high_cardinality_metadata() -> pl.DataFrame:
    # 74 distinct plate numbers — above the 50 default cap; 3 datasets.
    return pl.DataFrame(
        {
            "Metadata_PlateNum": [str(i % 74) for i in range(148)],
            "Metadata_Dataset": [f"ds{i % 3}" for i in range(148)],
            "Object_Label": list(range(148)),
            "Size_Area": [1.0] * 148,  # measurement-prefixed → always excluded
        }
    )


def _value_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    return {
        col: df.get_column(col).cast(pl.String).drop_nulls().unique().sort().to_list()
        for col in df.columns
    }


def test_default_cap_excludes_high_cardinality_metadata() -> None:
    df = _frame_with_high_cardinality_metadata()
    cols = selectable_axis_columns(df, _value_sets(df))  # default 50
    assert "Metadata_PlateNum" not in cols  # 74 > 50
    assert "Metadata_Dataset" in cols       # 3 in [2, 50]


def test_none_cap_is_uncapped_and_admits_high_cardinality() -> None:
    df = _frame_with_high_cardinality_metadata()
    cols = selectable_axis_columns(df, _value_sets(df), max_cardinality=None)
    assert "Metadata_PlateNum" in cols      # 74 now allowed
    assert "Metadata_Dataset" in cols
    # Exclusions still hold: measurement-prefixed + per-object id are dropped.
    assert "Size_Area" not in cols
    assert "Object_Label" not in cols


def test_none_cap_still_excludes_singleton_columns() -> None:
    # cardinality < 2 is excluded regardless of cap (a constant axis is useless).
    df = pl.DataFrame(
        {"Metadata_Const": ["x"] * 10, "Metadata_Dataset": ["a", "b"] * 5}
    )
    cols = selectable_axis_columns(df, _value_sets(df), max_cardinality=None)
    assert "Metadata_Const" not in cols
    assert "Metadata_Dataset" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/colony_view/test_grid_axis_columns.py -v`
Expected: FAIL (`test_none_cap_*` raise `TypeError: '>' not supported between 'int'
and 'NoneType'`).

- [ ] **Step 3: Write minimal implementation**

In `colony_view/_grid.py`, widen the signature and the guard. The current body
(`:119`, `:164`) is:

```python
    max_cardinality: int = 50,
) -> list[str]:
    ...
        if cardinality < 2 or cardinality > max_cardinality:
            continue
```

Change to:

```python
    max_cardinality: int | None = 50,
) -> list[str]:
```

and the guard:

```python
        if cardinality < 2 or (
            max_cardinality is not None and cardinality > max_cardinality
        ):
            continue
```

Update the docstring's `max_cardinality` line to: *"Upper bound on accepted
cardinalities, or ``None`` for no upper cap (the timeline Y axis passes ``None`` so
high-cardinality groupings like ``Metadata_PlateNum`` stay selectable — spec §16.5).
Defaults to 50."*

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/colony_view/test_grid_axis_columns.py -v`
Expected: PASS (3 tests).
Also confirm no colony regression: `uv run pytest tests/gui/results_viewer/colony_view -q`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/colony_view/_grid.py \
  tests/gui/results_viewer/colony_view/test_grid_axis_columns.py \
  tests/gui/results_viewer/colony_view/__init__.py
git commit -m "feat(gui-timeline): allow uncapped selectable_axis_columns (max_cardinality=None)"
```

---

### Task 3: Time-axis predicate + record adapter (`timeline_view/_grid.py`)

**Files:**
- Create: `src/phenotypic/gui/results_viewer/timeline_view/_grid.py`
- Test: `tests/gui/results_viewer/timeline_view/test_grid.py`

**Interfaces:**
- Consumes: `OutputRoot` (for `master_df` / `column_value_sets` / `image_pairs` /
  `has_overlay`), `selectable_axis_columns` (Task 2); `build_matrix` (Phase 1).
- Produces:
  - `selectable_time_columns(df: pl.DataFrame, column_value_sets: Mapping[str, list[str]]) -> list[str]`
    — eligible iff the name matches a `Metadata_Time`-like pattern **OR** the dtype
    is numeric/temporal; **no cardinality cap** (spec §15.2). Measurement-prefixed
    and `Object_Label` columns excluded. `Metadata_*` time-like names sort first.
  - `is_large_time_axis(n_values: int, threshold: int = 100) -> bool` — gates the
    bucketing-warning banner (spec §15.2; bucketing UI is out of scope, warning only).
  - `has_eligible_time_axis(df, column_value_sets) -> bool` — the empty-state predicate
    (D9): `True` iff `selectable_time_columns` is non-empty.
  - `build_timeline_records(output_root, df, *, row_col, time_col) -> list[dict]` —
    one record per `(dataset, stem)` image pair surviving `df`, carrying
    `{"row_value": <df[row_col] for that image>, "time_value": <df[time_col]>,
    "cell_ref": (dataset, stem)}`. Skips pairs lacking an overlay (`has_overlay`).
    `cell_ref` is the `(dataset, stem)` tuple the thumb route + DZI route consume.

**Time-name pattern (seed from Heatmap, verified):** Heatmap hardcodes the literal
`"Metadata_Time"` (`_heatmap_tab/_callbacks.py:367`). Generalize to a case-insensitive
name match for a `Metadata_Time`-like column **and** offer any numeric/temporal
column. For UCR_029 the user picks `Metadata_ImageNumber` (Int64, monotonic with
capture date+time per plate); `Metadata_Time` (time-of-day) is ALSO offered by
name-match but mis-orders across days — the empty-state/help text advises picking a
monotonic column (spec §16.6).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/results_viewer/timeline_view/test_grid.py`:

```python
"""Time-axis predicate + (dataset, stem) record adapter for the Timeline tab."""
from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer.timeline_view._grid import (
    has_eligible_time_axis,
    is_large_time_axis,
    selectable_time_columns,
)


def _value_sets(df: pl.DataFrame) -> dict[str, list[str]]:
    return {
        col: df.get_column(col).cast(pl.String).drop_nulls().unique().sort().to_list()
        for col in df.columns
    }


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Metadata_Dataset": ["ds"] * 4,
            "Metadata_ImageFile": ["a", "a", "b", "b"],
            "Metadata_ImageNumber": pl.Series([1, 2, 1, 2], dtype=pl.Int64),
            "Metadata_Time": ["09:00", "10:00", "09:00", "10:00"],
            "Metadata_PlateNum": ["1", "1", "2", "2"],
            "Object_Label": [10, 11, 12, 13],
            "Size_Area": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_numeric_column_is_an_eligible_time_axis() -> None:
    # M3: dtype is the AUTHORITATIVE eligibility path — Metadata_ImageNumber
    # (Int64, the UCR_029 reference column) is eligible by dtype alone.
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_ImageNumber" in cols   # Int64 dtype → eligible


def test_numeric_dtype_eligible_even_without_time_like_name() -> None:
    # M3: the dtype path admits a numeric column whose NAME does not match the
    # Metadata_Time-like regex — proving dtype is authoritative, name is a
    # string-typed FALLBACK (for String-stored time columns).
    df = pl.DataFrame({"Metadata_Generation": pl.Series([0, 1, 2], dtype=pl.Int64)})
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_Generation" in cols


def test_metadata_time_name_match_is_eligible_even_if_string() -> None:
    # M3: the name regex is the FALLBACK path — Metadata_Time stored as pl.String
    # (join_metadata casts join keys to String) has no numeric/temporal dtype, but
    # its name matches the Metadata_Time-like pattern → still offered.
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_Time" in cols


def test_measurement_and_object_label_columns_are_excluded() -> None:
    df = _df()
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Size_Area" not in cols       # measurement-prefixed
    assert "Object_Label" not in cols    # per-object id


def test_no_cardinality_cap_on_time_axis() -> None:
    # 200 distinct numeric timepoints must remain eligible (the 50-cap on the
    # row axis would have hidden a long course — spec §15.2).
    df = pl.DataFrame({"Metadata_ImageNumber": pl.Series(range(200), dtype=pl.Int64)})
    cols = selectable_time_columns(df, _value_sets(df))
    assert "Metadata_ImageNumber" in cols


def test_is_large_time_axis() -> None:
    assert is_large_time_axis(150) is True
    assert is_large_time_axis(50) is False
    assert is_large_time_axis(100) is False   # threshold is "> threshold"


def test_empty_state_predicate_false_without_any_time_column() -> None:
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds", "ds"],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_PlateNum": ["1", "2"],  # categorical, no name/dtype match
            "Object_Label": [1, 2],
            "Size_Area": [1.0, 2.0],
        }
    )
    assert has_eligible_time_axis(df, _value_sets(df)) is False


def test_empty_state_predicate_true_with_a_time_column() -> None:
    df = _df()
    assert has_eligible_time_axis(df, _value_sets(df)) is True
```

Then append a record-adapter test that uses a tiny on-disk `OutputRoot` via the
real `discover` over a tmp tree (mirrors `_seed_master_df_in_output`). Put it in the
same file:

```python
from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view._grid import build_timeline_records
from tests._output_layout import write_master, write_measurements_mirror


def _make_output_root(tmp_path: Path) -> OutputRoot:
    cli_out = tmp_path / "out"
    df = _df()
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    overlays = cli_out / "results" / "ds" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (40, 30), (10, 20, 30)).save(overlays / f"{stem}.png")
    return OutputRoot.discover(cli_out)


def test_build_timeline_records_emits_one_per_overlay_pair(tmp_path: Path) -> None:
    root = _make_output_root(tmp_path)
    records = build_timeline_records(
        root, root.master_df, row_col="Metadata_PlateNum", time_col="Metadata_ImageNumber"
    )
    refs = {r["cell_ref"] for r in records}
    assert ("ds", "a") in refs and ("ds", "b") in refs
    rows = {r["row_value"] for r in records}
    assert rows == {"1", "2"}        # plate numbers
    times = {r["time_value"] for r in records}
    assert times == {"1", "2"}       # image numbers (stringified)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_grid.py -v`
Expected: FAIL (`ModuleNotFoundError: ...timeline_view._grid`).

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/results_viewer/timeline_view/_grid.py`:

```python
"""Time-axis predicate + (dataset, stem) record adapter for the Timeline tab.

The Results Timeline draws its axes from ``OutputRoot.master_df`` (the
post-applied mirror, which already carries joined ``Metadata_*`` columns).
The Y (row) axis reuses the colony-view ``selectable_axis_columns`` with the
50-cap removed (spec §16.5); the X (time) axis uses ``selectable_time_columns``
here — name/dtype-gated, *uncapped* (a long time-course is the whole point,
spec §15.2). One record is emitted per ``(dataset, stem)`` image pair that has
an overlay PNG; ``cell_ref`` is the ``(dataset, stem)`` tuple the thumbnail
route and the deep-zoom DZI route both consume.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._filtered_state import KEY_DATASET, KEY_IMAGE_FILE
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view._grid import (
    _MEASUREMENT_PREFIXES,
    _OBJECT_LABEL_COL,
)

__all__ = [
    "selectable_time_columns",
    "is_large_time_axis",
    "has_eligible_time_axis",
    "build_timeline_records",
    "LARGE_TIME_AXIS_THRESHOLD",
]

#: Above this distinct-time-value count the toolbar shows a bucketing-warning
#: banner (bucketing UI itself is out of scope for v1 — spec §15.2).
LARGE_TIME_AXIS_THRESHOLD = 100

#: Case-insensitive name match for a "Metadata_Time-like" column. Seeded from
#: Heatmap's hardcoded ``"Metadata_Time"`` (``_heatmap_tab/_callbacks.py:367``)
#: but generalized so e.g. ``Metadata_Timepoint`` / ``Metadata_ImageNumber``
#: also surface. Numeric/temporal dtype is an independent eligibility path.
_TIME_NAME_RE = re.compile(r"(?:^|_)(time|timepoint|imagenumber|frame)(?:_|$|\d)", re.IGNORECASE)

def _is_time_like_name(col: str) -> bool:
    return bool(_TIME_NAME_RE.search(col))


def _is_ordered_dtype(dtype: pl.DataType) -> bool:
    # Numeric or temporal dtypes read as an ordered time axis without a name
    # match. Use the per-dtype predicates (pl.NUMERIC_DTYPES/TEMPORAL_DTYPES
    # are deprecated since polars 1.0).
    return bool(dtype.is_numeric() or dtype.is_temporal())


def selectable_time_columns(
    df: pl.DataFrame,
    column_value_sets: Mapping[str, list[str]],
) -> list[str]:
    """Return columns eligible as the timeline X (time) axis.

    A column is eligible iff it is NOT measurement-prefixed and NOT
    ``Object_Label``, AND either its name matches a ``Metadata_Time``-like
    pattern OR its dtype is numeric/temporal. There is **no cardinality cap**
    (spec §15.2). ``Metadata_*`` time-like names sort first, then everything
    else, alphabetically within each bucket.

    Args:
        df: The frame to inspect (typically the filtered master mirror).
        column_value_sets: Unused for eligibility but accepted for signature
            symmetry with ``selectable_axis_columns`` (callers thread the same
            pair through). Cardinality is intentionally NOT consulted.

    Returns:
        Eligible time-column names in bucketed sort order.
    """
    del column_value_sets  # eligibility is name/dtype-based, never cardinality
    eligible: list[str] = []
    schema = df.schema
    for col in df.columns:
        if col == _OBJECT_LABEL_COL:
            continue
        if any(col.startswith(prefix) for prefix in _MEASUREMENT_PREFIXES):
            continue
        dtype = schema[col]
        if _is_time_like_name(col) or _is_ordered_dtype(dtype):
            eligible.append(col)

    def _bucket(name: str) -> int:
        return 0 if (name.startswith("Metadata_") and _is_time_like_name(name)) else 1

    eligible.sort(key=lambda name: (_bucket(name), name))
    return eligible


def is_large_time_axis(n_values: int, threshold: int = LARGE_TIME_AXIS_THRESHOLD) -> bool:
    """Return ``True`` when the time axis has more than ``threshold`` distinct values."""
    return n_values > threshold


def has_eligible_time_axis(
    df: pl.DataFrame, column_value_sets: Mapping[str, list[str]]
) -> bool:
    """Return ``True`` iff at least one eligible time column exists (D9 empty state)."""
    return bool(selectable_time_columns(df, column_value_sets))


def build_timeline_records(
    output_root: OutputRoot,
    df: pl.DataFrame,
    *,
    row_col: str,
    time_col: str,
) -> list[dict[str, object]]:
    """Build ``build_matrix`` records from a filtered master slice.

    One record per ``(dataset, stem)`` pair surviving ``df`` that has an
    overlay PNG on disk. ``row_value``/``time_value`` are the row's values in
    ``row_col``/``time_col`` (stringified by ``build_matrix`` downstream).

    Args:
        output_root: The viewer's output handle (overlay membership lookup).
        df: The filtered master mirror (the active filter slice).
        row_col: Y-axis column name.
        time_col: X (time)-axis column name.

    Returns:
        A list of ``{"row_value", "time_value", "cell_ref": (dataset, stem)}``
        dicts. ``cell_ref`` is the ``(dataset, stem)`` tuple consumed by the
        thumbnail + DZI routes.
    """
    needed = [KEY_DATASET, KEY_IMAGE_FILE, row_col, time_col]
    have = [c for c in dict.fromkeys(needed) if c in df.columns]
    slim = df.select(have).drop_nulls(subset=[KEY_DATASET, KEY_IMAGE_FILE]).unique()
    records: list[dict[str, object]] = []
    for record in slim.iter_rows(named=True):
        dataset = str(record[KEY_DATASET])
        image_file = str(record[KEY_IMAGE_FILE])
        stem = Path(image_file).stem if Path(image_file).suffix else image_file
        if not output_root.has_overlay(dataset, stem):
            continue
        records.append(
            {
                "row_value": record.get(row_col, ""),
                "time_value": record.get(time_col, ""),
                "cell_ref": (dataset, stem),
            }
        )
    return records
```

> **Import-surface check (verify before writing).** `_MEASUREMENT_PREFIXES` and
> `_OBJECT_LABEL_COL` are private module-level names in `colony_view/_grid.py`
> (used by `selectable_axis_columns`). Confirm both exist there
> (`grep -n "_MEASUREMENT_PREFIXES\|_OBJECT_LABEL_COL" src/phenotypic/gui/results_viewer/colony_view/_grid.py`).
> If either is spelled differently, import the actual name (or, if cleaner, lift
> the prefix tuple + object-label constant into `_config.py` and import from there
> in both modules — a one-line shared-constant move, no behavior change). Do not
> re-spell the prefix list. **Verified:** `_MEASUREMENT_PREFIXES` (`colony_view/_grid.py:66`)
> and `_OBJECT_LABEL_COL` (`:77`, `= KEY_OBJECT_LABEL`) exist. The dtype check uses
> the per-dtype predicates `dtype.is_numeric()` / `dtype.is_temporal()` (verified:
> `pl.TEMPORAL_DTYPES` is **deprecated since polars 1.0** — do NOT use the module
> constants).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_grid.py -v`
Expected: PASS (all predicate + adapter tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/timeline_view/_grid.py \
  tests/gui/results_viewer/timeline_view/test_grid.py
git commit -m "feat(gui-timeline): time-axis predicate + (dataset,stem) record adapter"
```

---

### Task 4: Timeline-view component ids (`timeline_view/_ids.py`)

**Files:**
- Create: `src/phenotypic/gui/results_viewer/timeline_view/_ids.py`
- Test: `tests/gui/results_viewer/timeline_view/test_ids.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the static `TIMELINE_*` id constants for the tab body. The four
  `TIMELINE_NAV_*` edge buttons + `TIMELINE_POSITION` readout are **DOM targets
  for `timeline.js`** (no Dash callback — the controller binds clicks + keyboard
  + sets the readout in JS). All kebab-case, prefixed `timeline-` so they never
  cross-fire colony/QC pattern callbacks (spec §15.9).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/results_viewer/timeline_view/test_ids.py`:

```python
"""Timeline-view component ids: present, unique, namespaced."""
from __future__ import annotations

from phenotypic.gui.results_viewer.timeline_view import _ids


def test_timeline_ids_present_unique_and_namespaced() -> None:
    timeline_ids = [
        _ids.TIMELINE_GRID,
        _ids.TIMELINE_Y_DROPDOWN,
        _ids.TIMELINE_X_DROPDOWN,
        _ids.TIMELINE_TILE_SIZE_MINUS,
        _ids.TIMELINE_TILE_SIZE_PLUS,
        _ids.TIMELINE_TILE_SIZE_READOUT,
        _ids.TIMELINE_NAV_UP,
        _ids.TIMELINE_NAV_DOWN,
        _ids.TIMELINE_NAV_LEFT,
        _ids.TIMELINE_NAV_RIGHT,
        _ids.TIMELINE_POSITION,
        _ids.TIMELINE_LARGE_AXIS_WARNING,
        _ids.TIMELINE_EMPTY_STATE,
        _ids.TIMELINE_BODY,
        _ids.TIMELINE_STORE_TILE_SIZE,
        _ids.TIMELINE_POPOUT_MODAL,
        _ids.TIMELINE_POPOUT_OSD,
        _ids.TIMELINE_POPOUT_STORE,
        _ids.TIMELINE_POPOUT_INPUT,
        _ids.TIMELINE_POPOUT_OSD_SYNC,
    ]
    assert len(timeline_ids) == len(set(timeline_ids))
    assert all(isinstance(i, str) and i for i in timeline_ids)
    # Namespaced so colony/QC pattern callbacks never cross-fire (spec §15.9).
    assert all(i.startswith("timeline-") for i in timeline_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_ids.py -v`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/results_viewer/timeline_view/_ids.py`:

```python
"""Component ids for the results-viewer Timeline tab.

All ids are kebab-case and prefixed ``timeline-`` so the timeline's
pattern/clientside callbacks never collide with the colony/QC surfaces
(spec §15.9). The four ``TIMELINE_NAV_*`` edge buttons and the
``TIMELINE_POSITION`` readout are DOM targets driven by ``timeline.js`` —
they carry NO Dash callback (the controller wires clicks + keyboard in JS).
"""
from __future__ import annotations

#: Grid container (the focus-navigate controller's attach target).
TIMELINE_GRID = "timeline-grid"
#: The whole Timeline tab body wrapper.
TIMELINE_BODY = "timeline-body"

#: Y (row) axis dropdown — uncapped selectable_axis_columns.
TIMELINE_Y_DROPDOWN = "timeline-y-dropdown"
#: X (time) axis dropdown — selectable_time_columns.
TIMELINE_X_DROPDOWN = "timeline-x-dropdown"

#: Colony-style tile-size stepper.
TIMELINE_TILE_SIZE_MINUS = "timeline-tile-size-minus"
TIMELINE_TILE_SIZE_PLUS = "timeline-tile-size-plus"
TIMELINE_TILE_SIZE_READOUT = "timeline-tile-size-readout"
TIMELINE_STORE_TILE_SIZE = "timeline-store-tile-size"

#: Focus-and-navigate edge buttons + position readout (spec §16) — JS targets.
TIMELINE_NAV_UP = "timeline-nav-up"
TIMELINE_NAV_DOWN = "timeline-nav-down"
TIMELINE_NAV_LEFT = "timeline-nav-left"
TIMELINE_NAV_RIGHT = "timeline-nav-right"
TIMELINE_POSITION = "timeline-position"

#: Bucketing-warning banner for a very long time axis (spec §15.2).
TIMELINE_LARGE_AXIS_WARNING = "timeline-large-axis-warning"
#: Guided empty state shown when no eligible time column exists (D9).
TIMELINE_EMPTY_STATE = "timeline-empty-state"

#: Deep-zoom pop-out (reuses the viewer's /tiles DZI route + OSD).
TIMELINE_POPOUT_MODAL = "timeline-popout-modal"
TIMELINE_POPOUT_OSD = "timeline-popout-osd"
TIMELINE_POPOUT_STORE = "timeline-popout-store"        # {dataset, stem}
TIMELINE_POPOUT_INPUT = "timeline-popout-input"        # hidden JS→Dash bridge
TIMELINE_POPOUT_OSD_SYNC = "timeline-popout-osd-sync"  # clientside-callback sink

__all__ = [
    "TIMELINE_GRID",
    "TIMELINE_BODY",
    "TIMELINE_Y_DROPDOWN",
    "TIMELINE_X_DROPDOWN",
    "TIMELINE_TILE_SIZE_MINUS",
    "TIMELINE_TILE_SIZE_PLUS",
    "TIMELINE_TILE_SIZE_READOUT",
    "TIMELINE_STORE_TILE_SIZE",
    "TIMELINE_NAV_UP",
    "TIMELINE_NAV_DOWN",
    "TIMELINE_NAV_LEFT",
    "TIMELINE_NAV_RIGHT",
    "TIMELINE_POSITION",
    "TIMELINE_LARGE_AXIS_WARNING",
    "TIMELINE_EMPTY_STATE",
    "TIMELINE_POPOUT_MODAL",
    "TIMELINE_POPOUT_OSD",
    "TIMELINE_POPOUT_STORE",
    "TIMELINE_POPOUT_INPUT",
    "TIMELINE_POPOUT_OSD_SYNC",
]
```

> **Tile-size stepper helper.** Before Task 6, confirm whether Phase 2 added
> `step_timeline_tile_size` to `phenotypic.gui._config` (its review note W6 claims
> so). `grep -n "step_timeline_tile_size" src/phenotypic/gui/_config.py`. If absent,
> add it (and to `__all__`) mirroring `step_colony_tile_size` over the
> `TIMELINE_TILE_SIZE_*` constants — a ~5-line pure function, no `dash` import. Note
> it in this task's commit if added here.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_ids.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/timeline_view/_ids.py \
  tests/gui/results_viewer/timeline_view/test_ids.py
git commit -m "feat(gui-timeline): Results Timeline component ids"
```

---

### Task 5: Timeline thumbnail route (`timeline_view/_thumb_routes.py`)

**Files:**
- Create: `src/phenotypic/gui/results_viewer/timeline_view/_thumb_routes.py`
- Test: `tests/gui/results_viewer/timeline_view/test_thumb_routes.py`

**Interfaces:**
- Consumes: `register_thumbnail_route`, `ThumbUnavailable` (Phase 1);
  `VIEWER_THUMB_URL_SEGMENT`, `VIEWER_CACHE_DIRNAME` (Phase 1/`_config`);
  `OutputRoot.overlay_path` / `has_overlay`; `is_safe_path_component`
  (`gui/_shared/tiles`) for the identity guard.
- Produces:
  - `encode_cell_ref(dataset: str, stem: str) -> str` — encodes the `(dataset, stem)`
    pair into a single URL-path identity (the `ref_builder` the grid uses for
    `data-ref`/`data-src`). Use a separator the route can split unambiguously.
  - `decode_cell_ref(identity: str) -> tuple[str, str]`.
  - `register(app: dash.Dash, output_root: OutputRoot) -> None` — mounts the Phase 1
    factory under `VIEWER_THUMB_URL_SEGMENT` with `cache_base =
    output_root.root / VIEWER_CACHE_DIRNAME / "timeline_thumbs"` and a
    `resolve_source(identity)` that decodes → guards → resolves the overlay PNG via
    `overlay_path`, raising `FileNotFoundError` (→404) for an unknown/missing
    overlay. (Overlay PNGs are plain 8-bit RGB, always decodable, so `ThumbUnavailable`
    (→422) is reserved for a genuinely-undecodable source; resolve it if
    `downscale_to_thumb` would choke, but the happy path never raises it.) Per spec
    §15.6, do NOT route the warm sweep through the small `_load_overlay_rgb` LRU —
    the Phase 1 factory decodes the file and relies on its own disk cache.

**Identity encoding (verify the separator choice).** `register_thumbnail_route`
mounts `GET /<segment>/<path:identity>` (Phase 1 Task 6) — `<path:...>` allows `/`.
So `encode_cell_ref` can join as `f"{dataset}/{stem}"` and `decode_cell_ref` splits
on the **last** `/` (a stem cannot contain `/`; a dataset name is a single path
component per `_scan_overlay_index`). Guard both halves with `is_safe_path_component`
(reused from the DZI route) before touching disk. Confirm `<path:identity>` is what
the Phase 1 factory uses; if it uses a non-path converter, switch the separator to a
sentinel that survives a single path segment (e.g. base64url of `"dataset\x1fstem"`).

- [ ] **Step 1: Write the failing test**

Create `tests/gui/results_viewer/timeline_view/test_thumb_routes.py`:

```python
"""Timeline thumbnail route: (dataset, stem) → cached downscaled overlay PNG."""
from __future__ import annotations

import io
from pathlib import Path

import dash
import polars as pl
from PIL import Image as PILImage

from phenotypic.gui._config import BROWSE_THUMB_URL_SEGMENT, VIEWER_THUMB_URL_SEGMENT
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _thumb_routes
from tests._output_layout import write_master, write_measurements_mirror


def test_viewer_and_browse_thumb_segments_are_distinct() -> None:
    # S2: the two surfaces' thumbnail routes must not collide if ever co-mounted
    # on one Flask server; the Browse segment is "thumb", the viewer's is
    # "timeline-thumb" (Phase 1 _config).
    assert VIEWER_THUMB_URL_SEGMENT == "timeline-thumb"
    assert BROWSE_THUMB_URL_SEGMENT == "thumb"
    assert VIEWER_THUMB_URL_SEGMENT != BROWSE_THUMB_URL_SEGMENT


def _output_root(tmp_path: Path) -> OutputRoot:
    cli_out = tmp_path / "out"
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds", "ds"],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_ImageNumber": pl.Series([1, 2], dtype=pl.Int64),
            "Object_Label": [1, 2],
            "Size_Area": [1.0, 2.0],
        }
    )
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    overlays = cli_out / "results" / "ds" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (200, 100), (0, 64, 128)).save(overlays / "a.png")
    PILImage.new("RGB", (200, 100), (0, 64, 128)).save(overlays / "b.png")
    return OutputRoot.discover(cli_out)


def _client(tmp_path: Path):
    root = _output_root(tmp_path)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()  # layout-less Dash 500s on first request
    _thumb_routes.register(app, root)
    return app.server.test_client(), root


def test_cell_ref_round_trips() -> None:
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    assert _thumb_routes.decode_cell_ref(ident) == ("ds", "a")


def test_thumb_happy_path_returns_bucketed_png(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    resp = client.get(f"/timeline-thumb/{ident}?size=100")  # snaps to bucket 128
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    out = PILImage.open(io.BytesIO(resp.data))
    assert max(out.size) == 128


def test_thumb_unknown_pair_is_404(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    ident = _thumb_routes.encode_cell_ref("ds", "does-not-exist")
    assert client.get(f"/timeline-thumb/{ident}?size=128").status_code == 404


def test_thumb_unsafe_identity_is_404(tmp_path: Path) -> None:
    client, _root = _client(tmp_path)
    assert client.get("/timeline-thumb/..%2F..%2Fetc/passwd?size=128").status_code == 404


def test_thumb_cache_persists_under_viewer_cache(tmp_path: Path) -> None:
    client, root = _client(tmp_path)
    ident = _thumb_routes.encode_cell_ref("ds", "a")
    assert client.get(f"/timeline-thumb/{ident}?size=128").status_code == 200
    cache_dir = root.root / ".viewer_cache" / "timeline_thumbs"
    assert cache_dir.is_dir()
    assert list(cache_dir.glob("*.png"))  # a cached thumbnail was written
```

> The 422 path is covered by Phase 1's `register_thumbnail_route` route tests
> (the factory maps `ThumbUnavailable` → 422); here the overlay is always
> decodable, so this suite verifies happy/404/unsafe/cache. If you want an explicit
> 422 here, add a `resolve_source` monkeypatch test raising `ThumbUnavailable`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_thumb_routes.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/gui/results_viewer/timeline_view/_thumb_routes.py`:

```python
"""Mount the Results Timeline thumbnail route (overlay → cached downscale).

Thin adapter over the Phase 1 ``register_thumbnail_route`` factory: the
resolver decodes a ``(dataset, stem)`` identity, guards both halves with the
DZI route's path-component check, and resolves the overlay PNG via
``OutputRoot.overlay_path``. The factory downscales it to the requested
size bucket and serves a self-invalidating, atomically-written disk cache
under the output root's ``.viewer_cache/timeline_thumbs`` (persists with the
run). Per spec §15.6 the warm sweep decodes the file and relies on the disk
cache — it does NOT lean on the small ``_load_overlay_rgb`` LRU.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash

from phenotypic.gui._config import VIEWER_CACHE_DIRNAME, VIEWER_THUMB_URL_SEGMENT
from phenotypic.gui._shared.tiles import is_safe_path_component
from phenotypic.gui._shared.timeline import ThumbUnavailable, register_thumbnail_route
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

__all__ = ["register", "encode_cell_ref", "decode_cell_ref"]

#: Subdir of the output root's viewer cache for downscaled overlay thumbnails.
_THUMB_CACHE_SUBDIR = "timeline_thumbs"


def encode_cell_ref(dataset: str, stem: str) -> str:
    """Encode a ``(dataset, stem)`` pair into a single URL-path identity.

    A stem never contains ``/`` and a dataset is a single path component,
    so ``dataset/stem`` round-trips by splitting on the LAST ``/``.
    """
    return f"{dataset}/{stem}"


def decode_cell_ref(identity: str) -> tuple[str, str]:
    """Inverse of :func:`encode_cell_ref` (split on the last ``/``)."""
    dataset, _, stem = identity.rpartition("/")
    return dataset, stem


def register(app: dash.Dash, output_root: OutputRoot) -> None:
    """Mount the ``(dataset, stem)`` thumbnail route on ``app.server``."""

    def resolve_source(identity: str) -> Path:
        dataset, stem = decode_cell_ref(identity)
        if not (dataset and stem):
            raise FileNotFoundError(identity)
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            raise FileNotFoundError(identity)
        if not output_root.has_overlay(dataset, stem):
            raise FileNotFoundError(identity)
        overlay = output_root.overlay_path(dataset, stem)
        if not overlay.is_file():
            raise FileNotFoundError(identity)
        return overlay

    register_thumbnail_route(
        app,
        segment=VIEWER_THUMB_URL_SEGMENT,
        resolve_source=resolve_source,
        cache_base=output_root.root / VIEWER_CACHE_DIRNAME / _THUMB_CACHE_SUBDIR,
    )
    logger.debug(
        "Registered Results Timeline thumbnail route under /%s for root=%s",
        VIEWER_THUMB_URL_SEGMENT,
        output_root.root,
    )
```

> `ThumbUnavailable` is imported for symmetry/forward-compat (overlays decode
> reliably, so it is not raised here). If ruff flags it as unused, either drop the
> import or add an explicit monkeypatched 422 test that raises it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_thumb_routes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/timeline_view/_thumb_routes.py \
  tests/gui/results_viewer/timeline_view/test_thumb_routes.py
git commit -m "feat(gui-timeline): Results Timeline thumbnail route (overlay → downscale)"
```

---

### Task 6: Timeline tab body layout + callbacks (`_layout.py`, `_callbacks.py`)

**Files:**
- Create: `src/phenotypic/gui/results_viewer/timeline_view/_layout.py`
- Create: `src/phenotypic/gui/results_viewer/timeline_view/_callbacks.py`
- Test: `tests/gui/results_viewer/timeline_view/test_layout.py`
- Test: `tests/integration/gui/...` (a Flask-test-client render test — see Step 1b)

**Interfaces:**
- Consumes: Task 3 (`selectable_time_columns`, `is_large_time_axis`,
  `has_eligible_time_axis`, `build_timeline_records`), Task 4 ids, Task 5
  (`encode_cell_ref`), `selectable_axis_columns(max_cardinality=None)` (Task 2),
  `build_matrix` + `build_timeline_grid` (Phase 1), `FilterSpec` (filter slice),
  `step_timeline_tile_size` + `TIMELINE_TILE_SIZE_*` + `snap_thumb_bucket` +
  `VIEWER_THUMB_URL_SEGMENT` (`_config`), `window.__phenotypicAppPrefix` (via the
  thumbnail URL prefix).
- Produces:
  - `layout(output_root) -> Component` — the static tab body: toolbar (Y dropdown,
    X dropdown, tile-size stepper), the large-axis warning banner (hidden), the
    guided empty state (hidden by default; shown when no time column), the
    **no-scroll centered-window viewport** wrapping the `TIMELINE_GRID` container
    + the four edge nav buttons + position readout, the stores, and the pop-out
    modal (a `dbc.Modal` hosting an OSD div + the hidden `TIMELINE_POPOUT_INPUT`
    bridge + the `TIMELINE_POPOUT_OSD_SYNC` sink store). Mirrors `colony_view._layout`'s
    toolbar styling/tokens and the Browse Timeline body structure (Phase 2 Task 6).
  - `register_callbacks(app, output_root) -> None` — the server + clientside
    callbacks (see below).
  - A pure `build_timeline_grid_component(output_root, df, *, row_col, time_col,
    tile_size) -> tuple[Component, bool, int]` helper returning
    `(grid_component_or_empty_state, show_empty_state, n_time_values)` so the
    render is unit-testable without Dash callback machinery.

**Static data-attrs on the grid container (mirror Phase 2):** the `TIMELINE_GRID`
container carries `data-focus-margin = str(TIMELINE_FOCUS_MARGIN)`,
`data-mount-cap = str(TIMELINE_MOUNT_CAP)`, `data-warm-concurrency =
str(TIMELINE_WARM_CONCURRENCY)` as **static** attrs written once in `layout()` — the
render callback replaces only the container's CHILDREN, never these attrs (same
constraint Phase 2 documents).

**The controller-required CLASSES must be on the Results layout (CRITICAL — C2).**
The surface-agnostic `timeline.js` (Decision #1) finds every control **by class
scoped to `.timeline-body`**, never by id. The Results layout therefore must carry
each `timeline-*` class **alongside** its id (the same dual id+class pattern Phase 2
uses on Browse). Without these classes the controller silently finds nothing. Mirror
Phase 2 exactly:

| Element | id (`timeline_view/_ids.py`) | REQUIRED class |
|---|---|---|
| Tab body wrapper (scope anchor) | `TIMELINE_BODY` | `timeline-body` |
| No-scroll viewport (focusable, `tabIndex=0`) | (wrapper) | `timeline-viewport` |
| Grid container | `TIMELINE_GRID` | `timeline-grid-container` |
| Edge button ▲ | `TIMELINE_NAV_UP` | `timeline-nav-up` |
| Edge button ▼ | `TIMELINE_NAV_DOWN` | `timeline-nav-down` |
| Edge button ◀ | `TIMELINE_NAV_LEFT` | `timeline-nav-left` |
| Edge button ▶ | `TIMELINE_NAV_RIGHT` | `timeline-nav-right` |
| Position readout | `TIMELINE_POSITION` | `timeline-position` |
| Hidden pop-out bridge `dcc.Input(id=TIMELINE_POPOUT_INPUT)` | `TIMELINE_POPOUT_INPUT` | `timeline-popout-bridge` |

(The controller's `startReattachObserver` discovers the grid by
`.timeline-grid-container` and re-attaches via that element's own id — which is why
the grid id is `timeline-grid` so `attach("timeline-grid")` and the observer agree.)
Do **not** edit or parameterize the controller; the classes are the contract. Step 1
adds a pure component-tree-walk test (`test_controller_required_classes_present`)
asserting all five class families are present, so a missing class fails fast in unit
CI — the e2e is not the only net.

- [ ] **Step 1: Write the failing test (a) — layout ids + empty state**

Create `tests/gui/results_viewer/timeline_view/test_layout.py`:

```python
"""Timeline tab body: ids present + empty-state predicate wiring (pure)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _ids
from phenotypic.gui.results_viewer.timeline_view._layout import (
    build_timeline_grid_component,
    layout,
)
from tests._output_layout import write_master, write_measurements_mirror


def _walk(component):
    stack = [component]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)


def _walk_ids(component) -> set[str]:
    return {
        cid
        for node in _walk(component)
        if isinstance((cid := getattr(node, "id", None)), str)
    }


def _walk_classnames(component) -> set[str]:
    classes: set[str] = set()
    for node in _walk(component):
        cls = getattr(node, "className", None)
        if isinstance(cls, str):
            classes.update(cls.split())
    return classes


def _root(tmp_path: Path, *, with_time: bool) -> OutputRoot:
    cli_out = tmp_path / "out"
    cols = {
        "Metadata_Dataset": ["ds", "ds"],
        "Metadata_ImageFile": ["a", "b"],
        "Metadata_PlateNum": ["1", "2"],
        "Object_Label": [1, 2],
        "Size_Area": [1.0, 2.0],
    }
    if with_time:
        cols["Metadata_ImageNumber"] = pl.Series([1, 2], dtype=pl.Int64)
    df = pl.DataFrame(cols)
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    overlays = cli_out / "results" / "ds" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for stem in ("a", "b"):
        PILImage.new("RGB", (40, 30), (1, 2, 3)).save(overlays / f"{stem}.png")
    return OutputRoot.discover(cli_out)


def test_layout_mounts_all_focus_navigate_chrome(tmp_path: Path) -> None:
    ids = _walk_ids(layout(_root(tmp_path, with_time=True)))
    for cid in (
        _ids.TIMELINE_GRID,
        _ids.TIMELINE_Y_DROPDOWN,
        _ids.TIMELINE_X_DROPDOWN,
        _ids.TIMELINE_TILE_SIZE_READOUT,
        _ids.TIMELINE_NAV_UP,
        _ids.TIMELINE_NAV_DOWN,
        _ids.TIMELINE_NAV_LEFT,
        _ids.TIMELINE_NAV_RIGHT,
        _ids.TIMELINE_POSITION,
        _ids.TIMELINE_POPOUT_MODAL,
    ):
        assert cid in ids


def test_empty_state_when_no_time_column(tmp_path: Path) -> None:
    root = _root(tmp_path, with_time=False)
    _component, show_empty, _n = build_timeline_grid_component(
        root, root.master_df, row_col="Metadata_PlateNum", time_col=None, tile_size=150
    )
    assert show_empty is True


def test_grid_renders_when_time_column_present(tmp_path: Path) -> None:
    root = _root(tmp_path, with_time=True)
    _component, show_empty, n = build_timeline_grid_component(
        root,
        root.master_df,
        row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    assert show_empty is False
    assert n == 2  # two distinct image numbers


def test_controller_required_classes_present(tmp_path: Path) -> None:
    # C2: timeline.js is surface-agnostic and finds controls BY CLASS scoped to
    # .timeline-body. The Results layout must carry every controller-required
    # class or the controller silently finds nothing (the e2e is not the only net).
    classes = _walk_classnames(layout(_root(tmp_path, with_time=True)))
    required = {
        "timeline-body",
        "timeline-viewport",
        "timeline-grid-container",
        "timeline-nav-up",
        "timeline-nav-down",
        "timeline-nav-left",
        "timeline-nav-right",
        "timeline-position",
        "timeline-popout-bridge",
    }
    missing = required - classes
    assert not missing, f"layout is missing controller classes: {sorted(missing)}"


def test_one_column_matrix_when_filtered_to_single_time(tmp_path: Path) -> None:
    # S3: filtering down to a single image-number must still render a sensible
    # 1-column matrix (not an empty/degenerate grid).
    root = _root(tmp_path, with_time=True)
    single = root.master_df.filter(pl.col("Metadata_ImageNumber") == 1)
    _component, show_empty, n = build_timeline_grid_component(
        root, single, row_col="Metadata_PlateNum", time_col="Metadata_ImageNumber",
        tile_size=150,
    )
    assert show_empty is False
    assert n == 1  # one time column survives the filter


def test_representative_reaching_builders_is_the_raw_tuple(tmp_path: Path) -> None:
    # C3: build_matrix keeps the representative as the raw (dataset, stem) TUPLE
    # (min by str(...), object stored), so ref_builder=lambda r: encode_cell_ref(*r)
    # and the url_builder both receive a tuple — not a stringified pair. Assert the
    # emitted data-ref/data-src reflect encode_cell_ref(dataset, stem), proving the
    # representative was unpackable. A future Phase 1 change that stringified the
    # representative would break this.
    root = _root(tmp_path, with_time=True)
    component, _show_empty, _n = build_timeline_grid_component(
        root, root.master_df, row_col="Metadata_PlateNum",
        time_col="Metadata_ImageNumber", tile_size=150,
    )
    refs: set[str] = set()
    for node in _walk(component):
        # Dash components expose their props via to_plotly_json(); the cell
        # placeholders carry data-ref = encode_cell_ref(dataset, stem).
        to_json = getattr(node, "to_plotly_json", None)
        if to_json is None:
            continue
        props = to_json().get("props", {})
        ref = props.get("data-ref")
        if isinstance(ref, str):
            refs.add(ref)
    # At least one populated cell's data-ref is encode_cell_ref("ds", stem) ==
    # "ds/<stem>" — proving build_matrix kept the representative as the raw
    # (dataset, stem) tuple that ref_builder/url_builder unpacked.
    assert any(r.startswith("ds/") for r in refs)
```

> `test_representative_reaching_builders_is_the_raw_tuple` guards the C3 dependency:
> `build_timeline_records` puts the `(dataset, stem)` tuple in `cell_ref`, and Phase 1's
> `build_matrix` selects the representative with `min(members, key=lambda m: str(m))`
> but **stores the object itself**, so `ref_builder`/`url_builder` receive the tuple
> unstringified (`encode_cell_ref(*ref)` works only because of this). If a future
> Phase 1 change stringified the representative, `encode_cell_ref(*ref)` would receive
> a `str` and either `TypeError` or produce a wrong `data-ref` — this test catches it.
> Adjust the `to_plotly_json()` walk to whatever cleanly reads `data-ref` off the
> emitted cells in the actual Dash component build; the load-bearing assertion is that
> at least one populated cell's `data-ref` equals `encode_cell_ref(dataset, stem)`.

- [ ] **Step 1b: Write the failing test (b) — Flask-test-client thumb-URL render**

Add an integration render check under `tests/integration/gui/` (the existing
no-browser Flask-test-client lane). **Name the test function/file with `timeline` in
it** (e.g. `test_timeline_thumb_url_resolves`) so the Step-4 `-k timeline` selector
catches it (M5). Assert that, with a time column present, the rendered grid's first
cell `data-src` points at `/<VIEWER_THUMB_URL_SEGMENT>/...` and resolves 200 through
the registered route. (Reuse the Task 5 `_output_root` helper + register both
`_thumb_routes.register` and the layout; assert the `data-src` URL the grid emitted
returns a 200 PNG.) This proves the `url_builder` prefix + `encode_cell_ref` wiring
end-to-end without a browser.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_layout.py -v`
Expected: FAIL (`ModuleNotFoundError: ..._layout`).

- [ ] **Step 3: Write minimal implementation**

Create `timeline_view/_layout.py`. Key points:
- `build_timeline_grid_component(output_root, df, *, row_col, time_col, tile_size)`:
  - If `time_col is None` or `not has_eligible_time_axis(df, output_root.column_value_sets)`
    → return `(build_empty_state(), True, 0)`.
  - Else build records via `build_timeline_records(output_root, df, row_col=row_col,
    time_col=time_col)`, `matrix = build_matrix(records)`, then
    `grid, _order = build_timeline_grid(matrix, url_builder=..., display_size=tile_size,
    fetch_size=snap_thumb_bucket(tile_size), ref_builder=lambda ref:
    encode_cell_ref(*ref))`. The `url_builder(ref, fetch)` builds
    `f"{prefix}{VIEWER_THUMB_URL_SEGMENT}/{encode_cell_ref(*ref)}?size={fetch}"`
    where `prefix` is the mount prefix (read from `app.server.config[CFG_URL_PREFIX]`
    at callback time, or `"/"` for the pure unit path — pass it in). Return
    `(grid, False, len(matrix.columns))`.
  - `ref` here is the `(dataset, stem)` tuple `build_timeline_records` put in
    `cell_ref`; `build_matrix`'s representative is the smallest `str(cell_ref)`, so
    `ref_builder`/`url_builder` receive a `(dataset, stem)` tuple — unpack with
    `encode_cell_ref(*ref)`.
- `build_empty_state()` → a `dbc.Alert`/`html.Div` with id `TIMELINE_EMPTY_STATE`:
  *"The Timeline needs a time field. Re-run with `--metadata <csv>` (or add a post
  step like `ExpandMetadata`) so a column such as `Metadata_Time` / `Metadata_ImageNumber`
  is available. Pick a **monotonic** column (e.g. image number) — a time-of-day
  column mis-orders across days."* (spec §6.3/§16.6).
- The large-axis warning banner (id `TIMELINE_LARGE_AXIS_WARNING`, hidden) text:
  *"This time axis has N points — rendering may be dense; time-bucketing is not yet
  available."* shown by the callback when `is_large_time_axis(n)`.
- `layout(output_root)` assembles toolbar + warning + empty-state + the no-scroll
  viewport + the `TIMELINE_STORE_TILE_SIZE` store + the pop-out `dbc.Modal`. **Every
  controller-facing element MUST carry its `timeline-*` class alongside its id (C2 —
  the controller finds controls by class, see the table above):**
  - the tab body wrapper → `html.Div(id=TIMELINE_BODY, className="timeline-body", ...)`;
  - the focusable (`tabIndex=0`) no-scroll viewport wrapper → `className="timeline-viewport"`;
  - the grid container → `html.Div(id=TIMELINE_GRID, className="timeline-grid-container", ...)`
    with the static `data-focus-margin`/`data-mount-cap`/`data-warm-concurrency` attrs;
  - the four edge buttons → `className="timeline-nav-up|down|left|right"` (one each),
    positioned at the viewport edges;
  - the position readout → `html.Span(id=TIMELINE_POSITION, className="timeline-position", ...)`;
  - the hidden pop-out bridge → `dcc.Input(id=TIMELINE_POPOUT_INPUT,
    className="timeline-popout-bridge", type="text", value="", style={"display":"none"})`.
  The pop-out `dbc.Modal` hosts the OSD div `TIMELINE_POPOUT_OSD`, the
  `TIMELINE_POPOUT_OSD_SYNC` sink store, and the payload store `TIMELINE_POPOUT_STORE`.
  Reuse `_design` tokens and the colony toolbar styling. (If a class needs supporting
  CSS — e.g. `.timeline-viewport { overflow: hidden }`, the `:hover` ⤢ reveal — add it
  to `results_viewer/_assets/results_viewer.css`, mirroring Browse's `browse.css`
  timeline rules.)

Create `timeline_view/_callbacks.py` with `register_callbacks(app, output_root)`:
- `df = output_root.master_df`; `column_value_sets = output_root.column_value_sets`;
  `url_prefix = app.server.config.get(CFG_URL_PREFIX, MOUNT_HOME)`.
- **Populate dropdowns** (reacting to `Input(STORE_FILTER_SPEC, "data")`, like the
  colony dropdown callback): slice `filtered = FilterSpec.from_store(payload).apply_to(df)`;
  `y_opts = selectable_axis_columns(filtered, column_value_sets, max_cardinality=None)`;
  `x_opts = selectable_time_columns(filtered, column_value_sets)`. Preserve the
  current value if still valid; default Y to the first option and X to the first
  time option (prefer a numeric/`ImageNumber`-like default if present). Output the
  options + values for `TIMELINE_Y_DROPDOWN` / `TIMELINE_X_DROPDOWN`.
- **Render the grid** (reacting to filter spec, both dropdown values, the tile-size
  store, and `Input(TABS_ID, "active_tab")` so it (re)builds when the Timeline tab
  activates — gate on `active_tab == TAB_TIMELINE_ID` and `raise PreventUpdate`
  off-tab, mirroring the Error tab's `active_tab` gate): slice the filtered df, call
  `build_timeline_grid_component(output_root, filtered, row_col=y, time_col=x,
  tile_size=size)` (passing `url_prefix` into the closure), output the grid into
  `TIMELINE_GRID.children`, toggle the empty-state + large-axis warning visibility,
  and set the warning text.
- **Tile-size stepper**: a callback over `TIMELINE_TILE_SIZE_MINUS/PLUS` →
  `step_timeline_tile_size` → `TIMELINE_STORE_TILE_SIZE` + readout (mirror
  `stepped_colony_tile_size_from_trigger`).
- **Clientside attach**: after the grid updates, a clientside callback calls
  `window.__phenotypicTimeline.attach("timeline-grid")` (reset focus to first
  populated cell + render the centered window). Sink to a throwaway dcc.Store
  (declare one in the body or reuse `TIMELINE_POPOUT_OSD_SYNC` is wrong — use a
  dedicated `dcc.Store`; Dash requires every clientside callback to have an Output).
  Trigger on `Input(TIMELINE_GRID, "children")` + `Input(TABS_ID, "active_tab")`.
- **Pop-out** (the controller already owns both triggers — hover-⤢ click and
  Enter/Space on the focused cell — and writes the cell's `(dataset, stem)` identity
  into the `.timeline-popout-bridge` hidden input, scoped by class; here add only the
  Dash side):
  - A server callback `Input(TIMELINE_POPOUT_INPUT, "value")` → `raise PreventUpdate`
    on empty `""` (the first-load value) → decode the bridge value → write
    `{dataset, stem}` into `TIMELINE_POPOUT_STORE` + open the modal.
  - A **clientside** callback `Input(TIMELINE_POPOUT_STORE, "data")` mounts an OSD
    viewer in `TIMELINE_POPOUT_OSD` at the viewer's existing DZI URL. **Verified
    real idiom (M4):** the viewer's OSD bridge lives on
    `window.__phenotypicResultsViewer` — `ns.mountViewer(divId, dziUrl)` /
    `ns.applyImageSelection(states)` (`results_viewer.js`, the namespace section);
    `applyImageSelection` builds the URL as `appPrefix + "tiles/" +
    encodeURIComponent(dataset) + "/" + encodeURIComponent(stem) + ".dzi"`
    (`results_viewer.js` ≈`:298`; note `appPrefix` already ends in `/`, so it is
    `"tiles/…"` not `"/tiles/…"`). The Python side bridges it via the
    `ids.OSD_MOUNT_TRIGGER_ID` clientside-callback pattern
    (`results_viewer/_callbacks.py:123`, `_register_clientside_callbacks` →
    `Output(ids.OSD_MOUNT_TRIGGER_ID, "data")` calling `ns.applyImageSelection`). So
    the pop-out clientside callback should call `ns.mountViewer("timeline-popout-osd",
    appPrefix + "tiles/" + encodeURIComponent(dataset) + "/" + encodeURIComponent(stem)
    + ".dzi")` and sink to `TIMELINE_POPOUT_OSD_SYNC` (`return ""`). The DZI route is
    the SAME `/tiles/<dataset>/<stem>.dzi` blueprint Plate/Colony already use
    (`_tile_routes.py`, `VIEWER_TILES_PREFIX = "/tiles"`) — already registered, no new
    route.
  - **Repeat-open (OQ-6):** the controller writes the bridge `value` to trigger Dash.
    Writing the same `(dataset, stem)` twice would dedupe (no re-fire), so a second
    Enter on the same focused cell wouldn't reopen the modal. **Reuse Phase 2's exact
    bridge-write convention verbatim** — the Browse controller must already solve this
    (e.g. a nonce-suffixed value like `"dataset/stem#<ts>"`, or a Dash callback that
    clears `TIMELINE_POPOUT_INPUT.value` to `""` after consuming it). Whatever Phase 2
    does, the Results Dash side must (a) strip any nonce suffix before decoding and
    (b) match the same clear-or-nonce pattern so repeat-opens re-fire. Do **not**
    invent a new convention. The Task 7 e2e `test_repeat_enter_reopens_popout`
    (OQ-6) asserts two consecutive Enters on the same focused cell both open the modal.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_layout.py tests/integration/gui -k timeline -v`
Expected: PASS — the 6 `test_layout.py` cases (chrome ids present; empty state;
grid renders; **controller-required classes present**; **one-column matrix when
filtered to a single time**; **representative reaching builders is the raw tuple**)
plus the Step-1b integration thumb-URL render test.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/timeline_view/_layout.py \
  src/phenotypic/gui/results_viewer/timeline_view/_callbacks.py \
  tests/gui/results_viewer/timeline_view/test_layout.py \
  tests/integration/gui/
git commit -m "feat(gui-timeline): Results Timeline tab body + render/dropdown/popout callbacks"
```

---

### Task 7: Wire the tab into the viewer + e2e (focus-navigate over overlays, spec §16.9)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (add the 6th `dbc.Tab`)
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py` (dispatch timeline callbacks)
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (register the thumb route)
- Test: `tests/e2e/gui/test_results_timeline.py` (Playwright)

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: a fully-mounted Timeline tab served by `create_app`, with the focus-
  navigate controller live over real overlays.

- [ ] **Step 1: Write the failing test**

Create `tests/e2e/gui/test_results_timeline.py`. Mirror `test_heatmap_tab.py`'s
fixture wiring exactly (function-scoped `fake_sandbox` → `_build_sandbox` +
`_seed_master_df_in_output`-style seeding, function-scoped `live_server`, and a
`_hand_off_viewer` page helper POSTing to `/sandbox/api/viewer/output-root`). Seed a
matrix big enough that the focus window does NOT swallow it (≥ ~6 plates ×
≥ ~10 image numbers) so the bounded-window assertions are meaningful, with overlays
for every `(dataset, stem)` and a mirror parquet carrying `Metadata_ImageNumber`
(Int64 monotonic) + `Metadata_PlateNum`.

```python
"""Playwright e2e: Results-viewer Timeline tab focus-and-navigate (spec §16.9)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page

from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import _build_sandbox, _start_live_server

# Tight DOM-poll budget on a fresh Werkzeug server: stochastically slow on GHA.
pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "CliOutputExample"
_DATASET = "ds1"
_N_PLATES = 6
_N_TIMES = 12


def _timeline_master_df() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    label = 0
    for plate in range(1, _N_PLATES + 1):
        for img_no in range(1, _N_TIMES + 1):
            label += 1
            rows.append(
                {
                    "Metadata_Dataset": _DATASET,
                    "Metadata_ImageFile": f"p{plate}_t{img_no}",
                    "Metadata_ImageNumber": img_no,
                    "Metadata_PlateNum": str(plate),
                    "Object_Label": label,
                    "Size_Area": float(plate * 10 + img_no),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("Metadata_ImageNumber").cast(pl.Int64)
    )


def _seed(sandbox: Path) -> Path:
    cli_out = sandbox / "results" / _OUTPUT_NAME
    df = _timeline_master_df()
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    overlays = cli_out / "results" / _DATASET / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for plate in range(1, _N_PLATES + 1):
        for img_no in range(1, _N_TIMES + 1):
            PILImage.new("RGB", (160, 120), (20, 40, 60)).save(
                overlays / f"p{plate}_t{img_no}.png"
            )
    return cli_out


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    sandbox = _build_sandbox(tmp_path)
    _seed(sandbox)
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    return live_server


def _open_timeline(page: Page, hub_url: str) -> None:
    # Hand off the seeded output to the viewer, then open the Timeline tab.
    page.goto(hub_url + "/")
    page.wait_for_load_state("networkidle")
    resp = page.evaluate(
        """async (path) => {
            const r = await fetch('/sandbox/api/viewer/output-root', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path}),
            });
            return {status: r.status, body: await r.text()};
        }""",
        f"results/{_OUTPUT_NAME}",
    )
    assert resp["status"] == 200, resp
    page.goto(hub_url + "/results/")
    page.wait_for_selector("a.nav-link", timeout=15_000)
    page.locator("a.nav-link", has_text="Timeline").first.click()
    page.wait_for_selector(".timeline-cell[data-src]", timeout=15_000)


def test_y_and_x_dropdowns_populate(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    # Y offers the high-cardinality plate number (uncapped, spec §16.5);
    # X offers the numeric image number (selectable_time_columns).
    page.wait_for_selector("#timeline-y-dropdown")
    page.wait_for_selector("#timeline-x-dropdown")


def test_focus_starts_on_first_populated_cell(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    assert page.eval_on_selector_all(".timeline-cell--focused", "e => e.length") == 1


def test_arrow_right_moves_focus_and_mounts_neighborhood(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-col-index') === '1'"
    )
    page.wait_for_function(
        "document.querySelectorAll('#timeline-grid .timeline-cell img').length > 0"
    )


def test_edge_button_down_moves_focus(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click("#timeline-nav-down")
    page.wait_for_function(
        "document.querySelector('.timeline-cell--focused')"
        ".getAttribute('data-row-index') === '1'"
    )


def test_far_cell_unmounted_window_is_bounded(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    total = page.eval_on_selector_all(".timeline-cell[data-src]", "e => e.length")
    mounted = page.eval_on_selector_all("#timeline-grid .timeline-cell img", "e => e.length")
    assert 0 < mounted < total


def test_margin_ring_pre_mounted_offscreen(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    off = page.evaluate(
        """() => {
            const vp = document.querySelector('.timeline-viewport').getBoundingClientRect();
            let n = 0;
            document.querySelectorAll('#timeline-grid .timeline-cell img').forEach((img) => {
                const r = img.getBoundingClientRect();
                const vis = r.right > vp.left && r.left < vp.right && r.bottom > vp.top && r.top < vp.bottom;
                if (!vis) n += 1;
            });
            return n;
        }"""
    )
    assert off >= 1


def test_tab_reentry_reattaches_controller(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    # Leave the tab and come back — the <body> MutationObserver + attach
    # re-fire and re-establish focus (spec §15.7).
    page.locator("a.nav-link", has_text="Plate").first.click()
    page.wait_for_timeout(300)
    page.locator("a.nav-link", has_text="Timeline").first.click()
    page.wait_for_selector(".timeline-cell--focused")
    assert page.eval_on_selector_all(".timeline-cell--focused", "e => e.length") == 1


def test_enter_opens_popout(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector("#timeline-popout-modal.show, .modal.show", timeout=10_000)


def test_repeat_enter_reopens_popout(page: Page, hub_url: str) -> None:
    # OQ-6: opening the SAME focused cell's pop-out twice must re-fire (Dash
    # dedupes an identical bridge value, so Phase 2's bridge-write convention —
    # nonce-suffix or clear-after-consume — must carry through to Results).
    _open_timeline(page, hub_url)
    page.wait_for_selector(".timeline-cell--focused")
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector(".modal.show", timeout=10_000)
    # Dismiss, then Enter again on the SAME focused cell — must reopen.
    page.keyboard.press("Escape")
    page.wait_for_selector(".modal.show", state="detached", timeout=10_000)
    page.click(".timeline-viewport")
    page.keyboard.press("Enter")
    page.wait_for_selector(".modal.show", timeout=10_000)


def test_hover_reveals_popout_button(page: Page, hub_url: str) -> None:
    _open_timeline(page, hub_url)
    cell = page.locator(".timeline-cell[data-src]").first
    cell.hover()
    page.wait_for_selector(".timeline-cell:hover .timeline-cell-popout", timeout=5_000)


def test_empty_state_when_no_time_column(page: Page, hub_url: str, tmp_path: Path) -> None:
    # Documented as a SEPARATE module/param that seeds a master WITHOUT any
    # numeric/Metadata_Time column, so has_eligible_time_axis is False and the
    # guided empty state renders. (Implement via a df_factory-style override or
    # a second fixture; assert `#timeline-empty-state` is visible.)
    pytest.skip("Implement with a no-time-column fixture override; see Task 6 unit test.")
```

> **Fixture note:** `_build_sandbox` writes a stub `master_measurements.parquet`
> (`b""`) at `results/CliOutputExample/`; `_seed` OVERWRITES it with a real parquet
> via `write_master` (same path), exactly as `test_heatmap_tab._seed_master_df_in_output`
> does — verified against that module. The empty-state test needs its own fixture
> seeding a master with NO eligible time column (drop `Metadata_ImageNumber`, keep
> only categorical `Metadata_PlateNum`); wire it as a second fixture or a parametrized
> `df_factory` like `test_heatmap_tab.py` and replace the `skip` with the real assert.

- [ ] **Step 2: Run test to verify it fails**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_results_timeline.py -v`
Expected: FAIL (no Timeline tab; the nav-link click finds nothing).

- [ ] **Step 3: Write minimal implementation**

In `results_viewer/_layout.py`:
- Import `from phenotypic.gui.results_viewer.timeline_view import _layout as _timeline_layout`.
- Build `timeline_tab_body = _timeline_layout.layout(output_root)` in `build_app_layout`.
- Add a 6th `dbc.Tab(timeline_tab_body, label="Timeline", tab_id=ids.TAB_TIMELINE_ID)`
  to the `dbc.Tabs` children list (after the Error tab).

In `results_viewer/_callbacks.py`:
- Import `from phenotypic.gui.results_viewer.timeline_view import _callbacks as _timeline_callbacks`.
- Call `_timeline_callbacks.register_callbacks(app, output_root)` in
  `register_callbacks` (alongside the other per-tab `register_callbacks`).

In `results_viewer/_app.py`:
- Import `from phenotypic.gui.results_viewer.timeline_view import _thumb_routes as timeline_thumb_routes`.
- Call `timeline_thumb_routes.register(app, output_root)` next to
  `_tile_routes.register(app, output_root)` (only on the loaded path, `output_root
  is not None`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_results_timeline.py -v`
Expected: PASS (dropdowns populate; focus starts on first populated cell;
ArrowRight + the down edge button move focus and mount the new neighborhood; a far
cell stays unmounted while the margin ring is pre-mounted; tab re-entry re-attaches;
Enter opens the pop-out; **a second Enter on the same focused cell reopens it
(OQ-6)**; hover reveals ⤢; the no-time-column fixture renders the empty state). If
the timing budget flakes on CI only, keep the `ci_flaky` marker per `tests/CLAUDE.md`.

Also run the no-browser viewer smoke to confirm `create_app` still boots:
`uv run pytest tests/integration/gui -k "viewer or results" -q`.

- [ ] **Step 5: Live MCP verification (manual orchestrator gate, spec §16.9).**
Drive the *running* `phenotypic-gui` against the real reference data — Results over
`…/data/results/2026-06-16/` (the mirror reads `deliverables/measurements.parquet`,
X = `Metadata_ImageNumber`, Y = `Metadata_PlateNum`) — via the Playwright MCP:
switch to the Timeline tab, navigate with arrows/edge buttons, open a pop-out, and
capture screenshots confirming real-overlay rendering a fixture-only e2e cannot.
This is a manual gate (needs a live server + real data); the committed pytest e2e
above is the CI-enforced guard.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_layout.py \
  src/phenotypic/gui/results_viewer/_callbacks.py \
  src/phenotypic/gui/results_viewer/_app.py \
  tests/e2e/gui/test_results_timeline.py
git commit -m "feat(gui-timeline): mount Results Timeline tab + thumb route + e2e"
```

---

### Task 8: Package exports + asset-sync CI guard + FEATURES.md + WORKFLOWS.md + tutorial

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/timeline_view/__init__.py` (exports)
- Create: `tests/gui/results_viewer/timeline_view/test_public_api.py`
- Modify: `src/phenotypic/gui/FEATURES.md`
- Modify: `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py` (add `_capture_results_timeline`)
- Create: `docs/source/tutorials/gui/20_results_timeline.md` (match sibling format;
  Phase 2's Browse-timeline tutorial claims `19_*` after `18_browse.md`, so Results
  timeline is **20** — confirm Phase 2's actual number at execution and bump if needed)

**Interfaces:**
- Consumes: every public symbol from Tasks 3–6.
- Produces: the package public API, the asset-sync guard, and green CI gates.

- [ ] **Step 1: Write the failing test**

Create `tests/gui/results_viewer/timeline_view/test_public_api.py`:

```python
"""timeline_view package public API surface."""
from __future__ import annotations

import phenotypic.gui.results_viewer.timeline_view as tv


def test_public_api_is_exported() -> None:
    expected = {
        "layout",
        "register_callbacks",
        "selectable_time_columns",
        "is_large_time_axis",
        "has_eligible_time_axis",
        "build_timeline_records",
    }
    assert expected.issubset(set(tv.__all__))
    for name in expected:
        assert hasattr(tv, name)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/timeline_view/test_public_api.py -v`
Expected: FAIL (`AttributeError` / no `__all__`).

- [ ] **Step 3: Write minimal implementation**

Replace `timeline_view/__init__.py`:

```python
"""Results-viewer Timeline tab — overlay matrix over OutputRoot.master_df axes.

Renders the shared timeline engine (``gui/_shared/timeline``) + the
focus-and-navigate ``timeline.js`` controller over the run's overlay PNGs,
with a (row × time) axis pair drawn from ``OutputRoot.master_df`` (spec §6/§16).
"""
from __future__ import annotations

from phenotypic.gui.results_viewer.timeline_view._callbacks import register_callbacks
from phenotypic.gui.results_viewer.timeline_view._grid import (
    build_timeline_records,
    has_eligible_time_axis,
    is_large_time_axis,
    selectable_time_columns,
)
from phenotypic.gui.results_viewer.timeline_view._layout import layout

__all__ = [
    "layout",
    "register_callbacks",
    "selectable_time_columns",
    "is_large_time_axis",
    "has_eligible_time_axis",
    "build_timeline_records",
]
```

**FEATURES.md** — add rows under the `## Results Viewer integration` heading
(verified `FEATURES.md:282`), matching the table's exact column header
(`| Feature | Element | Expected behaviour | Status | Test layer | Test ref |` —
confirm the live header before editing). One `✅ shipping` row per affordance, each
with a resolvable `path::test`:
- **Timeline tab** (`TAB_TIMELINE_ID`, the 6th tab) → e2e
  `tests/e2e/gui/test_results_timeline.py::test_focus_starts_on_first_populated_cell`.
- **Y (row) dropdown** (uncapped axis columns) → `tests/gui/results_viewer/colony_view/test_grid_axis_columns.py::test_none_cap_is_uncapped_and_admits_high_cardinality`.
- **X (time) dropdown** (`selectable_time_columns`) → `tests/gui/results_viewer/timeline_view/test_grid.py::test_numeric_column_is_an_eligible_time_axis`.
- **Large-time-axis warning banner** → `tests/gui/results_viewer/timeline_view/test_grid.py::test_is_large_time_axis`.
- **No-time empty state (D9)** → `tests/gui/results_viewer/timeline_view/test_layout.py::test_empty_state_when_no_time_column`.
- **Overlay thumbnail route** → `tests/gui/results_viewer/timeline_view/test_thumb_routes.py::test_thumb_happy_path_returns_bucketed_png`.
- **Focus-navigate matrix** (centered no-scroll window + neighborhood/margin-ring mount + bounded offload) → `tests/e2e/gui/test_results_timeline.py::test_far_cell_unmounted_window_is_bounded`.
- **Four edge nav buttons + keyboard nav + position readout** → `tests/e2e/gui/test_results_timeline.py::test_edge_button_down_moves_focus`.
- **Tab re-entry re-attach** → `tests/e2e/gui/test_results_timeline.py::test_tab_reentry_reattaches_controller`.
- **Deep-zoom pop-out** (hover ⤢ / Enter on focused) → `tests/e2e/gui/test_results_timeline.py::test_enter_opens_popout`.

**WORKFLOWS.md** — add one row to the workflow registry table:

```markdown
| results_timeline | Results — trait emergence over time | Open the Timeline tab in the results viewer, pick a grouping for the Y axis (e.g. plate number) and a monotonic time column for the X axis (e.g. image number), then scan one plate's overlay time-course with ←/→ and compare plates with ↑/↓; Enter (or the hover ⤢) deep-zooms any plate. | `_capture_results_timeline` | `gui/20_results_timeline.md` | ✅ shipping |
```

Add `_capture_results_timeline(context, base_url)` in
`scripts/capture_gui_tutorial_screenshots.py` and **dispatch it** from
`capture_standalone_viewer_screenshots` (verified: that function boots
`python -m phenotypic.gui.results_viewer --output-root <OUTPUT_DIR>` against a real
CLI output — the right host for a loaded-viewer capture; `capture_workflow_screenshots`
runs against the stub hub sandbox and would hit the empty state). The capture: open
`/results/`, click the Timeline nav-link, set the X/Y dropdowns, screenshot the
matrix, press ArrowRight a few times, screenshot again, save under
`docs/source/_static/gui_images/results_timeline/`. Add the tutorial page
`docs/source/tutorials/gui/20_results_timeline.md` (mirror an existing
`gui/NN_*.md` page; Phase 2's Browse-timeline tutorial is `19_*`, so Results is `20`
— confirm Phase 2's actual number at execution and bump if it differs).

> **`check_workflows_md.py` round-trip (verified):** the gate requires the row's
> `` `_capture_results_timeline` `` to be both DEFINED and DISPATCHED (inside
> `capture_workflow_screenshots` or `capture_standalone_viewer_screenshots`), a
> non-empty `docs/source/_static/gui_images/results_timeline/*.png`, and the
> referenced tutorial page to exist. Satisfy all three before the gate runs.

- [ ] **Step 4: Run tests + gates**

Run: `uv run pytest tests/gui/results_viewer/timeline_view -v` → PASS (all unit).
Run: `uv run python scripts/check_workflows_md.py` → pass (row ↔ capture fn ↔ page).
Run: `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the
**full** regenerated PNG set wholesale (per CLAUDE.md — do not cherry-pick the
font-rendering collateral).
Run: `uv run ruff check src/phenotypic/gui/results_viewer/timeline_view src/phenotypic/gui/results_viewer/_ids.py src/phenotypic/gui/results_viewer/colony_view/_grid.py`
Run: `uv run mypy src/phenotypic/gui/results_viewer/timeline_view`
Expected: clean (fix any reported issue before committing).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gui-timeline): export Results Timeline API + FEATURES/WORKFLOWS/tutorial/screenshots"
```

---

## Phase 3 deliverable

A working **Results Timeline tab** (6th tab): the shared engine + the SAME
focus-and-navigate `timeline.js` controller rendered over **overlay images**, with a
**Y dropdown** (uncapped `selectable_axis_columns`, so `Metadata_PlateNum`'s 74
values are selectable), an **X time dropdown** (`selectable_time_columns` —
name/dtype-gated, uncapped), a **large-time-axis warning**, a **guided no-time empty
state** (D9), a `(dataset, stem)` **overlay thumbnail route** cached under the run's
`.viewer_cache/timeline_thumbs`, a **deep-zoom pop-out** reusing the viewer's
existing `/tiles` DZI route, and **tab-reentry re-attach**. Honors the active filter
sidebar (same `master_df` slice as the other tabs). FEATURES/WORKFLOWS/tutorial/
screenshots updated; full Playwright e2e + a live MCP verification gate.

## Cross-surface findings (recorded for reviewers)

- **Browse and Results are separate Dash apps on separate Flask servers** mounted at
  `/browse/` and `/results/` by the hub `DispatcherMiddleware`. They never share a
  `window`, so the Phase 2 `window.__phenotypicTimeline` "singleton" cannot collide
  in v1 — **no per-container keying needed** (resolved at default). The only shared
  artifact is `timeline.js`, **vendored per-app** (Task 1) with a byte-equality CI
  guard, matching the OSD vendoring precedent.
- **The Phase 2 `timeline.js` is surface-agnostic** (confirmed by the coordinator,
  post-authoring): it finds all sibling controls **by class scoped to `.timeline-body`**
  (`.timeline-nav-{up,down,left,right}`, `.timeline-position`,
  `.timeline-popout-bridge`, `.timeline-viewport`, `.timeline-grid-container`), and
  its re-attach observer discovers the grid by `.timeline-grid-container` and
  re-attaches via that element's own id. **There is no cross-surface controller
  coupling and no parameterization needed** — Phase 3 vendors the file byte-for-byte,
  puts the `timeline-*` classes on its layout (C2 / Task 6), and calls
  `attach("timeline-grid")`. **Phase 3 must NOT edit `timeline.js`** (it would break
  the byte-equality guard and contradict the Phase 2 contract).
- **`selectable_axis_columns(max_cardinality=None)` is a real code change** (Task 2):
  the current body `cardinality > max_cardinality` `TypeError`s on `None`. Spec
  §16.5/§15.1 anticipated this; the fix is a one-line guard, colony caller unchanged.

## Resolved decisions (no human action needed)

- **Per-container controller state — resolved: not needed.** Browse and Results are
  separate Dash apps → no shared `window`, so the controller's single `ns` namespace
  is safe. Leave the Phase 2 forward-note as the trigger if a future single-page
  surface ever embeds both timelines; adding per-container keying now is speculative
  complexity.
- **`timeline.js` sharing — resolved: per-app vendored copy + byte-equality CI guard**
  (Task 1). Matches the OSD vendoring precedent; Dash auto-loads `assets_folder`
  scripts; the guard prevents drift. Revisit only if a third surface appears.

## Open Questions (need a human decision)

- **OQ-6 — pop-out bridge repeat-open.** Opening the SAME focused cell's pop-out
  twice writes the same `.timeline-popout-bridge` `value`, which Dash dedupes (no
  re-fire), so a second Enter wouldn't reopen the modal. **Recommended default:**
  reuse Phase 2's exact bridge-write convention verbatim — the Browse controller must
  already solve this (a nonce-suffixed value, or a Dash callback that clears the input
  after consuming it). The Results Dash side strips any nonce before decoding and
  matches the same clear-or-nonce pattern. The `test_repeat_enter_reopens_popout` e2e
  (Task 7) is the guard. **Needs confirmation of Phase 2's actual convention at
  execution** (read `browse/_assets/timeline.js` once it exists; do not invent a new
  one).
- **OQ-5 — tutorial page number.** Recommended default **20** (`gui/20_results_timeline.md`):
  Browse is `18_browse.md` and Phase 2's Browse-timeline tutorial claims `19_*`.
  **Confirm Phase 2's actual number at execution** and bump if it differs.
