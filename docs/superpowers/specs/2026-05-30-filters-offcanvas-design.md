# Design — Filters offcanvas (right-docked, reclaim full width for all viewer tabs)

**Status:** Designed, approved — ready for an implementation plan.
**Supersedes:** the seed `2026-05-30-filters-popover-followup.md` (the offcanvas
hypothesis there is now validated against the codebase and resolved into this design).

## Context

The results-viewer body is a fixed two-column layout: a persistent filter **sidebar**
(`col-lg-3`, ~25%) on the left and tabbed content (`col-lg-9`) on the right. The sidebar
shows on **every** tab — Plate, Colony, Heatmap, QC (Configure + Review) — whether or not
the user is filtering. On dense tabs (notably the new QC Review master-detail worklist +
tile gallery) that always-on 25% is wasted: the user is curating groups, not editing a
filter spec.

This change makes the filter panel **on-demand**: a right-docked `dbc.Offcanvas` toggled
from the top bar, so all four tabs render full-width by default and filtering is one click
away — **without** regressing filter behavior or the store-driven re-hydration.

Exploration confirmed the move is low-risk: the filter panel is a self-contained
`dbc.Card` with no `col-lg-*`/sidebar-width assumptions inside it, its row re-hydration
fires at boot regardless of visibility, and no tab silently loses filtering (see
*Filtering data flow* below).

## Decisions (locked)

| Question | Decision |
|---|---|
| Mechanism | **Offcanvas**, `placement="end"` (right slide-in, dimming backdrop, click-away + ✕ dismiss). |
| Boot state | **Always boot closed.** No persistence store — the offcanvas's own `is_open=False` default *is* the state. |
| Toggle badge | **Both** — active-filter-row count on the top-bar button badge; keep the existing "N images match" chip inside the panel. |
| Tab scope | **Uniform** — one offcanvas shared by all four tabs. |
| Bulk-paste popover | Re-place from `bottom` → **`left`** so it stays on-screen inside the right-docked panel. |
| Ledgers | **Minimal** — `FEATURES.md` rows (mandatory) + refresh existing `view_results` screenshots (add one offcanvas-open shot). **No** new `WORKFLOWS.md` row. |

## Architecture & components

### 1. Right-docked offcanvas (replaces the left sidebar column)

- `dash-bootstrap-components>=2.0.4` (`pyproject.toml`) ships `dbc.Offcanvas` with
  `placement`, `is_open`, `backdrop`, `scrollable`, `title`.
- Mount **once** at the top level of `build_app_layout()`
  (`gui/results_viewer/_layout.py`, ~L365–461), as a sibling of `stores` / `header` /
  `body`, so it persists across tab switches and is globally reachable.
- Its body is the **unchanged** `_filter_panel.layout(output_root)` tree
  (`gui/results_viewer/_filter_panel.py`, L73–139 — a `dbc.Card`/`CardBody` with the
  filter-rows container, match-count chip, "+ Add filter" button, and per-row bulk-paste
  popovers). No edits to the panel tree itself except the popover placement (§4).
- Props: `placement="end"`, `is_open=False`, `scrollable=True`, `backdrop=True`,
  `title="Filter"`. The panel's existing `maxHeight: 70vh` scroll on
  `FILTER_ROWS_CONTAINER_ID` is fine inside the offcanvas.

### 2. Full-width content

- Collapse the two-column `dbc.Row` (`_layout.py` L436–455: `dbc.Col(sidebar, lg=3)` +
  `dbc.Col(tabs, lg=9)`) to a single **full-width** content container holding `tabs`
  (L409–434). Preserve the existing background/padding and the
  `minHeight: calc(100vh - 7rem)`.

### 3. Top-bar toggle button + count badge ("Both")

- Add a `dbc.Button` (id `BTN_FILTERS_TOGGLE`) to the header's right cluster in
  `_build_header()` (`_layout.py` L139–157, the `top_row` flex row that already holds the
  logo, title, pipeline chip, spacer, and Lock-views switch). Place it in the right
  cluster (near the Lock-views switch), aligned with the side the panel docks on.
- Button label: `≡ Filters` + a `dbc.Badge` (id `FILTER_TOGGLE_BADGE_ID`) showing the
  **active-filter-row count** (rows in `STORE_FILTER_SPEC` with a column set). Badge is
  hidden / empty when the count is 0.
- The existing **"N images match" chip stays inside the panel** (`FILTER_MATCH_COUNT_ID`,
  already maintained by `_derive_image_pairs`, `_filter_panel.py` L668–709). Result-size
  on the chip; applied-count on the button.

### 4. Bulk-paste popover placement

- The per-row bulk-paste `dbc.Popover` (`_filter_panel.py` L319–352) currently opens
  `placement="bottom"`. With the panel docked on the right edge, change it to
  **`placement="left"`** so it opens inward and never clips the viewport's right edge.
  This is the resolution of the "move the popover to the right side" request: keep it
  fully on-screen given the panel is now right-docked.

### 5. Logic — pure helpers + thin callbacks

New module `gui/results_viewer/_filter_offcanvas.py` (keeps `_layout.py` lean and the
behavior unit-testable, matching the smart-QC `worklist_row_metric_update` pattern):

- `next_offcanvas_state(n_clicks, is_open) -> bool` — toggle logic (guards `n_clicks`
  falsy → unchanged).
- `active_filter_count(spec) -> int` — counts spec rows with a non-empty `column`.
- `register_filter_offcanvas_callbacks(app)` — wires:
  - `Input(BTN_FILTERS_TOGGLE, "n_clicks")` + `State(OFFCANVAS_FILTER_ID, "is_open")` →
    `Output(OFFCANVAS_FILTER_ID, "is_open")` via `next_offcanvas_state`
    (`prevent_initial_call=True`). dbc's own backdrop/✕ close updates `is_open` so the
    next toggle reads correct State.
  - `Input(STORE_FILTER_SPEC, "data")` → `Output(FILTER_TOGGLE_BADGE_ID, "children")`
    (and a hidden/empty render when 0) via `active_filter_count`.
- Called from `build_app_layout()`'s registration alongside the existing
  `_filter_panel.register_callbacks(...)`.

### 6. New IDs (`gui/results_viewer/_ids.py`)

`OFFCANVAS_FILTER_ID`, `BTN_FILTERS_TOGGLE`, `FILTER_TOGGLE_BADGE_ID`. Existing filter IDs
(`STORE_FILTER_SPEC`, `BTN_ADD_FILTER_ROW` L68, `FILTER_ROWS_CONTAINER_ID` L93,
`FILTER_MATCH_COUNT_ID` L97) are unchanged — the panel moves intact, so its callbacks keep
wiring by id.

## Filtering data flow (the key de-risk)

Filtering survives the move because it is **store-driven and downstream of the panel**, not
dependent on the sidebar's DOM position:

- **Re-hydration:** `_render_rows` (`_filter_panel.py` L445–452) renders rows into
  `FILTER_ROWS_CONTAINER_ID` from `STORE_FILTER_SPEC` with **no `prevent_initial_call`** —
  it fires at boot and populates the rows even while the offcanvas is closed (offcanvas
  children stay in the DOM, just hidden). Verified to be visibility-independent.
- **Plate tab:** reads `STORE_IMAGE_PAIRS` (derived by `_derive_image_pairs` from the
  filter spec) → picker options (`_viewer_card.py`).
- **Colony tab:** reads `STORE_FILTER_SPEC` directly and applies `FilterSpec.apply_to(df)`
  (`colony_view/_callbacks.py` ~L114, L195).
- **Heatmap & QC tabs:** intentionally operate on the **curated** frame
  (`CFG_FILTERED_STATE` / `get_curated_frame`), independent of the filter spec — this is
  by design (they show all QC-augmented/curated rows), not a regression. Curation
  (`STORE_REMOVED_KEYS` / `FilteredMeasurements`) is orthogonal to filtering and unchanged.

No callback references the sidebar **container** position; all are keyed on stable ids.

## Files to modify

| File | Change |
|---|---|
| `gui/results_viewer/_ids.py` | Add `OFFCANVAS_FILTER_ID`, `BTN_FILTERS_TOGGLE`, `FILTER_TOGGLE_BADGE_ID`. |
| `gui/results_viewer/_layout.py` | `_build_header()`: add the Filters toggle + badge to the right cluster. `build_app_layout()`: full-width content body; mount the top-level `dbc.Offcanvas(_filter_panel.layout(...), placement="end", …)`; call `register_filter_offcanvas_callbacks(app)`. |
| `gui/results_viewer/_filter_panel.py` | Bulk-paste popover `placement="bottom"` → `"left"`. Otherwise untouched (keeps `FILTER_MATCH_COUNT_ID` chip per the "Both" decision). |
| `gui/results_viewer/_filter_offcanvas.py` *(new)* | `next_offcanvas_state`, `active_filter_count`, `register_filter_offcanvas_callbacks`. |
| `gui/FEATURES.md` | Rows for the toggle button, the active-filter badge, and the offcanvas (✅ shipping + unit Test refs). |
| Tutorial PNGs | Refresh `view_results/02_viewer_loaded.png`, `…/03_measurement_table.png`, `heatmap_exploration/*` via the capture script; add one offcanvas-open shot to the existing `view_results` capture. |

Design tokens (`_design.py`): use `COLOR_*` / `SPACING_*` / `RADIUS` / `SHADOW` for the
button + badge; badge color from the palette (e.g. `COLOR_BLUE` accent). No raw hex.

## Testing

- **Unit** (`tests/unit/gui/results_viewer/test_filter_offcanvas.py`):
  - `next_offcanvas_state`: falsy `n_clicks` → unchanged; click flips `is_open`.
  - `active_filter_count`: `[]` → 0; rows with/without `column` counted correctly; the
    badge render hides at 0 and shows the count otherwise.
  - Layout: `build_app_layout()` mounts the offcanvas with the filter panel inside, and
    all existing filter callback-target ids (`FILTER_ROWS_CONTAINER_ID`,
    `BTN_ADD_FILTER_ROW`, `FILTER_MATCH_COUNT_ID`) are present.
- **E2E** (`tests/e2e/gui/test_filter_offcanvas.py`, `ci_flaky`, reusing
  `tests/e2e/gui/conftest.py` `_build_sandbox` / `_start_live_server`):
  - Click `Filters` → offcanvas opens; ✕/backdrop → closes.
  - Open → add a filter row + pick a value → assert content narrows (e.g. Plate picker /
    Colony grid reflects the filter).
  - Badge reflects the active-filter count after adding/removing a row.
- **Regression (must stay green, untouched):**
  `tests/gui/results_viewer/test_filter_state.py`,
  `tests/gui/results_viewer/test_filtered_state.py` (pure data layer — no layout
  assertions). QC Review layout tests are a separate sidebar and unaffected.

## Ledgers (CI-gated)

- **`FEATURES.md` (mandatory):** any PR touching `src/phenotypic/gui/` must modify it
  (`features-md-gate`). Add rows (columns: Feature, Element, Expected behaviour, Status,
  Test layer, Test ref) for the toggle button, the active-filter badge, and the offcanvas;
  status `✅ shipping` with a real unit `Test ref` (pre-commit `check_features_md.py`
  validates the ref resolves).
- **`WORKFLOWS.md`: no new row.** Filtering already exists; this relocates it. The
  existing `view_results` workflow + `_capture_view_results` /
  `capture_standalone_viewer_screenshots()` cover the viewer — just refresh the affected
  PNGs (and add an offcanvas-open capture within the existing `view_results` flow so the
  tutorial still demonstrates filtering). Avoids the heavier round-trip
  (`check_workflows_md.py`: new row → new `_capture_*` → new tutorial page).
- Run `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit the **full**
  regenerated set (per CLAUDE.md: commit-everything, no cherry-picking the font-render
  collateral).

## Out of scope

The smart-QC feature (shipped: Phases A–D). The QC Review JS drag-splitter, worklist,
curation, and per-group recompute are done — not revisited here. No change to filter
semantics (`FilterSpec`), curation (`FilteredMeasurements`), or the per-tab consumers
beyond relocating the panel.

## Acceptance

- All four tabs render full-width by default; filters reachable in ≤1 click (right-docked
  offcanvas).
- Active-filter count visible on the toggle without opening the panel; "N images match"
  chip remains inside.
- No filter-behavior or re-hydration regression (Plate + Colony still narrow; Heatmap/QC
  curated-frame behavior unchanged).
- Bulk-paste popover stays on-screen inside the right-docked panel.
- `FEATURES.md` + filter tests green; tutorial PNGs refreshed.

## Verification (end-to-end)

1. `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py tests/gui/results_viewer/test_filter_state.py tests/gui/results_viewer/test_filtered_state.py`
2. `uv run pytest tests/e2e/gui/test_filter_offcanvas.py` (locally; `ci_flaky`).
3. `uv run ruff check --fix` ; `uv run mypy` on the changed files.
4. `uv run phenotypic-gui --root <output>` → top bar shows `Filters` toggle; content is
   full-width on every tab; click → panel slides in from the right; add a filter →
   Plate/Colony narrow, badge increments; close → state correct.
5. `uv run python scripts/capture_gui_tutorial_screenshots.py` → refreshed PNGs;
   `features-md-gate` + `workflows-md-gate` pass.
