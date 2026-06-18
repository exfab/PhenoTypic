# Results Viewer — Filter UX Redesign

- **Date:** 2026-06-18
- **Status:** Approved (design); pending implementation plan
- **Worktree / branch:** `results-viewer-improvements` / `worktree-results-viewer-improvements`
- **Surface:** `src/phenotypic/gui/results_viewer/`

## Context

The results viewer's filter sidebar narrows the master measurements table down
to the overlay images worth inspecting. Today:

- The **Filters** toggle lives in the header bar (`_build_header` in
  `_layout.py`), next to the *Lock views* switch. It opens a right-docked
  `dbc.Offcanvas` hosting the filter panel.
- Each filter **row** binds one column to a multi-value list and matches with
  `pl.col(column).cast(String).is_in(values)`. Rows AND together; values within
  a row OR together (`_filter_state.py::FilterSpec`).
- Filter-value dropdown options come from
  `_output_root.py::_LazyColumnValueSets._compute`, which casts every column to
  `String` and sorts **lexically** — so `"10"` sorts before `"2"`.

This redesign delivers three user-requested improvements.

## Goals

1. **A.** Move the Filters button onto the tab row, right-aligned, and keep it
   **sticky** so it (and the tab nav) stay pinned to the top as tab content
   scrolls.
2. **B.** Give each filter row a **Method** dropdown offering five matching
   methods (the current list, plus exclude / range / compare / contains).
3. **C.** When a column's values are all numeric, sort its filter-value options
   **numerically** instead of lexically.

## Non-goals

- No change to the offcanvas mechanism, the per-card / colony-grid views, QC,
  Heatmap, or Error tabs.
- No new analytical filters (top-N, percentile, is-null/not-null) — explicitly
  deferred as YAGNI.
- No change to curation, the `STORE_REMOVED_KEYS` flow, or the master/mirror
  data sourcing.
- *Lock views* stays in the header; only the Filters button relocates.

---

## Feature A — Filters button on a sticky tab row

### Current state

- `filters_toggle` (`dbc.Button`, id `BTN_FILTERS_TOGGLE`) with its count badge
  (`FILTER_TOGGLE_BADGE_ID`) is built in `_build_header` and placed in `top_row`
  after a flex spacer, next to `lock_switch`.
- `dbc.Tabs` (id `TABS_ID`) is a separate full-width row in `build_app_layout`,
  with all tab bodies mounted (CSS-only switching). Roughly a dozen callbacks
  read/write `TABS_ID.active_tab`.
- The offcanvas toggle + badge callbacks live in `_filter_offcanvas.py`, keyed
  on `BTN_FILTERS_TOGGLE` and `STORE_FILTER_SPEC`.

### Design

Keep `dbc.Tabs` (id `TABS_ID`) **unchanged** — same `active_tab` prop, same
mounted panes — so no tab-gated callback changes. Relocate the Filters button
and make the nav row sticky:

1. **Remove** `filters_toggle` (and its badge) from `_build_header`; *Lock
   views* and the rest of the header are untouched.
2. **Wrap** the `dbc.Tabs` in a positioned container (new
   `results-viewer-tabbar` wrapper) inside `build_app_layout`.
3. **Sticky nav:** a scoped rule in `results_viewer.css` makes the dbc nav row
   sticky — `#results-viewer-tabs .nav-tabs { position: sticky; top: 0;
   z-index: 1020; background: var(--color-bg); }` with right padding reserved
   for the button. Tab panes scroll beneath it.
4. **Sticky right-aligned button:** the Filters button is rendered into a
   right-aligned **zero-height sticky actions strip** layered above the nav
   (`position: sticky; top: 0; z-index: 1030; height: 0; display: flex;
   justify-content: flex-end; pointer-events: none;` with `pointer-events: auto`
   restored on the button). This pins the button on the nav row without
   consuming layout height.

The button keeps its id `BTN_FILTERS_TOGGLE` and badge id
`FILTER_TOGGLE_BADGE_ID`, so the `_filter_offcanvas.py` toggle/badge callbacks
need **no change**.

### Risk / mitigation

The sticky-overlay strip is the one CSS-fiddly piece. The exact offset math
(button vertical centering on the nav row) is finalized during implementation
and verified live in a browser (scroll test). **Fallback** if the overlay
proves brittle cross-browser: rebuild the tab header as a custom `dbc.Nav`
toolbar holding the tab links + button in one flex row — rejected as the
default because it touches `active_tab` wiring, but available as a contingency.

z-index note: the offcanvas (`position: fixed`, bootstrap z-index ~1045) stays
above both the sticky nav (1020) and the actions strip (1030); no conflict.

### Files touched

- `_layout.py` — remove button from header; add tab-bar wrapper + actions strip.
- `_assets/results_viewer.css` — sticky nav + actions-strip rules.
- `_ids.py` — no new ids (button id reused). Optional: a wrapper class constant.

---

## Feature B — Per-row Method dropdown (5 methods)

### Row data model (`STORE_FILTER_SPEC`)

Each row grows from `{id, column, values}` to a superset carrying every
method's payload; only the active method's fields are read:

```jsonc
{
  "id": "<uuid hex>",
  "column": "Size_Area",
  "method": "range",                 // is_any_of | is_none_of | range | compare | contains
  "values": [],                      // is_any_of / is_none_of
  "range":   {"min": 100, "max": 5000},   // range (either bound nullable)
  "compare": {"op": ">=", "value": 0.85}, // compare (op in >, >=, <, <=, ==, !=)
  "contains":{"pattern": "plate_02", "regex": false, "case_sensitive": false}
}
```

**Backward compatibility:** `_normalise_spec` / `FilterSpec.from_store` default
`method` to `is_any_of` when absent and default each method's payload to its
empty form. Session-stored specs from before this change keep working
unchanged.

### Method → predicate

`FilterRow.to_expr()` (new) in `_filter_state.py` returns a
`pl.Expr | None` (`None` = unset/skip), and `apply_to` ANDs the non-None
expressions:

| Method | Predicate | Input control |
|---|---|---|
| **Is any of** (default) | `col.cast(String).is_in(values)` | multi-select dropdown + Paste |
| **Is none of** | `~col.cast(String).is_in(values)` | multi-select dropdown + Paste |
| **Range (between)** | `col.cast(Float64, strict=False)` with `>= min` and/or `<= max` (either optional) | two numeric inputs |
| **Compare** | `col.cast(Float64, strict=False) <op> value` | operator dropdown + numeric input |
| **Contains** | `col.cast(String).str.contains(pat, literal=not regex)`; see case-insensitivity note below | text input + `regex` / `case-sensitive` checkboxes |

**Unset = skip (no-op), never "match nothing"** — consistent with today's
semantics. A row is skipped when: no column; or `is_any_of`/`is_none_of` with
empty `values`; or `range` with both bounds blank; or `compare` with blank
value; or `contains` with blank pattern. Rows still AND together.

Numeric casting uses `strict=False`, so non-numeric cells become null and fall
out of range/compare matches without raising — the viewer never crashes on a
mixed column. Columns absent from the frame log a warning and skip (as today).

**Contains case-insensitivity.** `str.contains` is case-sensitive by default and
regex flags only apply when `literal=False`, so the two modes differ:

- `regex=true` → `str.contains(pattern, literal=False)`; case-insensitive
  prepends the `(?i)` inline flag.
- `regex=false` (literal) → case-sensitive uses `str.contains(pattern,
  literal=True)`; case-insensitive lowercases **both** sides:
  `col.cast(String).str.to_lowercase().str.contains(pattern.lower(),
  literal=True)`. (Cannot use `(?i)` in literal mode — it would match nothing.)

An invalid user regex is caught (`try/except` around expr build) and the row is
skipped rather than 500-ing the callback.

### Numeric-column gating

Range and Compare are offered **only for numeric columns**. New
`OutputRoot.is_numeric_column(col) -> bool`:

1. `True` if `master_df.schema[col]` is a polars numeric dtype (free; covers all
   `Size_*` / `Shape_*` / `Intensity_*` measurement columns).
2. Else `True` if the column's value-set (already eager for `Metadata_*`) is
   non-empty and **every** value parses as a float (covers numeric-valued string
   metadata like `Metadata_Time`).
3. Else `False`.

Result is cached per column. The Method dropdown disables the Range / Compare
options for non-numeric columns; `is_any_of`, `is_none_of`, `contains` are
always available.

### UI / rendering

Reuse the existing **store-driven render**: the Method dropdown writes `method`
into `STORE_FILTER_SPEC`, the store change re-fires `_render_rows`, and
`_render_filter_row` renders only the control(s) for the active method. This
matches today's add/remove/rehydrate code path and avoids dynamic-children
races — pattern-matching `ALL` callbacks simply match whatever controls
currently exist.

- Changing **method or column resets** that row's payload (column change already
  clears `values` today; extend to clear all method payloads).
- **Paste** stays available only for the two list methods (`is_any_of`,
  `is_none_of`).
- `_filter_offcanvas.py::active_filter_count` is updated: a row counts as active
  when its column is set **and** its method has a usable constraint (via a new
  `row_is_active(row)` helper mirroring the skip rules above), so the toggle
  badge reflects range/compare/contains rows too.

### New component ids (`_ids.py`)

Pattern-matching id-builders, one `type` per control:

- `filter_row_method_id(idx)` → `filter-row-method`
- `filter_row_range_min_id(idx)` → `filter-row-range-min`
- `filter_row_range_max_id(idx)` → `filter-row-range-max`
- `filter_row_compare_op_id(idx)` → `filter-row-compare-op`
- `filter_row_compare_value_id(idx)` → `filter-row-compare-value`
- `filter_row_text_pattern_id(idx)` → `filter-row-text-pattern`
- `filter_row_text_regex_id(idx)` → `filter-row-text-regex`
- `filter_row_text_case_id(idx)` → `filter-row-text-case`

(`filter-row-values`, `-paste-*`, `-remove`, `-column` are unchanged.)

### New callbacks (`_filter_panel.py`)

- Method dropdown → spec (`method` write; clears payload).
- Range min/max (`ALL`) → spec `range`.
- Compare op + value (`ALL`) → spec `compare`.
- Contains pattern/regex/case (`ALL`) → spec `text`.

Existing callbacks (add/remove row, column→spec, values→spec, populate value
options, paste toggle/apply, derive image pairs) stay; column→spec extends its
payload-reset.

### Files touched

- `_filter_state.py` — extend `FilterRow`/`FilterSpec`; add `to_expr()`,
  per-method coercion; `apply_to` ANDs `to_expr()` results.
- `_filter_panel.py` — Method dropdown + per-method controls in
  `_render_filter_row`; new sync callbacks; `_normalise_spec` extension;
  numeric gating passed into render.
- `_output_root.py` — `is_numeric_column`.
- `_ids.py` — new id-builders above.
- `_filter_offcanvas.py` — `row_is_active` + `active_filter_count` update.

---

## Feature C — Numeric-aware sorting of value options

### Design

Fix at the single source: `_output_root.py::_LazyColumnValueSets._compute`.
After collecting the unique non-null **string** values for a column, if **every**
value parses as a float, return them sorted by float value; otherwise keep the
current lexical `.sort()`. Values remain strings, so dropdown options, the
spec payload, and `is_in` matching are unaffected — only the **order** changes
(`"2"` before `"10"`).

This naturally covers `Metadata_Time` and any numeric-valued column, and reuses
the same all-values-parse-as-float test as `is_numeric_column` (extract a shared
helper, e.g. `_all_parse_as_float(values) -> bool`).

### Files touched

- `_output_root.py` — numeric-aware branch in `_compute`; shared float-parse
  helper reused by `is_numeric_column`.

---

## Data flow (unchanged shape)

`STORE_FILTER_SPEC` (richer rows) → `_derive_image_pairs` builds a `FilterSpec`,
calls `apply_to(master_df)` (now method-aware), derives `(dataset, stem)` pairs
into `STORE_IMAGE_PAIRS`, and updates the match-count chip. The offcanvas
toggle/badge read the same spec store. No new stores.

## Testing strategy

**Unit (pure, no browser):**

- `FilterRow.to_expr()` / `FilterSpec.apply_to` for all five methods, including:
  optional/single-bound range, inclusive bounds, every compare operator,
  contains literal vs regex and case on/off, numeric cast of mixed columns,
  and unset-skip for each method.
- `_compute` numeric vs lexical ordering; `_all_parse_as_float`;
  `is_numeric_column` (numeric dtype, numeric-string metadata, non-numeric).
- `_normalise_spec` backward-compat (legacy `{column, values}` → `is_any_of`).
- `active_filter_count` / `row_is_active` across methods.

**Live Dash (Playwright + viewer log tail)** — per the project rule that
callback wiring bugs only surface on `/_dash-update-component`:

- Method dropdown swaps the row's control set; range/compare/contains actually
  narrow the picker and the match-count chip.
- Sticky tab bar + Filters button stay pinned while scrolling tab content;
  offcanvas still opens/closes.

## Documentation / ledger obligations

- **`src/phenotypic/gui/FEATURES.md`** (CI-gated `features-md-gate`): add rows
  for the relocated sticky Filters button and each new filter method, with
  `Test ref`s on shipping rows.
- **`src/phenotypic/gui/WORKFLOWS.md`:** filtering is an established surface, so
  no new tutorial flow is expected; confirm against existing entries during the
  plan. If unchanged, no `_capture_*` / tutorial page is required.
- Re-run `scripts/capture_gui_tutorial_screenshots.py` after the visible chrome
  change and commit the full refreshed PNG set (do not cherry-pick collateral).
- Respect `_design.py` / `_config.py` token discipline — no hardcoded colors or
  z-index/spacing magic numbers that belong in tokens.

## Risks

- **Sticky overlay CSS** (Feature A) — see mitigation above; live scroll test +
  custom-nav fallback.
- **Per-row re-render on method change** resets in-progress input focus —
  acceptable, matches today's add/remove behavior.
- **High-cardinality numeric parse** in `_compute`/`is_numeric_column` — bounded
  by the existing lazy materialization; `Metadata_*` are eager and small,
  measurement columns short-circuit on dtype.
