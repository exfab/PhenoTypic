# Followup prompt — Filters → popover (reclaim the viewer's shared left column)

**Status:** Followup seed, *not yet brainstormed or designed.* Pick this up with
the `superpowers:brainstorming` skill — treat the "Proposed approach" below as a
starting hypothesis to validate/reject, not a settled design.

**Origin:** Surfaced while shipping the smart-QC feature
(`2026-05-29-smart-qc-design.md`). Deliberately deferred to keep that feature
scoped — this is a **viewer-wide** change, not a QC change.

---

## Motivation

The results-viewer body is a fixed two-column layout: a persistent filter
**sidebar** (`col-lg-3`, ~25% width) on the left, and the tabbed content
(`col-lg-9`) on the right. The sidebar shows on **every** tab — Plate, Colony,
Heatmap, and QC (Configure + Review) — whether or not the user is filtering.

On tabs with their own dense layout (notably the new **QC Review** master-detail
worklist + tile gallery), that always-on 25% is mostly wasted: the user is
curating groups, not editing a measurement filter spec. The win is to make the
filter panel **on-demand** so all four tabs get full width by default.

## Goal

Convert the persistent left filter sidebar into an on-demand **popover/offcanvas**
toggled from the top bar — reclaiming full width for all tabs — **without**
regressing filter behavior or the store-driven re-hydration.

## Current state — key files

- `gui/results_viewer/_layout.py` — builds the two-column body
  (`dbc.Col(sidebar, …)` + content `dbc.Col`); `sidebar = _filter_panel.layout(output_root)`;
  `_build_stores(...)`.
- `gui/results_viewer/_filter_panel.py` — `layout(output_root)` returns the sidebar
  tree (filter-rows container, match-count, "add filter row" button, bulk-paste
  popover); `register_*` registers its callbacks. Rows are rendered into
  `FILTER_ROWS_CONTAINER_ID` by a callback that listens to `STORE_FILTER_SPEC`, so
  the panel **re-hydrates from the store** rather than from server-rendered children.
- `gui/results_viewer/_filter_state.py` / `_filtered_state.py` — filter spec store
  + curation (`FilteredMeasurements`).
- `gui/results_viewer/_ids.py` — `FILTER_ROWS_CONTAINER_ID`,
  `FILTER_MATCH_COUNT_ID`, `BTN_ADD_FILTER_ROW`, `STORE_FILTER_SPEC`.

## Proposed approach (validate during brainstorming)

Replace the left `dbc.Col` with a `dbc.Offcanvas` containing the **existing**
`_filter_panel.layout(...)` tree unchanged; add a "Filters" toggle button (with an
active-filter **count badge**) to the top bar; let the content column expand to
full width.

- **Offcanvas over Popover:** a multi-row filter form wants room, scroll, and
  backdrop-dismiss — an offcanvas (left slide-in) fits better than a small popover.
- **Re-hydration should be layout-only:** the panel already populates
  `FILTER_ROWS_CONTAINER_ID` from `STORE_FILTER_SPEC`. Offcanvas children stay in
  the DOM (just hidden), so the boot callback can still populate rows while it's
  closed. Verify this holds.
- **Keep filter state visible when closed:** surface the match-count / active-filter
  count on the toggle button so users see filtering state without opening.

## Constraints & gotchas

- **Viewer-wide:** every tab shares this panel. Verify Plate, Colony, Heatmap, and
  QC (Configure + Review) all still filter after the move.
- **CI-gated GUI ledgers:** touching `gui/` requires a `FEATURES.md` update (toggle
  button, offcanvas, badge). If the user flow changes enough to warrant it, add a
  `WORKFLOWS.md` row + a `_capture_*` screenshot function and refresh affected
  tutorial PNGs (the `gui-checks` workflow gates both).
- **Tests:** existing filter integration/e2e suites
  (`tests/integration/gui`, `tests/e2e/gui`) must stay green; add coverage that the
  toggle opens/closes the offcanvas and that filtering still narrows results.

## Out of scope

The smart-QC feature itself (shipped: Phases A–D). The QC Review tab's JS
drag-splitter, worklist, curation, and per-group recompute are done — don't revisit
them here.

## Open questions for brainstorming

- Offcanvas vs Popover vs collapsible-inline?
- Persist open/closed per session, or always boot closed?
- Auto-open when filters are already active on load?
- Uniform across all tabs, or keep inline anywhere?

## Acceptance

- All tabs full-width by default; filters reachable in ≤1 click.
- Active-filter count visible without opening the panel.
- No filter-behavior or re-hydration regression.
- GUI ledgers + filter tests green.
