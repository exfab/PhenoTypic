# Error-Triage Cutoffs — Phase 2: Tile UI + App Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the durable `CurationLabels` store into the running results viewer and replace the single per-tile remove toggle with categorized error triage — a nested **radial menu** (5 core categories + reserved `Other` + a `Custom` folder) on colony and QC-review tiles, a per-tile category badge, and a category-aware **bulk "Mark N selected as ▾"** extension of the existing bulk bar.

**Architecture:** `CurationLabels` is already duck-compatible with `FilteredMeasurements` at every call site, so integration is a store swap plus an mtime guard, not a rewrite. The radial is a per-tile ▾ trigger + a **lazily-populated** `dbc.Popover` (reusing the existing `_build_stack_popover` lazy pattern) whose body holds absolutely-positioned wedge buttons with pattern-matched ids `{type, image_file, label, category}`; a wedge click runs one server callback → `CurationLabels.mark(image_file, label, category)`. The existing selection plumbing (`STORE_COLONY_SELECTION` + JS shift-click) and bulk bar (`COLONY_BULK_*`) are extended, not replaced.

**Tech Stack:** Dash 4 (pattern-matching callbacks, `dbc.Popover`), Python 3.12, polars, the Phase-1 `CurationLabels` API, `uv` runner, pytest (+ Flask test client for callback integration; Playwright for one e2e smoke).

**Depends on:** Phase 1 (merged) — `CurationLabels` with `mark`/`unmark`/`mark_many`/`register_custom_category`/`categories`/`labels_payload` + back-compat surface; `ErrorCategory` (bare-label tokens) and `CURATION.ERROR_CATEGORY`; `_design.py` `OI_*` palette.

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` (§3 Tile UI, §6, §2 behavior change).

---

## Conventions for this plan

- `uv run` for everything. Unit/integration: `uv run pytest <path> -v`. GUI-widget/e2e tests need `QT_QPA_PLATFORM=offscreen` only if they touch Qt — these don't; Playwright e2e needs `PLAYWRIGHT=1` (see `tests/CLAUDE.md`).
- Commit per task, scoped `git add <paths>`. Worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`.
- GUI constants → `gui/_config.py`; colors/type/radius → `gui/_design.py`; tool-internal ids → the tool's `_ids.py`. Never re-spell a literal.
- **`FEATURES.md` is CI-gated:** any PR touching `src/phenotypic/gui/` must modify `src/phenotypic/gui/FEATURES.md`; `✅ shipping` rows need a resolvable `Test ref`. This plan updates it in Task 8 (and the gate is per-PR, so a single consolidation pass suffices for the branch).
- After any visible chrome change, the screenshot capture (`scripts/capture_gui_tutorial_screenshots.py`) and `WORKFLOWS.md` round-trip are **Phase 6**, not here — Phase 2 stays at FEATURES.md + integration/e2e tests.

## Key integration facts (from the codebase map — bind to these exact symbols)

- **Store swap:** `results_viewer/_app.py:181` `FilteredMeasurements.load(output_root.root, output_root.master_df)` → `CurationLabels.load(...)`; stays under `CFG_FILTERED_STATE = "filtered_state"` (`_config.py:249`). All ~12 call sites are duck-typed and keep working; only the static `FilteredMeasurements` type annotations need widening (Task 1).
- **`_lock` / `removed_keys`:** `_qc_tab/review/_callbacks.py:548-559` does `with filtered._lock: return set(filtered.removed_keys)` — `CurationLabels` exposes `_lock` (RLock) and `removed_keys` (property → fresh set), so the idiom holds.
- **Colony tile remove seam:** `colony_view/_grid.py:338 _colony_remove_button(image_file,label,is_removed)` builds `dbc.Button(id=colony_cell_remove_btn_id(...))`; injected via `build_tile_cell(remove_button=...)` (`_grid.py:443`). Callback `colony_view/_callbacks.py:260-299 _toggle_single_cell_removal` (pattern `{"type":"colony-cell-remove-btn","image_file":ALL,"label":ALL}`) → `filtered_state.mutate_and_payload(lambda s: s.toggle(image_file,label))` → writes `STORE_REMOVED_KEYS`.
- **QC review tile remove seam:** `_qc_tab/review/_callbacks.py` `_review_tile_remove_button` (id `review_tile_remove_btn_id` = `{"type":"qc-review-tile-remove",...}`); gallery via `build_tile_grid(remove_button_builder=...)`; callback `_toggle_review_tile` (MATCH) → `toggle_review_tile(filtered, image_file, label)` (pure helper in `review/_callbacks.py`, unit-tested in `tests/unit/gui/results_viewer/test_qc_callbacks_helpers.py`).
- **Selection + bulk bar (already shipping):** stores `STORE_COLONY_SELECTION` (`{anchor, selected:[[img,label]]}`), `STORE_COLONY_SELECTION_DELTA`, `STORE_COLONY_GRID_ORDER`; JS in `results_viewer/_assets/results_viewer.js` (section F) writes the delta on checkbox shift-click. Colony bulk ids `COLONY_BULK_BAR_ID`/`COLONY_BULK_COUNT_LABEL_ID`/`COLONY_BULK_REMOVE_BTN_ID`/`COLONY_BULK_RESTORE_BTN_ID`/`COLONY_BULK_CLEAR_BTN_ID` (`_ids.py:529-546`), callbacks `colony_view/_callbacks.py:405-451` using `remove_many`/`restore_many`. QC review bulk ids `QC_REVIEW_BULK_REMOVE_BTN_ID`/`QC_REVIEW_BULK_RESTORE_BTN_ID` (`review/_ids.py:201-204`), callback `_bulk_review_curation` (`review/_callbacks.py:1096-1122`), pure helper `bulk_review_curation`.
- **Lazy-popover precedent:** `colony_view/_grid.py:450 _build_stack_popover` ships an empty `dbc.PopoverBody` + a co-located `dcc.Store`, populated on first badge click by `build_stack_popover_rows` via a pattern-matched callback. The radial reuses this exact pattern (empty body → populate-on-open) to keep the DOM light across many tiles.
- **OI palette (`_design.py:237-293`):** `OI_ORANGE #E69F00`, `OI_SKY #56B4E9`, `OI_GREEN #009E73`, `OI_BLUE #0072B2`, `OI_PURPLE #CC79A7`, `OI_GREY #BBBBBB`, `OI_VERMILION #D55E00` (reserve for error/alert), `OI_YELLOW` (never as text on white); AA text variants `OI_ORANGE_TEXT`/`OI_SKY_TEXT`/`OI_GREEN_TEXT`/`OI_PURPLE_TEXT`. `COLOR_*` are UI-only.
- **FEATURES.md row format:** `| Feature | Element | Expected behaviour | Status | Test layer | Test ref |`; `Test layer ∈ {unit,integration,e2e,manual}`.
- **Tests live in two trees:** older GUI unit tests under `tests/gui/results_viewer/...`; newer under `tests/unit/gui/results_viewer/...`; integration under `tests/integration/gui/`; e2e under `tests/e2e/gui/` (`PLAYWRIGHT=1`).

## File structure (Phase 2)

- Modify: `src/phenotypic/gui/results_viewer/_app.py` — construct `CurationLabels`.
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py` — add the mtime guard (Q3/M3).
- Modify type annotations: `_callbacks.py`, `_layout.py`, `_filter_panel.py`, `colony_view/_callbacks.py`, `_viewer_card.py`, `_qc_tab/_callbacks.py`, `_qc_tab/review/_callbacks.py` (widen `FilteredMeasurements` → `CurationLabels` or a shared alias).
- Create: `src/phenotypic/gui/_design.py` addition — `ERROR_CATEGORY_COLORS` + `category_color(token, custom_index)`.
- Create: `src/phenotypic/gui/_shared/_radial.py` — the radial component + wedge id factory + populate helper.
- Create: `src/phenotypic/gui/_shared/_assets/` or extend `results_viewer/_assets/` CSS for radial geometry.
- Modify: `colony_view/_grid.py` (radial trigger + badge), `colony_view/_callbacks.py` (mark-category + populate + custom-add callbacks), `_qc_tab/review/_callbacks.py` (same for QC tiles), `_shared/tiles.py` (badge slot if needed).
- Modify: `_ids.py` / `review/_ids.py` — new pattern-matched ids + bulk category-dropdown ids.
- Modify: `src/phenotypic/gui/FEATURES.md`, `gui/CLAUDE.md`.
- Tests: extend `tests/unit/gui/results_viewer/test_curation_labels.py` (mtime guard); new `tests/gui/_shared/test_radial.py`, `tests/unit/gui/test_design_category_colors.py`; new `tests/integration/gui/test_triage_callbacks.py`; one `tests/e2e/gui/test_radial_triage.py` (`ci_flaky`-eligible).

---

### Task 1: Swap the app to `CurationLabels` + add the mtime guard

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (line ~59 import, ~181 construct, ~213 layout)
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py` (mtime guard)
- Modify (annotations): `_callbacks.py`, `_layout.py`, `_filter_panel.py`, `colony_view/_callbacks.py`, `_viewer_card.py`, `_qc_tab/_callbacks.py`, `_qc_tab/review/_callbacks.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py`, new `tests/integration/gui/test_triage_callbacks.py`

**Why:** Make the durable store the live curation backend (so plain removals become durable per spec §2), and restore the staleness protection `FilteredMeasurements` had (Q3/M3) that `CurationLabels` dropped.

- [ ] **Step 1: Mtime-guard test (write first)**

Add to `tests/unit/gui/results_viewer/test_curation_labels.py`:

```python
def test_save_refuses_after_external_reseed(tmp_path: Path):
    store = CurationLabels.load(tmp_path, _master())
    store.mark("plateA", 1, "debris")  # seeds measurements.parquet + records mtime
    # Simulate a CLI re-seed: rewrite measurements.parquet under the open session.
    import os, time
    mpath = tools_.measurements_parquet_path(tmp_path)
    os.utime(mpath, (time.time() + 5, time.time() + 5))  # bump mtime
    store.mark("plateA", 2, "merged")  # must refuse to clobber
    on_disk = pl.read_parquet(mpath)
    # The second mark did not rewrite the mirror (object 2 still present on disk).
    assert 2 in on_disk.get_column("Object_Label").to_list()
    assert store.stale is True  # a flag the viewer can surface
```

- [ ] **Step 2: Run → fail** (`AttributeError: ... 'stale'` / mirror was overwritten).
Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py::test_save_refuses_after_external_reseed -v`

- [ ] **Step 3: Implement the guard** in `_curation_labels.py`

Mirror `FilteredMeasurements._seed_mtime_ns` (`_filtered_state.py:188-543`). Add a `_seed_mtime_ns: int | None = field(default=None, repr=False)` and a public `stale: bool = field(default=False, repr=False)`. In `load`, after the first write or when the mirror exists, capture `measurements_parquet_path(root).stat().st_mtime_ns`. In `_save_locked`, before writing the curated mirror: if the mirror exists and its current `st_mtime_ns != self._seed_mtime_ns`, set `self.stale = True`, `logger.warning(...)`, and `return` without writing. After a successful mirror write, refresh `_seed_mtime_ns`. (Keep the labels-parquet-last ordering from Phase 1 — but the guard short-circuits before any write.)

- [ ] **Step 4: Swap the store in `_app.py`**

```python
# _app.py — replace the import + construction
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
...
filtered_state = CurationLabels.load(output_root.root, output_root.master_df)
app.server.config[CFG_FILTERED_STATE] = filtered_state
```
Keep `CFG_FILTERED_STATE` unchanged. Do NOT rename the config key (12 call sites read it).

- [ ] **Step 5: Widen the type annotations**

In each of `_callbacks.py`, `_layout.py`, `_filter_panel.py`, `colony_view/_callbacks.py`, `_viewer_card.py`, `_qc_tab/_callbacks.py`, `_qc_tab/review/_callbacks.py`, change the `FilteredMeasurements` annotation/import used for `filtered_state` to `CurationLabels` (import from `._curation_labels`). The `KEY_*` constants are still imported from `_filtered_state` (unchanged — leave those imports). Run mypy across the viewer package to confirm no annotation mismatch.

- [ ] **Step 6: Integration smoke (Flask test client)**

Create `tests/integration/gui/test_triage_callbacks.py` with a fixture that builds the viewer app against a tiny fixture output root (follow `tests/integration/gui/test_qc_review_recompute.py` for the app-construction + Flask-test-client pattern). Assert: app boots, `app.server.config[CFG_FILTERED_STATE]` is a `CurationLabels`, and a POST to the colony single-cell-remove callback route (`/_dash-update-component`) toggling object 2 results in object 2 absent from `deliverables/measurements.parquet`.

- [ ] **Step 7: Run + commit**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py tests/integration/gui/test_triage_callbacks.py -q` and `uv run mypy src/phenotypic/gui/results_viewer`.
```bash
git add src/phenotypic/gui/results_viewer/_app.py src/phenotypic/gui/results_viewer/_curation_labels.py src/phenotypic/gui/results_viewer/_callbacks.py src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_filter_panel.py src/phenotypic/gui/results_viewer/colony_view/_callbacks.py src/phenotypic/gui/results_viewer/_viewer_card.py src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py tests/unit/gui/results_viewer/test_curation_labels.py tests/integration/gui/test_triage_callbacks.py
git commit -m "feat(viewer): swap CurationLabels into the live app + mtime guard"
```

---

### Task 2: Category → color map in `_design.py`

**Files:** Modify `src/phenotypic/gui/_design.py`; Test `tests/unit/gui/test_design_category_colors.py`

**Why:** Category colors are shared between the radial wedges, the per-tile badge, and (Phase 4) the ANOVA boxplots, so they live once in the design tokens, sourced from the data palette (`OI_*`).

- [ ] **Step 1: Test first**

```python
# tests/unit/gui/test_design_category_colors.py
from phenotypic.gui._design import ERROR_CATEGORY_COLORS, category_color, OKABE_ITO
from phenotypic.schema import ErrorCategory

def test_every_core_category_has_an_oi_color():
    for token in ErrorCategory.labels():
        assert token in ERROR_CATEGORY_COLORS
        assert ERROR_CATEGORY_COLORS[token] in set(OKABE_ITO) | {"#BBBBBB"}

def test_other_is_grey():
    assert ERROR_CATEGORY_COLORS["other"] == "#BBBBBB"  # OI_GREY

def test_custom_color_cycles_palette_and_is_deterministic():
    assert category_color("halo", custom_index=0) == category_color("halo", custom_index=0)
    assert category_color("halo", custom_index=0) != category_color("halo", custom_index=1)
```

- [ ] **Step 2: Run → fail** (import error).
- [ ] **Step 3: Implement** in `_design.py` (near the OI block):

```python
ERROR_CATEGORY_COLORS: dict[str, str] = {
    "oversegmented": OI_ORANGE,
    "undersegmented": OI_SKY,
    "merged": OI_PURPLE,
    "background_noise": OI_BLUE,
    "debris": OI_GREEN,
    "other": OI_GREY,
}

#: Palette custom categories cycle through (OI data colors minus the reserved
#: core/Other slots and the alert vermilion / unreadable yellow).
_CUSTOM_PALETTE: tuple[str, ...] = (OI_ORANGE, OI_SKY, OI_GREEN, OI_BLUE, OI_PURPLE)

def category_color(token: str, custom_index: int = 0) -> str:
    """Return the display color for a category token.

    Core tokens map to their fixed OI slot; custom tokens cycle
    ``_CUSTOM_PALETTE`` by their registration index.
    """
    if token in ERROR_CATEGORY_COLORS:
        return ERROR_CATEGORY_COLORS[token]
    return _CUSTOM_PALETTE[custom_index % len(_CUSTOM_PALETTE)]
```
Add the names to `_design.py`'s `__all__` if present.

- [ ] **Step 4: Run → pass. Step 5: Commit** (`feat(design): error-category color map from the OI palette`).

---

### Task 3: Shared radial menu component (`_shared/_radial.py`)

**Files:** Create `src/phenotypic/gui/_shared/_radial.py`; CSS in `results_viewer/_assets/results_viewer.css` (or a shared asset); Test `tests/gui/_shared/test_radial.py`

**Why:** One radial implementation, consumed by both tile surfaces. Built as a ▾ trigger + a lazily-populated popover (reusing the `_build_stack_popover` pattern) so a grid of many tiles stays light.

Design (implement exactly):
- `radial_wedge_id(surface, image_file, label, category)` → `{"type": f"{surface}-cat-wedge", "image_file": image_file, "label": label, "category": category}` (surface ∈ `"colony"`/`"qc"` so the two tabs' callbacks don't collide).
- `radial_trigger_id(surface, image_file, label)` and `radial_popover_body_id(...)`, `radial_store_id(...)` (carries `{image_file,label,surface,categories,custom}` for lazy populate).
- `build_radial_trigger(surface, image_file, label, current_category)` → a small `dbc.Button("▾")` (or the colored badge when `current_category` is set) + an empty `dbc.Popover` (trigger="legacy", placement="right") + a co-located `dcc.Store`. Returns the component list to drop into `build_tile_cell(remove_button=...)`.
- `build_radial_body(surface, image_file, label, categories, custom_categories, current_category)` → the wedge layout: core wedges (positions on a circle via inline `left/top` from precomputed angles, colored by `category_color`), an `Other` wedge, a `Custom ▸` folder wedge, a center "✕ close"/current-state node. The folder, when opened (a clientside class toggle or a second store-driven body), swaps to custom wedges + an `＋ Add custom` wedge. Keep ≤7 primary wedges.
- A per-tile category **badge**: when `current_category` is set, the trigger renders as a colored dot/pill (using `category_color`) with a tooltip of the category label.

- [ ] **Step 1: Pure component-tree tests first** (`tests/gui/_shared/test_radial.py`): assert `build_radial_body(...)` yields one wedge per core category + Other + Custom-folder, each wedge carrying the correct pattern-matched id and `category_color`; assert `build_radial_trigger` renders a badge styled with the category color when `current_category` is set, and a neutral ▾ otherwise; assert wedge ids differ by `surface`.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `_radial.py` (component builders + id factories + the circle-geometry helper `_wedge_positions(n, radius)` returning `(left,top)` px for n wedges). Add CSS for `.radial-popover`, `.radial-wedge`, `.radial-center`, `.radial-badge` (sizes, circular shape, z-index, the inward-fan modifier for edge tiles).
- [ ] **Step 4: Run → pass. Step 5: Commit** (`feat(gui): shared nested radial category menu component`).

---

### Task 4: Wire the radial into colony tiles + mark/populate/custom callbacks

**Files:** Modify `colony_view/_grid.py` (replace `_colony_remove_button` with the radial trigger; pass `current_category` from `labels`), `colony_view/_callbacks.py` (new callbacks), `_ids.py` (any non-radial ids); Test: extend `tests/gui/results_viewer/colony_view/test_grid.py`, add to `tests/integration/gui/test_triage_callbacks.py`

**Why:** Replace the colony tile's binary ✕ with the radial; marking a wedge categorizes + removes in one gesture.

- [ ] **Step 1: Grid test** — `build_grid(...)` tiles now carry the radial trigger (not the old ✕) and a category badge for already-labeled cells. Assert ids/colors. (`current_category` comes from a new `labels` map argument threaded from the callback that calls `build_grid` — note `build_grid` currently takes `removed_keys`; widen to also accept a `key→category` map, defaulting to `{}` so existing callers/tests pass.)
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement**:
  - `_grid.py`: replace `_colony_remove_button(...)` usage with `build_radial_trigger("colony", image_file, label, current_category)`; thread `category_of: dict[tuple[str,int],str]` into `build_grid`/`_build_cell` (derive from `filtered_state.labels` in the callback).
  - `colony_view/_callbacks.py`: add
    - **mark callback** (pattern `{"type":"colony-cat-wedge","image_file":ALL,"label":ALL,"category":ALL}` → Output `STORE_REMOVED_KEYS`): `cat = triggered["category"]`; if `cat == "__restore__"` call `unmark` else `mark`; return `filtered_state.mutate_and_payload(lambda s: s.mark(img,label,cat))`.
    - **lazy populate callback** (radial popover open → fill body via `build_radial_body`, reading `filtered_state.categories()` + custom list + current label).
    - **custom-add callback** → `filtered_state.register_custom_category(name)` then refresh the body (covered more in Task 7).
  - Keep `_toggle_single_cell_removal` removed/retired (the radial subsumes it) OR repurpose its Output. Ensure the grid re-renders category badges after a mark (the existing grid-refresh-on-`STORE_REMOVED_KEYS` callback must also reflect category — thread `labels`).
- [ ] **Step 4: Integration test** — POST a wedge click for object 2 / `debris`; assert `deliverables/errors/debris.parquet` gains object 2 and the curated mirror drops it. **Step 5: Run + commit** (`feat(viewer): radial category triage on colony tiles`).

---

### Task 5: Wire the radial into QC review tiles

**Files:** Modify `_qc_tab/review/_callbacks.py` (replace `_review_tile_remove_button` with the radial; surface `"qc"`), `review/_ids.py`; Test: `tests/unit/gui/results_viewer/test_qc_callbacks_helpers.py`, integration.

**Why:** The second tile surface gets the same triage. Reuses the shared component with `surface="qc"` so callbacks don't collide with colony.

- [ ] **Step 1: Test** the pure helper (a new `mark_review_tile(filtered, image_file, label, category)` mirroring the existing `toggle_review_tile`) + that the gallery builder injects the radial trigger.
- [ ] **Step 2: fail → Step 3: implement** the `surface="qc"` radial in `build_tile_grid(remove_button_builder=...)`, the QC mark/populate/custom callbacks (pattern `{"type":"qc-cat-wedge",...}`), and a `mark_review_tile` helper. **Step 4: run + commit** (`feat(viewer): radial category triage on QC review tiles`).

---

### Task 6: Category-aware bulk "Mark N selected as ▾"

**Files:** Modify `_ids.py`/`review/_ids.py` (bulk category dropdown ids), `colony_view/_callbacks.py` + `_qc_tab/review/_callbacks.py` (extend bulk callbacks), the bulk-bar layout; Test: integration + the pure `bulk_*` helpers.

**Why:** The bulk bar already does remove/restore via `remove_many`/`restore_many`; add a category dropdown so a selection can be marked as a chosen category via `mark_many(selected, category)`.

- [ ] **Step 1: Test** a pure `bulk_mark(filtered, selected, category)` helper (mirrors `bulk_review_curation`) → returns the payload, asserts each selected key gets `category`.
- [ ] **Step 2: fail → Step 3: implement**: add a `dbc.DropdownMenu`/select to both bulk bars (`COLONY_BULK_*` and QC review bulk) listing `filtered_state.categories()`; a callback reads the active selection + chosen category → `mutate_and_payload(lambda s: s.mark_many(selected, category))`. Keep the existing remove (= mark `other`) / restore buttons. **Step 4: run + commit** (`feat(viewer): bulk mark-as-category on the selection bar`).

---

### Task 7: Custom-category add UI

**Files:** Modify `_shared/_radial.py` (the `＋ Add custom` affordance + a small name input/popover), the colony + QC callbacks; Test: integration.

**Why:** The Custom folder's `＋ Add custom` lets a user name a new category at runtime → `register_custom_category`, which then appears in every wheel + bulk dropdown.

- [ ] **Step 1: Test** (integration): submit a custom name "Halo" via the add callback → `filtered_state.categories()` includes `halo`, the registry json persists, and a subsequent wedge populate shows the new custom wedge.
- [ ] **Step 2: fail → Step 3: implement** the add-input (a `dbc.Input` + confirm inside the folder body, or a small modal), the submit callback (sanitizes via `register_custom_category`, handles the `ValueError` on collision/empty with an inline message), and body refresh. **Step 4: run + commit** (`feat(viewer): add-custom-category from the radial folder`).

---

### Task 8: FEATURES.md, CLAUDE.md, and an e2e smoke

**Files:** Modify `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/CLAUDE.md`; Create `tests/e2e/gui/test_radial_triage.py`.

**Why:** Satisfy the `features-md-gate`, document the new chrome, and get one real-browser confirmation (the memory rule: verify Dash callbacks live, not just unit tests).

- [ ] **Step 1: e2e smoke** (`PLAYWRIGHT=1`, mark `ci_flaky` per `tests/CLAUDE.md` if the budget is tight): boot the viewer, open a colony tile's radial, click `Debris`, assert the tile shows the debris badge and `deliverables/errors/debris.parquet` appears. Follow `tests/e2e/gui/conftest.py` `live_server`/`fake_sandbox`.
- [ ] **Step 2: FEATURES.md rows** — one `✅ shipping` row per affordance (radial trigger, each core wedge, Other wedge, Custom folder, Add-custom, per-tile category badge, bulk mark-as dropdown, durable-curation store swap), each with a resolvable `Test ref` into the Task 1/4/5/6/7/8 tests.
- [ ] **Step 3: CLAUDE.md** — note the store swap (`CFG_FILTERED_STATE` now holds `CurationLabels`), the `deliverables/errors/*` live-write + finalize-re-emit dual-ownership (the Q4 note), and the radial component location.
- [ ] **Step 4: Run the e2e + the full Phase-2 suite; commit** (`docs(gui): FEATURES/CLAUDE rows + e2e for radial triage`).

---

## Open design questions (to resolve before/at implementation)

1. **Plate-view table-cell remove (`_viewer_card.py`):** keep it as plain `Other` remove (radial is tiles-only) for v1, or also categorize? (Recommend: keep plain for v1.)
2. **QC review selection UI:** the QC gallery currently passes `selected=set()` (no selection visuals). Does bulk "Mark as" need selection enabled on QC review tiles, or is bulk-mark colony-only for v1 with QC review staying single-tile? (Recommend: enable selection parity on QC review.)
3. **Type annotations:** widen each `FilteredMeasurements` annotation to `CurationLabels`, or introduce a shared `Protocol`/alias to avoid churn and keep `FilteredMeasurements` importable? (Recommend: direct widen — `FilteredMeasurements` is being retired.)
4. **Radial open/close mechanism:** `dbc.Popover(trigger="legacy")` (server-free open) + clientside folder toggle, vs. a store-driven server populate. (Recommend: legacy popover + lazy populate-on-open, matching `_build_stack_popover`.)
5. **Existing ✕ / bulk-remove semantics:** the old per-tile ✕ and bulk "Remove" now map to `mark(..., "other")` (durable). Confirm the bulk bar keeps an explicit "Remove (Other)" alongside the category dropdown.
6. **Test depth:** integration (Flask client) for every callback + ONE e2e smoke, vs. broader Playwright coverage. (Recommend: integration-per-callback + 1 e2e.)
7. **`build_grid` signature change:** threading a `key→category` map widens `build_grid`/`build_tile_grid`. Acceptable, or pass categories via the existing `removed_keys` channel reshaped? (Recommend: add an optional `category_of` map, default `{}`.)

## Self-review (against spec §3/§6)

- §6.1 radial menu → Tasks 3–5. §6.2 bulk → Task 6. §6.3 colors → Task 2. Custom folder/add → Tasks 3,7. Store swap + durable plain-remove (§2) → Task 1. Mtime guard (Q3) → Task 1. FEATURES/docs → Task 8.
- Not in Phase 2 (correctly deferred): the cutoff ANOVA engine (Phase 3), the Error-analysis tab (Phase 4), CLI finalize per-category/error_analysis emit (Phase 5), WORKFLOWS.md + screenshots (Phase 6).
