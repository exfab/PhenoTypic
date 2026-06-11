# Design: Error-category triage → live cutoff finder

- **Date:** 2026-06-10
- **Status:** Draft for user review, no implementation plan yet
- **Author:** Alexander Nguyen with Claude
- **Scope:** Results viewer (colony view + QC tiles + a new analysis tab), a
  durable curation-labels store, an `analysis/` cutoff engine, and the CLI
  finalize/deliverables wiring. No changes to detection/measurement algorithms.

## 1. Summary

Today the results viewer offers a single binary affordance per colony tile:
remove from the dataset (`✕`) or restore (`↺`). Removed keys live in
`FilteredMeasurements` as a flat set of `(Metadata_ImageFile, Object_Label)`
tuples, mirrored **live on each mark** to `deliverables/measurements.parquet`
(curated = master − removed) and `deliverables/measurements.csv` — via the
`measurements_parquet_path`/`measurements_csv_path` helpers, which already
resolve under `deliverables/`. (The current `_filtered_state.py` module
docstring says "output root"; that wording is stale — the helpers point at
`deliverables/`.)

This feature replaces "remove" with **categorized triage**: the user marks
*why* an object is bad — oversegmented, undersegmented, merged/touching,
background noise, debris/artifact, a reserved "other" bucket, or a custom
category — via a **nested radial menu** on each tile (and a bulk "mark
selected as…" bar). Each marked object leaves the curated set (exactly as
remove does today) and is partitioned into a per-category parquet under
`deliverables/errors/`.

The labels become hand-labeled training data for a **live cutoff finder**: a
new "Error analysis" tab runs, for the selected error category, a per-measurement
one-way ANOVA (good vs. that category) and surfaces a **ROC/Youden's J**
threshold per discriminative measurement, ranked by separability (AUC). The
output is a ranked table the user reads to identify measurement cutoffs they can
adopt to filter similar bad data, recomputed live (debounced) on every mark and
persisted to `deliverables/error_analysis.{parquet,csv,html}`.

A single **durable labels store** (`<root>/qc/curation_labels.parquet`) is the
source of truth; the curated mirror and the per-category exports are both
derived from it. Unlike today's curation, it is **not** wiped by CLI re-runs —
it is re-keyed onto the fresh master with a stale banner.

## 2. Decisions captured during brainstorming

All confirmed with the user via the visual companion:

| Decision | Choice |
| --- | --- |
| Taxonomy | Fixed core enum + custom escape hatch |
| Core categories | Oversegmented, Undersegmented, Merged/touching, Background noise, Debris/artifact |
| Plain remove | Folded into a reserved `OTHER` category (every removal is categorized) |
| Custom categories | A "Custom ▸" radial *folder* that expands to user categories + an "＋ Add custom" action |
| Tile affordance | Per-tile **radial menu** (A) **+** select-then-mark **bulk bar** (C) |
| Recompute trigger | Live, debounced, as objects are marked in-session |
| Result surface | Live in-viewer panel **+** persisted deliverable + HTML report |
| ANOVA grouping | Per-category vs. good (binary), ranked by effect size (AUC) |
| Cutoff method | ROC / Youden's J; AUC ranking; recall/precision at the cutoff; BH-FDR adjusted p |
| Label durability | Durable sidecar, re-keyed on re-run, with a stale banner |
| Panel placement | New "Error analysis" tab |
| v1 apply scope | Identify + export a filter spec; one-click apply is phase 2 |
| ⚠️ Plain "Other" durability | **Confirmed:** plain removals become durable too (unified store) |
| ⚠️ Re-key safety | **Confirmed:** fingerprint each label by centroid/bbox; drop ambiguous re-keys |

## 3. Evidence basis (established from local code)

- The colony grid and the QC review gallery both render tiles through the
  single shared `build_tile_cell()` in `gui/_shared/tiles.py`, which already
  takes a caller-supplied `remove_button`. This is the dedup boundary the new
  radial trigger plugs into — both surfaces get the feature from one change.
- `FilteredMeasurements` (`gui/results_viewer/_filtered_state.py`) owns the
  removal set keyed by `KEY_COLUMNS = (Metadata_ImageFile, Object_Label)`,
  derives the curated frame via a polars anti-join, persists atomically
  (`.tmp` + `os.replace`) under a re-entrant lock, and already guards against
  clobbering a CLI-reseeded mirror via `_seed_mtime_ns`. The new store
  generalizes this class.
- `scipy` is a first-class dependency; `scipy.stats.f_oneway` and the building
  blocks for ROC/AUC are available without a new dependency. `analysis/qc/_icc.py`
  already hand-computes a two-way ANOVA from numpy, establishing the in-repo
  precedent for ANOVA math.
- Analyzers are pydantic `SetAnalyzer`/`QualityCheck` models invoked via
  `.analyze(df)`, exported from `analysis/__init__.py`. The memory note
  "new checks must be exported to be discoverable" applies — the cutoff engine
  must be re-exported.
- Deliverable filenames/paths are single-sourced in `tools_/_io_constants.py`
  (`deliverables_dir`, `measurements_parquet_path`, `measurements_by_feature_dir`,
  `analysis_parquet_path`, `qc_dir`, `qc_review_state_path`, …); new artifacts
  add helpers there.
- `finalize_post_master_outputs` is the canonical FINAL master writer; the
  per-category exports + error-analysis chain hook here. `qc/review_state.json`
  is already GUI-owned-but-CLI-reset; the labels store is GUI-owned-and-CLI-
  *preserved*, a deliberate contrast.
- GUI palette rule (`gui/CLAUDE.md`, `DESIGN.md`): `COLOR_*` are UI-only, `OI_*`
  (Okabe-Ito) are data-only. Category colors double as plot series colors, so
  they are sourced from `OI_*`.

## 4. Terminology

- **Label** — an assignment `(image_file, object_label) → category`.
- **Category** — a member of the core `ErrorCategory` enum, the reserved
  `OTHER`, or a registered custom name.
- **Good** — an object present in the master frame with **no** label.
- **Curated frame** — master minus every labeled object (today's
  `measurements.parquet`, now derived from the labels store).
- **Separability** — AUC of the good-vs-category split on one measurement.

## 5. Data model

### 5.1 `ErrorCategory` enum (`phenotypic.schema`)

A closed value set following the project's `MeasurementInfo`/`ConstantLabels`
convention (`(label, description)` members, `category()` classmethod). Members:

```
OVERSEGMENTED     "one colony split into many detections"
UNDERSEGMENTED    "a colony detected as too small / fragmented low"
MERGED            "multiple touching colonies detected as one"
BACKGROUND_NOISE  "not a colony — agar texture, reflection, vignette"
DEBRIS            "dust, scratch, bubble, or plate artifact"
OTHER             "removed without a specified reason (catch-all)"
```

A `Literal` alias is added in `tools_/typing_.py` only if string-typed input
crosses a boundary (e.g., the radial callback's wedge id); paired with an
alignment test if so, per the project's enum/Literal convention.

### 5.2 Custom-category registry

Custom categories are user-named at runtime. Stored as `{name, color, created}`
records persisted alongside the labels store (same parquet's sidecar metadata or
a small `<root>/qc/custom_categories.json`). Names are sanitized to the
`[A-Za-z0-9._-]` class used by `is_safe_path_component` so they are safe as a
parquet filename component. Colors are cycled from the `OI_*` palette after the
core categories consume their fixed slots.

### 5.3 Durable labels store

New `CurationLabels` dataclass (`gui/results_viewer/_curation_labels.py`),
generalizing `FilteredMeasurements`:

- **Persistence:** `<root>/qc/curation_labels.parquet` — columns
  `Metadata_ImageFile`, `Object_Label`, `Curation_Category`, and a fingerprint
  (`Bbox_CenterRR`, `Bbox_CenterCC` captured at mark time). Atomic write under a
  re-entrant lock, same discipline as `FilteredMeasurements`.
- **CLI does not wipe it.** This is the durability contrast with
  `qc/review_state.json`.
- **Derived outputs — written LIVE on each mutation** (same discipline and the
  same `deliverables/` location as today's curated mirror):
  - `deliverables/measurements.parquet` + `.csv` = master anti-join all labeled
    keys (curated good set — unchanged contract for the viewer/analysis chain).
  - `deliverables/errors/<sanitized_category>.parquet` = master rows whose label
    is that category, carrying a `Curation_Category` column. (This is the
    "marked objects land in separate parquet files in deliverables" ask.)
  - These live writes are the GUI's responsibility; CLI finalize (§9)
    idempotently re-emits the same files so a headless re-run stays consistent.
- **Mutators** mirror `FilteredMeasurements` but carry a category:
  `mark(image_file, label, category)`, `unmark(...)`, `mark_many(keys, category)`,
  `relabel(...)`, plus `mutate_and_payload(action)` for the lock-consistent Dash
  store write. `removed_keys` becomes a derived property (= all labeled keys) so
  existing crop-dimming/`is_removed` call sites keep working.

### 5.4 Migration & re-keying

- **Migration:** on first load against a `<root>` that has a legacy
  `measurements.parquet` but no labels store, infer removed keys (master −
  curated) and import them as `OTHER`. These are tallied as `migrated` (a
  distinct `RekeyReport` field, not `kept`) so the stale banner stays accurate.
- **Re-keying (⚠️ confirmed):** on load/recompile, re-attach stored labels to
  the fresh master:
  - **Bbox present** (`Bbox_CenterRR/CC`): if the exact `(image_file,
    object_label)` survives and its centroid is within a small pixel tolerance →
    keep. If the exact key survives but the centroid moved beyond tolerance →
    **drop immediately** (do *not* search neighbours — avoids mis-attaching to an
    adjacent colony). If the exact key is gone (renumbered) → re-key only to a
    *unique* object within tolerance of the stored centroid, else drop.
  - **Bbox absent** (no `MeasureBounds`): fingerprinting is impossible →
    **degrade gracefully** — keep labels whose exact key survives, drop the rest,
    emit one WARNING. `Bbox_*` is already a de-facto requirement of the
    colony/QC crop tiles, so this is an edge case, not the norm.
  - Counts `{kept, re-keyed, dropped, migrated}` feed the stale banner.
- ⚠️ **Behavior change (confirmed):** because `OTHER` (plain remove) now lives
  in this durable store, plain removals survive CLI re-runs instead of being
  wiped. Re-keying runs at **load** time only; a viewer session left open
  *across* a CLI re-run still needs an explicit `measurements.parquet`
  **mtime guard** (restoring the protection `FilteredMeasurements._seed_mtime_ns`
  gave) to avoid clobbering the fresh seed — added in **Phase 2**, not Phase 1.

## 6. Tile UI

### 6.1 Radial menu (shared)

New shared component in `gui/_shared/` (e.g. `_radial.py`): a wheel of
absolutely-positioned wedge buttons placed on a circle via precomputed angles,
popped as a high-`z-index` overlay anchored to the tile's ▾ trigger, dimming the
tile behind it. One pattern-matched callback resolves the chosen wedge →
category. Edge tiles fan the wheel inward to avoid viewport clipping.

- **Layer 1:** 5 core categories + `OTHER` + `Custom ▸` (folder) + center close.
- **Layer 2 (custom folder):** registered custom categories + `＋ Add custom`
  (opens a small name input → registers a category) + center "back".
- The ▾ trigger replaces the current `✕`/`↺` button passed into
  `build_tile_cell(remove_button=…)`; both colony view and QC review supply it.
- **Tile state:** a small colored corner badge indicates the assigned category
  (in addition to the existing dim). Color from the category→`OI_*` map.

### 6.2 Bulk "mark selected as…"

Reuse the existing multi-select checkbox + shift-click range (`expand_range`,
`STORE_*` selection plumbing). When ≥1 tile is selected, a sticky action bar
shows "Mark N selected as ▾" opening the **same** radial/menu; the chosen
category applies to the whole selection via `mark_many`.

### 6.3 Colors & design tokens

A category→color map in `_design.py` assigns each core category a fixed `OI_*`
slot; custom categories cycle the remaining `OI_*` colors. No `COLOR_*`
(UI-chrome) color is used as a category swatch.

## 7. Cutoff engine (`analysis/_error_cutoffs.py`)

A pydantic model (subclassing `SetAnalyzer` for consistency, or a focused
analyzer if the `on/groupby` fields don't fit cleanly — to be settled in the
plan), exported from `analysis/__init__.py`.

- **Inputs:** `master_measurements.parquet` (all objects + measurements), the
  labels store, and (verified mode) the QC review state. The engine takes the
  **good** baseline frame + the per-category **error** rows as inputs (so it
  stays mode-agnostic and unit-testable); the panel chooses which good set.
- **Good-baseline modes (a panel toggle).** The error set is always the rows
  labeled with the target category (unchanged). The *good* baseline has two
  modes:
  - **All unlabeled** (default) — every object with no curation label
    (`master − labeled`). Simple, but includes any *un-triaged* noise the user
    hasn't reached yet, which dilutes the good distribution.
  - **Verified-only** — only unlabeled objects in QC-review groups the user has
    **marked reviewed in any QC module** (resolved decision: any-module). This is
    the `deliverables/verified.parquet` set: a reviewed group's unlabeled members
    are *confirmed* good, so un-triaged noise (in not-yet-reviewed groups) can't
    contaminate the baseline. **Only the good set is restricted** — error
    examples from any group still count (a marked error is confirmed regardless
    of its group's review state; resolved decision: good-only).
  - **Verified-good derivation:** an object is verified-good iff it is unlabeled
    AND its `(image_file, object_label)` is a member of ≥1 group whose key
    appears in `review_state.json`'s `reviewed` set for *any* module. The
    group→member mapping comes from `qc/qc_members.parquet`
    (`_data.group_member_keys`); the review state from `ReviewState`
    (`qc/review_state.json`). A GUI/data helper builds this frame and (verified
    mode) materializes `deliverables/verified.parquet`; the engine just consumes
    the resulting good frame.
- **Measurement selection:** numeric measurement columns auto-detected by the
  category prefixes (`Size_`, `Shape_`, `Intensity_`, `TextureGray_`,
  `SymZones_`, `GridSpatial_`, `Bbox_` extents), excluding metadata/grid-key and
  non-numeric columns.
- **Per measurement:**
  - One-way ANOVA `scipy.stats.f_oneway(good, error)` → F, p.
  - Effect size + **AUC** (good-vs-error binary separability), used as the
    ranking key (effect size, not raw p — p is sample-size driven).
  - **ROC/Youden's J** optimal threshold (direction-aware), with recall and
    precision reported at that cutoff.
  - **Benjamini-Hochberg** FDR adjustment of p across all tested measurements.
- **Output:** a tidy frame `[measurement, auc, f, p, p_bh, cutoff, direction,
  recall, precision, good_n, error_n]` sorted by AUC desc.
- **Guard:** below a minimum error-n (default 8; configurable) — or, in
  verified-only mode, below a minimum verified-good-n — return an
  "insufficient labels / review more QC groups" sentinel rather than unstable
  statistics.

## 8. "Error analysis" tab (`gui/results_viewer/_error_tab/`)

A new tab alongside Colony view / QC / Heatmap:

- **Category switcher** — chips per category with live label counts; defaults to
  the highest-count non-`OTHER` category.
- **Good-baseline toggle** — switches the *good* comparison set between
  **All unlabeled** (default) and **Verified-only** (the `verified.parquet` set).
  Shows the live verified-good count; when verified mode is on but that count is
  below the guard, a "review more QC groups to use verified mode" state. Flipping
  it recomputes the ANOVA against the chosen baseline.
- **Ranked table** — the engine's frame (measurement, AUC, suggested cutoff,
  BH-p), top-discriminative first.
- **Distribution plot** — box/violin of good vs. the selected error category for
  the focused measurement, with a **draggable cutoff line**; dragging updates the
  recall/precision readout live.
- **Copy filter spec** — exports the chosen cutoff(s) as a reusable filter
  snippet (JSON and/or a `SetAnalyzer`/post-filter form). *Apply-as-filter is
  phase 2.*
- **Reactivity** — recomputes (debounced) when the labels store changes, driven
  by the same store-write the tiles already trigger. A "need more labels" state
  shows below the engine's min-n guard.
- **Persistence timing** — the panel computes in memory on each recompute; the
  persisted `deliverables/error_analysis.{parquet,csv}` is written debounced as
  labels change, while the heavier `error_analysis.html` report is regenerated
  only at CLI finalize and via an explicit "Save analysis report" action (not on
  every click), to avoid report churn. **`deliverables/verified.parquet`** is
  (re)written debounced from the QC review state whenever verified mode is active
  and the reviewed-group set changes.
- **Stale banner** — surfaces the re-keying `{kept, re-keyed, dropped}` summary
  after a CLI re-run.

## 9. CLI / deliverables wiring

- New `io_constants` constants + path helpers: `DIR_ERRORS` (`errors`),
  `errors_dir(output)`, `error_category_parquet_path(output, category)`,
  `ERROR_ANALYSIS_PARQUET`/`_CSV`/`_HTML` + `error_analysis_*_path(output)`,
  `VERIFIED_PARQUET` (`verified.parquet`) + `verified_parquet_path(output)`,
  `curation_labels_parquet_path(output)`, and the custom-registry path.
- **`deliverables/verified.parquet` is GUI-written, not CLI-emitted.** It is
  derived from `qc/review_state.json`, which `finalize_post_master_outputs`
  **resets** on every CLI re-run — so a headless finalize has no reviewed groups.
  The GUI owns this file (live/debounced while verified mode is active); finalize
  leaves it untouched rather than overwriting it with an empty set.
- `finalize_post_master_outputs` (and the `--recompile` worker path) is the
  **authoritative, idempotent** writer of the per-category
  `deliverables/errors/*.parquet` and `deliverables/error_analysis.{parquet,csv,html}`
  from the labels store (the GUI writes the same files live; finalize re-emits
  them so a headless run with no open viewer still produces them). It also
  **preserves + re-keys** the labels store (no wipe). Mid-run chunk writers are
  untouched (the existing rule: don't add finalize work to the chunk writer).
- The curated `measurements.parquet` continues to be the post-applied mirror the
  viewer/analysis chain reads; it is now derived from the labels store.

## 10. Docs & CI ledgers (gated)

- `gui/FEATURES.md` rows: radial trigger, each core wedge, Other wedge, Custom
  folder, Add-custom, per-tile category badge, bulk mark bar, Error-analysis tab,
  category switcher, good-baseline (all-unlabeled / verified-only) toggle, ranked
  table, draggable cutoff, copy-filter-spec, stale banner. Each `✅ shipping` row
  needs a `Test ref`.
- `gui/WORKFLOWS.md` row for the end-to-end triage→cutoffs flow + a matching
  `_capture_<id>` in `scripts/capture_gui_tutorial_screenshots.py` + a
  walkthrough page under `docs/source/tutorials/gui/`. Re-run the capture and
  commit the full refreshed PNG set.
- `gui/CLAUDE.md` and `tools_/_io_constants.py` docstring updates for the new
  artifacts.

## 11. Risks & edge cases

- **Class imbalance / small n** — few error labels make ANOVA/ROC unstable;
  gated by the min-n guard with an explicit "need more labels" UI state.
- **Object-label instability across re-detection** — mitigated by the centroid
  fingerprint re-key; ambiguous labels are dropped and counted, never silently
  re-attached.
- **Per-measurement independence** — v1 reports single-measurement cutoffs;
  correlated measurements and combined multi-condition rules are out of scope
  (phase 2+).
- **`OTHER` as a grab-bag** — selectable in the switcher but low-signal by
  nature; not the default focus.
- **Radial scale** — wheels read best at ≤7 items; many custom categories
  overflow into the folder's own scroll/secondary list rather than crowding the
  wheel.
- **Wiped vs. durable contrast** — `qc/review_state.json` is CLI-reset while
  `qc/curation_labels.parquet` is CLI-preserved; both live under `qc/` and must
  not be conflated by finalize.

## 12. Phasing (for the implementation plan)

1. **Data model** — `ErrorCategory` enum, custom registry, `io_constants` paths,
   the durable `CurationLabels` store (migration, fingerprint re-keying, derived
   curated + per-category outputs). Unit-tested in isolation.
2. **Tile UI** — shared radial component + nested custom folder + category
   colors + per-tile badge + bulk mark bar, wired on both colony and QC tiles.
3. **Cutoff engine** — `analysis/_error_cutoffs.py` (ANOVA + AUC + Youden +
   BH-FDR + min-n guard) with tests; exported from `analysis/__init__.py`.
4. **Error-analysis tab** — live panel (switcher, ranked table, boxplot,
   draggable cutoff, copy-filter-spec, debounced recompute).
5. **CLI finalize** — per-category exports + `error_analysis.*` + HTML report +
   re-keying + stale banner.
6. **Docs/ledgers** — FEATURES/WORKFLOWS/tutorial/screenshots/CLAUDE updates.

## 13. Out of scope (v1)

- One-click "apply this cutoff as a filter" (phase 2).
- Multi-measurement / combined-rule cutoffs.
- Cross-run pooling of labeled errors or a folder-watcher ingesting foreign
  parquet files (the trigger is in-session marking only).
- Omnibus across-all-categories ANOVA (per-category-vs-good only in v1).
