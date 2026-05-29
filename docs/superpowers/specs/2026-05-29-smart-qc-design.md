# Smart QC — design spec

**Date:** 2026-05-29
**Branch:** `feature/smart-qc-gui`
**Status:** design approved in brainstorming; pending user spec review → implementation plan.
**Alternatives log:** [`2026-05-29-smart-qc-design-alternatives.md`](2026-05-29-smart-qc-design-alternatives.md)

## Goal

Make poor detection and disagreeing biological replicates fast to find and
curate. A QC check takes a `groupby` and an `on` column, computes a
statistical metric per group, and flags groups whose members disagree.
Checks run during CLI recompile/remeasure and persist a compact artifact.
A new **Review** mode in the results-viewer QC tab walks the user through
the worst-agreeing groups worst-first, shows each group's colonies as a
tile gallery (reusing the colony-view tiles), lets the user curate, and
recomputes after each group.

## Current state (what already exists)

- **`SetAnalyzer`** (`analysis/abc_/_set_analyzer.py`) — grouped-analysis
  root; already provides `on: ColumnRef`, `groupby: ColumnRefList`,
  `agg_func`, `n_jobs`.
- **`QualityCheck`** (`analysis/abc_/_quality_check.py`) — extends
  `SetAnalyzer`; subclass `_compute(group)` emits `QC_<name>_Severity`;
  base derives `Flag`/`Status` from `severity_warn`/`severity_fail`;
  has `summary()`, `flagged_keys()`, `dash()`.
- **`ReplicateAgreement`** (`name="SE"`) — relative SE per `(group, time)`.
- **`ExpectedVsDetectedCount`** (`name="Count"`) — detected vs metadata.
- **`MADOutlierRemover` / `TukeyOutlierRemover`** — plain `SetAnalyzer`s
  that *remove rows*; reusable MAD/IQR math (`_MAD_CONSISTENCY = 0.6745`).
- **Schema enums** `QUALITY_CHECK`, `QUALITY_SE`, `QUALITY_COUNT` in
  `schema/`.
- **`QcRecipe`** (`gui/_qc_recipe.py`) — serializable check list persisted
  to `.viewer_cache/qc_recipe.json` (GUI-owned).
- **QC tab** (`gui/results_viewer/_qc_tab/`) — chart-centric cards
  (figure + summary + "mark all flagged"); add/edit/duplicate/delete via
  modal; class picker from `OperationRegistry.get_by_category("quality_check")`.
- **Colony view** (`gui/results_viewer/colony_view/`) — server-side crop
  route `/crops/<dataset>/<stem>/<label>.png?size=`, `crop_overlay()` +
  LRU overlay cache, 2D `build_grid()`, multi-select + bulk remove/restore
  via `FilteredMeasurements`.
- **CLI finalize** — `finalize_post_master_outputs()` applies post, joins
  metadata, writes `measurements`/`analysis`/per-feature splits. **No QC
  is computed in the CLI today.**

---

## Component A — QC metrics engine (`analysis/`)

### A.1 `QualityCheck` contract refactor

Drop the normalized `severity` abstraction in favor of a single raw
**headline metric** per check plus a direction flag.

- Rename emitted `QC_<name>_Severity` → **`QC_<name>_Metric`** (raw value
  in the metric's own units). Keep `QC_<name>_Flag` (bool) and
  `QC_<name>_Status` (`pass`/`warn`/`fail`).
- Add **`_HIGHER_IS_BAD: ClassVar[bool]`** on every concrete subclass —
  intrinsic to the metric, not user-tunable.
- Thresholds become **directional** instance fields **`warn_threshold`**
  / **`fail_threshold`** (metric units, class-default values,
  per-instance overridable):
  - `_HIGHER_IS_BAD=True`  → `fail` when `metric ≥ fail_threshold`,
    `warn` when `metric ≥ warn_threshold` (with `warn ≤ fail`).
  - `_HIGHER_IS_BAD=False` → `fail` when `metric ≤ fail_threshold`,
    `warn` when `metric ≤ warn_threshold` (with `fail ≤ warn`).
- `NaN` metric ⇒ `Status="pass"`, `Flag=False` (under-powered /
  degenerate bins never gate curation — unchanged behavior).
- **`summary()`** returns one row per `groupby` key:
  `[*groupby, n_members, n_flagged, worst_metric, status]`, where
  `worst_metric` is the extreme in the bad direction (max if
  higher-is-bad, else min), matching today's per-group aggregation.
- Add **`group_members()`** helper: maps each `groupby` key →
  list of member `(Metadata_ImageFile, Object_Label)` pairs (+ each
  member's contributing value), for the worklist and tile gallery.
- Expose `_HIGHER_IS_BAD`, `warn_threshold`, `fail_threshold`, and `name`
  as machine-readable metadata (already partly via `model_json_schema()`).

**Migration:** the two existing checks already emit a raw ratio as
"severity", so migration ≈ rename + directional comparison; both are
`_HIGHER_IS_BAD=True`. Update the chart QC tab and tests referencing
`severity_col()`/`severity_warn`/`severity_fail`.

### A.2 v1 check roster (6)

| Class | `name` | family | metric | `_HIGHER_IS_BAD` | default warn/fail | reuse |
|---|---|---|---|---|---|---|
| `ReplicateAgreement` | `SE` | replicate | rel-SE = \|SE\|/\|mean\| per (group,time) | True | 0.10 / 0.20 | existing |
| `ExpectedVsDetectedCount` | `Count` | detection | \|detected−expected\|/expected (∞ if no metadata) | True | 0.05 / 0.10 | existing |
| `RelativeMAD` (new) | `MAD` | replicate (robust) | MAD/\|median\| | True | 0.10 / 0.20 | MAD math |
| `MaxModifiedZScore` (new) | `ZMax` | detection/outlier | max over members of 0.6745·\|x−med\|/MAD | True | 3.5 / 5.0 | MAD math |
| `ICC` (new) | `ICC` | replicate reliability | ICC(2,1) two-way random, absolute agreement | **False** | warn ≤ 0.75, fail ≤ 0.50 | — |
| `TukeyOutlierFraction` (new) | `Tukey` | detection | fraction of members outside Q1−k·IQR … Q3+k·IQR (k=1.5) | True | 0.10 / 0.25 | IQR fences |

- New MAD/ZMax/Tukey checks reuse the math in `MADOutlierRemover` /
  `TukeyOutlierRemover` (extract shared helpers if convenient) but are
  `QualityCheck` subclasses that **flag groups**, not remove rows.
- `ICC` validates the `_HIGHER_IS_BAD=False` path end-to-end.
- Each new check gets a sibling schema enum (`QUALITY_MAD`,
  `QUALITY_ZMAX`, `QUALITY_ICC`, `QUALITY_TUKEY`) documenting its emitted
  columns, mirroring `QUALITY_SE` / `QUALITY_COUNT`.
- Thresholds above are **defaults**; tune during implementation against
  `load_synth_yeast_plate()` and real plates.

### A.3 Registry

No registry edits: `OperationRegistry.get_by_category("quality_check")`
already auto-discovers `QualityCheck` subclasses exported from
`phenotypic.analysis`. New checks are added to `analysis/__init__.py`.

### A.4 Time-aware grouping nuance

Time-course checks (`SE`, `ICC`) compute per `(group, time)` internally
but the **review unit is the `groupby` key**. `summary()` reduces
per-time metrics to a worst-per-group value for ranking — the group is
ranked by its **worst timepoint**. In the Review detail pane the tile
gallery is **faceted into one row per timepoint** (see D.2 / D.5).

---

## Component B — CLI compute & persistence (`_cli/`, pipeline)

### B.1 QC config in `pipeline.json`

`ImagePipeline` gains a serializable **`qc: list[QualityCheck]`** section
(sibling to `operations` and `post`); `to_json`/`from_json` round-trip
it via each check's pydantic params. Consistent with the pipeline already
carrying an analysis `model`.

```jsonc
{ "operations": [...], "post": [...],
  "qc": [
    {"class": "ReplicateAgreement", "params": {"on":"Size_Area","groupby":["Plate","Strain","Time"],"time_label":"Time"}},
    {"class": "RelativeMAD", "params": {"on":"Size_Area","groupby":["Plate","Strain","Time"]}}
  ] }
```

**Concurrency:** `pipeline.json` is CLI-written with a staleness guard.
The QC tab edits only the `qc` array via a **scoped atomic
read-modify-write** (preserving `operations`/`post`), and performs a
**one-time migration** of any existing `.viewer_cache/qc_recipe.json` into
the `qc` array on first load. `QcRecipe` moves from a standalone sidecar
to a thin adapter over the pipeline's `qc` section.

### B.2 Shared runner `run_qc(...)`

A **GUI-free** function (home: `phenotypic/_cli/_cli_qc.py` or a neutral
`phenotypic/qc/` module — decide in planning):

```
run_qc(measurements_df, pipeline, output_dir) -> None
```

- Instantiates each `pipeline.qc` check.
- Runs them on the **post-applied + metadata-joined** frame
  (`measurements.parquet`, not the clean master — QC `groupby` cols often
  come from joined metadata).
- Writes the compact `qc/` artifact (B.3).
- Pure: writes metrics only; **does not** touch `review_state.json`.

### B.3 `qc/` artifact

```
<output>/qc/
  qc_summary.parquet   # one row per (instance_id, group)
    instance_id, class, <groupby cols...>, metric, status, flag,
    n_members, n_flagged, rank        # rank = worst-first within instance
  qc_members.parquet   # group -> member colonies
    instance_id, <groupby cols...>, Metadata_ImageFile, Object_Label,
    member_value
  qc_config.json       # snapshot of the checks that produced this run
```

`measurements.parquet` is left untouched. The compact artifact makes the
GUI's after-each-group recompute cheap to rewrite.

### B.4 CLI integration

- `finalize_post_master_outputs()` calls `run_qc(...)` after
  `_emit_analysis_outputs` / `split_master_by_feature`, so QC is computed
  on **every recompile and remeasure** when `pipeline.qc` is non-empty.
- Add `--no-qc` to skip; computing is otherwise implied by a non-empty
  `qc` section.
- **Reset-on-rerun:** because a CLI run is "a different run", the finalize
  path clears/regenerates `qc/review_state.json` (D.6). `run_qc` itself
  does not — so the GUI's in-session recompute preserves review progress.
- Same `run_qc` is the in-process seam the GUI calls for the
  after-each-group recompute.

---

## Component C — shared tile imaging (`gui/`)

### C.1 New `gui/_shared/tiles.py`

One implementation, two consumers (colony view + QC review):

- `crop_overlay(png_path, center_rr, center_cc, size, pad_value)` and the
  LRU `_load_overlay_rgb(path, mtime_ns)` cache — moved verbatim from
  `colony_view/_cropper.py` (already pure, no colony knowledge).
- `is_safe_path_component(name)` — moved from `_tile_routes.py`.
- `register_crop_route(app, output_root, segment)` — crop-route factory,
  generalized so both `/crops` and a QC segment can mount it; looks up
  `Bbox_CenterRR/CC` by `(Metadata_ImageFile, Object_Label)` and serves a
  centered PNG.
- `build_tile_cell(...)` — per-tile chrome (img, checkbox, remove/restore,
  badges, removed/selected state) parameterized on a `url_builder` and a
  `(dataset, image_file, label)` key. **This is the dedup boundary.**
- `build_tile_grid(keys, url_builder, *, selected, removed, display_size, ...)`
  — flat gallery of tiles + a row-major key list for shift+click range
  selection (`expand_range` reused).

### C.2 Refactor `colony_view`

`colony_view/_grid.py` keeps only its 2D axis-header arrangement and
builds each cell via `build_tile_cell`; its cropper/route import from
`_shared/tiles.py`. Behavior unchanged; re-run colony-view tests
(`test_cropper`, `test_crop_routes`, `test_grid`).

---

## Component D — GUI QC Review (`gui/results_viewer/`)

### D.1 QC tab: Configure | Review toggle

A single **QC tab** with a segmented switch:
- **Configure** — existing per-check cards (now editing `pipeline.json`'s
  `qc` array via B.1) + overview charts + per-check summary.
- **Review** — the master-detail walkthrough (D.2–D.6).

New code under `gui/results_viewer/_qc_tab/review/` (or a `qc_review/`
sibling); reuses `_shared/tiles.py`.

### D.2 Layout (master–detail)

- **Top toolbar:** module picker (`Module: <check> ▾`, populated from the
  enabled `qc` entries), `on`/`groupby` chips, a "Show: unreviewed / all /
  fail+warn" filter, and **↻ Re-sort queue**.
- **Summary header:** stat tiles — total groups, fail, warn, pass,
  reviewed, colonies removed, median metric (per the selected module).
- **Left worklist sidebar:** groups for the selected module sorted
  worst-first; each row shows group key, metric, status badge, a reviewed
  ✓ (dimmed when reviewed), and a "moved/changed" hint after recompute.
- **Right detail pane:** group header (key, metric with delta after
  recompute e.g. `0.42 → 0.21`, status, `n` members, `n` removed), the
  tile gallery (`build_tile_grid`; **faceted into one row per timepoint**
  for time-course checks), and actions.

### D.3 Module picker

Switches the worklist + detail to the selected check's `qc_summary` /
`qc_members` slice (filtered by `instance_id`). Each module has its own
worklist and its own review progress (D.6).

### D.4 Curation

Per-tile remove (×) / restore (↺) and multi-select bulk remove/restore,
reusing the **same `FilteredMeasurements` removal store** as colony view —
edits are consistent across the whole viewer. Removed tiles dim. v1 ships
**no dedicated whole-group/whole-image exclude buttons** — excluding a
group is select-all + remove.

### D.5 Recompute (per-group) + frozen order

- Recompute runs **when the user finishes a group** (mark reviewed / next)
  **if changes were made** — not on every tile click. It calls `run_qc`
  in-process on the curated (removal-applied) frame and rewrites the `qc/`
  artifact.
- The recomputed group's metric/badge update **in place**; its row keeps
  its position with a "moved/changed" hint. Order only changes when the
  user clicks **↻ Re-sort queue**. The detail header shows the
  before→after metric delta and new status.

### D.6 Review progress

- Stored in **`<output>/qc/review_state.json`**, **per-module** (keyed by
  check `instance_id` → `{reviewed: [group-key…], last: group-key}`).
- A group becomes reviewed on explicit **mark reviewed**, and is
  auto-marked when the user advances past a group they curated.
- **CLI recompile/remeasure resets it** (finalize regenerates/clears it —
  a fresh run). The GUI's in-session recompute preserves it.

### D.7 GUI ledgers (CI-gated — see project CLAUDE.md)

- **`gui/FEATURES.md`** — add a row for every new affordance (Review
  toggle, module picker, worklist, re-sort button, summary header,
  per-tile remove in review, mark-reviewed, recompute indicator). The
  `features-md-gate` job blocks PRs touching `gui/` without this.
- **`gui/WORKFLOWS.md`** — add a "Curate QC review queue" flow with a
  matching `_capture_qc_review` in
  `scripts/capture_gui_tutorial_screenshots.py` and a tutorial page under
  `docs/source/tutorials/gui/`; `workflows-md-gate` enforces the
  round-trip.

---

## Build order & phasing

`A → {B, C in parallel} → D`. One review gate per phase (per user's
workflow conventions): code review after each phase, a code-simplifier
pass after each phase plus a final pass, and a regression run.

1. **Phase A** — `QualityCheck` contract refactor + 6 checks + schema
   enums; migrate existing checks/tab/tests off `severity`.
2. **Phase B** — `pipeline.qc` (de)serialization + migration; `run_qc`;
   `qc/` artifact; finalize integration + `--no-qc` + reset-on-rerun.
3. **Phase C** — extract `gui/_shared/tiles.py`; refactor colony_view to
   consume it (no behavior change).
4. **Phase D** — Configure|Review toggle, module picker, summary header,
   worklist, detail/curation, per-group recompute, review state; ledgers
   + tutorial + screenshots.

## Testing strategy

- **A:** unit tests per check on `load_synth_yeast_plate()`-derived
  frames — metric values, directional thresholds (both `_HIGHER_IS_BAD`
  polarities), NaN/degenerate guards, `summary()` and `group_members()`.
  Doctests runnable per project convention.
- **B:** round-trip `pipeline.qc` through `to_json`/`from_json`; `run_qc`
  artifact schema + content; finalize integration (recompile/remeasure
  produce `qc/`); `--no-qc`; reset-on-rerun vs in-session preserve.
- **C:** existing colony-view tests pass unchanged; new `_shared/tiles.py`
  unit tests (crop, safe-path, grid keys).
- **D:** callback/integration tests for module switch, worklist sort,
  curate→recompute→in-place update, manual re-sort, mark-reviewed
  persistence, reset-on-rerun.

## Open items to confirm during planning

- Exact home for `run_qc`/migrated `QcRecipe` (`_cli/_cli_qc.py` vs a
  neutral `phenotypic/qc/` package).
- `ImagePipeline` serialization mechanics for a list of `QualityCheck`
  pydantic models (mirror how `post`/`model` serialize).
- Where `FilteredMeasurements` persists its removal store (to align
  review_state placement and the in-session recompute input).
- ICC(2,1) small-n / degenerate-bin guards (under-powered bins → NaN →
  pass, mirroring the SE check).
- Default thresholds (A.2) tuned against real plates.
```
