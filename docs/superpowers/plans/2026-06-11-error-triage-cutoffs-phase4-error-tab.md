# Error-Triage Cutoffs — Phase 4: Error-analysis tab + verified-good toggle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new "Error analysis" tab to the results viewer that, for the selected error category, runs `ErrorCutoffFinder` against a good baseline and surfaces a ranked cutoff table + a good-vs-error distribution plot with a draggable cutoff line + a live recall/specificity readout + a copy-able filter spec, recomputed live as the user marks objects — with a **good-baseline toggle** (All-unlabeled vs Verified-only) and the durable persistence of `deliverables/error_analysis.{parquet,csv}` and (verified mode) `deliverables/verified.parquet`.

**Architecture:** A new `gui/results_viewer/_error_tab/` package mirroring `_heatmap_tab/`/`_qc_tab/` (layout / pure data layer / figure builder / report renderer / callbacks / ids). The pure data layer (`_data.py`) builds the good/error pandas frames — including the **verified-good** derivation from `ReviewState` + `qc_members` — and is fully unit-testable without Dash. The Phase-3 `ErrorCutoffFinder` is the only stats engine; this phase is all wiring + framing + persistence. No new stats.

**Tech Stack:** Python 3.12, Dash 4 + dash-bootstrap-components, Plotly `graph_objects`, polars (master/curated frames) → pandas (engine boundary), pydantic v2, pytest + pytest-qt/offscreen-free (these are pure-data + Dash-component tests), `uv`.

**Depends on:** Phase 1 (`CurationLabels` store, `ErrorCategory`, io_constants error paths), Phase 3 (`ErrorCutoffFinder`). Feeds Phase 5 (CLI finalize re-emits the same `error_analysis.*` headlessly + reuses the HTML renderer from this phase).

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` — §7 (engine inputs / good-baseline modes / verified derivation), §8 (the tab), §9 (`verified.parquet` is GUI-written, not CLI-emitted), §10 (FEATURES.md rows).

---

## Conventions for this plan

- `uv run` for everything. GUI/component tests need a Qt binding only if they touch napari — these do **not**; they are pure-data + Dash-layout/callback-helper tests. Run the GUI suite with `QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/...` to be safe.
- Commit per task, scoped `git add <paths>`. Worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`.
- ⚠️ **NEVER** `git stash` / `git checkout <ref>` / branch-switch — four unrelated user stashes live in this worktree. Be the sole committer; scope every `git add`.
- Google-style docstrings everywhere. Pure-data doctests are fine with tiny inline frames (the engine + data layer take raw frames).
- **Palette rule (DESIGN.md):** category fills use `category_color(token, custom_index)` (OI data palette); the **good** series uses a neutral data tone (`COLOR_INFO` = OI_SKY). Never a `COLOR_*` chrome hue as a data series. Call `apply_theme(fig)` (from `phenotypic.viz.figures`) before returning any figure.
- **FEATURES.md is CI-gated:** any touch under `src/phenotypic/gui/` requires a `FEATURES.md` row; `✅ shipping` rows need a resolvable `Test ref`. Add rows in Task 7.
- Import shared constants, never re-spell: `category_color` (`gui/_design.py`); `CFG_FILTERED_STATE`, `CFG_OUTPUT_ROOT` (`gui/_config.py`); io path helpers from `phenotypic.tools_`.

## Design decisions settled here (the spec left these to the plan)

1. **Tab placement & module shape.** A 5th `dbc.Tab` ("Error") after Heatmap, `tab_id = ids.TAB_ERROR_ID`. New package `_error_tab/` with the same split as `_heatmap_tab/`: `__init__.py` (exports `build_error_tab_body`, `register_error_callbacks`), `_ids.py`, `_data.py` (pure), `_figure.py` (pure), `_report.py` (pure HTML), `_layout.py`, `_callbacks.py`.

2. **Recompute trigger = the existing curation store.** Per the curation design (decision A) there is **no** `STORE_LABELS` Dash store — the grid reads `filtered_state.labels` server-side under the lock. The Error tab follows suit: its recompute callback takes `Input(ids.STORE_REMOVED_KEYS, "data")` (which the tiles bump on every mark/unmark) **plus** `Input(ids.TABS_ID, "active_tab")` (so navigating back to the tab refreshes verified-mode state that QC marks changed) and reads `filtered_state.labels` + `output_root.master_df` server-side. This is the natural debounce — Dash coalesces rapid store writes. No `dcc.Interval`.

3. **Good/error frames are built in `_data.py` (polars→pandas at the boundary).** The engine is pandas-typed; the viewer's master is polars. `_data.build_good_error_frames(...)` returns `(good_pdf, error_pdf)` already converted.

4. **Verified-good derivation (spec §7, resolved: any-module, good-only).** An object is *verified-good* iff it is **unlabeled** AND its `(image_file, object_label)` belongs to ≥1 QC group marked reviewed in **any** module. Built purely from `qc_summary.parquet` (for per-module `groupby_cols_for`), `qc_members.parquet` (group→members), and `ReviewState` (reviewed encoded keys), reusing `groupby_cols_for` + `decode_group_key`. **Only the good set is restricted; the error set is always every object labeled the target category, regardless of group review state.**

5. **Draggable cutoff = a real editable Plotly horizontal line** (honors the spec literally), with a **numeric input** beside it as an accessible/precise alternative. The figure is a vertical box+strip of good vs error (measurement value on the **y-axis**); the cutoff is a horizontal line shape. `dcc.Graph(config={"editable": True, "edits": {"shapePosition": True}})`; the drag emits `relayoutData` carrying `shapes[0].y0`/`y1`. The good/error value arrays for the focused measurement are stashed in a `dcc.Store` so the readout recomputes recall/specificity at *any* dragged cutoff without re-reading parquet.

6. **Persistence timing.** On each recompute (tab active + store/baseline change) write `deliverables/error_analysis.{parquet,csv}` for the focused category (atomic temp+replace). In **verified** mode also (re)write `deliverables/verified.parquet`. The heavier `error_analysis.html` is written **only** by an explicit "Save analysis report" button (this phase) and by CLI finalize (Phase 5) — both call the shared `_report.render_error_analysis_html`. Never write HTML on every click.

7. **Min-n / "need more labels" state.** When `finder.enough_data(good, error)` is `False` — or in verified mode when the verified-good count is below `finder.min_good_n` — render an explanatory empty-state card (no table/plot, no parquet write) instead of unstable stats.

8. **Copy-filter-spec = JSON + a human query string.** For the focused measurement + current cutoff + direction, emit `{"measurement", "op", "cutoff"}` JSON and a `Size_Area > 123.40` style expression, in a read-only `dcc.Textarea` paired with a `dcc.Clipboard`. (Apply-as-filter is phase 2 / out of scope.)

## Plan-review resolutions (folded in — authoritative)

An independent plan review validated the verified-good derivation as correct against the QC layer and confirmed every referenced API exists. Its findings are resolved as follows; **these override anything above that conflicts**:

- **R1 — HTML/filter-spec renderer lives in `analysis/`, NOT the GUI** (resolves the gui→cli layering inversion). Create `src/phenotypic/analysis/_error_report.py` with `render_error_analysis_html`, `filter_spec_json`, `filter_spec_query`; re-export them from `analysis/__init__.py`. Both the Error tab (this phase) and CLI finalize (Phase 5) import *down* into `analysis/`. Task 4 is retargeted accordingly. It stays Dash-free and Plotly-free (pure pandas/json/string).
- **R2 — Recompute trigger is sound as designed; relabels ARE caught.** The Error tab has **no marking affordance** — every mark/unmark/relabel happens on the Colony/QC/viewer-card tabs. So the `active_tab → TAB_ERROR_ID` Input is the effective trigger: returning to the Error tab always recomputes `_recompute` reading `filtered_state.labels` fresh under the lock, which reflects relabels (category reassignments) even though `STORE_REMOVED_KEYS` is byte-identical for a relabel. Keep the off-tab `PreventUpdate` gate (it prevents ANOVA on every colony-view mark — a real perf win). Triggers: `active_tab`, `STORE_REMOVED_KEYS`, the good-mode toggle, the focused-category chip store, the table row-select. Document this reasoning in the callback. **Do not** add an Output to the colony/QC mark callbacks (avoids destabilizing proven `allow_duplicate` wiring).
- **R3 — `error_analysis.parquet` carries a leading `category` column.** The engine's `RESULT_COLUMNS` is category-free (per-category by design); the **persistence layer** prepends a `category` column so the on-disk shape is identical between the GUI's focused-category write and Phase 5's all-category emit. Contract: the GUI writes **the focused category's** rows (transient, last-viewed-in-session); **Phase 5 finalize owns the authoritative all-category concatenation**. State this in Task 6 so Phase 5 inherits a consistent shape. (No race with the `CurationLabels` mtime guard — that guard watches `measurements.parquet` only; `error_analysis.parquet` is a distinct file.)
- **R4 — `verified.parquet` is written only when the verified-good frame is non-degenerate** (i.e. when content actually renders, not in the empty/insufficient state) and via atomic temp+replace (reuse `_atomic_write_parquet` from `_curation_labels.py`, or a local equivalent). An empty `verified.parquet` must never be written (it would mislead Phase 5 / a later session).
- **R5 — verified-mode guard reuses `enough_data`, special-casing only the message.** Pass the **verified-good** frame as `good` to `finder.enough_data(good, error)` (which already enforces `len(good) >= min_good_n`). Do not re-implement the count guard; the verified branch only swaps the empty-state copy to "review more QC groups to use verified mode."
- **R6 — `default_category` may land on `OTHER` only when OTHER is the sole label class** (the common legacy-migration case). Highest-count non-OTHER wins; fall back to OTHER (focusing *something* beats nothing); `None` when there are no labels at all. Keep the §11 "low-signal" note visible when focused on OTHER.
- **R7 — Reuse, don't duplicate, the QC null-key predicate.** Import `_eq_or_null` from `phenotypic.gui.results_viewer._qc_tab.review._data` for the verified-member filter rather than re-implementing the null/NaN check, so the two never drift. (The inline snippet in Task 2 is illustrative only.)
- **R8 — Drag-parse robustness.** The relayout callback must tolerate `relayoutData` lacking a `shapes[0].y0` key (drags can emit partial/other keys) and use `dash.ctx` to resolve drag-vs-numeric-input; `raise PreventUpdate` when neither a shape-y key nor a numeric value is present. Plotly emits editable-shape drags as **flat dotted-string keys** (`{"shapes[0].y0": …, "shapes[0].y1": …}`), not a nested dict.
- **R9 — Test fixtures must match the QC artifact schema.** `qc_summary.parquet` = `instance_id, class, <groupby…>, metric, status, flag, n_members, n_flagged, rank`; `qc_members.parquet` = `instance_id, <groupby…>, Metadata_ImageFile, Object_Label, member_value`. `groupby_cols_for` recovers group columns *structurally* (non-fixed, non-all-null for the instance) — the fixture's groupby column MUST have ≥1 non-null value for the instance or it is silently dropped and the derivation returns empty. Name `from dash import dash_table, dcc` explicitly in the layout; `dcc.Clipboard(target=<textarea id>)`.

## File structure (Phase 4)

- Create: `src/phenotypic/gui/results_viewer/_error_tab/__init__.py`
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_ids.py`
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_data.py` — pure: verified-good derivation, good/error frame construction, category counts, classification-at-cutoff metrics.
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_figure.py` — pure: box/strip good-vs-error figure + cutoff line shape.
- Create: `src/phenotypic/analysis/_error_report.py` — pure: `render_error_analysis_html`, `filter_spec_json`, `filter_spec_query` (Dash-free + Plotly-free; reused by CLI finalize in Phase 5). **(R1: moved out of the GUI to avoid a gui→cli import inversion.)**
- Modify: `src/phenotypic/analysis/__init__.py` — re-export the three renderers.
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_layout.py` — `build_error_tab_body(output_root, schema) -> Component`.
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_callbacks.py` — `register_error_callbacks(app, output_root, filtered_state) -> None`.
- Modify: `src/phenotypic/gui/results_viewer/_ids.py` — add `TAB_ERROR_ID` (+ `__all__`).
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` — import + mount the 5th tab.
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py` — call `register_error_callbacks`.
- Modify: `src/phenotypic/tools_/_io_constants.py` — add `VERIFIED_PARQUET` + `verified_parquet_path`.
- Modify: `src/phenotypic/tools_/__init__.py` — import + `__all__` re-export of the two new names.
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.css` — error-tab chrome.
- Modify: `src/phenotypic/gui/FEATURES.md` — one row per new affordance.
- Tests:
  - Create: `tests/unit/tools_/test_io_constants_verified.py` (or extend the existing io-constants test) — path helper.
  - Create: `tests/gui/results_viewer/error_tab/test_error_data.py` — verified-good derivation, good/error frames, counts, cutoff metrics.
  - Create: `tests/gui/results_viewer/error_tab/test_error_figure.py` — figure structure + cutoff shape.
  - Create: `tests/gui/results_viewer/error_tab/test_error_report.py` — HTML renderer.
  - Create: `tests/gui/results_viewer/error_tab/test_error_tab_integration.py` — tab body builds; callbacks register on a real `dash.Dash`; recompute callback returns a table/figure for a seeded output root; verified toggle path.

---

### Task 1: `io_constants` — `verified.parquet` path

**Files:**
- Modify: `src/phenotypic/tools_/_io_constants.py`
- Modify: `src/phenotypic/tools_/__init__.py`
- Test: `tests/unit/tools_/test_io_constants_verified.py`

**Why:** The Error tab + Phase 5 both resolve `deliverables/verified.parquet` via a single helper, never a hand-joined name. Mirrors the existing `ERROR_ANALYSIS_*` helpers added in Phase 1.

- [ ] **Step 1: Write the failing test** — `tests/unit/tools_/test_io_constants_verified.py`:

```python
from pathlib import Path

from phenotypic.tools_ import VERIFIED_PARQUET, verified_parquet_path, deliverables_dir


def test_verified_parquet_filename():
    assert VERIFIED_PARQUET == "verified.parquet"


def test_verified_parquet_path_under_deliverables(tmp_path: Path):
    out = tmp_path / "run"
    assert verified_parquet_path(out) == deliverables_dir(out) / "verified.parquet"
```

- [ ] **Step 2: Run → fail** (`ImportError: cannot import name 'VERIFIED_PARQUET'`).
Run: `uv run pytest tests/unit/tools_/test_io_constants_verified.py -v`

- [ ] **Step 3: Implement.** In `_io_constants.py`, next to `ERROR_ANALYSIS_PARQUET` (the deliverable-filename block) add:

```python
#: Filename of the GUI-written verified-good baseline archive (spec §9). It is
#: derived from ``qc/review_state.json`` (which CLI finalize RESETS), so it is
#: GUI-owned and never CLI-emitted; finalize leaves any existing file untouched.
VERIFIED_PARQUET: Final[str] = "verified.parquet"
```

and next to `error_analysis_parquet_path` add:

```python
def verified_parquet_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/verified.parquet`` (GUI-written, §9)."""
    return deliverables_dir(output_dir) / VERIFIED_PARQUET
```

In `tools_/__init__.py` add `VERIFIED_PARQUET` to the filename-constants import group + `verified_parquet_path` to the path-helpers import group, and add both to `__all__` (keep alphabetical within their existing groupings, beside `verified`/`v` neighbours and the `error_*`/`measurements_*` helpers).

- [ ] **Step 4: Run → pass; gate; commit.**
Run: `uv run pytest tests/unit/tools_/test_io_constants_verified.py -v`, `uv run mypy src/phenotypic/tools_/_io_constants.py`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/tools_/_io_constants.py src/phenotypic/tools_/__init__.py tests/unit/tools_/test_io_constants_verified.py
git commit -m "feat(io): verified.parquet path helper (deliverables/verified.parquet)"
```

---

### Task 2: `_error_tab/_data.py` — verified-good derivation + good/error frames + counts

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_error_tab/__init__.py` (stub export now; filled in Task 6)
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_data.py`
- Test: `tests/gui/results_viewer/error_tab/test_error_data.py`

**Why:** This is the load-bearing, Dash-free core: turn the master frame + labels + QC review state into the exact `(good, error)` pandas frames the engine consumes, plus the per-category counts the switcher needs and the at-cutoff classification metrics the drag readout needs. All unit-tested without a browser.

**Key types / signatures** (`_data.py`):

```python
"""Pure data layer for the Error-analysis tab.

Dash-free, side-effect-free except the optional disk *reads* of the QC
review artifacts. Builds the good/error frames ErrorCutoffFinder consumes
(spec §7) in both good-baseline modes, plus category counts and the
at-cutoff classification metrics the draggable readout needs.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from phenotypic.gui.results_viewer._qc_tab.review._data import (
    groupby_cols_for, load_qc_members, load_qc_summary,
)
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState, decode_group_key,
)

if TYPE_CHECKING:
    import pandas as pd
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

KEY_IMAGE_FILE = "Metadata_ImageFile"
KEY_OBJECT_LABEL = "Object_Label"
LabelKey = tuple[str, int]
GoodMode = Literal["all_unlabeled", "verified"]


def category_counts(labels: dict[LabelKey, str]) -> dict[str, int]:
    """Count labeled objects per category token (Counter over labels.values())."""


def default_category(counts: dict[str, int], other_token: str) -> str | None:
    """Highest-count NON-other category; fall back to other; None if no labels."""


def verified_good_keys(
    output_root: "OutputRoot", labeled_keys: set[LabelKey],
) -> set[LabelKey]:
    """Return the unlabeled objects that sit in ≥1 reviewed QC group (any module).

    Reads qc_summary/qc_members/review_state from <root>/qc. For each module
    instance_id with a non-empty reviewed set, recover its groupby columns
    (groupby_cols_for) and, for each reviewed encoded key, decode it and
    filter qc_members to those (image_file, label) members. Union across all
    modules → reviewed-member keys; subtract labeled_keys → verified-good.
    Empty set when the QC artifacts are absent (logged once).
    """


def build_good_error_frames(
    output_root: "OutputRoot",
    labels: dict[LabelKey, str],
    category: str,
    good_mode: GoodMode,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return (good_pdf, error_pdf) for one category in the chosen good mode.

    error  = master rows whose key is labeled `category`.
    good   = all_unlabeled: master rows whose key is NOT in labels;
             verified:      master rows whose key is in verified_good_keys(...).
    Polars filter on the master, then .to_pandas() at the boundary.
    """


def classify_at_cutoff(
    good_values: np.ndarray, error_values: np.ndarray, cutoff: float, direction: str,
) -> dict[str, float]:
    """Recall / specificity / good_flagged for an arbitrary cutoff (drag readout).

    direction ">" flags values strictly above cutoff as error; "<" flags below.
    recall = flagged_error / n_error; specificity = kept_good / n_good;
    good_flagged = count of good on the flagged side. NaN-safe (drop NaN first).
    """
```

**Verified-member filtering detail** (write this helper; null group keys handled like the QC layer's `_eq_or_null`):

```python
def _module_reviewed_member_keys(
    members_df: pl.DataFrame, summary_df: pl.DataFrame,
    instance_id: str, reviewed_encoded: set[str],
) -> set[LabelKey]:
    cols = groupby_cols_for(summary_df, instance_id)
    out: set[LabelKey] = set()
    for encoded in reviewed_encoded:
        key_values = decode_group_key(encoded)
        predicate = pl.col("instance_id") == instance_id
        for col, value in zip(cols, key_values):
            if col not in members_df.columns:
                continue
            if value is None or (isinstance(value, float) and value != value):
                predicate = predicate & pl.col(col).is_null()
            else:
                predicate = predicate & (pl.col(col).cast(pl.String) == str(value))
        sl = members_df.filter(predicate)
        for img, lbl in zip(
            sl.get_column(KEY_IMAGE_FILE).to_list(),
            sl.get_column(KEY_OBJECT_LABEL).to_list(),
        ):
            out.add((str(img), int(lbl)))
    return out
```

- [ ] **Step 1: Write failing tests** — `tests/gui/results_viewer/error_tab/test_error_data.py`. Use a fake `OutputRoot`-like object exposing `.root` (a `tmp_path`) and `.master_df` (a polars frame), and write tiny `qc/qc_summary.parquet`, `qc/qc_members.parquet`, `qc/review_state.json` by hand. Cover:
  - `category_counts` tallies per token; `default_category` picks highest non-other, falls back to other, returns None on empty.
  - `verified_good_keys`: a master of 6 objects across 2 groups; mark group "A" reviewed in one module; members of A that are **unlabeled** are returned, labeled members of A are excluded, members of un-reviewed group "B" are excluded. Multi-column group key (`["plate1","A"]`) round-trips via `decode_group_key`. Absent QC artifacts → empty set.
  - `build_good_error_frames`: all_unlabeled good = master − labeled; verified good = verified_good_keys; error = labeled-as-category rows. Returns pandas; key columns present.
  - `classify_at_cutoff`: a hand-checkable split (good `[1,2,3]`, error `[8,9,10]`, cutoff 5, ">" → recall 1.0, specificity 1.0, good_flagged 0; cutoff 2.5 → specificity 1/3 etc.), NaN-safe.

- [ ] **Step 2: Run → fail** (module/functions absent).
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/results_viewer/error_tab/test_error_data.py -v`

- [ ] **Step 3: Implement `_data.py`** per the signatures above. Stub `_error_tab/__init__.py` with a docstring only (real exports land in Task 6) so the package imports.

- [ ] **Step 4: Run → pass; gate; commit.**
Run the test, `uv run mypy src/phenotypic/gui/results_viewer/_error_tab/_data.py`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/gui/results_viewer/_error_tab/__init__.py src/phenotypic/gui/results_viewer/_error_tab/_data.py tests/gui/results_viewer/error_tab/test_error_data.py
git commit -m "feat(viewer): error-tab data layer — verified-good derivation + good/error frames"
```

---

### Task 3: `_error_tab/_figure.py` — good-vs-error distribution + cutoff line

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_figure.py`
- Test: `tests/gui/results_viewer/error_tab/test_error_figure.py`

**Why:** A pure `go.Figure` builder, isolated for unit testing (trace count, axis title, the cutoff line shape). No callbacks here.

**Signature:**

```python
def build_distribution_figure(
    good_values: np.ndarray,
    error_values: np.ndarray,
    measurement: str,
    category: str,
    cutoff: float,
    custom_index: int = 0,
) -> go.Figure:
    """Vertical box+strip of good vs error for one measurement, value on y.

    - good trace: go.Box(..., name="Good kept", marker_color=COLOR_INFO) + jittered points.
    - error trace: go.Box(..., name=category, marker_color=category_color(category, custom_index)).
    - cutoff: a horizontal line *shape* (editable) at y=cutoff spanning the x range,
      added via fig.add_shape(type="line", ..., editable=True) so the drag emits relayoutData.
    - apply_theme(fig); y-axis title = measurement; x categorical; legend top.
    """
```

- [ ] **Step 1: Write failing tests** — assert: 2 box traces present; y-axis title == measurement; exactly one line shape at `y0==y1==cutoff`; the error trace color == `category_color(category)`. (`fig.layout.shapes`, `fig.data`.)

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement** with `go`, `apply_theme`, `category_color`, `COLOR_INFO`. Keep the shape `editable=True` and store its index 0 (the callback reads `relayoutData["shapes[0].y0"]`).

- [ ] **Step 4: Run → pass; gate; commit.**
```bash
git add src/phenotypic/gui/results_viewer/_error_tab/_figure.py tests/gui/results_viewer/error_tab/test_error_figure.py
git commit -m "feat(viewer): error-tab distribution figure with editable cutoff line"
```

---

### Task 4: `analysis/_error_report.py` — shared HTML renderer + filter spec

**Files:**
- Create: `src/phenotypic/analysis/_error_report.py`
- Modify: `src/phenotypic/analysis/__init__.py` (re-export `render_error_analysis_html`, `filter_spec_json`, `filter_spec_query`)
- Test: `tests/unit/analysis/test_error_report.py`

**Why (R1):** The "Save analysis report" button (this phase) and CLI finalize (Phase 5) must render the **same** HTML from a result frame. Placing it in `analysis/` (beside `_error_cutoffs.py`) keeps both the GUI tab and the CLI importing *down* into `analysis/` — no `gui→cli` layering inversion. Pure, Dash-free **and Plotly-free** (just pandas/json/string), so the headless CLI never imports the Dash stack to write a table.

**Signatures:**

```python
def render_error_analysis_html(category: str, result_df: pd.DataFrame) -> str:
    """Self-contained HTML: a heading + the ranked result table.

    Pure string build (pandas `.to_html` for the table + a small inline <style>).
    No Plotly/Dash import required; safe for the CLI to call at finalize.
    """

def filter_spec_json(measurement: str, direction: str, cutoff: float) -> str:
    '''`{"measurement": "...", "op": ">", "cutoff": 123.4}` (indent=2).'''

def filter_spec_query(measurement: str, direction: str, cutoff: float) -> str:
    """Human expression, e.g. `Size_Area > 123.40`."""
```

- [ ] **Step 1: Write failing tests** (`tests/unit/analysis/test_error_report.py`) — HTML contains the category, the measurement names, and is non-empty `<html>`; `filter_spec_json` round-trips via `json.loads` to the right dict; `filter_spec_query` formats `f"{measurement} {direction} {cutoff:.2f}"`. Import from the public `phenotypic.analysis`.

- [ ] **Step 2: Run → fail.** `uv run pytest tests/unit/analysis/test_error_report.py -v`

- [ ] **Step 3: Implement** `analysis/_error_report.py` (pure pandas/json/string; no Dash, no Plotly) + re-export the three names from `analysis/__init__.py`.

- [ ] **Step 4: Run → pass; gate; commit.** `uv run mypy src/phenotypic/analysis/_error_report.py`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/analysis/_error_report.py src/phenotypic/analysis/__init__.py tests/unit/analysis/test_error_report.py
git commit -m "feat(analysis): error-analysis HTML report + filter-spec renderers (shared with CLI)"
```

---

### Task 5: `_error_tab/_ids.py` + `_layout.py` — the tab body

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_ids.py`
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_layout.py`
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.css`

**Why:** Assemble the static layout — category-chip container, good-baseline toggle, verified-count badge, ranked `dash_table.DataTable`, the `dcc.Graph` (editable), cutoff `dcc.Input` (number), recall/specificity readout, copy-spec `dcc.Textarea` + `dcc.Clipboard`, "Save analysis report" button + toast, stale banner, "need more labels" empty-state — plus the per-tab `dcc.Store`s (focused-measurement value arrays for the drag readout; focused category; good_mode).

**`_ids.py`** — define + `__all__`: `ERROR_CATEGORY_CHIPS_ID`, `ERROR_GOOD_MODE_TOGGLE_ID`, `ERROR_VERIFIED_COUNT_ID`, `ERROR_TABLE_ID`, `ERROR_FIGURE_ID`, `ERROR_CUTOFF_INPUT_ID`, `ERROR_READOUT_ID`, `ERROR_FILTER_SPEC_ID`, `ERROR_CLIPBOARD_ID`, `ERROR_SAVE_REPORT_BTN_ID`, `ERROR_SAVE_TOAST_ID`, `ERROR_STALE_BANNER_ID`, `ERROR_EMPTY_STATE_ID`, `ERROR_CONTENT_ID`, `STORE_ERROR_FOCUS_ID` (`{category, measurement, direction, cutoff, good_values, error_values}`), `STORE_ERROR_GOOD_MODE_ID`.

**`build_error_tab_body(output_root, schema) -> Component`** — ships the containers empty; the recompute callback (Task 6) fills chips/table/figure/badge. Include both `dcc.Store`s. Use `dbc` cards + the design tokens. The good-baseline toggle is a `dbc.RadioItems`/segmented control with options `All unlabeled` / `Verified only`.

- [ ] **Step 1:** Implement `_ids.py` (+ tiny test that all ids are unique strings, optional — fold into Task 6 integration test).
- [ ] **Step 2:** Implement `build_error_tab_body`. Keep it import-light; no data reads at build time.
- [ ] **Step 3:** Add error-tab CSS (chip row, readout pills, empty-state card) using `var(--*)` tokens only — no `:root` redefinition.
- [ ] **Step 4: Gate; commit.** `uv run mypy` the two files, `uv run ruff check --fix`.
```bash
git add src/phenotypic/gui/results_viewer/_error_tab/_ids.py src/phenotypic/gui/results_viewer/_error_tab/_layout.py src/phenotypic/gui/results_viewer/_assets/results_viewer.css
git commit -m "feat(viewer): error-analysis tab layout + ids + css"
```

---

### Task 6: `_error_tab/_callbacks.py` + package wiring + tab registration

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_error_tab/_callbacks.py`
- Modify: `src/phenotypic/gui/results_viewer/_error_tab/__init__.py` (export `build_error_tab_body`, `register_error_callbacks`)
- Modify: `src/phenotypic/gui/results_viewer/_ids.py` (`TAB_ERROR_ID` + `__all__`)
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (import + 5th `dbc.Tab`)
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py` (call `register_error_callbacks`)
- Test: `tests/gui/results_viewer/error_tab/test_error_tab_integration.py`

**Why:** Wire reactivity. Callbacks read `filtered_state` + `output_root` server-side, run `ErrorCutoffFinder`, render chips/table/figure/readout/spec, persist deliverables, and handle the toggle / row-select / drag / save-report.

**`register_error_callbacks(app, output_root, filtered_state)`** registers:

1. **Recompute** — `Output`: chips, verified-count badge, table data+columns, figure, focus store, empty-state, stale banner, content visibility. `Input`: `STORE_REMOVED_KEYS.data`, `TABS_ID.active_tab`, `ERROR_GOOD_MODE_TOGGLE.value`, `STORE_ERROR_GOOD_MODE.data`; `State`: focused category (chip). Body:
   - If `active_tab != TAB_ERROR_ID`: `raise PreventUpdate` (don't compute off-tab — avoids ANOVA on every mark while the user is in Colony/QC; the tab refreshes on activation, satisfying §8's live intent without per-mark cost).
   - Read `labels = dict(filtered_state.labels)` under `filtered_state._lock`; `counts = category_counts(labels)`; build chips (selected = focused category or `default_category`).
   - `good_pdf, error_pdf = build_good_error_frames(output_root, labels, category, good_mode)`.
   - `finder = ErrorCutoffFinder()`; if `not finder.enough_data(...)` (or verified count `< finder.min_good_n` in verified mode): show the empty-state, hide content, **skip** the parquet write. Else `res = finder.analyze(good_pdf, error_pdf)`.
   - Focused measurement = top row (`res.iloc[0]`); build figure from that measurement's good/error arrays; stash arrays + direction + cutoff in `STORE_ERROR_FOCUS`.
   - **Persist (R3):** prepend a `category` column to `res` and write `error_analysis.{parquet,csv}` atomically for the focused category (transient, last-viewed-in-session; Phase 5 finalize owns the authoritative all-category concat with the same `[category, *RESULT_COLUMNS]` shape). **(R4)** in verified mode, (re)write `verified.parquet` from the good frame **only** in this non-degenerate branch, via atomic temp+replace.
   - Stale banner from `filtered_state.rekey_report` + `filtered_state.stale`.

2. **Category select** — chip click (`ALL` pattern) → write focused category into the recompute path (a small store or a `State` echo); triggers recompute.

3. **Row select** — `Input ERROR_TABLE.active_cell`/`selected_rows` → set focused measurement → rebuild figure + reset cutoff input/line to that measurement's suggested cutoff + refresh readout + filter-spec.

4. **Cutoff change (drag OR numeric)** — `Input ERROR_FIGURE.relayoutData` (parse `shapes[0].y0`) **and** `Input ERROR_CUTOFF_INPUT.value`; `State STORE_ERROR_FOCUS`. Recompute `classify_at_cutoff(good_values, error_values, cutoff, direction)` → update readout pills + the numeric input (keep them in sync) + `filter_spec_json/query` text. Use `dash.ctx` to resolve which input fired. Do **not** rewrite parquet on drag (the suggested cutoff in the table is the persisted one; the drag is exploratory).

5. **Save report** — `Input ERROR_SAVE_REPORT_BTN.n_clicks` → `from phenotypic.analysis import render_error_analysis_html` → `render_error_analysis_html(category, res)` → write `error_analysis_html_path(root)` → show the toast. (Recompute `res` for the focused category inside this callback or read it back from the just-written parquet.)

**Integration test** (`test_error_tab_integration.py`):
- `build_error_tab_body(output_root, schema)` returns a Component containing the table + figure ids.
- On a real `dash.Dash()`, `register_error_callbacks(app, output_root, filtered_state)` registers without raising; assert key callbacks exist in `app.callback_map`.
- Drive the recompute helper directly (extract the body into a module-level `_recompute(...)` so it's unit-testable without `_dash-update-component`, per the memory note "extract callback bodies into module-level helpers"): seed an `OutputRoot` whose master has a clearly-separating `Size_Area`, label ≥`min_error_n` objects as `debris`, assert the returned table's top row is `Size_Area` and that `error_analysis.parquet` was written. Verified-mode branch: seed `qc/*` + a reviewed group, flip mode, assert `verified.parquet` written and the good set shrank.

- [ ] **Step 1: Write the failing integration test** (above).
- [ ] **Step 2: Run → fail.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/results_viewer/error_tab/test_error_tab_integration.py -v`
- [ ] **Step 3: Implement** `_callbacks.py` (extract `_recompute` + `_render_chips` + `_persist` module-level helpers), fill `__init__.py` exports, add `TAB_ERROR_ID`, mount the tab in `_layout.py` (5th `dbc.Tab(error_tab_body, label="Error", tab_id=ids.TAB_ERROR_ID)`; build the body next to `heatmap_tab_body`), and call `register_error_callbacks(app, output_root, filtered_state)` in the orchestrator `_callbacks.register_callbacks` (it already binds `filtered_state` from `CFG_FILTERED_STATE`).
- [ ] **Step 4: Run → pass; full gate; commit.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/results_viewer/error_tab/ -v`, `uv run mypy src/phenotypic/gui/results_viewer/_error_tab`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/gui/results_viewer/_error_tab/_callbacks.py src/phenotypic/gui/results_viewer/_error_tab/__init__.py src/phenotypic/gui/results_viewer/_ids.py src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_callbacks.py tests/gui/results_viewer/error_tab/test_error_tab_integration.py
git commit -m "feat(viewer): wire Error-analysis tab — recompute, toggle, drag readout, save report"
```

---

### Task 7: FEATURES.md rows + final gate

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`

**Why:** CI gate. Every new affordance gets a row; `✅ shipping` rows need a resolvable `Test ref`.

- [ ] **Step 1:** Add rows (columns: `Feature | Element | Expected behaviour | Status | Test layer | Test ref`) for: Error-analysis tab; category switcher chips; good-baseline toggle (All-unlabeled / Verified-only); verified-good count badge; ranked cutoff table; distribution figure; draggable cutoff line; numeric cutoff input; recall/specificity readout; copy-filter-spec (textarea + clipboard); save-analysis-report button; "need more labels" empty-state; stale banner. Point `Test ref` at the new tests (`tests/gui/results_viewer/error_tab/...::...`). Use `🧪 internal`/`✅ shipping` honestly; for any affordance only exercised live, mark `manual` test layer + `n/a (manual)` and lean on Phase-6 e2e.

- [ ] **Step 2: Whole-feature gate.**
Run: `QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/results_viewer/error_tab tests/unit/analysis/test_error_cutoffs.py tests/unit/tools_/test_io_constants_verified.py -v`, `uv run mypy src/phenotypic/gui/results_viewer/_error_tab src/phenotypic/tools_/_io_constants.py`, `uv run ruff check`.
Also run the FEATURES gate locally if available: `uv run python scripts/check_workflows_md.py` is for WORKFLOWS (Phase 6); the FEATURES gate is the pre-commit `features-md` hook — at minimum confirm every `✅ shipping` row's `Test ref` resolves.

- [ ] **Step 3: Commit.**
```bash
git add src/phenotypic/gui/FEATURES.md
git commit -m "docs(gui): FEATURES.md rows for the Error-analysis tab"
```

---

## Self-review (against spec §7–§9)

- New "Error analysis" tab alongside Colony/QC/Heatmap → Task 5/6. ✅
- Category switcher with live counts, default highest-count non-OTHER → `category_counts`/`default_category` (Task 2) + chips (Task 5/6). ✅
- Good-baseline toggle (All-unlabeled default / Verified-only) recomputes → toggle + recompute (Task 6); verified derivation any-module, good-only (Task 2). ✅
- Ranked table (measurement, AUC, cutoff, BH-p, …) → `ErrorCutoffFinder.analyze` rendered to DataTable (Task 6). ✅
- Distribution box/violin + draggable cutoff + live recall/specificity readout → `_figure` editable shape + `classify_at_cutoff` + drag/numeric callback (Tasks 3/6). ✅
- Copy filter spec (JSON + query) → `_report` builders + textarea/clipboard (Tasks 4/5/6). ✅
- Reactivity debounced on label change → recompute on `STORE_REMOVED_KEYS` + tab-active gate (Task 6). ✅
- Persistence: `error_analysis.{parquet,csv}` live; `error_analysis.html` only on explicit save (+ Phase 5 finalize) via shared renderer; `verified.parquet` GUI-written debounced in verified mode → Tasks 4/6 + Task 1 path. ✅
- Min-n / "need more labels" state → empty-state (Task 6). ✅
- Stale banner from rekey report → Task 6. ✅
- `verified.parquet` is GUI-only (not CLI-emitted) → Task 1 docstring; Phase 5 leaves it untouched. ✅

Deferred to later phases: CLI finalize re-emitting `errors/*` + `error_analysis.*` headlessly and reusing `render_error_analysis_html` (Phase 5); WORKFLOWS row + tutorial + screenshots + CLAUDE/io docstrings (Phase 6); one-click apply-as-filter (out of scope, phase 2).
