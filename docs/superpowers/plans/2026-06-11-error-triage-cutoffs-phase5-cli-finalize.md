# Error-Triage Cutoffs — Phase 5: CLI finalize re-emits error deliverables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `finalize_post_master_outputs` (the authoritative, idempotent CLI finalize, reached by both the forward `aggregate_measurements` path and the `--recompile` worker) **re-emit the per-category `deliverables/errors/*.parquet` and `deliverables/error_analysis.{parquet,csv,html}` from the durable labels store, and preserve + re-key that store onto the fresh master** — so a headless run with no open viewer still produces the error deliverables, exactly as the GUI writes them live.

**Architecture:** A new Dash-free CLI helper `reemit_error_deliverables(output_dir, master_df)` that (1) re-keys the durable `qc/curation_labels.parquet` onto the fresh **clean** master via the existing `CurationLabels.load`, (2) re-writes the per-category error parquets + the re-keyed labels parquet via a new Dash-free `CurationLabels.write_error_partitions()` (which deliberately does **not** rewrite the curated `measurements.parquet` mirror — that stays the GUI's live concern), and (3) runs `ErrorCutoffFinder` per category against the all-unlabeled good baseline and writes `error_analysis.{parquet,csv,html}` (HTML via the Phase-4 `analysis/_error_report` renderer). It is keyed off the **same clean `master_measurements.parquet` frame the GUI's `CurationLabels` loads** (`_app.py:181`), so headless output is byte-consistent with live curation.

**Tech Stack:** Python 3.12, polars (master/labels), pandas (engine boundary), `ErrorCutoffFinder` + `_error_report` (Phase 3/4, in `analysis/`), `CurationLabels` (Phase 1, Dash-free), pytest, `uv`.

**Depends on:** Phase 1 (`CurationLabels`, io_constants error paths), Phase 3 (`ErrorCutoffFinder`), Phase 4 (`render_error_analysis_html` / `render_error_analysis_report` in `analysis/_error_report.py`).

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` §9 (CLI/deliverables wiring), §5.3-§5.4 (durable store, re-keying).

---

## Conventions for this plan

- `uv run` for everything. These are CLI/IO + numeric tests — **no Qt, no browser**: `uv run pytest tests/unit/cli/... -v`.
- Commit per task, scoped `git add <paths>`. Worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`.
- ⚠️ **NEVER** `git stash` / `git checkout <ref>` / branch-switch — four unrelated user stashes live here. Sole committer; scoped `git add`.
- Google-style docstrings; doctests where natural.

## Grounding facts (verified against the code — these drive the design)

- `finalize_post_master_outputs(output_dir, master_df: pl.DataFrame, pipeline, metadata_csv=None, no_qc=False) -> pl.DataFrame` (`_cli/_cli_output_manager.py:507`). It receives the **clean, pre-post** master as `master_df`, applies post to a working copy (`post_df`), seeds `measurements.{csv,parquet}` from `post_df`, emits `analysis.*`, **resets `qc/review_state.json`** (`_reset_qc_review_state`, ~L616 — unlink), runs QC, and splits per-feature. Returns `post_df`.
- Both finalize callers — `aggregate_measurements` (`:891`) and the `--recompile` worker `_run_post_master_steps` (`_cli/_cli_recompile_worker.py:389`) — call `finalize_post_master_outputs`, so wiring the re-emit **inside** finalize covers both paths automatically.
- The mid-run chunk writer `_aggregate_chunks_locked` (`_cli/_cli_chunk_writer.py`) deliberately bypasses finalize — **do not touch it** (the project rule).
- **The GUI loads `CurationLabels.load(output_root.root, output_root.master_df)` (`_app.py:181`), where `output_root.master_df` is the CLEAN `master_measurements.parquet`** (`_output_root.py:105/135`). So errors/* and the cutoff frames are built from the clean master in the GUI; finalize uses the **same clean `master_df`** → identical output.
- `CurationLabels` (`gui/results_viewer/_curation_labels.py`) and `gui/__init__` / `gui/results_viewer/__init__` import **no Dash at module load** (verified) — the CLI can import `CurationLabels` cheaply.
- io path helpers all importable from `phenotypic.tools_`: `errors_dir`, `error_category_parquet_path`, `error_analysis_parquet_path/_csv_path/_html_path`, `curation_labels_parquet_path`, `verified_parquet_path`.

## Design decisions settled here

1. **Keyed off the clean `master_df`, not `post_df`.** Re-keying needs every object (post can drop rows, e.g. outlier removal) and the clean master has `Bbox_*` for fingerprinting; it is also the exact frame the GUI's `CurationLabels` uses, so headless == live. The error rows + good/error cutoff frames are therefore clean-master rows (matching the GUI's errors/* and Error tab).

2. **Finalize re-emits errors/* + error_analysis.* + the re-keyed labels store; it does NOT rewrite `measurements.parquet`.** The curated mirror stays the post-applied seed `_seed_measurements` wrote; curation of the mirror remains the GUI's live responsibility (re-derived on next viewer load via the re-keyed labels). This honors spec §9 bullet 3 (the explicit finalize directive: errors/* + error_analysis.* + preserve/re-key), avoids stripping post columns from `measurements.parquet` headlessly, and avoids a measurements-vs-analysis inconsistency. A dedicated Dash-free `CurationLabels.write_error_partitions()` writes errors/* + the labels parquet **without** the mirror.

3. **Guard on `curation_labels.parquet` existence — skip migration headlessly.** `CurationLabels.load`'s legacy-migration path infers removed = `master_keys − measurements_keys`; in finalize `measurements.parquet` is the **post-applied** seed (fewer rows if post drops outliers), so migration would falsely import post-dropped rows as `other` removals. The re-emit therefore **returns early when `curation_labels.parquet` does not exist** — migration stays a GUI-load-only concern. The re-emit only acts when there is a genuine durable store.

4. **Good baseline = all-unlabeled (verified mode is GUI-only).** Finalize just reset `review_state.json`, so it has no reviewed groups; `verified.parquet` is GUI-written and finalize **leaves it untouched** (spec §9). The headless `error_analysis.*` is computed with the all-unlabeled good set.

5. **`error_analysis.*` covers ALL categories** (headless is the authoritative all-category emit; the GUI writes the single focused category transiently — R3 of Phase 4). The parquet/csv carry the leading `category` column (`[category, *RESULT_COLUMNS]`); the HTML is one report with a section per category.

## File structure (Phase 5)

- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py` — add the Dash-free `write_error_partitions()` method.
- Modify: `src/phenotypic/analysis/_error_report.py` — add `render_error_analysis_report(results_by_category)` (multi-category HTML); re-export from `analysis/__init__.py`.
- Create: `src/phenotypic/_cli/_cli_error_outputs.py` — `reemit_error_deliverables(output_dir, master_df)`.
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` — call `reemit_error_deliverables` inside `finalize_post_master_outputs` (after the per-feature split, before `return post_df`).
- Tests:
  - Create: `tests/unit/gui/results_viewer/test_curation_labels_write_error_partitions.py` (or extend the existing curation-labels test) — the new method writes errors/* + labels, not the mirror.
  - Create: `tests/unit/analysis/test_error_report_multi.py` — the multi-category report.
  - Create: `tests/unit/cli/test_cli_error_outputs.py` — the helper (happy path, guard/no-op, idempotency, verified.parquet untouched).
  - Modify: `tests/unit/cli/test_cli_output_manager.py` — extend an `aggregate_measurements`/finalize test to seed a labels store and assert the error deliverables appear.

---

### Task 1: `CurationLabels.write_error_partitions()` (Dash-free, no mirror)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py`
- Test: `tests/unit/gui/results_viewer/test_curation_labels_write_error_partitions.py`

**Why:** Finalize must re-emit the per-category error parquets + the re-keyed labels parquet **without** rewriting the curated `measurements.parquet` mirror (decision 2). The store already has `_write_category_parquets()` + `_write_labels_parquet()`; expose a public, lock-held composition of just those two.

- [ ] **Step 1: Write the failing test.** Build a `CurationLabels` over a small clean master (with `Bbox_CenterRR/CC`), `mark` a few objects across two categories, then:
  - capture the current `measurements.parquet` mtime (write the mirror once via `save()` first, or assert it is absent), call `write_error_partitions()`, assert `errors/<cat>.parquet` exist for each non-empty category and carry `Curation_Category`, `curation_labels.parquet` exists, and **the curated `measurements.parquet` mirror was not (re)written by this call** (mtime unchanged / still absent if never saved).

- [ ] **Step 2: Run → fail** (`AttributeError: write_error_partitions`).
Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels_write_error_partitions.py -v`

- [ ] **Step 3: Implement.** Add to `CurationLabels`:

```python
    def write_error_partitions(self) -> None:
        """Write the per-category error parquets + the (re-keyed) labels parquet.

        Deliberately does **not** rewrite the curated ``measurements.parquet``
        mirror — used by CLI finalize to re-emit the durable error deliverables
        headlessly while leaving the post-applied measurements seed untouched
        (curation of the mirror stays the GUI's live responsibility). Bypasses
        the mirror mtime guard for the same reason. Dash-free.
        """
        with self._lock:
            self._write_category_parquets()
            self._write_labels_parquet()
```

- [ ] **Step 4: Run → pass; gate; commit.** `uv run mypy src/phenotypic/gui/results_viewer/_curation_labels.py`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/unit/gui/results_viewer/test_curation_labels_write_error_partitions.py
git commit -m "feat(viewer): CurationLabels.write_error_partitions — errors/* + labels, no mirror (for CLI finalize)"
```

---

### Task 2: `render_error_analysis_report` — multi-category HTML

**Files:**
- Modify: `src/phenotypic/analysis/_error_report.py`
- Modify: `src/phenotypic/analysis/__init__.py` (re-export)
- Test: `tests/unit/analysis/test_error_report_multi.py`

**Why:** Headless finalize writes one `error_analysis.html` covering **all** categories (decision 5); the Phase-4 `render_error_analysis_html(category, df)` is single-category. Add a multi-category report that reuses the per-category table rendering.

- [ ] **Step 1: Write the failing test.** `render_error_analysis_report({"debris": df1, "background_noise": df2})` returns one non-empty `<html>` document containing **both** category headings and their measurement names; an empty dict yields a valid "no categories" document.

- [ ] **Step 2: Run → fail.** `uv run pytest tests/unit/analysis/test_error_report_multi.py -v`

- [ ] **Step 3: Implement.** Refactor `render_error_analysis_html` so the per-category table+heading is a private `_render_section(category, df) -> str`, have `render_error_analysis_html` wrap one section in a full doc, and add:

```python
def render_error_analysis_report(results_by_category: dict[str, "pd.DataFrame"]) -> str:
    """Render one self-contained HTML report with a section per category.

    Args:
        results_by_category: Mapping ``category token -> ErrorCutoffFinder
            result frame`` (category-free ``RESULT_COLUMNS``).

    Returns:
        A full ``<html>`` document; a "no error categories" body when empty.
    """
```

Re-export `render_error_analysis_report` from `analysis/__init__.py`. Keep it Dash-free + Plotly-free.

- [ ] **Step 4: Run → pass; gate; commit.**
```bash
git add src/phenotypic/analysis/_error_report.py src/phenotypic/analysis/__init__.py tests/unit/analysis/test_error_report_multi.py
git commit -m "feat(analysis): render_error_analysis_report — multi-category HTML for CLI finalize"
```

---

### Task 3: `reemit_error_deliverables` + wire into finalize

**Files:**
- Create: `src/phenotypic/_cli/_cli_error_outputs.py`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
- Test: `tests/unit/cli/test_cli_error_outputs.py`
- Test: `tests/unit/cli/test_cli_output_manager.py` (extend)

**Why:** The orchestration: re-key + preserve the labels store, re-emit errors/*, compute + persist `error_analysis.*` — keyed off the clean master, guarded on a real durable store, called from finalize so both CLI paths get it.

**`_cli_error_outputs.py`:**

```python
"""Headless re-emit of the error-triage deliverables at CLI finalize.

Dash-free. Re-keys the durable ``qc/curation_labels.parquet`` onto the fresh
clean master (the SAME frame the GUI's CurationLabels loads), re-writes the
per-category ``deliverables/errors/*.parquet`` + the re-keyed labels parquet,
and computes ``deliverables/error_analysis.{parquet,csv,html}`` across every
labeled category (all-unlabeled good baseline; verified mode is GUI-only).
``deliverables/measurements.parquet`` and ``deliverables/verified.parquet`` are
left untouched (spec §9 decisions 2 + 4).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from phenotypic.analysis import ErrorCutoffFinder, render_error_analysis_report
from phenotypic.analysis._error_cutoffs import RESULT_COLUMNS
from phenotypic.tools_ import (
    curation_labels_parquet_path,
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
)

logger = logging.getLogger(__name__)

_PERSIST_COLUMNS: tuple[str, ...] = ("category", *RESULT_COLUMNS)


def reemit_error_deliverables(output_dir: Path, master_df: pl.DataFrame) -> None:
    """Re-emit errors/* + error_analysis.* from the durable labels store.

    No-op when there is no durable ``curation_labels.parquet`` (migration is a
    GUI-load concern — see decision 3). Idempotent.

    Args:
        output_dir: The run output directory.
        master_df: The clean (pre-post) master frame being finalized.
    """
    if not curation_labels_parquet_path(output_dir).exists():
        return
    # Local import keeps the GUI package off the hot CLI import path; it is
    # Dash-free (verified) so this stays cheap.
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

    store = CurationLabels.load(output_dir, master_df)  # re-keys onto fresh master
    if not store.labels:
        return
    store.write_error_partitions()  # errors/*.parquet + re-keyed labels parquet (no mirror)
    _write_error_analysis(output_dir, store, master_df)


def _write_error_analysis(output_dir: Path, store, master_df: pl.DataFrame) -> None:
    """Run ErrorCutoffFinder per category; write error_analysis.{parquet,csv,html}."""
    good_pdf = store.filtered_df(master_df).to_pandas()  # all-unlabeled good
    finder = ErrorCutoffFinder()
    frames: list[pd.DataFrame] = []
    reports: dict[str, pd.DataFrame] = {}
    for category in sorted(set(store.labels.values())):
        error_keys = [k for k, c in store.labels.items() if c == category]
        error_pdf = _rows_for_keys(master_df, error_keys)
        res = finder.analyze(good_pdf, error_pdf)
        reports[category] = res
        if not res.empty:
            tagged = res.copy()
            tagged.insert(0, "category", category)
            frames.append(tagged[list(_PERSIST_COLUMNS)])
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames else pd.DataFrame(columns=list(_PERSIST_COLUMNS))
    )
    error_analysis_parquet_path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(combined).write_parquet(error_analysis_parquet_path(output_dir))
    pl.from_pandas(combined).write_csv(error_analysis_csv_path(output_dir))
    error_analysis_html_path(output_dir).write_text(
        render_error_analysis_report(reports), encoding="utf-8"
    )
```

`_rows_for_keys(master_df, keys)` — a small polars helper: build the `(Metadata_ImageFile, Object_Label)` key frame (String/Int64), semi-join `master_df`, `.to_pandas()`. (Mirror `_curation_labels._join_on_keys(..., "semi")`; reuse if cleanly importable, else inline.)

**Wire into finalize:** in `finalize_post_master_outputs`, after the per-feature split block and before `return post_df`, add:

```python
    # Re-emit the durable error-triage deliverables (errors/* + error_analysis.*)
    # from the labels store, keyed off the clean master (GUI-consistent). No-op
    # without a durable curation_labels.parquet. (spec §9)
    try:
        from phenotypic._cli._cli_error_outputs import reemit_error_deliverables

        reemit_error_deliverables(output_dir, master_df)
    except Exception:  # defensive: a curation re-emit must never fail finalize
        logger.warning("Failed to re-emit error-triage deliverables", exc_info=True)
```

(The broad guard matches the existing finalize style — a curation re-emit failure must not abort the run's primary outputs.)

- [ ] **Step 1: Write the failing tests** (`test_cli_error_outputs.py`). Use `tmp_path`:
  - **Happy path:** write a clean `master_measurements.parquet` (≥ `min_error_n` objects, a clearly-separating `Size_Area`), a `qc/curation_labels.parquet` labeling ≥8 objects as `debris` (with `Bbox_CenterRR/CC`), call `reemit_error_deliverables(out, master_df)`; assert `errors/debris.parquet` exists + carries `Curation_Category`; `error_analysis.parquet` exists, first column is `category`, and `Size_Area` ranks top for `debris`; `error_analysis.csv` + `.html` exist; `curation_labels.parquet` still exists (preserved); `verified.parquet` does **not** exist.
  - **Guard/no-op:** no `curation_labels.parquet` → helper returns without creating `errors/` or `error_analysis.*`, and does not touch `measurements.parquet`.
  - **Idempotency:** call twice → same files; a category removed from labels between calls leaves no stale `errors/<oldcat>.parquet`.

- [ ] **Step 2: Run → fail.** `uv run pytest tests/unit/cli/test_cli_error_outputs.py -v`

- [ ] **Step 3: Implement** `_cli_error_outputs.py` + the finalize wiring.

- [ ] **Step 4: Extend the finalize integration test.** In `tests/unit/cli/test_cli_output_manager.py`, in an `aggregate_measurements` (or direct `finalize_post_master_outputs`) test, additionally stage a `qc/curation_labels.parquet` and assert `errors/*.parquet` + `error_analysis.parquet` exist after finalize, and that `review_state.json` was still reset (existing behavior) while `curation_labels.parquet` survived (no wipe).

- [ ] **Step 5: Full gate + commit.**
Run: `uv run pytest tests/unit/cli/test_cli_error_outputs.py tests/unit/cli/test_cli_output_manager.py tests/unit/analysis -v`, `uv run mypy src/phenotypic/_cli/_cli_error_outputs.py src/phenotypic/_cli/_cli_output_manager.py`, `uv run ruff check --fix`.
```bash
git add src/phenotypic/_cli/_cli_error_outputs.py src/phenotypic/_cli/_cli_output_manager.py tests/unit/cli/test_cli_error_outputs.py tests/unit/cli/test_cli_output_manager.py
git commit -m "feat(cli): finalize re-emits errors/* + error_analysis.* from the durable labels store"
```

---

## Self-review (against spec §9)

- finalize (both `aggregate_measurements` + `--recompile`) is the authoritative idempotent writer of `errors/*.parquet` + `error_analysis.{parquet,csv,html}` from the labels store → Task 3 (wired inside finalize; both callers covered). ✅
- Preserves + re-keys the labels store (no wipe) → `CurationLabels.load` re-keys; `write_error_partitions` re-writes the labels parquet; the store is never deleted. ✅
- Mid-run chunk writer untouched → not modified. ✅
- `verified.parquet` GUI-only / untouched by finalize → the helper never writes it; documented. ✅
- `measurements.parquet` stays the post-applied seed (mirror curation remains GUI-live) → decision 2; `write_error_partitions` omits the mirror. ✅
- Headless == GUI output → keyed off the same clean `master_df` the GUI's `CurationLabels` loads. ✅
- Migration not mis-fired headlessly → guard on `curation_labels.parquet` existence (decision 3). ✅

Deferred to Phase 6: WORKFLOWS.md row + tutorial + screenshots + `gui/CLAUDE.md` / io_constants docstring updates. (No `FEATURES.md` row needed — Phase 5 touches no `gui/` user-affordance; the `CurationLabels` method is internal plumbing. Confirm the `features-md-gate` does not trip on the `_curation_labels.py` edit; if it does, add an internal `🧪` row or a Test ref for `write_error_partitions`.)
