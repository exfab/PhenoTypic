# Error-Triage Cutoffs — Phase 6: Docs / ledgers / screenshots — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the end-to-end "Error analysis" tutorial: a `WORKFLOWS.md` row + a `_capture_error_analysis` function (with seeded curation labels so the tab renders populated) + a walkthrough page under `docs/source/tutorials/gui/`, the regenerated full GUI screenshot set, and the `gui/CLAUDE.md` + `tools_/_io_constants.py` docstring updates — closing the CI-gated round-trip for the feature.

**Architecture:** Mirror the existing `qc_review` / `heatmap_exploration` tutorial pattern. Because the Error tab only renders a ranked table when ≥`min_error_n` objects are labeled, the capture script seeds a synthetic `qc/curation_labels.parquet` (labeling the most-extreme-`Size_Area` synthetic objects as an error category) before the **standalone** viewer captures the loaded shots; the hub-mounted viewer captures the empty state.

**Tech Stack:** Markdown docs, `scripts/capture_gui_tutorial_screenshots.py` (Playwright/Chromium), `scripts/check_workflows_md.py` round-trip gate, `CurationLabels` for seeding, `uv`.

**Depends on:** Phases 1–5 (the tab + the CLI finalize). FEATURES.md rows already landed in Phase 4/5, so this phase adds **no** FEATURES rows.

**Spec:** `docs/superpowers/specs/2026-06-10-error-category-triage-cutoff-finder-design.md` §10 (docs/ledgers).

---

## Conventions for this plan

- `uv run` for everything. The capture needs Chromium: `uv run playwright install chromium` if absent.
- Commit per task, scoped `git add`. Worktree: `/Users/alex/Projects/PhenoTypic/.claude/worktrees/error-triage-cutoffs`.
- ⚠️ **NEVER** `git stash` / `git checkout <ref>` / branch-switch — four unrelated user stashes live here. Sole committer; scoped `git add`.
- **Commit the FULL refreshed PNG set** — the capture regenerates every workflow's PNGs and cross-platform font rendering shifts unrelated ones by a few bytes; per `CLAUDE.md`, commit them all, do not cherry-pick or `git checkout --` the collateral.

## Grounding facts (verified)

- The capture script (`scripts/capture_gui_tutorial_screenshots.py`) builds a synthetic 3-plate dataset, runs the CLI once (`run_cli_once`) into `docs/source/_static/gui_images/_dataset/results/` (`OUTPUT_DIR`), boots the hub for empty-state shots (`capture_workflow_screenshots`), then boots a **standalone** results viewer with a real `output_root` for **loaded** shots (`capture_standalone_viewer_screenshots`).
- Results-viewer captures click a tab via `page.locator('a[role="tab"]:has-text("<Label>")')` and save via `_save(page, "<workflow>", "NN_step.png")`. Loaded shots live in `_<id>_loaded_shots(page)` helpers dispatched from `capture_standalone_viewer_screenshots`.
- **No existing capture seeds curation/QC state** — they rely on real CLI output. The Error tab needs labels, so this phase ADDS a seeding step.
- `check_workflows_md.py` enforces, per `✅ shipping` row: the `_capture_<id>` is **defined AND dispatched**, the folder `docs/source/_static/gui_images/<id>/` has ≥1 PNG, and the tutorial page exists. Orphan capture fns (defined+dispatched but unreferenced) also fail.
- `WORKFLOWS.md` table ends at `tune_copilot` (#16); the new row is `error_analysis` → `gui/17_error_analysis.md`. `docs/source/tutorials/gui/index.md` carries the table + a `:hidden:` toctree that must both gain the new page.
- `gui/CLAUDE.md` has an "Error-category triage (curation)" section (~L341-369); add the Error-tab + CLI-finalize notes right after it (before "Builder preview cache").
- The Error tab's component ids (from `_error_tab/_ids.py`): tab label "Error"; `#error-category-chips`, `#error-good-mode-toggle`, `#error-verified-count`, `#error-table`, `#error-figure`, `#error-empty-state`. The recompute only fires when the tab is active (off-tab `PreventUpdate`) — the capture must click the tab and wait for `#error-table` rows.

## Design decisions settled here

1. **Seed synthetic labels for a meaningful screenshot.** After the CLI run, write `qc/curation_labels.parquet` into `OUTPUT_DIR` labeling the ~12 objects with the smallest `Size_Area` (likely debris/noise) as `background_noise`, via `CurationLabels.load(OUTPUT_DIR, master).mark_many(keys, "background_noise")`. This gives the finder real separation → a populated ranked table + distribution plot. Idempotent (re-mark is a no-op set-write). Pick a count ≥ `ErrorCutoffFinder().min_error_n` (8) — use 12 for headroom.
2. **Three loaded screenshots:** (a) the ranked cutoff table + category chips, (b) the distribution plot with the cutoff line + recall/specificity readout, (c) the good-baseline toggle (All-unlabeled / Verified-only) with the verified-good count. Plus one empty-state shot from the hub. (The verified count may read 0 without seeded review state — that is fine; the toggle + the "review more QC groups" message are the point. Optionally also seed one reviewed QC group so the verified count is non-zero — nice-to-have, not required.)
3. **No new FEATURES.md rows** (already added in Phase 4/5). This phase touches `gui/` only via the capture script + CLAUDE.md; the `features-md-gate` triggers on `src/phenotypic/gui/` source changes — `WORKFLOWS.md` and `scripts/` are outside that gate, and `gui/CLAUDE.md` is documentation. If the gate trips on the `gui/CLAUDE.md` edit, the existing FEATURES rows already satisfy it (no new affordance).

## File structure (Phase 6)

- Modify: `scripts/capture_gui_tutorial_screenshots.py` — `_seed_error_triage_labels()`, `_capture_error_analysis(context, base_url)`, `_error_analysis_loaded_shots(page)`, + both dispatch sites + the seeding call.
- Modify: `src/phenotypic/gui/WORKFLOWS.md` — the `error_analysis` row.
- Create: `docs/source/tutorials/gui/17_error_analysis.md` — the walkthrough.
- Modify: `docs/source/tutorials/gui/index.md` — table row + toctree entry.
- Create (by running the capture): `docs/source/_static/gui_images/error_analysis/*.png` + the regenerated full set.
- Modify: `src/phenotypic/gui/CLAUDE.md` — Error-tab + CLI-finalize notes.
- Modify: `src/phenotypic/tools_/_io_constants.py` — docstring note for the error/verified artifacts (the `VERIFIED_PARQUET` / `ERROR_ANALYSIS_*` block).

---

### Task 1: capture function + label seeding + WORKFLOWS row

**Files:**
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Modify: `src/phenotypic/gui/WORKFLOWS.md`

- [ ] **Step 1: Add the seeding helper.** Near `run_cli_once`, add:

```python
def _seed_error_triage_labels() -> None:
    """Label the smallest-Size_Area synthetic objects so the Error tab renders.

    The Error-analysis tab only ranks measurements once a category has
    >= ErrorCutoffFinder().min_error_n labels; the synthetic run carries none,
    so seed ~12 'background_noise' labels (the smallest colonies — plausible
    debris) for a populated, meaningful screenshot. Idempotent.
    """
    import polars as pl
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.tools_ import master_measurements_parquet_path

    master_path = master_measurements_parquet_path(OUTPUT_DIR)
    if not master_path.is_file():
        return
    master = pl.read_parquet(master_path)
    smallest = (
        master.sort("Size_Area").head(12)
        .select(["Metadata_ImageFile", "Object_Label"])
    )
    keys = [(str(f), int(l)) for f, l in smallest.iter_rows()]
    store = CurationLabels.load(OUTPUT_DIR, master)
    store.mark_many(keys, "background_noise")
```

Call it in `main()` right after `run_cli_once()` succeeds (and after any `--skip-cli` reuse), so `OUTPUT_DIR/qc/curation_labels.parquet` + `deliverables/errors/background_noise.parquet` exist before the viewer boots.

- [ ] **Step 2: Add the empty-state capture** (dispatched from `capture_workflow_screenshots`, mirroring `_capture_heatmap_exploration`):

```python
def _capture_error_analysis(context, base_url: str) -> None:
    """Capture the Error-analysis walkthrough (empty-state via the hub viewer)."""
    print("[shot] workflow=error_analysis")
    page = _new_page(context, base_url, "/results/")
    page.wait_for_timeout(800)
    tab = page.locator('a[role="tab"]:has-text("Error")').first
    if tab.count() > 0:
        try:
            tab.click(timeout=2000)
            page.wait_for_timeout(600)
        except Exception:
            pass
    _save(page, "error_analysis", "01_empty_state.png")
    page.close()
```

- [ ] **Step 3: Add the loaded-state helper** (dispatched from `capture_standalone_viewer_screenshots`, mirroring `_qc_review_loaded_shots`). Click the Error tab, wait for `#error-table` to have rows, screenshot the ranked table; then screenshot the distribution plot region (`#error-figure`); then open/toggle the good-baseline control and screenshot it:

```python
def _error_analysis_loaded_shots(page) -> None:
    """Capture the populated Error-analysis tab in the standalone viewer."""
    tab = page.locator('a[role="tab"]:has-text("Error")').first
    if tab.count() == 0:
        print("[shot]   error_analysis: Error tab not found — loaded captures skipped")
        return
    try:
        tab.click(timeout=3000)
        page.wait_for_selector("#error-table .dash-cell", timeout=6000)
        page.wait_for_timeout(800)
    except Exception:
        pass
    _save(page, "error_analysis", "02_ranked_table.png")
    if page.locator("#error-figure").count() > 0:
        page.wait_for_timeout(400)
        _save(page, "error_analysis", "03_distribution_cutoff.png")
    # Good-baseline toggle (All unlabeled / Verified only) + verified count.
    toggle = page.locator("#error-good-mode-toggle").first
    if toggle.count() > 0:
        try:
            toggle.scroll_into_view_if_needed(timeout=1500)
            page.wait_for_timeout(300)
        except Exception:
            pass
        _save(page, "error_analysis", "04_good_baseline_toggle.png")
```

(Tune the exact selectors to the rendered DOM during the capture run; `#error-table .dash-cell` is the `dash_table` cell class. If the Error tab's content sits inside `#error-content`, wait for that instead.)

- [ ] **Step 4: Register dispatch.** Add `_capture_error_analysis(context, base_url)` to the `capture_workflow_screenshots` dispatch list (after `_capture_heatmap_exploration` or near the other viewer captures), and `_error_analysis_loaded_shots(page)` to the standalone-viewer dispatch (beside `_heatmap_exploration_loaded_shots(page)`).

- [ ] **Step 5: Add the WORKFLOWS.md row** (status `🔭 planned` for now; flipped to `✅ shipping` in Task 4 once PNGs + page exist):

```markdown
| error_analysis | Error analysis | `_capture_error_analysis` | `gui/17_error_analysis.md` | 🔭 planned |
```

(Match the exact column set of the existing table.)

- [ ] **Step 6: Commit** (the script + WORKFLOWS row; PNGs land in Task 2).
```bash
git add scripts/capture_gui_tutorial_screenshots.py src/phenotypic/gui/WORKFLOWS.md
git commit -m "feat(docs): error_analysis capture function + label seeding + WORKFLOWS row"
```

---

### Task 2: regenerate + commit the full screenshot set

**Files:**
- Create/modify: `docs/source/_static/gui_images/**` (full set)

- [ ] **Step 1: Ensure Chromium.** `uv run playwright install chromium` (no-op if present).
- [ ] **Step 2: Regenerate.** `uv run python scripts/capture_gui_tutorial_screenshots.py` (add `--force` if the `_dataset` is stale). Watch the log for `[shot] workflow=error_analysis` and the `error_analysis/0{1..4}_*.png` saves.
- [ ] **Step 3: Verify** the new shots exist and are non-empty: `ls -la docs/source/_static/gui_images/error_analysis/` shows `01_empty_state.png` … `04_good_baseline_toggle.png`. Open `02_ranked_table.png` to confirm the table actually rendered (not an empty-state) — if empty, the seeding/selector needs fixing (re-run Task 1 Step 1/3) before proceeding.
- [ ] **Step 4: Commit the FULL set** (all refreshed PNGs, including unrelated font-noise churn — do NOT cherry-pick):
```bash
git add docs/source/_static/gui_images
git commit -m "docs(gui): regenerate tutorial screenshots incl. error_analysis"
```

---

### Task 3: tutorial page + index

**Files:**
- Create: `docs/source/tutorials/gui/17_error_analysis.md`
- Modify: `docs/source/tutorials/gui/index.md`

- [ ] **Step 1: Write `17_error_analysis.md`** mirroring `15_qc_review.md`'s structure: title + intro (what the tab does — rank measurements that separate a chosen error category from the good baseline, read off a cutoff), **Prerequisites** (a finished CLI run with ≥8 objects labeled in some error category — point at the QC curation loop / colony triage tutorials for how to label), **Walkthrough** embedding the four screenshots with prose:
  - `![…](../../_static/gui_images/error_analysis/01_empty_state.png)` — empty until an output with labels is bound.
  - `![…](../../_static/gui_images/error_analysis/02_ranked_table.png)` — category chips + ranked cutoff table (AUC, suggested cutoff, recall/specificity, BH-p).
  - `![…](../../_static/gui_images/error_analysis/03_distribution_cutoff.png)` — good-vs-error box/strip with the draggable cutoff line + live recall/specificity readout; mention the copy-filter-spec.
  - `![…](../../_static/gui_images/error_analysis/04_good_baseline_toggle.png)` — All-unlabeled vs Verified-only baseline + the verified-good count; explain when to use verified mode.
  - **Common gotchas:** the "need more labels" / "review more QC groups" empty states; `error_analysis.{parquet,csv,html}` + `errors/*.parquet` are written to `deliverables/` and re-emitted by CLI finalize; `verified.parquet` is GUI-only; the cutoff is single-measurement (multi-measure rules are out of scope).
  - **Where to next:** QC review, QC curation loop, View results.

- [ ] **Step 2: Index** — add the table row and the toctree entry `17_error_analysis` to `docs/source/tutorials/gui/index.md` (keep numeric order).

- [ ] **Step 3: Commit.**
```bash
git add docs/source/tutorials/gui/17_error_analysis.md docs/source/tutorials/gui/index.md
git commit -m "docs(gui): Error-analysis tutorial walkthrough page"
```

---

### Task 4: flip status + round-trip gate

**Files:**
- Modify: `src/phenotypic/gui/WORKFLOWS.md`

- [ ] **Step 1:** Flip the `error_analysis` row's status `🔭 planned` → `✅ shipping`.
- [ ] **Step 2: Run the gate.** `uv run python scripts/check_workflows_md.py` → expect `WORKFLOWS.md OK -- 17 workflows, …` with no errors (capture fn defined+dispatched, PNGs present, page exists).
- [ ] **Step 3: Commit.**
```bash
git add src/phenotypic/gui/WORKFLOWS.md
git commit -m "docs(gui): mark error_analysis workflow shipping"
```

---

### Task 5: code-doc updates (CLAUDE.md + io_constants docstrings)

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`
- Modify: `src/phenotypic/tools_/_io_constants.py`

- [ ] **Step 1: `gui/CLAUDE.md`** — after the "Error-category triage (curation)" section, add an "Error-analysis tab" subsection: the new `_error_tab/` package (data layer / figure / report / callbacks); the good-baseline toggle (All-unlabeled vs Verified-only, the verified-good derivation from `review_state.json` + `qc_members`); that it reads `filtered_state.labels` server-side under the lock + recomputes on tab activation (off-tab `PreventUpdate`); that it persists `deliverables/error_analysis.{parquet,csv}` live + `verified.parquet` in verified mode, and the HTML only on explicit save / CLI finalize; that CLI finalize re-emits `errors/*` + `error_analysis.*` from the durable labels store via `reemit_error_deliverables` (keyed off the clean master, GUI-consistent), while `verified.parquet` stays GUI-only. Resolve paths via `phenotypic.tools_` helpers.

- [ ] **Step 2: `tools_/_io_constants.py`** — extend the docstrings on the error-artifact block to note: `verified.parquet` is GUI-written/CLI-untouched; `error_analysis.*` is written live by the GUI (focused category) and authoritatively re-emitted (all categories, `[category, *RESULT_COLUMNS]`) by CLI finalize; `errors/<category>.parquet` is dual-owned (GUI live + CLI finalize).

- [ ] **Step 3: Gate + commit.** `uv run ruff check` (docstring-only change, but keep it clean); confirm the `features-md-gate` is satisfied (no new affordance). 
```bash
git add src/phenotypic/gui/CLAUDE.md src/phenotypic/tools_/_io_constants.py
git commit -m "docs: Error-analysis tab + finalize notes in gui/CLAUDE.md + io_constants"
```

---

## Self-review (against spec §10)

- `WORKFLOWS.md` row + matching `_capture_error_analysis` (defined + dispatched in both capture entry points) + tutorial page → Tasks 1/3/4 (round-trip gated by `check_workflows_md.py`). ✅
- Full screenshot set regenerated + committed (all PNGs, no cherry-pick) → Task 2. ✅
- Tutorial under `docs/source/tutorials/gui/` + indexed in the toctree → Task 3. ✅
- `gui/CLAUDE.md` + `io_constants` docstring updates → Task 5. ✅
- FEATURES.md rows already present (Phase 4/5) — no new rows needed; gate stays green. ✅
- Seeded labels make the screenshots meaningful (the tab's whole value is the ranked table) → Task 1 decision 1. ✅
