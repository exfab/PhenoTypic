# Design Spec — Move final CLI outputs into `<output>/deliverables/`

- **Date:** 2026-05-30
- **Branch:** `refactor/cli-output-structure`
- **Status:** Draft for review
- **Author:** spec generated from a full codebase access-point sweep

---

## 1. Goal

Relocate the run's **user-facing deliverables** out of the output-directory
root and into a single `<output>/deliverables/` folder, so a user (or a
`scp`/`rsync` of "the results") gets one obvious folder of finished artifacts
instead of having to pick them out from the machine-state sidecars
(`progress/`, `processing_state.json`, `slurm_scripts/`, …).

### Locked decisions (from product review)

1. **Scope = all user-facing outputs.** The following move into `deliverables/`:
   - `master_measurements.csv` / `master_measurements.parquet`
   - `measurements.csv` / `measurements.parquet`
   - `measurements_by_feature/<feature>.{csv,parquet}` (the per-`MeasureFeatures` split)
   - `analysis.csv` / `analysis.parquet`
   - `dashboard.html`
   - `analysis.html`
   - `processing_report.html`
   - `README.md`
   - `pipeline.json`
2. **Hard cutover.** Producers write only to `deliverables/`. Consumers read
   only from `deliverables/`. No dual-read fallback, **no** migration shim.
   Output directories produced by older versions will not open in the GUI
   until re-run or `--recompile`d. This is accepted.

### Target layout

```
output_folder/
├── deliverables/                 # NEW — all user-facing outputs
│   ├── master_measurements.csv
│   ├── master_measurements.parquet
│   ├── measurements.csv
│   ├── measurements.parquet
│   ├── measurements_by_feature/
│   │   ├── MeasureSize.csv
│   │   ├── MeasureSize.parquet
│   │   └── …
│   ├── analysis.csv              # only when pipeline has a model
│   ├── analysis.parquet
│   ├── dashboard.html
│   ├── analysis.html
│   ├── processing_report.html
│   ├── pipeline.json
│   └── README.md
├── results/                      # UNCHANGED (per-image artifacts)
│   └── <dataset>/{hdf,measurements,overlays,inspect}/…
├── qc/                           # UNCHANGED (Review GUI artifacts)
├── progress/                     # UNCHANGED (machine state, sidecars, chunks)
├── slurm_scripts/                # UNCHANGED
├── logs/                         # UNCHANGED
└── processing_state.json         # UNCHANGED (resume state)
```

### Explicitly **not** moved (stays at root)

`results/` (per-image HDF / per-image `measurements/` parquets / overlays /
inspect), `qc/`, `progress/` (manifest, events log, failures, chunks, recompile
shards, analysis sidecars, vendored `plotly.min.js`/`hyparquet.min.js`),
`slurm_scripts/`, `logs/`, `processing_state.json`.

> Rationale: these are **inputs to** aggregation or **machine state**, not
> finished deliverables. In particular `results/<ds>/measurements/` (per-image
> parquets) is the *source* the master is aggregated from — distinct from the
> top-level `measurements.{csv,parquet}` post-applied mirror, which **does** move.

---

## 2. The leverage point

Almost all of these paths are already centralized in
**`src/phenotypic/tools_/_io_constants.py`** as filename constants + path-builder
helper functions, re-exported through `phenotypic.tools_` and (for the GUI)
`phenotypic.gui._config`. This is the single source of truth the existing
refactor established, and it makes the move tractable.

**The core change is two new symbols plus re-rooting the existing helpers:**

```python
# _io_constants.py
DIR_DELIVERABLES: Final[str] = "deliverables"
README_MD: Final[str] = "README.md"          # NEW constant (currently a bare literal)

def deliverables_dir(output_dir: Path) -> Path:
    """Return <output>/deliverables/."""
    return output_dir / DIR_DELIVERABLES

def readme_md_path(output_dir: Path) -> Path:
    """Return <output>/deliverables/README.md."""
    return deliverables_dir(output_dir) / README_MD
```

Then re-root these existing helpers from `output_dir / X` to
`deliverables_dir(output_dir) / X`:

| Helper (in `_io_constants.py`) | New target |
|---|---|
| `master_measurements_csv_path` | `deliverables/master_measurements.csv` |
| `master_measurements_parquet_path` | `deliverables/master_measurements.parquet` |
| `measurements_csv_path` | `deliverables/measurements.csv` |
| `measurements_parquet_path` | `deliverables/measurements.parquet` |
| `measurements_by_feature_dir` | `deliverables/measurements_by_feature/` |
| `analysis_csv_path` | `deliverables/analysis.csv` |
| `analysis_parquet_path` | `deliverables/analysis.parquet` |
| `dashboard_html_path` | `deliverables/dashboard.html` |
| `analysis_html_path` | `deliverables/analysis.html` |
| `processing_report_html_path` | `deliverables/processing_report.html` |
| `pipeline_json_path` | `deliverables/pipeline.json` |
| `load_master_measurements` | (transitive — uses `master_measurements_csv_path`) |

> **`deliverables/` is created where files are written.** `_atomic_write`
> already does `target.parent.mkdir(parents=True, exist_ok=True)`, so the CSV /
> parquet / HTML writers provision the folder for free. The per-feature split
> already calls `split_dir.mkdir(parents=True, exist_ok=True)`. The README
> generator uses `Path.write_text` directly — it must gain an explicit
> `mkdir` (see §4.1).

**But helpers are not the whole story.** Many call sites bypass the helpers and
join the *constant* inline (`output_dir / MASTER_MEASUREMENTS_CSV`, `root /
MEASUREMENTS_PARQUET`). Re-rooting the helpers does **not** fix those — they must
be converted to call the helper (or `deliverables_dir(output_dir) / CONST`). The
full inventory in §3 marks each site **[helper]** (transitively fixed) or
**[inline]** (must convert).

---

## 3. Complete access-point inventory

### 3.1 Producers (write)

| File | Line(s) | Artifact | Mechanism | Action |
|---|---|---|---|---|
| `_cli/_cli_output_manager.py` | 872–873 | master csv/parquet | **[inline]** `output_dir / MASTER_MEASUREMENTS_*` | convert to helper |
| `_cli/_cli_output_manager.py` | 488, 494 | measurements csv/parquet (`_seed_measurements`) | **[inline]** | convert to helper |
| `_cli/_cli_output_manager.py` | 409–410 | analysis csv/parquet (`_emit_analysis_outputs`) | **[inline]** | convert to helper |
| `_cli/_cli_output_manager.py` | 714, 727–728 | `measurements_by_feature/` dir + `<key>.{csv,parquet}` | **[inline]** `output_dir / DIR_MEASUREMENTS_BY_FEATURE` | convert to `measurements_by_feature_dir()` |
| `_cli/_cli_output_manager.py` | 346 | pipeline.json (`_persist_pipeline_to_output_dir`) | **[inline]** `output_dir / PIPELINE_JSON` | convert to `pipeline_json_path()` |
| `_cli/_cli_output_manager.py` | 291, 303–311 | pipeline.json **read** (`_load_pipeline_from_output_dir`) | **[inline]** | re-root canonical read to `pipeline_json_path()`; see §6 note on legacy `processing_state.json`→`output_dir/<name>.json` fallback (that sibling is **not** a deliverable; leave at root) |
| `_cli/_cli_chunk_writer.py` | 157, 161 | master csv/parquet (**mid-run** partial) | **[inline]** | convert to helper (mid-run master also lands in `deliverables/`) |
| `_cli/_cli_recompile_worker.py` | 343, 349 | master csv/parquet (recompile finalize) | **[inline]** | convert to helper |
| `_cli/_cli_readme_generator.py` | 49 | README.md | **[inline bare literal]** `output_dir / "README.md"` | convert to `readme_md_path()` + add `mkdir` |
| `_cli/_dashboard/_generator.py` | 80, 84 | dashboard.html, analysis.html | **[helper]** `dashboard_html_path` / `analysis_html_path` | transitive ✓ — **but embedded JS URLs need rebasing, see §5** |
| `phenotypicCLI.py` | 1262 | processing_report.html | **[helper]** `processing_report_html_path` | transitive ✓ |

### 3.2 Consumers (read) — CLI / dashboard

| File | Line(s) | Artifact | Mechanism | Action |
|---|---|---|---|---|
| `_cli/_cli_checkpoint_handler.py` | 164 | measurements.parquet mirror (analysis plugins) | **[inline]** `output_dir / MEASUREMENTS_PARQUET` | convert to `measurements_parquet_path()` |
| `phenotypicCLI.py` | 1766 | measurements.parquet mirror (analysis plugins) | **[inline]** | convert to helper |
| `_cli/_dashboard/_manifest_builder.py` | 421 | master_measurements.parquet mtime | **[helper]** `master_measurements_parquet_path(progress_dir.parent)` | transitive ✓ |

### 3.3 Consumers (read/write) — GUI

| File | Line(s) | Artifact | Mechanism | Action |
|---|---|---|---|---|
| `gui/shell/_classifier.py` | 209, 215 | `master_measurements.parquet` + `dashboard.html` markers | **dir scan by name** | **rework** — markers are now nested under `deliverables/` (see §4.4) |
| `gui/results_viewer/_output_root.py` | 102 | master_measurements.parquet (discovery sentinel) | **[inline]** `root / MASTER_MEASUREMENTS_PARQUET` | convert to helper |
| `gui/results_viewer/_output_root.py` | 119 | measurements.parquet (mirror) | **[inline]** `root / MEASUREMENTS_PARQUET` | convert to helper |
| `gui/results_viewer/_output_root.py` | 175 | pipeline.json (summary) | **[inline]** `root / PIPELINE_JSON` | convert to `pipeline_json_path()` |
| `gui/results_viewer/_filtered_state.py` | 229–230 | measurements.parquet/csv (**read + curation write-back**) | **[inline]** `root / MEASUREMENTS_*` | convert to helpers — note this path **writes** the mirror back |
| `gui/_schema_cache.py` | 40–43, 93–94 | measurements + master_measurements {parquet,csv} | **[inline]** `output_root / files[i]` (filename map) | re-root joins through `deliverables_dir(output_root)` |
| `gui/analysis/_callbacks.py` | 416 | measurements.parquet (preview analyze) | **[inline]** `Path(output_root.root) / MEASUREMENTS_PARQUET` | convert to helper |
| `gui/analysis/_callbacks.py` | 622 | measurements.parquet (inline run) | **[inline]** `output_dir / MEASUREMENTS_PARQUET` | convert to helper |
| `gui/analysis/_callbacks.py` / `_recipe_state.py` | (recipe save) | pipeline.json **rewrite** on recipe edit | verify path source | ensure recipe save targets `pipeline_json_path()` (now in `deliverables/`) |
| `gui/results_viewer/_qc_tab/review/_data.py` | 522, 535 | measurements.parquet, master_measurements.parquet fallback | **[helper]** already uses `measurements_parquet_path` / `master_measurements_parquet_path` | transitive ✓ |

### 3.4 Consumers — URL/link construction (the `/runs/<rel>` HTTP surface)

The Flask blueprint `gui/shell/_runs_blueprint.py` serves **any** sandbox-relative
path (`/runs/<path:rel_file>`), so it needs **no change** — but every site that
*constructs* a URL or existence-check path to a moved artifact does:

| File | Line(s) | What | Action |
|---|---|---|---|
| `gui/run_console/_callbacks.py` | 366 | builds iframe URL `f"{RUNS_BLUEPRINT_PREFIX}/{safe_rel}/{DASHBOARD_FILENAME}"` | insert `deliverables/` segment |
| `gui/run_console/_callbacks.py` | 1034, 1216 | `sandbox.resolve(Path(rel_path) / DASHBOARD_FILENAME)` existence check | insert `DIR_DELIVERABLES` segment |
| `gui/run_console/_layout.py`, `_recent_runs.py` | docstrings | mention `/runs/<rel>/dashboard.html` | update text |

### 3.5 Dashboard embedded-JS relative URLs — `_cli/_dashboard/_generator.py`

`dashboard.html` itself moves into `deliverables/`, so its **relative** fetch URLs
re-base relative to `deliverables/`. Two classes:

| Line(s) | URL (relative) | Resolves to (today) | After move | Fix |
|---|---|---|---|---|
| 1121 | `README.md` | `<output>/README.md` | sibling in `deliverables/` ✓ | **none** (moves together) |
| 1161 | `…+ base + 'measurements.csv'` (wget hint) | mirror | sibling ✓ | **none** if `base` stays the dashboard's own dir; verify `base` derivation (uses `window.location`) |
| 1628 | `_loadParquetFile('measurements.parquet')` | mirror | sibling ✓ | **none** |
| 1163 | `…+ base + 'results/'` (wget hint) | `<output>/results/` | now one level up | re-base to `../results/` |
| 1318 | `progress/failures.jsonl` | `<output>/progress/…` | up one level | `../progress/failures.jsonl` |
| 1388, 1406, 1700 | `progress/manifest.json` | ″ | ″ | `../progress/manifest.json` |
| 1563, 1564 | `progress/plotly.min.js`, `progress/hyparquet.min.js` | ″ | ″ | `../progress/…` |
| 1634 | `progress/analysis_full.parquet` | ″ | ″ | `../progress/…` |
| 1669, 1673 | `progress/analysis_stats.json`, `overlay_manifest.json` | ″ | ″ | `../progress/…` |

> **Net rule for the dashboard JS:** sibling deliverables (`measurements.*`,
> `README.md`) keep their bare relative names; everything pointing at `results/`
> or `progress/` gains a `../` prefix. The `wget` download hints in the
> dashboard's "Download" panel (1161, 1163) are user-copy-paste strings — verify
> the computed `base`/`cutDirs` still produce correct recursive-download commands
> for the new depth (`--cut-dirs` count changes by one for `results/`).

`analysis.html` (also generated by `_generator.py`, also moving) must get the
**same `../progress/` rebasing** if it fetches any `progress/` sidecars. Audit
its template block alongside the dashboard block.

### 3.6 `tools_/__init__.py` re-exports + `gui/_config.py`

- `tools_/__init__.py`: add `DIR_DELIVERABLES`, `deliverables_dir`,
  `README_MD`, `readme_md_path` to imports + `__all__`.
- `gui/_config.py`: re-export `DIR_DELIVERABLES` (as `DELIVERABLES_DIRNAME` to
  match the `*_DIRNAME` GUI convention) and `deliverables_dir` so GUI code has
  the ergonomic import. Update the module docstring's filename list.

### 3.7 `tools_/_column_ref.py`

References the **logical** source names `"measurements"` / `"master_measurements"`
(`ColumnSource` literal), not filesystem paths. **No change** — the path mapping
lives in `_schema_cache.py` (§3.3).

---

## 4. Change detail by area

### 4.1 `_io_constants.py`
Add `DIR_DELIVERABLES`, `README_MD`, `deliverables_dir()`, `readme_md_path()`.
Re-root the 11 helpers in §2's table. Add directory-docstring entries describing
`deliverables/`. Keep filename constants (`MASTER_MEASUREMENTS_CSV`, …)
**unchanged** — only the *path composition* changes, so any code still importing
the bare constant for display continues to work.

### 4.2 CLI producers
Convert every **[inline]** site in §3.1 to the helper. The README generator
(§3.1) additionally needs `readme_md_path(output_dir).parent.mkdir(parents=True,
exist_ok=True)` before `write_text`, because (unlike the parquet/CSV writers) it
does not go through `_atomic_write`'s implicit `mkdir`. Update the README's
embedded **Output Structure** ASCII tree (`_cli_readme_generator.py:86–108`) to
show the new `deliverables/` layout.

### 4.3 CLI consumers
Convert checkpoint handler (164) and `phenotypicCLI` (1766) mirror reads to
`measurements_parquet_path()`.

### 4.4 GUI classifier (`shell/_classifier.py`) — needs real logic, not a rename
Today `_classify_dir` does a single `listdir` of the candidate dir and flags:
- `is_cli_output` = has `master_measurements.parquet` child **and** `results/` child
- `has_dashboard` = has `dashboard.html` child

After the move, `master_measurements.parquet` and `dashboard.html` live under
`deliverables/`, while `results/` stays at root. Options:

- **Recommended:** after the root `listdir`, if a `deliverables/` child dir
  exists, do one `(deliverables/master_measurements.parquet).is_file()` and
  `(deliverables/dashboard.html).is_file()` stat. Keep `results/` detection at
  root. This is two extra stats on dirs that have a `deliverables/` child —
  negligible, and the classifier already stats per child.
- Keep the LRU cache keyed on the **root** dir mtime. Note a subtle caveat:
  dropping a file *inside* `deliverables/` bumps `deliverables/`'s mtime, not the
  root's, so a freshly-finalized run may need the sidebar **Refresh** (which
  calls `invalidate_cache()`) to light up. Document this; it matches existing
  behavior for `results/` subdir changes.

Update the `Capabilities` docstring (`is_cli_output`, `has_dashboard`) and the
marker comments (38–40).

### 4.5 GUI results viewer
`_output_root.discover` (§3.3): re-root the master sentinel, the mirror, and
`pipeline.json` through helpers. Update the docstring's "expected layout" block
(81–86, 93–99) to `<root>/deliverables/master_measurements.parquet` etc. The
`results/<dataset>/overlays|measurements` references stay at root.
`_filtered_state` (curation write-back) re-roots both mirror paths through helpers
— this is the one consumer that **writes** into `deliverables/`, so confirm the
folder exists (it will, since finalize seeded it; but the atomic writer mkdirs
anyway).

### 4.6 GUI run console
Insert the `deliverables/` segment into the iframe URL builder (366) and the two
existence checks (1034, 1216). Centralize via `DELIVERABLES_DIRNAME` from
`_config.py` rather than a bare literal.

---

## 5. Highest-risk item: moving `dashboard.html` / `analysis.html`

This is the only part where re-rooting helpers is **insufficient** — the HTML
artifacts contain ~10 hardcoded relative URLs (§3.5). The move is still safe
because the artifacts they *share* a folder with (`measurements.*`, `README.md`)
move together, so only the `progress/`- and `results/`-targeting URLs need a
`../` prefix. Concretely:

1. In `_generator.py`, introduce a single JS-side constant for the "path back to
   output root" (`const ROOT = '../';`) and prefix the `progress/`+`results/`
   fetches with it, rather than hand-editing ~10 string literals (keeps it
   greppable and lets a future relocation flip one value).
2. Re-verify the **Download panel** wget commands (1152–1163): `base` is derived
   from `window.location` (the dashboard's own URL), and `--cut-dirs` is computed
   from the path depth. Moving the dashboard one level deeper changes that depth —
   the `results/` recursive-download command and `cutDirs` math must be re-checked
   against the new `…/deliverables/dashboard.html` location.
3. Apply the identical `../progress/` rebasing to the `analysis.html` template
   block if present.

There is an **end-to-end smoke** for this: launch the GUI, open a finalized run's
dashboard in the iframe, and confirm the charts (which fetch `progress/manifest.json`,
`progress/analysis_full.parquet`, vendored `plotly.min.js`) and the measurements
table (`measurements.parquet`) all load with no console 404s.

---

## 6. `pipeline.json` coupling (call out explicitly)

`pipeline.json` is more entangled than the other artifacts — it is **read and
rewritten** by the analysis GUI on every recipe edit, read by the viewer for the
pipeline summary, and read by the recompile worker's
`_load_pipeline_from_output_dir`. All those go through `pipeline_json_path()`
once §3 is applied, so they stay in lockstep. Two non-obvious points:

- The classifier's `has_pipeline_json` / `cfg` badge is for **standalone** pipeline
  JSON files the user authored (peeked via `_peek_for_pipeline_marker`), **not**
  the output-dir `pipeline.json`. Moving the output-dir copy does **not** affect
  that badge. No change.
- `_load_pipeline_from_output_dir`'s **legacy fallback** reads
  `processing_state.json` → `output_dir / <original-pipeline-name>.json`. That
  sibling JSON is the user's *input* pipeline copied next to state, not a
  generated deliverable. Leave it at root; only the canonical `pipeline.json`
  read/write re-roots.

---

## 7. Tests to update

Hard cutover means every test asserting a root-level path for a moved artifact
flips to `deliverables/`. Prefer asserting via the **helper** (`measurements_parquet_path(out)`)
so future moves don't re-break them.

- `tests/unit/cli/test_cli_output_manager.py` — master/measurements/by-feature/analysis/pipeline.json write locations.
- `tests/unit/cli/test_cli_v2.py`, `test_cli_recompile.py`, `test_cli_recompile_slurm.py` — aggregate + recompile output paths.
- `tests/integration/cli/test_finalize_qc.py` — finalize side-effect paths (note: `qc/` stays at root; only the deliverables move).
- `tests/unit/tools_/test_io_constants.py` — add coverage for `deliverables_dir`, `readme_md_path`, and the re-rooted helpers; add a test that every moved helper is under `deliverables/`.
- `tests/gui/results_viewer/test_output_root.py`, `test_filtered_state.py` — viewer discovery + curation write-back layout.
- `tests/unit/gui/shell/test_classifier.py` — `is_cli_output` / `has_dashboard` now require nested `deliverables/` markers; add fixtures with the new layout and a negative test for the legacy (root-level) layout (should classify as **not** CLI output, per hard cutover).
- `tests/unit/gui/test_schema_cache.py` — re-rooted source paths.
- `tests/unit/gui/shell/test_runs_registry.py` — dashboard URL/existence.
- `tests/unit/gui/results_viewer/test_qc_review_data.py` — helper-based, should pass once helpers move; add a layout fixture.
- `tests/integration/gui/test_app.py`, `test_analysis_*`, `test_viewer_handoff.py`, `test_recent_runs_rehydrate.py`, `test_qc_review_recompute.py` — handoff + recents + analysis read paths.
- `tests/e2e/gui/*` (`conftest.py`, `test_analysis_app.py`, `test_heatmap_tab.py`, `test_qc_tab.py`, `test_qc_review_splitter.py`) — fixtures that synthesize an output dir must emit the new layout; add a dashboard-iframe no-404 assertion if feasible.

### Test fixture audit
Grep test fixtures that **construct** a fake output dir by writing
`master_measurements.parquet` / `measurements.parquet` at root — these builders
must place files under `deliverables/`. This is the most likely source of broad
test breakage. A shared `make_output_dir(...)` test helper that writes via the
production path-helpers would prevent drift.

---

## 8. Docs to update

- `_cli_readme_generator.py` output-structure tree (the run's own README).
- `docs/source/tutorials/gui/{01_setup,02_file_explorer,06_view_results,08_analysis,10_qc_curation_loop,15_qc_review}.md` — any reference to file locations.
- `docs/source/how_to/pages/gui_hub.md`.
- `scripts/capture_gui_tutorial_screenshots.py` — if it asserts/reads artifact paths; re-run the capture per `CLAUDE.md` after chrome changes (the file-explorer screenshot of an output dir will visibly change).
- **`CLAUDE.md`** "Master vs. mirror outputs" + "Finalize via `finalize_post_master_outputs`" gotchas — update every `master_measurements.{csv,parquet}` / `measurements.{csv,parquet}` / `measurements_by_feature/` path to `deliverables/…`.
- `src/phenotypic/gui/CLAUDE.md` and `src/phenotypic/gui/_config.py` docstrings — filename/location references; add `DELIVERABLES_DIRNAME`.
- `src/phenotypic/gui/FEATURES.md` — if any row's `Test ref`/description names a path (CI `features-md-gate` will force a touch here anyway since `gui/` changes).
- `docs/build/**` (generated HTML) and `docs/source/api_reference/api/*.rst` stubs regenerate from docstrings — no manual edit.

---

## 9. Edge cases & gotchas

1. **Mid-run partial master.** The chunk writer writes `master_measurements.*`
   mid-run for early download; these now land in `deliverables/`. The viewer's
   discovery sentinel (`output_root.discover`) also looks in `deliverables/`, so
   the mid-run "open early" behavior is preserved. Consistent.
2. **`results/` is NOT a deliverable** but is required by the viewer and by the
   classifier's `is_cli_output`. It stays at root; only the master sentinel moved.
   Don't accidentally move per-image `results/<ds>/measurements/`.
3. **`qc/` stays at root** — it is not in the chosen scope and is heavily consumed
   by the Review GUI; leaving it reduces blast radius. Confirm no spec reviewer
   expected QC under `deliverables/`.
4. **Classifier cache staleness** (§4.4): finalizing a run mutates
   `deliverables/`'s mtime, not the root's; the sidebar may need Refresh. Matches
   existing `results/` behavior; document, don't fix.
5. **`_atomic_write` mkdir** provisions `deliverables/` for parquet/CSV/HTML; the
   README generator and any direct `write_text`/`write_bytes` must mkdir explicitly.
6. **Hard-cutover UX:** an old output dir silently fails to open in the viewer
   (`FileNotFoundError: Master measurements parquet not found at
   <root>/deliverables/master_measurements.parquet`). Make that error message name
   the new location and suggest `--recompile` (the message in `_output_root.py:104`
   already mentions re-running; update the path it prints).

---

## 10. Suggested implementation phasing

1. **Core constants + helpers** — `_io_constants.py` (new symbols, re-root
   helpers), `tools_/__init__.py` + `gui/_config.py` re-exports, plus the
   `test_io_constants.py` assertions. Land first; nothing else compiles cleanly
   without it.
2. **CLI producers** — convert all §3.1 inline sites; README generator mkdir +
   tree text. Verify with a real `python -m phenotypic` run on
   `load_synth_yeast_plate()`-style fixtures; confirm `deliverables/` is fully
   populated and root is clean.
3. **CLI consumers** — checkpoint handler + `phenotypicCLI` mirror reads.
4. **Dashboard/analysis HTML rebasing** (§5) — the `../` prefix + wget `cutDirs`
   re-check; browser smoke for no-404.
5. **GUI consumers** — output_root, filtered_state, schema_cache, analysis
   callbacks, classifier rework, run-console URL/existence sites.
6. **Tests + fixtures** — update builders to emit the new layout (shared helper),
   flip assertions, add legacy-layout negative tests.
7. **Docs + capture** — README tree, tutorials, CLAUDE.md gotchas, re-run
   screenshot capture, commit all refreshed PNGs (per `CLAUDE.md`, commit the
   full set, don't cherry-pick).

Per-phase: run `uv run mypy src/phenotypic` + `uv run ruff check --fix`, and the
targeted test module(s) for that phase. After phase 5, run the GUI e2e suite.

---

## 11. Residual risks / open questions

- **Dashboard `wget` hint correctness** (§5.2) is the easiest thing to silently
  get wrong (it's a copy-paste string, not exercised by unit tests). Flag for a
  manual check.
- **Anything reading these paths outside this repo** (user scripts, notebooks,
  downstream pipelines on the cluster) breaks under hard cutover. That's inherent
  to the chosen policy; worth a one-line CHANGELOG/release-note callout.
- **`pipeline.json` as resume input:** confirm no SLURM resubmission path expects
  `pipeline.json` at the output root before `deliverables/` exists (the sentinel
  can run before finalize). `_load_pipeline_from_output_dir` already tolerates a
  missing canonical file via the legacy fallback, so this is low-risk, but worth
  a targeted recompile-from-checkpoint test.
```
