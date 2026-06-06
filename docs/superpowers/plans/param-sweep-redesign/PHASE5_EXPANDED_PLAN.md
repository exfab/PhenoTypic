# Phase 5 — `/tune/` Dash Co-Pilot — EXPANDED TDD PLAN (held for execution)

> Untracked working artifact (like RESUME.md). Produced by the code-architect
> expansion on 2026-06-05, then annotated with the two user decisions that
> revise it. **Do not execute until Phase 4.5 (task #15) is integrated** — Phase
> 5 depends on the gap/suspicious/splits data Phase 4.5 produces.
>
> **🛑 GATE before executing this plan (user-requested 2026-06-05, task #17):**
> after the backend simplify (#16), STOP and **brainstorm the GUI interface WITH
> the user** (superpowers:brainstorming) — refine the UX/layout/interaction
> intent on top of this technical structure BEFORE dispatching any chunk. This
> plan is the engineering skeleton; the brainstorm sets the design intent. Fold
> the brainstorm outcomes back into the chunk tasks before implementing.

## POST-DECISION REVISIONS (apply these on top of the plan below)

The expansion surfaced two OQs; the user resolved them 2026-06-05:

1. **OQ-1 = "build engine prereqs first" (NOT stub-and-gate).** Phase 4.5
   (task #15) builds the generalization-gap flag, the anti-gaming-suspicious
   flag, the `splits/` calibration/held-out writer, and the `gap`/`suspicious`
   fields on `Trial`. **So Phase 5 CONSUMES the real fields** — delete the
   `gap_available()` / `suspicious_available()` dormant-gating from C1/C2. The
   shortlist's default mix is the full **top-5 + Pareto members + gap-flagged +
   anti-gaming-suspicious** (dash §3/§5). Overlay calibration plates come from
   the **persisted `splits/calibration.json`** (Phase 4.5 writes it via a
   `tools_` path helper) — NOT an ad-hoc id-sorted input-dir subset. The
   monitor's generalization-gap badge shows the **real winner gap** from Phase
   4.5's `deliverables/generalization.json`.
2. **OQ-2 = human winner OVERWRITES `best_pipeline.json` + provenance stamp.**
   C3's `write_winner` writes the human pick to `best_pipeline_path(output_dir)`
   (the file the production run consumes), stamped `source="human"` (provenance
   sidecar or a top-level key), and surfaces a "CLI auto-finalize would
   overwrite" warning. NOT a side `curated_pipeline.json`.

Everything else below is settled by §0a/§0b and stands as written.

---

## 0. Verified ground-truth (live code, not spec prose)

**Tune public API** (`src/phenotypic/tune/__init__.py`): `JournalStudyStore`,
`StudyStore` (alias→journal), `Trial`, `compute_param_importance`,
`compute_param_importance_report`, `ImportanceReport`, `ImportanceMethod`,
`infer_search_space`, `InferredSearchSpace`, `Excluded`, `TuneSpec`,
`SearchSpace`, `Knob`, `build_pipeline`, `TuningSpec`, `run_tuning`,
`run_auto_space`.

- `OptunaStudyStore` (`tune/_study/_optuna_store.py`) opens
  `create_study(storage=…, study_name="tune", load_if_exists=True)`, always
  read-write + `PRAGMA journal_mode=WAL`. Trial `user_attrs` carry `pheno_*`
  keys (~L28-36). **No read-only adapter yet** — C0 builds one.
- `Trial` (`tune/_study_store.py`): `number, params, score, terms, n_images,
  objectives (Optional[dict]), failed, pruned`. (Phase 4.5 ADDS `gap` +
  `suspicious`.)
- Pareto: `tune/_study/_pareto.py::pareto_front_of(trials)` /
  `knee_point_of(front)`; `StudyStore.pareto_front()==[]` for single-objective.
- Multi-objective: `tune/_multi_objective.py::is_multi_objective(scorer)` /
  `objective_names(scorer)` — inferred from the scorer, never a flag.
- Importance: `compute_param_importance_report(store, *, random_state=0,
  objective=None) -> ImportanceReport(importances, method, interactions_estimated)`,
  `method ∈ {"fanova","rf-permutation"}`.
- Path helpers (`tools_/_io_constants.py`, all present): `study_db_path`,
  `trials_parquet_path`, `best_pipeline_path`, `param_importance_path`,
  `tuning_spec_path`, `deliverables_dir`, `pareto_dir`,
  `pareto_front_parquet_path`, `pareto_best_pipeline_path(out, objective)`,
  `pareto_importance_path(out, objective)`. (Phase 4.5 ADDS `DIR_SPLITS` +
  `splits_*_path` + `generalization_path`.)
- CLI (`tune/__main__.py`): `python -m phenotypic.tune run <spec> -i <input> -o
  <output> [--strategy grid|random|tpe|cmaes|gp|nsga2] [--n-trials N]
  [--screen|--no-screen] [--storage-url URL] [--slurm]`. **No** `--objective`,
  `--multi-objective`, `--calibration-frac`, `--stability-weight` on the live
  `run` (those are aspirational in e2e-workflows). The command card emits **live
  flags only**.

**GUI shell** (verified):
- `gui/_config.py`: `MOUNT_HOME="/"`, `MOUNT_BUILDER="/builder/"`,
  `MOUNT_VIEWER="/results/"`, `MOUNT_RUN="/run/"`, `MOUNT_ANALYSIS="/analysis/"`;
  `CFG_RUNNER`, `CFG_URL_PREFIX`; `TITLE_*`.
- `gui/shell/_ids.py`: `ToolName = Literal["viewer","analysis","builder","run"]`;
  `SHELL_TAB_HOME/BUILDER/VIEWER/RUN/ANALYSIS`.
- `gui/shell/_layout.py`: `_TAB_HREFS`, `_TAB_LABELS`, `TAB_DISPLAY_ORDER`
  (`HOME, BUILDER, RUN, VIEWER, ANALYSIS`). `_build_tab` reads HREFS/LABELS;
  `build_top_bar` iterates DISPLAY_ORDER.
- `gui/shell/_app.py::compose_hub`: builds each sub-app, `wrap_in_chrome`, mounts
  via `DispatcherMiddleware` keyed `MOUNT_*.rstrip("/")` (dict ~L308-316).
- Overlay renderer: `gui/builder/_image_renderer.py::to_overlay_png_bytes(image,
  *, max_dim=512, alpha=0.4) -> bytes` (~L295) — reuse directly (NOT
  `render_node_preview`).
- Command-card mirror source: `gui/run_console/_state.py::to_argv(state)` (~L234)
  + `gui/run_console/_callbacks.py::_local_argv_for`.
- `_param_forms.py`: `param_form(...)` (~L721), `_param_row` (~676),
  `_widget_for_param` (~510) — 6c reuses these.
- `InferredSearchSpace`/`Excluded`/`Knob` (`tune/_search_space/_inferred.py`,
  `_space.py`): `Knob.source ∈ KnobSource` (manual/…/`presence_optin`),
  `Knob.needs_review: bool`, `Knob.description`, `Knob.conditional_on`;
  `Excluded.reason ∈ ExcludeReason`, `Excluded.field_type`;
  `InferredSearchSpace.to_search_space()`.

---

## 1. Chunking (C0 seam-owned first, then fan out)

- **C0 (orchestrator, FIRST, alone)** — registration + skeleton + read-adapter.
  Owns the shared seam files (`gui/_config.py`, `gui/shell/_ids.py`,
  `gui/shell/_layout.py`, `gui/shell/_app.py`) + new
  `gui/tune/{__init__,_app,_ids,_layout,_study_reader}.py`. Ships an
  empty-but-mounted `/tune/` tab + the read-only WAL study reader.
- **C1 — 6a Monitor** (`gui/tune/_monitor.py`,`_callbacks.py`): objective curve,
  importance bars+badge, run status, **real generalization-gap badge** (from
  Phase 4.5 `generalization.json`), Pareto panel feature-flagged off single-obj.
- **C2 — 6b Shortlist + overlays** (`_shortlist.py`,`_overlay_worker.py`,
  `_candidate_detail.py`): shortlist = top-5 + Pareto + **gap-flagged +
  suspicious** (real Phase-4.5 fields); background overlay worker + disk-LRU via
  `to_overlay_png_bytes`; plates from persisted `splits/calibration.json`;
  per-image score table.
- **C3 — 6b Curation write-back + winner** (`_curation.py`): `user_attrs`
  write-back (last-write-wins + attribution; local-single-node only,
  monitor-only on NFS/Postgres); **winner → `best_pipeline.json` overwrite +
  provenance stamp** (OQ-2).
- **C4 — 6c Space-edit** (`_space_edit.py`,`_spec_emit.py`): `pipeline.json` →
  `infer_search_space` → `param_form` reuse (flat + presence only; nested
  read-only); → `tuning_spec.json`; **copy-paste command card** (live flags
  only; NO `LocalRunner` spawn).

Integration order: `C0 → (C1 ∥ C4) → C2 → C3`. Seam files orchestrator-owned;
downstream chunks add only files inside `gui/tune/` + append their own
FEATURES.md/WORKFLOWS.md rows.

## 2. Registration-site checklist (verified anchors)

| # | File | Edit |
|---|------|------|
| 1 | `gui/_config.py` | `MOUNT_TUNE="/tune/"`, `TITLE_TUNE`, `CFG_TUNE_STUDY="pheno_tune_study"`, `SANDBOX_TUNE_OVERLAYS_SUBDIR`; extend `__all__` |
| 2 | `gui/shell/_ids.py` | `SHELL_TAB_TUNE="shell-tab-tune"`; append `"tune"` to `ToolName`; `__all__` |
| 3 | `gui/shell/_layout.py` | add to `_TAB_HREFS`/`_TAB_LABELS`; insert into `TAB_DISPLAY_ORDER` (Run→**Tune**→Viewer); import `MOUNT_TUNE`,`SHELL_TAB_TUNE` |
| 4 | `gui/shell/_app.py::compose_hub` | build `tune_app`, `wrap_in_chrome(active_tab=SHELL_TAB_TUNE)`, add `MOUNT_TUNE.rstrip("/"): tune_app.server` to dispatcher; imports + `logger.info` |
| 5 | new `gui/tune/__init__.py` | export `create_app` |
| 6 | new `gui/tune/_app.py` | `create_app(sandbox, *, url_prefix=MOUNT_HOME, study_path=None)` (mirror `run_console/_app.py`) |
| 7 | new `gui/tune/_ids.py` | `TUNE_ROOT`, `TUNE_POLL_INTERVAL`, `TUNE_REFRESH_BUTTON`, shortlist/detail/curation/space ids |

Body files: `_layout.py`, `_callbacks.py`, `_study_reader.py`, `_monitor.py`,
`_shortlist.py`, `_overlay_worker.py`, `_candidate_detail.py`, `_curation.py`,
`_space_edit.py`, `_spec_emit.py`.

## 3. Per-chunk TDD tasks (abridged — full detail in the 2026-06-05 architect
transcript; reconstruct each as failing-test → minimal-impl → `-n 8` gate)

**C0:** T1 config consts · T2 `SHELL_TAB_TUNE`+`ToolName` · T3 tab slot (3 dicts +
order, Run→Tune→Viewer) · T4 sub-app factory + hub mount (Flask test-client
`GET /tune/`==200) · T5 read-only WAL reader `TuneStudyReader.open_readonly`
(`sqlite:///file:{path}?mode=ro&uri=true` + busy_timeout; `trials()`/`best()`/
`pareto_front()`/`incremental_since(n)`; non-existent path → `NoStudyState`
sentinel; module docstring: SLURM/NFS=monitor-only, local=write-back). FEATURES
row for the Tab; WORKFLOWS `tune_copilot` row as `🔭 planned` (no capture yet).

**C1:** T1 `summarize_run_status` · T2 `build_objective_curve` (per-trial + best-so-far
monotone; `OI_*` colors) · T3 `build_importance_panel` (bars + method badge +
`interactions_estimated` note) · T4 `build_pareto_panel`+`pareto_panel_enabled`
(multi-obj fixture; single-obj→hidden) · T5 incremental poll (3s `dcc.Interval`)
+ manual refresh callback (thin adapter over `refresh_monitor(reader,last_n)`).
**+ real gap badge** from `generalization.json` (revision 1). FEATURES: 5 rows.

**C2:** T1 `select_shortlist(trials, *, top_n=5, front, gap_threshold=0.15)` →
top-5 + Pareto + **gap-flagged + suspicious** (real fields; dedup by number,
reason tags) · T2 `rebuild_candidate_pipeline(spec, trial)` via `build_pipeline`
(spec from `tuning_spec.json`) · T3 `OverlayCache`(disk-LRU, key
`(trial_number,plate_id)`)+`OverlayWorker`(ThreadPoolExecutor; inject sync
executor in tests); plates from **`splits/calibration.json`** (revision 1) ·
T4 candidate detail panel + per-image score table (integration). FEATURES: 4 rows.

**C3:** T1 `build_curation_attr`+`merge_curation` (last-write-wins, ISO ts,
attribution) · T2 `curation_writable(study_path)` (local SQLite only; Postgres/NFS
→ monitor-only) + `write_curation` (read-write `OptunaStudyStore`, namespaced
`user_attr`, completed trials only) · T3 `write_winner(out, spec, trial)` →
**overwrite `best_pipeline_path` + provenance stamp `source="human"` +
`winner_conflict_warning`** (revision 2) · T4 accept/reject/rank/notes +
pick-winner callbacks (integration; monitor-only banner on non-writable) ·
T5 e2e Playwright flow (`tests/e2e/gui/test_tune_copilot.py`, `ci_flaky`, serial).
FEATURES: 5 rows. **Flip WORKFLOWS `tune_copilot` → `✅`** + capture + tutorial page.

**C4:** T1 `load_inferred_space(pipeline_json)` · T2 `build_space_form` (domain
editors; provenance badges from `KnobSource`; `⚠ needs_review`; excluded list +
"add a TuneSpec" hint; **nested knobs read-only/disabled**) · T3
`emit_tuning_spec(...)` → `TuningSpec` round-trip · T4 `render_tune_command(...)`
(private `Final[str]` template + typed fn; **live flags only; assert no
subprocess/LocalRunner import**) · T5 space-edit panel (integration). FEATURES: 7 rows.

## 4. CI gates
- FEATURES.md: 1(C0)+5(C1)+4(C2)+5(C3)+7(C4)=**22 rows**, each `✅ shipping` with a
  real `Test ref` (test before row; pre-commit validates).
- WORKFLOWS.md: one `tune_copilot` flow — `🔭 planned` in C0 (no capture), flips
  `✅` in C3 with all three round-trip pieces: `_capture_tune_copilot` **defined
  AND dispatched** in `scripts/capture_gui_tutorial_screenshots.py` (dispatch
  block ~L293-307) + tutorial page `docs/source/tutorials/gui/15_tune_copilot.md`
  + `docs/source/_static/gui_images/tune_copilot/`.
- **Hermetic synthetic study for capture:** `_build_synthetic_tune_study(out)` —
  append ~8 `Trial`s via `OptunaStudyStore` over synthetic plates + write
  `tuning_spec.json`/`param_importance.json`/`generalization.json`/`splits/`;
  point `CFG_TUNE_STUDY` at it; snap monitor + shortlist+overlay + curation.
- Regenerate + **commit ALL PNGs** (full set; do not cherry-pick churn).

## 5. Tests
- unit `-n 8`: `tests/unit/gui/tune/`, `tests/unit/tune/test_study_reader.py`,
  `tests/unit/gui/test_config_tune.py`, `tests/unit/gui/shell/test_*`.
- integration `-n 8` (Flask client): `tests/integration/gui/test_tune_*.py`
  (drive overlay worker via injected sync executor for determinism).
- e2e serial (`PLAYWRIGHT=1`): `tests/e2e/gui/test_tune_copilot.py`, module
  `ci_flaky` (Dash chain + overlay poll budget).

## 6. Whole-phase gate
```
uv run pytest tests/unit/gui/tune tests/unit/tune/test_study_reader.py \
  tests/unit/gui/test_config_tune.py tests/unit/gui/shell tests/integration/gui -n 8
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_tune_copilot.py            # local (incl ci_flaky)
PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_tune_copilot.py -m "not ci_flaky"  # CI parity
uv run mypy src/phenotypic/gui/tune src/phenotypic/gui/shell src/phenotypic/gui/_config.py
uv run ruff check --fix
uv run python scripts/check_workflows_md.py
uv run pre-commit run --all-files
uv run python scripts/capture_gui_tutorial_screenshots.py   # after C3; commit ALL PNGs
```
