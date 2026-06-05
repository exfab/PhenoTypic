# Phase 5 — Tune Co-Pilot (`/tune/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Per-task commits with the
> `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer; scoped
> `git add` (never `-A`); pytest `-n 8` (NEVER `-n auto`); writing/editing agents use Opus.

**Goal:** Ship the `/tune/` Dash co-pilot as a 5th sub-app mounted in the `phenotypic-gui`
hub — a flat sub-tab toggle (Monitor / Curate / Space / Launch) over a shared run picker — so a
user can watch a live tune study, A/B-compare candidate segmentations and pick a winner, edit a
search space, and copy the launch command. The GUI never spawns or re-optimizes.

**Architecture:** A new `gui/tune/` Dash sub-app mirroring `gui/results_viewer/`’s
`create_app(root=None, *, url_prefix)` factory, registered in `compose_hub`’s
`DispatcherMiddleware`. Reads come from a tune-specific `TuneRunRoot` (NOT `OutputRoot` — a tune
run has no `master_measurements.parquet`). Monitor is pure Plotly reads on a 3 s `dcc.Interval`;
Curate renders dynamic `label2rgb` overlays into Plotly `go.Image` figures with clientside
linked pan/zoom + a `match_iou_greedy` difference view, behind a background overlay worker + disk-LRU
cache; Space renders a custom `_knob_form` per knob; Launch renders the CLI string clientside.

**Tech stack:** Dash 4, dash-bootstrap-components, Plotly `go.Image`, `dcc.Interval`,
`dash.clientside_callback`; `skimage.color.label2rgb` (via `gui/builder/_image_renderer`);
`phenotypic.tune` (`infer_search_space`, `build_pipeline`, the `StudyStore`s, `match_iou_greedy`);
pytest + the Dash test client; Playwright for e2e + the screenshot capture script.

---

## Design decisions — locked (brainstorm + plan-review OQs + post-merge)

See memory `[[tune-gui-design]]` and the mockup. Brainstorm decisions (1–8) stand; the
plan-reviewer OQ resolutions (9–16) and the signature corrections below are now folded in.

1. **5th hub sub-app** at `/tune/`, mounted in `compose_hub` like builder/results/run/analysis.
2. **Flat sub-tab toggle** `[Monitor] [Curate] [Space] [Launch]` under a shared header.
3. **Run source = browse-to-dir**, recognized by the new `.pht-tune-cache/run.json` marker (OQ6);
   the live-study URL + images dir are read from that marker.
4. **Monitor** (read-only): objective curve, fANOVA importance, gap badge (rel. dispersion > 0.15),
   trials table; Pareto + knee **multi-objective only**.
5. **Curate** (writes `deliverables/best_pipeline.json`): shortlist = top-5 ∪ Pareto ∪ gap-flagged;
   A/B Plotly `go.Image` overlays + clientside linked relayout + a difference toggle
   (only-A / only-B / both via `match_iou_greedy`, τ=0.5).
6. **Space** (writes `tuning_spec.json`): `infer_search_space` → a custom `_knob_form` per knob;
   flat + presence editable, nested read-only.
7. **Launch:** form → live-rendered `python -m phenotypic.tune run …` copy card.
8. **Overlay tech:** Plotly `go.Image` + clientside linked-relayout (not OSD/DZI).

**Plan-reviewer OQ resolutions (folded in):**
9. **OQ1 — eager mount** + a `dcc.Store` run-root (no ToolSession; the tab pre-loads nothing heavy).
10. **OQ2 — Image Source (Tune-Curate-only, sandbox-bounded)**: the Image Source picker lives in the
    **Tune Curate view ONLY** — NOT a hub-shared / builder control. (The review proved cross-server
    sharing is broken — builder + shell are separate Flask servers — and a free image path bypasses
    the builder's `sandbox.root` boundary, an arbitrary-image-load risk on a shared SSH tunnel.)
    Reuse the existing **sandbox-bounded directory-browser modal** (not free text); Curate loads
    `<Image Source>/<plate_name>`; pre-fill from the active `run.json` `images_dir`. The builder
    keeps its own root unchanged.
11. **OQ3 — single winner**: write `deliverables/best_pipeline.json` for single AND multi-objective
    + a TODO for multi-obj Pareto disambiguation.
12. **OQ4 — graceful live + banner (rec A, degrade on failure)**: Monitor reads the live study when
    `optuna` is importable AND the storage URL **connects**; on a missing extra OR a failed
    connection it falls back to the finished `trials.parquet` + a note (`couldn't reach the live
    study — check network / ~/.pgpass`) and the "install the `tune` extra" banner. (Distributed-PG
    live monitoring needs the GUI host to reach the storage URL + resolve creds; sqlite `study.db`
    is always local.)
13. **OQ5 — spec location**: `TuneRunRoot.discover` reads `.pht-tune-cache/run.json` first, then
    `deliverables/tuning_spec.json`, then legacy root.
14. **OQ6 — run picker**: a standalone path input + the `.pht-tune-cache/run.json` marker; extend
    `gui/shell/_classifier.py::classify` to recognize a tune run by the marker (mirrors main's
    `.phenotypic/progress/manifest.json` process-only detection).
15. **OQ7 — writability**: catch `PermissionError` on the winner write → toast.
16. **OQ8 — Space config**: preserve the run's scorer/strategy/budget when a `tuning_spec.json`
    exists; default only when starting fresh from a `pipeline.json`.
17. **Self-review recs (minor)**: `.pht-tune-cache` uses read-fallback to the legacy root, NO active
    migration — **EXCEPT `split.json`**, where `read_split` checks **both** locations (new then
    legacy root) so a resume NEVER re-derives the held-out split (rec C, review-refined: a re-derived
    split is a reproducibility leak; a cold sampler restart from a missing `study.db` is harmless).
    **Curate allowed on a live/unfinished run** with an "in progress" note (rec D); overlays
    **render-on-demand + spinner**, no pre-warm (rec E).

**Signature corrections (plan-reviewer; bake into every task):**
- `compose_hub(sandbox, *, …) -> (dash.Dash, ToolSession)` is a **tuple** — mount the tune app inside it.
- mirror `results_viewer.create_app(output_root=None, *, url_prefix)`.
- overlay source is `builder/_image_renderer.to_overlay_png_bytes(...) -> bytes`; add a small
  array-returning helper (or decode) for `go.Image` (which takes the RGB array).
- A/B difference uses `phenotypic.tune._scoring._matching.match_iou_greedy(pred, gt, tau=0.5)` —
  there is **no** `_greedy_pair`, and it lives in `_scoring`, not `_evaluation`.
- `infer_search_space -> InferredSearchSpace` (`.knobs` + `.excluded`); drop `Excluded` → `SearchSpace`.
- `storage_url` is only on `OptunaConfig` → `getattr(strategy, "storage_url", None)`.
- `objective_directions` / `is_multi_objective` are private (`phenotypic.tune._multi_objective`).
- per-cell diff: `GridImage.grid.get_section_counts() -> pd.Series` (absent cells = zero count).
- `param_form` is **OperationInfo-only** — Space needs a custom `_knob_form(knob)` widget builder.
- clientside relayout needs a triggered-prop loop guard; the LRU needs an `RLock`; winner write atomic.

**Post-merge alignment (main → branch, merge `dd8a5d8f`):** main added `.phenotypic` machine-state
+ a GUI `classify()` that detects runs via `.phenotypic/progress/manifest.json`, plus `Focus*`
enhancer renames (reconciled). `.pht-tune-cache` mirrors the `.phenotypic` pattern (sibling hidden
dir + a marker the classifier reads); the tune cache helpers join the `phenotypic_cache_*` family
in `_io_constants`. **`.pht-tune-cache` is a tune-run prerequisite — built in Chunk 0 below.**

---

## File structure

**New module `src/phenotypic/gui/tune/`** (mirrors `results_viewer/`):

| File | Responsibility |
|------|----------------|
| `__init__.py` | re-export `create_app` |
| `_app.py` | `create_app(root=None, *, url_prefix=MOUNT_HOME) -> dash.Dash` factory |
| `_ids.py` | all component-id constants, `tune-` prefixed |
| `_run_root.py` | `TuneRunRoot` — discover/validate a tune output dir; live-study handle |
| `_study_read.py` | pure read helpers: load trials, running-best, importance, gap, shortlist |
| `_layout.py` | `build_layout(...)`, the header + sub-tab shell + 4 sub-view containers |
| `_overlays.py` | overlay worker: `render_candidate_overlay`, `render_difference`, disk-LRU |
| `_command.py` | `render_launch_command(opts) -> str` (Python mirror of the clientside JS) |
| `_callbacks.py` | `register_callbacks(app, ...)`: sub-tab switch, poll, pin A/B, write winner/spec |
| `_assets/tune.css` | tab-local styling (tokens only; no `:root`) |
| `_assets/tune_sync.js` | clientside linked-relayout for the A/B figures |

**Seam edits (orchestrator-owned):**
- `gui/_config.py`: `MOUNT_TUNE = "/tune/"`, `TITLE_TUNE`.
- `gui/shell/_ids.py`: `SHELL_TAB_TUNE`.
- `gui/shell/_layout.py`: `_TAB_HREFS` / `_TAB_LABELS` / `TAB_DISPLAY_ORDER` (after Run).
- `gui/shell/_app.py` (`compose_hub`): build + `wrap_in_chrome` + mount in the dispatcher dict.
- `gui/FEATURES.md`, `gui/WORKFLOWS.md`, `scripts/capture_gui_tutorial_screenshots.py`,
  `docs/source/tutorials/gui/<n>_tune_copilot.md`.

**Tests:** `tests/unit/gui/tune/` (logic units), `tests/integration/gui/test_tune_*.py` (Dash
test-client), `tests/e2e/gui/test_tune_*.py` (Playwright; `@pytest.mark.ci_flaky` if DOM-poll).

> **Confirm-at-task signatures** (the implementer must open these and adapt the call — do not
> trust this plan's remembered signature): `results_viewer/_app.py::create_app`,
> `builder/_image_renderer` overlay-PNG helper, `_param_forms.param_form`,
> `run_console` `dcc.Interval` wiring, `shell/_app.py::compose_hub` mount block.

---

## Chunk 0 — backend prerequisites (`.pht-tune-cache` marker + classifier)

Two backend enablers the GUI depends on; land them FIRST (TDD, per-task commits). They mirror
main's `.phenotypic` machine-state pattern (merge `dd8a5d8f`). *(The Image Source picker is NOT
here — it's Tune-Curate-only, built in Chunk B.)*

### Task 0.1: `.pht-tune-cache/` path helpers + `run.json` marker

**Files:** `tools_/_io_constants.py` (+ `tools_/__init__.py` re-exports); `tune/_tune_cli/_run.py`,
`tune/_evaluation/_split.py`, `tune/_study/_optuna_store.py` callers; Test
`tests/unit/tools_/test_io_constants.py`, `tests/unit/tune/test_run_marker.py`.

Add, beside main's `phenotypic_cache_*`: `tune_cache_dir(out) -> out/.pht-tune-cache`,
`tune_cache_run_marker_path -> …/run.json`, `tune_cache_study_db_path`, `tune_cache_splits_dir`,
`tune_cache_split_assignment_path`. `run_tuning` writes `run.json` at run START (right after the
`deliverables/` mkdir, BEFORE the engine/slurm branch — so it marks a live run before any
deliverable exists): `{version, study_name="tune", storage_url, images_dir (from -i), strategy,
n_trials, is_multi_objective, slurm, start_time}`. **CRITICAL (review):** **resolve `storage_url`
FIRST** with the same 3-way fallback `_submit_slurm_fleet` uses (`storage_url or
$PHENOTYPIC_TUNE_STORAGE_URL or sqlite:///<cache>/study.db`) — for an env-var-driven SLURM run the
`run_tuning` param is `None`, and a null URL in `run.json` would silently force Monitor into
parquet-only mode for exactly the distributed-PG case. (`study_name` is the constant `"tune"`;
`start_time` is stamped in Python, not a workflow script.)

**Move into `.pht-tune-cache/`:** `study.db`(+`-wal`) and `splits/split.json` — update
`_open_store`, `_submit_slurm_fleet`, and `resolve_split`/`read_split`/`write_split`. **`study.db`:**
read-fallback to the legacy root, no migration (a cold sampler restart is harmless). **`split.json`
(review-critical):** `read_split` MUST check **both** locations — `.pht-tune-cache/splits/split.json`
then the legacy `splits/split.json` — because a missing split silently re-derives a fresh held-out
partition on resume (a reproducibility leak). Mirror `resolve_progress_dir`'s both-location pattern.
**Keep `trials.parquet` at the output root** (dual-purpose resume + user journal). `deliverables/`
stays entirely user-facing.

- [ ] **Update the ~6 existing tests** that assert root locations (`test_optuna_study_store.py::
  test_study_db_path_resolves_to_output_root`, `test_robust_eval_io_constants.py` split path,
  `test_tune_cli.py` study-db existence, …) to the new `.pht-tune-cache/` paths — the plan's TDD adds
  new tests AND fixes these.
- [ ] TDD: `run.json` exists at run start with the right keys + a **resolved non-null** storage_url;
  study/split paths resolve under `.pht-tune-cache/`; a resume from a **legacy-root** `split.json`
  reuses the original split (no re-derive). Re-run the live PG + SLURM smokes.

### Task 0.2: tune-run detection in the GUI classifier

**Files:** `gui/shell/_classifier.py`; Test `tests/unit/gui/shell/test_classifier_tune.py`.

Extend `classify(path) -> Capabilities` with an `is_tune_output` flag set when
`tune_cache_run_marker_path(path).is_file()` (mirrors the `resolve_manifest_json_path(...).is_file()`
process-only detection at `_classifier.py:265`). v1 only needs the capability; the sidebar
"Open in Tune" affordance can follow.

- [ ] TDD: a dir with `.pht-tune-cache/run.json` classifies `is_tune_output=True`; a plain dir / a
  forward-run dir does not.

> **Image Source (was Task 0.3) — moved to Chunk B.** Per the review, a hub-shared / builder
> Image Source is broken (separate Flask servers) + a sandbox-bypass risk. It's now **Tune-Curate-only**
> (Task B-IMG): a sandbox-bounded directory-browser modal in the Curate view, pre-filled from
> `run.json` `images_dir`. No shell/builder change. Chunk 0 is the two backend tasks above only.

> **Chunk 0 gate:** `pytest tests/unit/tools_ tests/unit/tune tests/unit/gui/shell -n 8` + `mypy`
> + `ruff` green; the live PG + SLURM smokes still pass with the relocated study/splits. **This
> chunk touches the tune backend + the shell classifier — review + push before starting Chunk A.**

---

## Chunk A — mount + run-root + Monitor (ships on the existing backend)

### Task A1: `MOUNT_TUNE` + title constants

**Files:** Modify `src/phenotypic/gui/_config.py`; Test `tests/unit/gui/test_tune_config.py`.

- [ ] **Step 1: failing test**
```python
def test_mount_tune_constant_and_title():
    from phenotypic.gui._config import MOUNT_TUNE, TITLE_TUNE
    assert MOUNT_TUNE == "/tune/"
    assert "Tune" in TITLE_TUNE
```
- [ ] **Step 2:** `uv run pytest tests/unit/gui/test_tune_config.py -v` → FAIL (ImportError).
- [ ] **Step 3:** add `MOUNT_TUNE: str = "/tune/"` and `TITLE_TUNE: str = "PhenoTypic Tune Co-Pilot"`
  next to the other `MOUNT_*` / `TITLE_*` constants.
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** `git add` both files; commit `gui(tune): add MOUNT_TUNE + TITLE_TUNE constants`.

### Task A2: `TuneRunRoot.discover` — validate a tune output dir

**Files:** Create `gui/tune/_run_root.py`, `gui/tune/__init__.py`; Test
`tests/unit/gui/tune/test_run_root.py`.

`TuneRunRoot` is a frozen dataclass: `path`, `trials_path: Path | None`, `storage_url: str | None`,
`study_name: str`, `directions: list[str] | None`, `images_dir: Path | None`, `best_pipeline_path`.
`discover(path)` reads markers in order — **`.pht-tune-cache/run.json` first** (OQ5/OQ6: carries
`study_name`/`storage_url`/`images_dir`), then `deliverables/tuning_spec.json`, then the legacy
output root — and locates `trials.parquet`; raises `TuneRunRootError` if none of run.json / a study
URL / a trials parquet exists. **Signature notes:** read the URL via
`getattr(spec.strategy, "storage_url", None)` (only `OptunaConfig` has the field) and directions via
`phenotypic.tune._multi_objective.objective_directions` (a **private** import — not in
`phenotypic.tune.__all__`). `images_dir` pre-fills the Curate Image Source picker (OQ2, Tune-only).

- [ ] **Step 1: failing test** (build a fake run dir with a `tuning_spec.json` + `trials.parquet`):
```python
def test_discover_reads_storage_url_and_trials(tmp_path):
    from phenotypic.tune._spec import Budget, TuningSpec
    # ... build a minimal spec with an OptunaConfig(storage_url="sqlite:///x.db") ...
    out = tmp_path / "out"; (out / "deliverables").mkdir(parents=True)
    (out / "deliverables" / "tuning_spec.json").write_text(spec.model_dump_json())
    (out / "trials.parquet").write_bytes(b"")  # presence is enough for discover
    from phenotypic.gui.tune._run_root import TuneRunRoot
    root = TuneRunRoot.discover(out)
    assert root.storage_url == "sqlite:///x.db"
    assert root.study_name == "tune"
    assert root.trials_path == out / "trials.parquet"

def test_discover_raises_when_neither_study_nor_trials(tmp_path):
    import pytest
    from phenotypic.gui.tune._run_root import TuneRunRoot, TuneRunRootError
    with pytest.raises(TuneRunRootError):
        TuneRunRoot.discover(tmp_path)
```
- [ ] **Step 2:** run → FAIL. **Step 3:** implement `TuneRunRoot` + `discover` + `TuneRunRootError`,
  resolving paths via `phenotypic.tools_._io_constants` helpers (`trials_parquet_path`,
  `tuning_spec_path`, `best_pipeline_path`). **Step 4:** PASS. **Step 5:** commit
  `gui(tune): TuneRunRoot.discover — validate a tune output dir`.

### Task A3: pure study-read helpers

**Files:** Create `gui/tune/_study_read.py`; Test `tests/unit/gui/tune/test_study_read.py`.

Pure functions over a `StudyStore` (so they unit-test against a `JournalStudyStore` with no
GUI): `running_best(trials) -> list[float]`, `gap_badge(store) -> tuple[str, bool]`
(label + is_flagged at rel-dispersion > 0.15), `shortlist(store, k=5) -> list[Trial]`
(top-k ∪ pareto_front ∪ gap-flagged, de-duped, score-desc), `is_multi_objective(root) -> bool`.

- [ ] **Step 1: failing test** — build a `JournalStudyStore` with a handful of `Trial`s
  (varying score/gap; one gap > 0.15) and assert `running_best` is monotone non-decreasing,
  `gap_badge` flags the right one, and `shortlist` includes the gap-flagged trial + top scorers,
  deduped, length ≤ 5 + extras.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit
  `gui(tune): pure study-read helpers (running-best / gap / shortlist)`.

### Task A4: `create_app` factory + empty-state layout + mount

**Files:** Create `gui/tune/_app.py`, `gui/tune/_ids.py`, `gui/tune/_layout.py`,
`gui/tune/_assets/tune.css`; Modify `gui/shell/_ids.py`, `gui/shell/_layout.py`,
`gui/shell/_app.py`; Test `tests/integration/gui/test_tune_mount.py`.

`create_app(root=None, *, url_prefix=MOUNT_HOME) -> dash.Dash` — mirror `results_viewer`:
`inject_design_tokens(app)`, `requests_pathname_prefix=url_prefix`,
`routes_pathname_prefix=MOUNT_HOME`, `title=TITLE_TUNE`; `root is None` → empty-state layout
(prompt to pick a run). `build_layout` renders the header + the 4 sub-tab buttons + 4 view
containers (only the active shown).

- [ ] **Step 1: failing test**
```python
def test_create_app_empty_state_has_subtabs():
    from phenotypic.gui.tune import create_app
    app = create_app(root=None, url_prefix="/tune/")
    html = app.index_string  # tokens injected
    layout = str(app.layout)
    for tid in ("tune-subtab-monitor", "tune-subtab-curate",
                "tune-subtab-space", "tune-subtab-launch"):
        assert tid in layout

def test_hub_mounts_tune(tmp_path):
    from phenotypic.gui.shell._app import compose_hub
    from phenotypic.gui.shell._sandbox import SandboxRoot  # confirm the SandboxRoot path
    # compose_hub(sandbox, *, …) returns a (dash.Dash, ToolSession) TUPLE — not a bare app.
    # start_idle_thread=False or the test leaks a daemon thread (review).
    hub_app, _session = compose_hub(SandboxRoot(tmp_path), start_idle_thread=False)
    mounts = hub_app.server.wsgi_app.mounts          # DispatcherMiddleware.mounts dict
    assert any(m.rstrip("/") == "/tune" for m in mounts)  # adapt to existing-mount assertions
```
- [ ] **Step 2:** FAIL. **Step 3:** implement the factory + layout + ids; add `SHELL_TAB_TUNE`,
  the `_TAB_HREFS`/`_TAB_LABELS`/`TAB_DISPLAY_ORDER` entries, and the `compose_hub` mount
  (`wrap_in_chrome(tune_app, active_tab=SHELL_TAB_TUNE, ...)` + dispatcher dict entry). Copy the
  exact mount idiom from the builder/run lines. **Step 4:** PASS. **Step 5:** commit
  `gui(tune): create_app factory + empty-state layout + hub mount`.

### Task A5: sub-tab switch callback

**Files:** Create `gui/tune/_callbacks.py`; Modify `gui/tune/_app.py` (call `register_callbacks`);
Test `tests/integration/gui/test_tune_subtabs.py` (Dash test client or callback-context unit).

One callback: clicking a sub-tab button sets the active view + the active-button class. Keep it a
thin adapter around a pure `active_view(trigger_id) -> str` helper so the logic is unit-tested
without Dash.

- [ ] **Step 1: failing test** for `active_view("tune-subtab-curate") == "curate"` and that an
  unknown/None trigger defaults to `"monitor"`.
- [ ] **Step 2:** FAIL. **Step 3:** implement `active_view` + the Dash callback wiring (style/class
  swap). **Step 4:** PASS. **Step 5:** commit `gui(tune): sub-tab switching`.

### Task A6: Monitor — poll + figures

**Files:** Modify `gui/tune/_layout.py` (monitor view), `gui/tune/_callbacks.py`,
`gui/tune/_study_read.py`; add `dcc.Interval(id="tune-study-poll", interval=3000)`; Test
`tests/integration/gui/test_tune_monitor.py`.

The poll callback (Interval `n_intervals` + the run-root from a `dcc.Store`) re-reads the study →
updates: objective `go.Figure` (running-best line + raw scatter, Okabe-Ito colors via
`phenotypic.gui._design.OI_*`), importance bar figure, gap badge, trials `dash_table`. **OQ4 —
graceful live read (degrade on failure):** read the live study (`OptunaStudyStore`) only when optuna
is importable (`importlib.util.find_spec("optuna")`) **and the storage URL connects** — wrap the
open/read in `try/except` **with a short connect timeout** (`connect_timeout=3` in the psycopg URL,
or a thread-timeout around the `OptunaStudyStore` constructor) so an unreachable PG doesn't stall
the 3 s poll for ~30 s. A failed connection (login node can't reach the compute-node Postgres,
missing `~/.pgpass`, etc.) falls back to the finished `trials.parquet`
(`JournalStudyStore.from_parquet`) + a "couldn't reach the live study — check network / ~/.pgpass"
note. A missing extra shows the "install the `tune` extra for live monitoring" banner instead. The
optuna import stays inside the callback body so `import phenotypic.gui.tune` never drags optuna in
(the lazy-import lock). A still-running study renders normally (rec D — Curate stays usable on a live
run, just flagged "in progress"). Pareto card renders only when `is_multi_objective(root)`
(private `_multi_objective` import). Build each figure in a pure
`build_objective_figure(trials) -> go.Figure` etc. so they unit-test headless.

- [ ] **Step 1: failing tests** — `build_objective_figure(trials)` returns a `go.Figure` whose
  best-trace y is monotone; `build_importance_figure(importances)` has one bar per param;
  `monitor_pareto_visible(root)` is False for a single-objective root.
- [ ] **Step 2:** FAIL. **Step 3:** implement the figure builders (import OI palette from
  `_design`, font from `FONT_FAMILY_*`) + the Interval poll callback. **Step 4:** PASS.
- [ ] **Step 6 (gate):** `uv run pytest tests/unit/gui/tune tests/integration/gui/test_tune_* -n 8`
  · `uv run mypy src/phenotypic/gui/tune` · `uv run ruff check src/phenotypic/gui/tune` →
  green. **Commit** `gui(tune): Monitor — 3s poll + objective/importance/gap/trials`.

> **Chunk A green gate:** the tune tab mounts, the run picker loads a finished/live study, Monitor
> renders and auto-refreshes. Update `FEATURES.md` rows for the mount + poll (Task C-ledgers).
> **Post-A:** `feature-dev:code-reviewer` over the diff; apply high-confidence fixes; push.

---

## Chunk B — Curate + overlays (the heavy flow)

### Task B-IMG: Curate Image Source picker (sandbox-bounded — OQ2)

**Files:** `gui/tune/_layout.py`, `gui/tune/_callbacks.py`, `gui/tune/_ids.py`; Test
`tests/integration/gui/test_tune_image_source.py`.

A directory picker **in the Curate view** that sets the plate **Image Source** for overlay loading
(Tune-only; NOT shell/builder). Reuse the existing **sandbox-bounded directory-browser modal** (the
component the builder/run-console already use — it enforces the `sandbox.root` boundary so plate
loads can't escape the sandbox on a shared SSH tunnel); do **not** accept free-text paths. Pre-fill
from the active run's `run.json` `images_dir`. The overlay render (B1/B4) loads
`<Image Source>/<plate_name>` as a `GridImage`; when the Image Source is unset, Curate shows a
"point me at the plate images" prompt instead of overlays (the run dir holds no input images).

- [ ] TDD: selecting an Image Source updates the Curate plate-source store; an overlay request
  resolves `<Image Source>/<plate_name>`; an out-of-sandbox path is rejected by the modal; unset →
  the prompt (no overlay attempt).

### Task B1: overlay render (single candidate on a plate)

**Files:** Create `gui/tune/_overlays.py`; Test `tests/unit/gui/tune/test_overlays.py`.

`render_candidate_overlay(base_pipeline, params, plate_image, *, max_dim=640) -> np.ndarray`:
`build_pipeline(base, params)` → `pipeline.apply(plate)` → an RGB `label2rgb(objmap over detect_mat)`
overlay array for `go.Image`. **Signature note:** the builder's
`_image_renderer.to_overlay_png_bytes(...)` returns **PNG bytes**, not an array — factor a small
array-returning helper out of it (preferred; reuse the same `skimage.color.label2rgb` call) rather
than decoding the PNG round-trip. `go.Image` takes the array directly (no base64). Pure +
deterministic on `load_synth_yeast_plate()`.

- [ ] **Step 1: failing test**
```python
def test_render_candidate_overlay_returns_rgb(tmp_path):
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic import ImagePipeline
    from phenotypic.enhance import GaussianBlur
    from phenotypic.detect import OtsuDetector
    from phenotypic.gui.tune._overlays import render_candidate_overlay
    base = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])
    img = render_candidate_overlay(base, {"0.sigma": 2.0}, load_synth_yeast_plate())
    assert img.ndim == 3 and img.shape[2] in (3, 4)
```
- [ ] **Step 2:** FAIL. **Step 3:** implement (reuse the builder overlay helper — confirm its
  name/signature at `builder/_image_renderer.py`). **Step 4:** PASS. **Step 5:** commit
  `gui(tune): candidate overlay render (build_pipeline → apply → label2rgb)`.

### Task B2: difference overlay (A vs B)

**Files:** Modify `gui/tune/_overlays.py`; Test `tests/unit/gui/tune/test_difference.py`.

`difference_objects(objmap_a, objmap_b, *, tau=0.5) -> DiffResult` with `both`, `only_a`, `only_b`
object-id lists via **`phenotypic.tune._scoring._matching.match_iou_greedy(pred, gt, tau=0.5)`** —
it returns `list[MatchPair]` (`(pred_id, gt_id)`): both non-`None` → `both`, `(pred, None)` →
`only_a`, `(None, gt)` → `only_b`. **There is no `_greedy_pair` and it is NOT in `_evaluation`** —
it's `match_iou_greedy` in `_scoring`. `render_difference(plate, objmap_a, objmap_b) -> np.ndarray`
colors outlines both=grey / only-A=sky / only-B=orange (Okabe-Ito). `cell_disagreement(grid_a,
grid_b) -> int` compares `GridImage.grid.get_section_counts() -> pd.Series` (a section absent from a
series = zero count — handle the missing keys).

- [ ] **Step 1: failing test** — synthetic objmaps where A has an extra object and B merges two;
  assert `only_a`/`only_b`/`both` partition correctly and `render_difference` returns an RGB array.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit
  `gui(tune): A/B difference matching + colored render`.

### Task B3: overlay worker + disk-LRU cache

**Files:** Modify `gui/tune/_overlays.py`; Test `tests/unit/gui/tune/test_overlay_cache.py`.

`OverlayCache(cache_dir, capacity=64)`: key `(trial_number, plate_name, mode)`; `get_or_render`
runs the overlay on a background thread (`ThreadPoolExecutor`, name via
`_config.THREAD_NAME_PREFIX`) and memoizes the array to disk (`.npy`); evicts LRU past capacity.
**Guard the LRU/eviction dict with a `threading.RLock`** — concurrent Dash callback threads
(Werkzeug is multi-threaded) can race on insert/evict. Pure enough to test the cache hit/miss +
eviction with a stub render fn.

- [ ] **Step 1: failing test** — a counting stub render fn; assert second `get_or_render` of the
  same key does NOT re-invoke it (cache hit), and a third distinct key past capacity evicts the
  oldest (re-render on next access).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit
  `gui(tune): overlay disk-LRU cache + background render`.

### Task B4: Curate layout + shortlist + pin A/B + figures

**Files:** Modify `gui/tune/_layout.py`, `gui/tune/_callbacks.py`,
`gui/tune/_assets/tune_sync.js`; Test `tests/integration/gui/test_tune_curate.py`,
`tests/e2e/gui/test_tune_overlay_sync.py`.

Layout: shortlist cards (from `shortlist()`), the A/B segment (Side-by-side ↔ Difference), two
`dcc.Graph` (`go.Image`) for side-by-side + one for difference, the plate picker, the winner bar.
Pin callback: click a card → set A or B in a `dcc.Store`. Render callback: on pin/plate/mode
change → `OverlayCache.get_or_render` → set figure `data`. The **clientside** linked-relayout in
`tune_sync.js`: a `dash.clientside_callback` that mirrors `relayoutData` (xaxis/yaxis range)
between the A and B graphs — **with a triggered-prop guard** (only propagate from the graph the user
actually zoomed: check `dash_clientside.callback_context.triggered[0].prop_id`), or the A→B→A relay
loops infinitely. The render callback uses the **non-blocking** poll pattern, NOT `future.result()` (which would block
Werkzeug's thread pool). Concrete state machine: on pin/plate/mode change, submit the render and
stash the `Future` in a **module-level dict keyed on `(session_id, trial, plate, mode)`**, returning
a spinner/placeholder figure; a short `dcc.Interval` (the overlay-readiness poll) checks
`future.done()` and swaps in the figure once ready (the builder-preview cache is the precedent).
Overlays are **render-on-demand** (no pre-warming the shortlist — rec E) with a spinner/placeholder
figure until the future resolves; Curate stays usable on a **live, unfinished** run (rec D) — render
from whatever trials exist so far, with an "in progress" badge.

- [ ] **Step 1: failing tests** — integration (Dash test client): the Curate view exposes the
  shortlist card ids + the two graph ids; a pure `pinned_pair(clicks, store)` helper assigns
  A then B then re-pins. e2e (`@pytest.mark.ci_flaky` if it polls): zoom graph A → graph B’s
  xaxis range matches (assert via `browser_evaluate` on the figure layout).
- [ ] **Step 2:** FAIL. **Step 3:** implement layout + callbacks + the clientside sync JS.
  **Step 4:** PASS. **Step 5:** commit `gui(tune): Curate shortlist + A/B figures + linked zoom`.

### Task B5: difference toggle + write winner

**Files:** Modify `gui/tune/_callbacks.py`; Test `tests/integration/gui/test_tune_winner.py`.

Toggle swaps side-by-side ↔ difference view (pure `curate_mode(trigger)` helper). "Set as winner"
→ `write_winner(root, base, trial)`: `build_pipeline(base, winner.params).to_json()` written
**atomically** (temp file + `os.replace`) to `root.best_pipeline_path`
(`deliverables/best_pipeline.json`) for both single- AND multi-objective runs (OQ3 — single
override; `# TODO` multi-obj Pareto disambiguation). **OQ7:** catch `PermissionError` (HPCC
read-only output dirs) and surface it in the toast (the helper re-raises; the callback catches).

- [ ] **Step 1: failing test** — call the winner-write helper
  `write_winner(root, base, trial) -> Path` and assert `deliverables/best_pipeline.json` now
  contains the candidate’s params (round-trips via `ImagePipeline.from_json`).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS.
- [ ] **Step 6 (gate):** chunk-B suite `-n 8` + `mypy` + `ruff` green; commit
  `gui(tune): difference toggle + write best_pipeline.json`.

> **Post-B:** `feature-dev:code-reviewer`; apply fixes; push.

---

## Chunk C — Space + Launch + ledgers + screenshots

### Task C1: Launch command renderer (Python + clientside parity)

**Files:** Create `gui/tune/_command.py`; Modify `gui/tune/_layout.py`, `gui/tune/_assets/`;
Test `tests/unit/gui/tune/test_command.py`.

`render_launch_command(spec_path, input_dir, output_dir, *, strategy, n_trials, storage_url,
screen, slurm) -> str` builds the exact `python -m phenotypic.tune run …` string (only appends
`--storage-url` when set, `--screen`/`--slurm` when toggled). The Launch form drives a clientside
callback that mirrors this logic into the command card (the mockup’s JS is the reference); the
Python function is the unit-tested source of truth + powers the copy payload.

- [ ] **Step 1: failing tests** — assert the string for a Postgres tpe run includes
  `--strategy tpe --n-trials 50 --storage-url postgresql+psycopg://…`; a local grid run omits
  `--storage-url` and `--screen`/`--slurm`.
- [ ] **Step 2:** FAIL. **Step 3:** implement + the form layout + clientside callback. **Step 4:**
  PASS. **Step 5:** commit `gui(tune): Launch command renderer + live card`.

### Task C2: Space — infer + forms + export

**Files:** Modify `gui/tune/_layout.py`, `gui/tune/_callbacks.py`; Test
`tests/integration/gui/test_tune_space.py`.

Load a `pipeline.json` → `infer_search_space(pipeline)` → render flat + presence knobs; nested knobs
read-only/disabled. **Signature notes:** `infer_search_space -> InferredSearchSpace` (`.knobs` +
`.excluded`) — convert to `SearchSpace` by dropping the `Excluded` items before embedding. `param_form`
is **OperationInfo-only** — it cannot render a `Knob`; write a small **`_knob_form(knob) -> dbc.Row`**
that maps `FloatRange`→two numeric inputs (low/high) + a log toggle, `IntRange`→two int inputs,
`Categorical`→a checklist, plus a per-knob `tunable` toggle. **OQ8 — preserve config:** when a
`tuning_spec.json` already exists in the run dir, `space_to_spec` replaces **only** its `search_space`
(keeping the run's scorer/strategy/budget); only when starting fresh from a bare `pipeline.json` does
it default `QCScorer`/strategy/budget (+ a "review these in Launch" note). Keep
`space_to_spec(pipeline_or_spec, edits) -> TuningSpec` pure + unit-tested.

- [ ] **Step 1: failing test** — `space_to_spec` over a `load_synth_yeast_plate()`-runnable
  pipeline yields a `TuningSpec` whose `search_space.knobs` match the inferred flat/presence
  targets, and round-trips through `model_dump_json` → `model_validate_json`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit
  `gui(tune): Space — infer_search_space → forms → tuning_spec.json`.

### Task C3: FEATURES.md + WORKFLOWS.md + screenshots + tutorial

**Files:** Modify `gui/FEATURES.md`, `gui/WORKFLOWS.md`,
`scripts/capture_gui_tutorial_screenshots.py`; Create `docs/source/tutorials/gui/<n>_tune_copilot.md`;
Test `tests/unit/gui/test_features_md.py` / `test_workflows_md.py` (existing gates).

Add a `FEATURES.md` row per affordance (tab mount, run picker, study poll, sub-tab switch,
shortlist pin, A/B linked zoom, difference toggle, write winner, space export, launch card) with a
real `Test ref`. Add a `WORKFLOWS.md` row `tune_copilot` → `_capture_tune_copilot` + the tutorial
page. Add `_capture_tune_copilot(context, base_url)` booting a hermetic tune run over the
synthetic dataset (a tiny `run_tune_once` fixture) and screenshotting each sub-tab. **This fixture
is the HEAVIEST item in Chunk C** (review), not a minor task: it must produce a tune output dir with
`.pht-tune-cache/run.json` + `trials.parquet` (≥5 trials for the shortlist) + a study, AND plate
images reachable from the Curate Image Source, AND a successful `build_pipeline(...).apply(plate)`
for the overlays. Budget for it: reuse the synthetic `load_synth_yeast_plate()` plates on disk +
a short real grid/random run (no Optuna needed for the capture).

- [ ] **Step 1:** run the existing `scripts/check_workflows_md.py` / `check_features_md.py` (or
  their pre-commit hooks) → FAIL (round-trip gap). **Step 2:** add the rows + capture fn +
  tutorial page. **Step 3:** `uv run python scripts/capture_gui_tutorial_screenshots.py` →
  regenerate **ALL** PNGs; commit every changed PNG (do NOT cherry-pick the collateral churn).
  **Step 4:** the gates pass. **Step 5:** commit
  `gui(tune): FEATURES/WORKFLOWS rows + capture fn + tutorial + screenshots`.

> **Chunk C green gate:** full `gui` + `gui/tune` suites `-n 8` + `mypy src/phenotypic/gui/tune` +
> `ruff` + the FEATURES/WORKFLOWS round-trip gates + a clean
> `uv run python scripts/capture_gui_tutorial_screenshots.py`. **Post-C:** `feature-dev:code-reviewer`
> over the whole `gui/tune/` diff; apply fixes; **regression** (`tests/unit/gui tests/integration/gui`)
> `-n 8`; push. Phase 5 is then complete → proceed to the single end-of-feature simplify (#11).

---

## Cross-cutting locks & gates (hold after their owning chunk)

- **No re-optimize:** the tune sub-app never imports the engine’s run loop in a callback — it only
  reads stores + writes `best_pipeline.json` / `tuning_spec.json`. (Grep gate in review.)
- **Tokens only:** no hard-coded hex / font / size in `gui/tune/` — import from `_design` (the
  annotation-adherence / review check). `tune.css` declares no `:root`.
- **Lazy optuna:** importing `phenotypic.gui.tune` must NOT import optuna (the study read goes
  through the `StudyStore` protocol; Optuna stays behind the `tune` extra). Add an import-lock test
  mirroring `tests/unit/tune/test_lazy_import_lock.py`.
- **GUI ledgers:** every `gui/tune/` affordance in `FEATURES.md` with a real `Test ref`; the
  `tune_copilot` flow round-trips in `WORKFLOWS.md`; all PNGs regenerated + committed.
- **`-n 8` always; never `-n auto`.**

## Definition of done (Phase 5)

`phenotypic-gui` shows a Tune tab; pointing it at a tune output dir loads the study; Monitor
auto-refreshes a live study; Curate A/B-compares two candidates with linked zoom + a difference
view and writes `best_pipeline.json`; Space emits a `tuning_spec.json`; Launch renders the exact
CLI command. `gui` + `gui/tune` suites + `mypy` + `ruff` + FEATURES/WORKFLOWS/screenshot gates all
green; code-review + regression passed; pushed. (Then #11 simplify → #12 docs → #13 PR.)
