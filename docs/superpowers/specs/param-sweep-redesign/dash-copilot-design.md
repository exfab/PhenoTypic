# Dash Tuning Co-pilot (the `/tune/` view)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
Deep dive on **master §6** and **decision D5**: the interactive, human-in-the-loop Dash
surface for the tuning engine — monitor a run, review candidates with overlays, curate
the winner, and visualise the search — mounted in the GUI hub alongside
builder/results/run.

- **Status:** Design settled (pre-implementation). **Phase 6** (the last phase; a GUI
  surface, so it trips the `FEATURES.md` / `WORKFLOWS.md` / screenshot CI gates).
- **Maps to:** master §6 (drivers, shared study), D5 (Dash co-pilot), D6 (shared study),
  §8 (outputs). Consumes
  [`search-space-inference.md`](search-space-inference.md) §8 (the `InferredSearchSpace`
  proposal), [`robust-evaluation.md`](robust-evaluation.md) §8–9 (the gap flag, the
  Pareto axes), [`screening-importance.md`](screening-importance.md) §7
  (`param_importance.json`), [`qc-objective-mapping.md`](qc-objective-mapping.md) §5/§7
  (anti-gaming-suspicious trials, the proxy limitation),
  [`optuna-integration.md`](optuna-integration.md) §7 (the shared `study.db`, `user_attrs`,
  WAL), and [`reference-free-segmentation-metrics.md`](reference-free-segmentation-metrics.md)
  §E (the meta-validation gate). Follows the GUI conventions in
  `src/phenotypic/gui/CLAUDE.md`.

---

## 1. Purpose and where it fits

The `/tune/` view is **not** another headless optimizer — it is the **interactive
surface** a human uses to *steer and curate* a tuning run that the engine (CLI/SLURM/MCP)
drives. Its three jobs: **(a)** monitor the run + visualise the search; **(b)** review the
optimizer's surfaced candidates with **detection overlays** + per-image scores and **curate
the winner**; **(c, optional)** review/edit the inferred search space before launch. The
napari `gui/sweep/` viewer stays for power-user exploration of raw grid outputs but is
**not** a tuning driver (master D5).

Its reason to exist: the `QCScorer` is a **proxy** (qc §7 — *plausible + reproducible ≠
correct*). A human looking at overlays catches the proxy's blind spot, so the co-pilot's
core value is **human eyes on the candidates the machine ranked highest (and the ones it
flagged as suspicious)**.

---

## 2. What master §6 / D5 lock (documented, not re-litigated)

A new **`/tune/` Dash mount** (sibling to builder/results/run, **not** napari); shows
suggested candidates with **overlays + per-image scores**; **rank/accept writes back to
the study**; visualises **objective-vs-trial, the Pareto front, and importance bars**; the
napari sweep viewer stays for raw-grid power-users; it trips the **`FEATURES.md` /
`WORKFLOWS.md` / screenshot** gates; **its own GUI phase**.

---

## 3. The view in the hub

Follows the established hub pattern (gui/CLAUDE.md): a new Dash sub-app under
`src/phenotypic/gui/tune/`, mounted by `compose_hub` via `DispatcherMiddleware`.

- **`MOUNT_TUNE = "/tune/"`** in `_config.py`; registered in `shell/_app.py`; added to
  `TAB_DISPLAY_ORDER` (Home → Pipelines → Run → **Tune** → Viewer → Analysis) and
  `_TAB_HREFS`.
- Sub-app factory uses `requests_pathname_prefix=url_prefix,
  routes_pathname_prefix=MOUNT_HOME` (prefix-stripping under the dispatcher); calls
  `inject_design_tokens(app)`.
- Shared handles via `app.server.config`: **`CFG_TUNE_STUDY`** (the resolved `study.db`
  path / Optuna study handle) and reuse of **`CFG_RUNNER`** (the process-wide
  `LocalRunner`, for delegated launch). New `CFG_*` go in `_config.py`.
- **Design tokens:** UI chrome uses `COLOR_*`; **Plotly data series use `OI_*`** in the
  fixed order (navy, orange, sky, green, blue, purple; vermilion = error/flag) and
  `FONT_FAMILY_*` (DESIGN.md). Never hardcode a hex or font literal.
- Tool-internal component IDs live in `tune/_ids.py`; cross-tool path-store IDs in
  `shell/_ids.py`.

---

## 4. Driver relationship — attach + delegate-launch

The view **attaches** (read-mostly) to a `study.db` and provides the interactive surface;
it does **not** run the ask-and-tell loop itself.

- **Launch** delegates to the existing `run_console` `LocalRunner`: the "Launch tuning
  run" action spawns `python -m phenotypic.tune <tuning_spec.json> <input_dir> [opts]` (a
  CLI process the run console already manages + tails). No second optimizer driver lives
  in the Dash process.
- **Attach** opens an existing `study.db` (one being driven by a local run, a SLURM array,
  or an MCP/agent session) for read + write-back — **this is the D6 shared study**, so the
  human can co-curate an agent's overnight run.

---

## 5. Panels (sub-phased)

### 6a — Monitor + visualizations (read-only; lowest risk, ships first)

Live, read-only views over `study.db` + `param_importance.json`:

- **Objective-vs-trial** curve (best-so-far + per-trial), with pruned trials marked.
- **Pareto front** (multi-objective) — the level-vs-dispersion / quality-vs-runtime axes
  (robust-eval §9); knee-point highlighted.
- **Importance bars** (screening §7) — the two-tier conditional importance (top-level +
  per-group; *not* cross-ranked), with the method badge (fANOVA vs RF-permutation
  fallback) and `insufficient-data` markers.
- **Run status** — trial counts (completed/pruned/failed/running), budget progress, the
  **generalization-gap flag** (robust-eval §8) surfaced loudly.

### 6b — Candidate review + write-back (the curation co-pilot)

- **Shortlist** — the **top-N by objective** (single-obj) or the **Pareto front**
  (multi-obj), **plus** the **generalization-gap-flagged** (robust-eval §8) and
  **anti-gaming-suspicious** (qc §5) trials — so the human reviews the winners *and* the
  trials most worth a skeptical eye. `N` configurable. (Diversity-aware dedup is a noted
  future refinement.)
- **Per-candidate overlays** — rendered **on demand** when the human opens a candidate, on
  a few calibration plates, **reusing the builder's `render_node_preview`** (`label2rgb`
  of `objmap` over `detect_mat` → PNG bytes) into a tune-session overlay cache (like
  `IntermediatesCache`); per-image scores shown alongside.
- **Write-back** — accept / reject / rank + notes, stored as Optuna **`user_attrs`** on the
  trial (§6); the **winner** pick writes `deliverables/best_pipeline.json`.

### 6c — Space review/edit (optional)

Reuse **`_param_forms`** / `_operation_registry` to render the `InferredSearchSpace`
proposal (search-space §8): per-knob **domain editors** (sliders for `Int`/`FloatRange`,
multi-select for `Categorical`, a toggle for presence `__enabled__`), **provenance badges**
(`tune_spec`/`bool`/`enum`/`bounded`/`unbounded_heuristic`/`presence_optin`), the
**`⚠ needs_review`** flag, the docstring `description`, and the **excluded list** (reasons
+ "add a `TuneSpec`" hints). Edits emit a **`tuning_spec.json`** that the Launch button
hands to the delegated `python -m phenotypic.tune` — closing the loop visually. This is a
*third* review surface alongside search-space §8's CLI `--auto-space` and MCP
`tune_infer_space`; the CLI path stays the headless default.

---

## 6. Human write-back semantics

The human signal is a **curation / winner-selection** signal, **not** the optimizer's
per-trial objective (the automated `Scorer` keeps driving — this refines master D5's
"objective signal" wording).

- **Storage:** Optuna `user_attrs` on the reviewed trial:
  `{"curation": {"verdict": "accept"|"reject", "rank": int|None, "notes": str,
  "author": str, "ts": <iso>}}`.
- **Conflict policy:** **last-write-wins with attribution** — the latest verdict is
  authoritative; `author` + `ts` record who/when. Human writes land on **completed** trials
  (which the engine never re-touches), so engine↔human conflict is nil; human↔human/agent
  is last-write-wins. No locking at this scale.
- **Winner:** the accepted/top-ranked trial's pipeline is written to
  `deliverables/best_pipeline.json` (drops straight into `python -m phenotypic`).
- **Meta-validation feed:** the human accept/reject is a small ground-truth-ish quality
  label, so it optionally feeds the **reference-free meta-validation gate** (reference-free
  §E) — directly addressing qc §7's "proxy ≠ correct" limitation.
- **Deferred power-mode:** human-*as*-Scorer (preference-based optimization driving the
  optimizer) is explicitly out of scope for v1.

---

## 7. Live-refresh & concurrency

- **Polling**, not push (idiomatic Dash; matches `run_console`'s log-tail): a `dcc.Interval`
  re-reads the study on a cadence + a manual refresh button. Reads are **incremental** (new
  trials since the last poll) so a large study doesn't re-load each tick.
- **Concurrency:** the study is SQLite in **WAL** mode (optuna §7), so the GUI's reads run
  concurrently with the engine/CLI/SLURM/MCP writers; the GUI's only writes are the
  `user_attrs` curation signal.
- **State-aware:** the view handles **no study yet** (prompt to configure/launch), a
  **running** study (live), and a **completed** study (final review).

---

## 8. CI obligations, phasing & reuse

- **`FEATURES.md`** — every affordance gets a row (the Tune tab; each visualization; the
  shortlist; the overlay viewer; accept/reject/rank; the winner action; the space-edit
  form; the launch button) with a `Test ref`. The `gui-checks` `features-md-gate` rejects
  any `gui/` change without a `FEATURES.md` edit.
- **`WORKFLOWS.md`** — one end-to-end flow ("Tune a pipeline with the co-pilot"): a
  `docs/source/tutorials/gui/<NN>_tune_copilot.md` page, a `_capture_tune_copilot` in
  `scripts/capture_gui_tutorial_screenshots.py`, and the screenshot folder; the
  `workflows-md-gate` enforces the round-trip. Re-run the capture script + commit the
  refreshed PNGs (full set; commit-everything per root CLAUDE.md).
- **Phasing:** **6a** (read-only monitor/visualizations) → **6b** (candidate review +
  overlays + write-back) → **6c** (optional space-edit). Each sub-phase is independently
  shippable and adds its own `FEATURES.md` rows.
- **Reuse, don't re-spell:** `_config.py` for constants/mounts, `_design.py` for
  colors/type, `_param_forms`/`_operation_registry` for the space-edit form, the builder's
  `render_node_preview` for overlays, `run_console`'s `LocalRunner` for launch. The napari
  `gui/sweep/` viewer is untouched.

---

## 9. Testing

- **Unit (helpers, not raw callbacks).** Per the GUI memory-gotcha, **extract callback
  bodies into module-level helpers** and unit-test those: shortlist selection
  (top-N/Pareto + flagged), the `user_attrs` write-back payload (verdict/rank/author/ts),
  the `tuning_spec.json` emission from an edited space, the incremental study-read diff.
- **Integration.** Attach to a `study.db` fixture (a tiny finished study): visualizations
  render; the importance panel reads `param_importance.json`; the overlay cache renders a
  candidate via `render_node_preview`.
- **e2e (Playwright, live browser — required).** Drive the flow: launch (delegated runner
  stub) → monitor → open a candidate → overlays + scores → accept → winner →
  `best_pipeline.json` written. Callback wiring bugs only fire on a live
  `/_dash-update-component`, so a browser e2e is mandatory (GUI memory).
- **Gotchas honored:** import flask `request` **function-local**; reuse-curation closures
  take the state arg (no zero-arg lambdas); design-token + `FEATURES.md` `Test ref`
  validation.

---

## 10. Resolved choices / open questions

**Resolved:**

1. **Driver** — attach + write-back; launch delegated to `run_console`'s `LocalRunner`
   (`python -m phenotypic.tune tuning_spec.json`); D6 shared study.
2. **Write-back** — curation/winner-selection signal as `user_attrs` (verdict/rank/notes/
   author/ts), last-write-wins + attribution; winner → `best_pipeline.json`; feeds
   meta-validation; human-as-Scorer deferred.
3. **Overlays** — on-demand, reuse `render_node_preview` + a session cache, only for
   reviewed shortlist candidates.
4. **Shortlist** — top-N/Pareto + flagged (gap) + suspicious (anti-gaming); configurable
   `N`.
5. **Space-edit** — in-scope as optional 6c, emits `tuning_spec.json`; CLI `--auto-space`
   stays headless default.
6. **Refresh/concurrency** — `dcc.Interval` polling + incremental reads; SQLite WAL
   concurrent reads + `user_attr` writes; state-aware.
7. **Phasing** — 6a (visualizations) → 6b (review/write-back) → 6c (space-edit); FEATURES
   rows + one WORKFLOWS flow + screenshot round-trip.

**Still open (planning / GUI):**

- The polling cadence default + an incremental-read cap for very large studies.
- Diversity-aware shortlist dedup (a future refinement over top-N).
- The `Tune` tab's exact slot in `TAB_DISPLAY_ORDER` (after `Run`, before `Viewer` —
  confirm against the shell's workflow ordering at build time).
- Whether 6c's space-edit grows a live "what-if" preview (render a candidate from the
  edited space before launching) — deferred.
