# Tune Engine — Phase 5: Dash Co-Pilot (Structured Outline)

> **Status: OUTLINE.** A structured task map, not a full TDD plan. Expand into bite-sized TDD
> tasks before implementing. **GUI work is CI-gated** — every task must honor the `FEATURES.md`
> / `WORKFLOWS.md` ledgers + the screenshot capture (see project `CLAUDE.md` → "Adding GUI
> features"; `gui/CLAUDE.md`).

**Goal:** The interactive, human-in-the-loop surface — a `/tune/` view in the GUI hub that
**attaches to a shared `study.db`** to monitor a run, curate the winner (with detection
overlays in view — the proxy proves *plausible+reproducible*, not *correct*; the human catches
that gap), and edit the search space. The GUI **never reimplements the optimizer** — it
delegates launching to the CLI runner.

**Maps to:** `dash-copilot-design.md` (whole doc); `e2e-workflows.md` §3 (the frontend flow);
master §6 (drivers), D5/D6.

**Depends on:** Phase 2 (the shared `study.db` to attach to), Phase 3 (`infer_search_space` +
`_param_forms` for 6c), Phase 4 (Pareto data for the front view). Reuses the existing GUI hub
(`gui/`), the builder's detection renderer, and `run_console`'s `LocalRunner`.

---

## Scope — sub-phased 6a → 6b → 6c (dash-copilot §5)

| Sub-phase | Surface | Core |
|-----------|---------|------|
| **6a · Monitor** | live read of `study.db` (WAL, `dcc.Interval`) | objective-vs-trial curve, Pareto front (knee highlighted), importance bars (fANOVA/RF method badge), run status + budget + the **generalization-gap flag** surfaced loudly |
| **6b · Curate** | the shortlist + winner pick (the co-pilot's whole point) | top-N/Pareto **+** gap-flagged + anti-gaming-suspicious trials; on-demand detection overlays (reuse builder renderer) + per-image scores; accept/reject/rank+notes → Optuna `user_attrs` (last-write-wins + attribution); pick winner → `best_pipeline.json` |
| **6c · Space-edit** | load `pipeline.json` → `InferredSearchSpace` as forms | reuse `_param_forms`; domain editors, provenance badges, `⚠ needs_review` flags, the excluded "add a `TuneSpec`" hints → emit `tuning_spec.json` |

Launch: **"Launch tuning run" delegates to `run_console`'s `LocalRunner`** (spawns the *same*
`python -m phenotypic.tune tuning_spec.json` and tails it). Co-drive: because the view
*attaches* to a `study.db`, a human can curate a run an agent/SLURM job is writing right now.

## Key components (interfaces — bodies TBD)

- A `/tune/` Dash app mounted in the hub (`DispatcherMiddleware`, like builder/results/console).
- `study.db` read adapter (WAL-safe polling; trials/scores/`user_attrs`/Pareto).
- Visuals: objective curve, Pareto+knee, importance bars (method badge), gap flag.
- Shortlist + candidate detail (reuse the builder's overlay renderer + per-image score table).
- Curation write-back to `user_attrs` (attribution, last-write-wins) → `best_pipeline.json`.
- 6c space-form (reuse `gui/.../_param_forms`) → `tuning_spec.json`.
- Launch button → `LocalRunner` (reuse, do not reimplement).

## Task breakdown (high-level)

1. **App skeleton + hub mount + `study.db` read adapter** (6a foundation).
2. **6a monitor visuals** (curve / Pareto / importance / status + gap flag).
3. **6b shortlist + candidate overlays + per-image scores**.
4. **6b curation write-back + winner → `best_pipeline.json`**.
5. **6c space-edit form → `tuning_spec.json`** (reuse `_param_forms`).
6. **Launch delegation to `LocalRunner`**.
7. **Gates per task:** update `gui/FEATURES.md` (every affordance) + `gui/WORKFLOWS.md` (each
   end-to-end flow) + add `_capture_<id>` in `scripts/capture_gui_tutorial_screenshots.py` +
   a walkthrough page under `docs/source/tutorials/gui/`; regenerate + commit **all** PNGs.

## Deferred / out of scope
- MCP co-drive of the same study → deferred MCP (the attach-model already supports it).
- In-GUI optimizer logic → never (delegation only; there is exactly one optimizer).

## Review findings (address at full-planning)

Opus plan-review (seams verified against live `gui/`) flagged these — fix when expanding to TDD:

- **The candidate→overlay data path is the heaviest unknown.** Rendering overlays needs reconstructing each candidate's pipeline from its trial params + running it on N calibration plates (seconds–minutes) — a dedicated task must decide sync-in-callback vs. background compute, where the calibration plates come from (the run's `splits/`?), and the cache key/eviction. Reuse **`to_overlay_png_bytes(image, max_dim=...)` directly**, not the category-routed `render_node_preview` (which only overlays for `Detector`/`Refiner` classes).
- **CI gate specifics:** each `_capture_<id>` must be **defined AND dispatched** (added to the dispatch block in `scripts/capture_gui_tutorial_screenshots.py`) or `workflows-md-gate` fails; a `✅ shipping` row needs committed PNGs + a tutorial page + a real `Test ref` (write the test *before* the ledger row). A `🔭 planned` row can ship 6a without screenshots, flipping to `✅` when the capture lands.
- **Enumerate the registration sites** (≈5): a `SHELL_TAB_TUNE` constant + the three `shell/_layout.py` dicts (`TAB_DISPLAY_ORDER`/`_TAB_HREFS`/`_TAB_LABELS`) + the `compose_hub` `DispatcherMiddleware` mount dict, plus a `tune/_ids.py`. None are in the current task list.
- **WAL-reader contract:** open the study **read-only** for monitoring (`?mode=ro`/`busy_timeout`); SQLite-WAL on a networked FS (SLURM multi-node) is unsafe → document "SLURM/NFS = monitor-only (no `user_attrs` write-back); local single-node = full write-back."
- **The Phase-4 dependency is conditional** — only the Pareto panel + Pareto-mode shortlist need Phase 4; **single-objective 6a/6b can ship on Phase 2 alone** (feature-flag the Pareto pieces). 6c genuinely needs Phase 3.
- Resolve all study/deliverables paths via the `phenotypic.tools_` helpers (never hand-join); define `best_pipeline.json` write precedence between the human pick (6b) and the CLI auto-winner.

## Open questions for the full plan
- Shortlist ranking weights (top-N vs Pareto vs flagged) — dash-copilot decision B; confirm the
  default mix.
- Does 6c's space-edit support nested-op knobs (Phase 3) in v1, or flat + presence only?
