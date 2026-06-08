# Tune Config Builder — Decisions & Open Questions (working log)

> **Status: brainstorming / mockup phase. The spec is NOT written yet.**
> This folder will hold the design spec as a *set* of documents (placement,
> search-space UX, deploy/monitor, data model) rather than one file. This log
> tracks decisions and open questions as we refine them in the interactive
> mockup (`mockups/tune-config-builder.html`).

## Context

The GUI already mounts a read-only `/tune/` co-pilot (Monitor / Curate / Space /
Launch). It can analyze a finished run but cannot **author** a tuning spec or
**deploy** a run. Goal: full `TuningSpec` authoring + deploy (local + SLURM),
folded into the existing `/tune/` surface.

## Decisions locked so far

| # | Decision | Notes |
|---|----------|-------|
| D1 | **Placement: evolve `/tune/`, not a separate top-level page.** | Grows the embryonic Space/Launch views instead of duplicating them; preserves the option to promote to its own mount later. |
| D2 | **In-tab IA = hamburger → Setup / Run / Monitor.** | Three verbs: author → deploy → inspect. Resolves the "bind to existing run" empty-state conflict. |
| D3 | **Setup = "what to tune & how to judge"** → pipeline + search space + **scorer**. | Scorer defines the experiment/objective, so it lives with the search space (moved back out of Run). |
| D4 | **Run = "how to run it"** → strategy + budget + advanced eval + compute target + deploy. | Strategy/budget are genuine per-launch knobs (the CLI already treats `--strategy`/`--n-trials` as overrides). |
| D5 | **Monitor = the existing read-only co-pilot**, plus an **Export-best-pipeline** result zone. | Closes the loop: best params applied to the base pipeline → ready-to-run `pipeline.json`. |
| D6 | **Progressive-disclosure single page over a wizard.** | Expert audience; editing an existing spec is first-class. (Wizard overlay deferred — see below.) |
| D7 | **Per-knob domain editor**: Range / Choices mode, optional **step**, "Sample across orders of magnitude" toggle. | Two routes to discreteness: uniform `step` vs arbitrary `Choices`. |
| D8 | **Add `FloatRange.step`** (quniform) as a model extension. | Side benefit: a stepped float becomes grid-enumerable, removing the hard "grid can't do float" error. |
| D9 | **Naming**: control reads "Sample across orders of magnitude"; chip tag `· by-magnitude`; "log scale" kept only as the tooltip alias. | Plainer than "log"; "scale-invariant" rejected as higher jargon. |
| D10 | **Empty state**: Search space + Scorer sections are folded, name-bar dimmed, and unselectable until a pipeline is chosen. | Pipeline is the gate; entry also via Builder hand-off or "Open spec…". |
| D11 | **Adopt the typed config-suffix naming** (merged from `origin/main`, implemented in `_io_constants.py`). | Tuning specs save/load as `tuning_spec.json.pht-tune`; exported tuned pipelines as `best_pipeline.json.pht-pipe`. Pickers show both typed + legacy `.json`; UI copy clarifies "JSON, type-tagged by suffix." Use `TUNING_SPEC_JSON`/`PIPELINE_JSON`/`BEST_PIPELINE_JSON` + `ensure_typed_json_suffix()`; never spell the suffix literals outside `_io_constants.py`. See `docs/superpowers/specs/2026-06-08-config-json-suffix-migration-design.md`. |
| D13 | **Spec-review resolutions** (plan-reviewer pass, 2026-06-08). | **Run is a new executing surface**, not a repurposing of the locked read-only `_launch.py` command-mirror — reuse `render_launch_command` for preview, keep `optuna` lazy (Q1). **Image source** for `-i <images>` comes from the merged shared source-image-root (`SHELL_SOURCE_IMAGE_ROOT_STORE` / `resolve_source_image_root`) with a per-run override (Q2). **Runner**: import `LocalRunner`/SLURM helper from `run_console/` in place — the move to `gui/_runner.py` is optional, not required (Q3). **Multi-objective Pareto export is IN scope**: Monitor shows a Pareto front + per-trial export to `pareto/best_<objective>.json.pht-pipe` (Q5). **SLURM monitoring polls when the store is reachable, else degrades** to a job-id/task-count card (Q6). **`FloatRange.values()`** uses `linspace`-style generation, not `arange` (Q7). **`phenotypic_version`** uses a `default_factory` (Q8). Add **`SANDBOX_TUNE_PRESETS_SUBDIR`** constant (Q10). Builder→Tune hand-off carries the **pipeline path via a shell-level store** (Q4). **Open spike (Q9):** confirm `_param_forms.param_form` renders non-`ImageOperation` scorer pydantic models, else add a scorer registry — resolve during planning. |
| D12 | **Spec save/load + versioning** (closes the save-location open thread). | **Two destinations, different purposes:** (a) explicit **Save/Load** manages a reuse **library** that defaults to **`.phenotypic-gui/presets/tune/`** — the sandbox preset convention already used by the builder/run-console (`SANDBOX_GUI_DIRNAME` + `SANDBOX_PRESETS_SUBDIR`), *not* `.phenotypic/` (CLI machine-state cache) or `.pht-tune-cache/` (per-run Optuna state) — with a **Browse…** file-picker escape hatch to any folder. (b) **Deploy** always auto-writes the run's own copy to `deliverables/tuning_spec.json.pht-tune` via `tuning_spec_path(output)` (reproducibility record, non-optional). **Version stamping:** add a top-level **`phenotypic_version`** to `TuningSpec` serialization (root has none today; only the embedded pipeline carries `version`) → compat warning when an old spec loads on a newer build. This is version *provenance*, not version *history* (multiple named specs in the library cover lightweight history for v1). |

## Deferred / non-goals (v1)

### Wizard overlay for first-timers — DEFERRED
The progressive single-page form (D6) is the v1 surface. A guided step-by-step
**wizard overlay** for first-time users (pipeline → space → scorer → strategy →
deploy, with back/next gating) is a **possible later add**, not v1. Rationale:
the audience is expert-leaning, `infer_search_space` means the biggest section
arrives pre-filled, and a wizard makes *editing an existing spec* slow. Revisit
if non-expert lab members become the primary users.

### Conditional nesting (define-by-run) — engine-supported, UI deferred
**What it is:** a knob is only sampled when a *parent* knob holds a specific
value — the classic Optuna define-by-run pattern. Modeled as
`Knob.conditional_on: tuple[(KnobTarget, value), ...]`. Example: only tune
`refiner.min_size` when the refiner's presence knob is `enabled`; only tune a
TPE-specific sub-parameter when `sampler == tpe`.

**Engine status:** **fully wired** in all three strategies via
`Knob.is_active(chosen)` — grid filters inactive knobs before enumeration,
random filters before sampling, and Optuna's `_materialize()` checks
`is_active` before each `suggest_*`. So the runtime can already honor
conditional knobs.

**Why the UI is deferred:** inference (`_infer.py`) always emits
`conditional_on = None` in v1 — the presence-opt-in path (`_tune_optional`) is
off, so *nothing populates conditional links automatically*. Exposing a UI to
hand-author parent→child gates is a real authoring surface (a small dependency
graph) with low payoff until inference produces them. **v1 = independent
per-knob domains.** Presence knobs (`__enabled__`) are shown, but their
would-be conditional children are flagged "advanced, v1-deferred" in the
domain editor.

**Future:** when presence-opt-in inference lands, the presence row would reveal
auto-gated child knobs (indentation / "depends on `enabled`" chips); no new
engine work needed, only the authoring UI.

### Relational constraints between knobs — not supported, validation only
**What it is:** constraints *across* knobs, e.g. `min_area < max_area`, or
`a + b <= N`. Optuna can express these via a sampler `constraints_func` (returns
per-trial feasibility; the sampler then steers away from / penalizes infeasible
trials).

**Engine status:** **not implemented anywhere** — no `constraints_func` wired,
no cross-knob validator, no apply-time enforcement.

**UX implication:** the GUI can only do **client-side validation** — red-flag
`min_area ≥ max_area` before deploy and block launch. It **cannot** push a true
constraint into the search, so the optimizer won't *avoid* the infeasible
region; it would just waste or fail those trials. v1 therefore treats
optimizer-level relational constraints as an **explicit non-goal**, providing
pre-launch validation only.

**Future:** if needed, wire Optuna's `constraints_func` plus a constraint-builder
UI (pick knobs + relation). Significant; revisit on demand.

## Open UX questions still being explored in the mockup

- **#1 Export tuned pipeline** — ✅ *prototyped* (Monitor "Best so far" card → Export / Open in Builder / Send to Run Console; writes `best_pipeline.json.pht-pipe`).
- **#2 Entry / empty / open-spec** — ✅ *prototyped* (locked sections, pipeline gate, Builder hand-off + "Open spec…").
- **#3 Blocked / invalid Deploy state** — ✅ *prototyped* (missing-metadata-CSV, zero-knobs, low≥high bounds, grid+float; red section badges + inline field errors + aggregated footer that disables Continue/Deploy).
- **#4 SLURM vs Local divergence after launch** — ✅ *prototyped* (run-switcher swaps Local log-tail vs SLURM array-task card + study-store polling note; Deploy honors compute target).
- **#5 Cancel / stop a run** — ✅ *prototyped* (✕ on run pills → mode-aware confirm dialog: SIGTERM vs scancel, trials kept + resumable; live count decrements).
- **#6 Knob table at real scale** — ✅ *prototyped* (filter box + needs-review-only + count; Re-infer with manual-edit-preservation note; bulk actions).
- **Save / spec management** — ✅ *resolved* by D11 (names) + D12 (library location, Browse picker, version stamping, Deploy-writes-run-copy).

**All six identified open UX questions are prototyped, and the save/location/versioning thread is closed (D12). No open UX threads remain — ready to write the spec set.**
