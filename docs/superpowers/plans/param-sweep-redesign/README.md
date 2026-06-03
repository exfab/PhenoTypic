# Tune Engine — Implementation Plans

Phased implementation of the parameter-tuning engine (`src/phenotypic/tune/`). The
**design bundle** lives in [`../../specs/param-sweep-redesign/`](../../specs/param-sweep-redesign/)
— start with the master spec + [`engine-architecture.md`](../../specs/param-sweep-redesign/engine-architecture.md)
(the ABC/interface layer the plans build on).

> **Deprecation:** `tune` replaces `sweep` via a **hard cutover** — `sweep` is **deleted
> wholesale at the end of Phase 1** (master §9). The two are not coupled; they only share
> the extracted `_execution` module.

## Phase roadmap

| Phase | Scope | Plan | Status |
|-------|-------|------|--------|
| **0 · Prerequisites** | `polymorphic_field` factory · registry `+= tune` · `LocalExecutor` · capture the grid golden fixture | [`phase-0-prerequisites.md`](phase-0-prerequisites.md) | ✅ written |
| **1 · Engine core** *(no new deps)* | SearchSpace types · `Grid`/`Random` strategies · `Evaluator` (CV-only MVP) · Count-only `QCScorer` · RF-permutation importance · `TuningEngine` · `TuningSpec` · CLI (`-i/-o`); **deletes `sweep`** at the end | _planned_ | ⬜ |
| **2 · Optuna backend** *(`tune` extra)* | `OptunaStrategy` (TPE/CMA-ES/NSGA-II) · ASHA pruning · SQLite persist/resume · fANOVA · two-round freeze · distributed · `SlurmExecutor` | _planned_ | ⬜ |
| **3 · Auto-space + reference-free** | `infer_search_space` · `TuneSpec` markers · `--auto-space` · `ReferenceFreeScorer` + meta-validation gate | _planned_ | ⬜ |
| **4 · Supervised + multi-objective** | `SupervisedScorer` · `CompositeScorer` · Pareto reporting / `--multi-objective` | _planned_ | ⬜ |
| **5 · Dash co-pilot** | the `/tune/` view: 6a monitor → 6b curate → 6c space-edit; FEATURES/WORKFLOWS/screenshot gates | _planned_ | ⬜ |
| *Parallel* · **Operation annotations** | `Field(ge,le)` + `TuneSpec` on operation fields (decoupled workstream) | _planned_ | ⬜ |
| *Deferred* · **MCP** | the `tune_*` agentic surface (out of scope per the param-sweep focus) | — | 🚫 |

## Dependency notes

- **Phase 0 precedes Phase 1** — without `polymorphic_field` + the registry edit, `TuningSpec` can't round-trip.
- **`infer_search_space` is Phase 3**, not Phase 1 — Phase 1 hand-authors the space (Python-first → `tuning_spec.json`); `--auto-space` is the later convenience.
- **Phases 3–5 fan out from a working Phase 1–2**; the operation-annotations workstream is fully decoupled.

## Convention

Phase plans use plain `phase-N-<topic>.md` filenames (the master spec §12 owns the canonical
phase numbering). Each plan is self-contained TDD tasks; execute via
`superpowers:subagent-driven-development` (fresh subagent per task + review) or
`superpowers:executing-plans` (inline with checkpoints).
