# Tune Engine — Phase 2: Optuna Backend (Structured Outline)

> **Status: OUTLINE.** A structured task map, not a full TDD plan. Before implementing,
> expand this into bite-sized TDD tasks (à la `phase-1a`…`phase-1d`), grounding each step
> against the live code + the `optuna` API. The Phase-1 seams this plugs into
> (`SearchStrategy`/`StrategyConfig`/`PruningChannel`/`StudyStore`/`Executor`) are stable.

**Goal:** Make tuning *sample-efficient* and *distributed* — swap the brute-force Grid/Random
strategies for an Optuna-backed `OptunaStrategy` (TPE/CMA-ES/NSGA-II), add ASHA pruning, a
persistent SQLite `study.db`, fANOVA importance, the two-round screening freeze, distributed
ask-and-tell, and the `SlurmExecutor` — all behind a `tune` extra (lazy import; Phase 1 keeps
working without it).

**Maps to:** `optuna-integration.md` (whole doc); `screening-importance.md` (fANOVA + freeze);
`engine-architecture.md` §4.1/§4.2/§7 (Strategy/Channel/Executor seams), §12 (phasing); master
§10 (dependency policy / extra), §8 (`study.db`, `screening/`, `splits/`).

**Depends on:** Phase 1 complete (the engine loop + the Protocol seams). **Adds the first new
third-party dep** (`optuna`), gated behind `pip install phenotypic[tune]`.

---

## Scope — what it adds / changes

| Phase-1 piece | Phase-2 change |
|---------------|----------------|
| `StrategyConfigUnion` (grid/random) | widen to include `OptunaConfig`; `TuningSpec.strategy` likely switches to `polymorphic_field(base=StrategyConfig)` (engine-arch §6) so the open set extends cleanly |
| `NoOpChannel` | add `OptunaPruningChannel` (ASHA via Optuna's pruner) — the `Evaluator` already calls `channel.report`/`should_prune` |
| `StudyStore` (parquet journal) | add an Optuna-backed store over `study.db` (SQLite WAL→RDB); the journal stays as the no-extra fallback |
| `compute_param_importance` (RF) | add fANOVA when Optuna is present; RF stays the fallback |
| `LocalExecutor` only | add `SlurmExecutor` (array scripts, drip-feed, event-log monitoring — reuse the forward CLI's SLURM machinery) |
| single-round loop | add the opt-in **two-round freeze** (explore → fANOVA → freeze low-importance knobs → focused round) |

## Key components (interfaces — bodies TBD in the full plan)

- **`OptunaStrategy`** (`_strategies/_optuna.py`, lazy `import optuna`) — implements the
  `SearchStrategy` Protocol via Optuna **ask-and-tell** (`study.ask()`/`study.tell()`).
  `suggest()` materializes only *active* conditional knobs (`suggest_categorical/int/float`,
  `log=` honored); returns `(params, OptunaPruningChannel(trial))`. `register_result` →
  `study.tell(trial, value, state)`. `is_exhausted` → budget-driven.
  - TPE default; CMA-ES native-fallback for all-continuous spaces; **NSGA-II auto-selected
    when the scorer is multi-objective** (Phase 4 hook).
- **`OptunaConfig`** (`StrategyConfig` subclass) — `kind="optuna"`, `sampler`, `n_trials`,
  `pruner`, `seed`; `build(space, store)` binds an `OptunaStrategy` to the store's study.
- **`OptunaPruningChannel`** (`PruningChannel`) — `report(value, step)` →
  `trial.report`; `should_prune()` → `trial.should_prune()` (ASHA via the configured pruner).
- **`OptunaStudyStore`** (`_study_store.py` sibling) — trials persisted in `study.db`;
  `best()`/Pareto from the study; supports **distributed** workers + Dash attach (shared WAL).
- **fANOVA** in `_screening.py` — `compute_param_importance` dispatches to fANOVA when the
  study is Optuna-backed; hierarchical per-activation importance over the conditional space.
- **`SlurmExecutor`** (`_execution/`) — the distributed half of the `Executor` Protocol.

## Task breakdown (high-level)

1. **`tune` extra + lazy import scaffolding** — `optuna` optional dep; a `_require_optuna()`
   guard with a clear install message; CI matrix job with the extra.
2. **`OptunaConfig` + `OptunaStrategy`** — ask-and-tell, conditional materialization, sampler
   selection; conformance to the `SearchStrategy` Protocol (the engine is untouched).
3. **`OptunaPruningChannel` + ASHA** — wire `Evaluator`'s multi-fidelity reporting (the
   per-image/per-fold loop reports intermediate scores) to the pruner.
4. **`OptunaStudyStore` + `study.db`** — SQLite WAL persistence, resume, Pareto/best.
5. **fANOVA importance** — dispatch + the `param_importance.json` upgrade.
6. **Two-round freeze** (screening §3) — explore (unpruned) → fANOVA → cumulative-tail freeze
   over total importance at top-k → focused (ASHA-pruned). Wrong-freeze recovery.
7. **`SlurmExecutor` + distributed** — per-worker strategy bound to the shared study;
   dead-worker/reproducibility-in-distribution policy (optuna §8).
8. **CLI** — `--strategy {tpe,cmaes}`, `--screen/--no-screen`, `--slurm`; `study.db`/`screening/`
   under `OUTPUT_DIR` (master §8 path helpers).

## Deferred / out of scope
- Multi-objective scorers + Pareto *reporting UI* → Phase 4 (the NSGA-II sampler hook lands
  here, the scorers + `pareto/` deliverables land there).
- Dash live-monitor of `study.db` → Phase 5 (this phase makes the shared study *exist*).

## Review findings (address at full-planning)

Opus plan-review (read-only, against live Phase-1 plans) flagged these — fix when expanding to TDD:

- **The `PruningChannel` is a no-op scaffold in Phase 1 — ASHA does NOT "drop in free."** Phase-1c's `Evaluator.evaluate` takes **no `channel` arg** and the 1d engine **discards** the channel from `suggest()`. Phase 2 must add `*, channel` to `Evaluator.evaluate`, thread it through the engine loop (`register_result(..., pruned=result.pruned)`), and add a `pruned` flag to `EvaluationResult`. Reframe the "zero Evaluator changes" claim.
- **ASHA needs a fidelity/rung ladder the CV-only MVP lacks** (it yields one score at the end, no per-plate intermediate). Building the rung ladder (report the running aggregate after each block of `step` plates) is a prerequisite — or ASHA is partly blocked on the deferred robust-eval §6–§8 multi-fidelity work. Decide if ASHA is genuinely Phase-2 scope.
- **`StudyStore` is a concrete class, not an interface.** Adding an Optuna-backed store needs an extracted `StudyStore` Protocol/ABC + re-typing `TuningEngine`; and the `len(store)`-replay resume is **meaningless for a stateful Optuna study** (Optuna resumes from `study.db`), so the engine needs an Optuna-aware resume branch.
- **`TuningSpec.strategy` widening is a prerequisite, not an open question** — recommend switching to `polymorphic_field(base=StrategyConfig)` (the registry plumbing already exists) before the `OptunaConfig` task; add a skip-if-no-extra round-trip test.
- **SQLite-WAL is unsafe on networked filesystems (NFS/Lustre)** — exactly where SLURM arrays put the shared `study.db`. Document Postgres/MySQL as the supported distributed backend; SQLite = local single-node only.
- **Lazy import:** `tune/__init__.py` must re-export `OptunaConfig` (registry needs it) **without** `import optuna` at module load — keep `import optuna` in method bodies / `_require_optuna()`, with a test asserting `phenotypic.tune` imports clean when the extra is absent. Pin an `optuna>=` floor.
- fANOVA dispatch needs the live Optuna study object (not the journal) — fold it into the `StudyStore`-interface decision (a polymorphic `param_importances()`).

## Open questions for the full plan
- `TuningSpec.strategy`: switch to `polymorphic_field(base=StrategyConfig)` now, or keep the
  union and add `OptunaConfig` to it? (engine-arch §6 prefers the polymorphic field for the
  open set; the union is simpler but closed.)
- Does the journal `StudyStore` stay the canonical record (with `study.db` as the live
  optimizer state), or does `study.db` become canonical when the extra is installed?
