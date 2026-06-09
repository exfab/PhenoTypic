# Tune Engine — Execution Entrypoint

> **Start here to implement the engine.** This is the runbook that turns the phase plans into
> executed code: the **implementation strategy** (how to dispatch + build) and the
> **verification strategy** (how to prove each phase is correct) in one place.
>
> **Read order:** this doc → [`README.md`](README.md) (phase roadmap) → the specific
> `phase-N-*.md` you're executing → its companion spec(s) under
> [`../../specs/param-sweep-redesign/`](../../specs/param-sweep-redesign/).

---

## 0. TL;DR

- **Goal of Phase 1 (the MVP milestone):** a runnable `python -m phenotypic.tune spec.json -i … -o …`
  with grid/random search, the Count-only `QCScorer`, robust CV evaluation, RF importance — and
  `sweep` deleted, its grid path locked to a golden fixture.
- **Build order:** `0 → 1a → 1b → 1c → 1d` **strictly sequential** (type + `__init__` dependencies),
  then `2`, `3`, `4`, `5` fan out; the **operation-annotations** workstream runs fully in parallel.
- **Cadence:** **one subagent per phase** (it runs the plan's TDD tasks internally, red→green→commit)
  → orchestrator review gate when it returns → **code-review agent** after the phase, plus an
  **annotation-adherence agent** at the model-introduction phases (§3.4). A **single simplify agent +
  full regression runs once, after the FINAL phase only**. Non-sequential phases run **in parallel,
  each in its own isolated worktree**.
- **Each phase is "done" only when its green gate passes** (§4.4) — never mark a task complete on a
  subagent's summary alone; read the diff and run the gate yourself.

---

## 1. Before you start

```bash
# In the redesign worktree (NOT the default checkout):
uv sync --group dev                     # full dev env (pytest, mypy, ruff)
uv run pytest -q tests/unit/tune        # expect: no tests yet (Phase 0 creates the package)
git switch -c <feature-branch>          # never implement on the default branch
```

- **`uv` is the sole runner** — never bare `python`/`pip` (project `CLAUDE.md`).
- The **design bundle is the source of truth**; the plans operationalize it. If a plan and a spec
  disagree, the spec wins — stop and reconcile before coding.
- Plans use **superpowers** skills. Execute each via
  `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`
  (inline). Finish a branch with `superpowers:finishing-a-development-branch`.

---

## 2. Dependency graph & execution order

```
        ┌──────────────────────── Phase 0 (prereqs) ────────────────────────┐
        │  polymorphic_field(base=) · registry += tune · LocalExecutor ·     │
        │  grid golden fixture (captured WHILE sweep still exists)           │
        └───────────────────────────────┬───────────────────────────────────┘
                                         │ (strict)
            1a search-space ─► 1b strategies ─► 1c scoring+eval ─► 1d engine+CLI+cutover
                                         │  (deletes sweep at the end; golden lock)
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
        Phase 2 (Optuna)          Phase 3 (auto-space +         Phase 4 (supervised +
        [needs 1]                 reference-free) [needs 1]      multi-obj) [needs 1 + 2]
            └──────────────┬───────────────┴──────────────┬─────────────┘
                           ▼                               ▼
                     Phase 5 (Dash co-pilot) [needs 2 + 3 + 4*]
                     (*Pareto pieces need 4; single-objective 6a/6b need only 2)

   ════ fully parallel, decoupled ════
   Operation-annotations workstream  [needs only the TuneSpec marker from Phase 3]
```

**Sequential vs parallel:**
- **0 → 1a → 1b → 1c → 1d is strictly ordered.** Each sub-phase imports types the previous defined,
  and `tune/__init__.py` is **cumulative** (1c writes the full file *including* 1b's exports — see
  the Phase-1c `__init__` note). Do **not** parallelize within Phase 1.
- **2 / 3 fan out from a working 1.** **4** also needs **2** (the NSGA-II sampler). **5** needs **2**
  (the shared `study.db`) + **3** (`_param_forms`) + **4** (Pareto data, for the Pareto pieces only).
- **Annotations** is decoupled — land it incrementally per operation any time after the Phase-3
  `TuneSpec` marker exists.

---

## 3. Implementation strategy

### 3.1 One subagent per phase; parallel non-sequential phases in isolated worktrees

Act as **orchestrator** — dispatch, review, integrate; implement inline only when a step is trivial
(< 30 lines, one file).

- **One subagent owns an entire phase** (a `phase-N-*.md` plan), not one task. It runs that plan's
  tasks in order within its own context (the TDD loop, §3.2) and returns when the phase's **green
  gate** (§4.4) passes. Brief it like a colleague: the plan file, the spec sections, the constraints,
  and *"stop and report if the green gate can't be reached — don't improvise past a blocker."*
- **The `0 → 1a → 1b → 1c → 1d` chain runs sequentially** — one agent per sub-phase, in order. Each
  imports the previous, and the cumulative `tune/__init__.py` + the cross-sub-phase type contract make
  it a strict chain, not a fan-out. (Optionally collapse 1a–1d into a single Phase-1 agent so the
  cumulative `__init__` + type consistency live in one context — at the cost of a larger working set.)
- **The `2 / 3 / 4 / 5` + annotations fan-out runs in DAG-respecting parallel waves**, each agent in
  its **own isolated git worktree** (`isolation: "worktree"` on the dispatch) so concurrent writes
  never share an index:
  - **Wave A (parallel):** Phase 2 · Phase 3 · annotations.
  - **Then Phase 4** (needs Phase 2's NSGA-II sampler).
  - **Then Phase 5** (needs 2 + 3 + 4*; the single-objective 6a/6b can start once Phase 2 lands).
- **The orchestrator owns the shared seam files** — `tune/__init__.py` exports, `pyproject.toml`
  (`[tune]` extra), the `TuningSpec.strategy` widening, the `EvaluationResult`/`Trial` multi-objective
  widening, the registry. Parallel agents own their **disjoint new** modules only; the orchestrator
  **merges their worktrees back one at a time** and resolves every seam-file edit centrally. *Never let
  two parallel agents edit a shared seam file live* — that is the documented oscillation/stage-deletion
  trap in this repo.
- **Commit cadence:** the phase agent commits per task inside its (isolated) worktree as the plan
  specifies; the orchestrator is the **sole committer on the integration branch**, merging with scoped
  paths (never `git add -A` during a fan-out).

### 3.2 The per-task TDD loop (already encoded in every Phase-0/1 task)

```
write the failing test → run it, confirm it FAILS for the stated reason
   → write the minimal implementation → run it, confirm it PASSES
   → ruff --fix + (at task/phase boundary) mypy → commit
```

Never write implementation before its test. Never skip the "confirm it fails" step (it validates the
test actually exercises the new code).

### 3.3 Review cadence (per phase) + the one end-of-feature simplify

**When a phase agent returns** (claiming its green gate passes):

0. **Integration review gate (first, every time):** **read the diff yourself** and **re-run the phase
   green gate** (§4.4) — never trust the agent's summary alone. For a parallel phase this happens as
   you merge its isolated worktree back onto the integration branch.
1. **Code-review agent** (`feature-dev:code-reviewer` / `implementation-test-reviewer`) over the
   phase's diff — bugs, edge cases, test/impl alignment — **plus the annotation-adherence agent
   (§3.4)** at the model-introduction phases. Apply high-confidence fixes, then **re-run the phase
   green gate + the regression suite** (§4.3).

**Once, after the FINAL phase** (not per phase): run a single **`code-simplifier`** over the whole
`tune/` (+ `gui/tune/`) surface, apply its simplifications, then run the **full** regression (§4.3).
Deferring simplify to the end avoids re-simplifying code that later phases reshape (e.g. the
`StudyStore` interface extraction or the `EvaluationResult`/`Trial` multi-objective widening). A
simplify that breaks a test is reverted, not shipped.

> This mirrors the standing rule: *"For each major phase, spawn a code review agent after. After all
> code implementation, launch a simplify agent and apply fixes. Run a regression test."*

### 3.4 Annotation-adherence agent (specialized review)

A dedicated reviewer that audits **only** the project's annotation/typing conventions — narrower and
deeper than the general code-reviewer (logic/bugs) and orthogonal to the simplify pass. It checks what
`mypy` cannot: *convention* adherence, not just type-correctness.

**Charter** — every new model/field is checked against the root + module `CLAUDE.md` rules:
- **Annotated class-level fields**, no hand-written `__init__`, keyword-only construction;
  input normalization + guards in a `field_validator`, never `__init__`.
- **Closed value sets never typed as bare `str`:** `EnumType | Literal[...]` normalized in a
  `field_validator(mode="before")` with the `Literal` aliased once as a `TypeAlias` + an Enum↔Literal
  alignment test; or a `Literal` `TypeAlias` in `tools_/typing_.py` for type-only enforcement; or
  `MeasurementInfo`/`ConstantLabels` for documented sets.
- **Special field types used correctly** — `polymorphic_field(base=…)`/`OperationField` for
  operation/scorer/strategy-valued params (and they **round-trip via the registry**), `NdArrayField`
  for raw arrays, `Field(discriminator="kind")` discriminated unions for closed model sets,
  `frozen=True` value-models where required.
- **Google-style docstrings with `Args:`** on every model — field descriptions auto-derive into
  `model_json_schema()` (the machine-readable contract); doctests runnable on `load_synth_yeast_plate()`.
- **(annotations workstream)** the validity-vs-search distinction (`Field(ge,le)` vs `TuneSpec`), the
  **`⊆` invariant** + its validator-blindness backstop, the coverage allowlist, "leave normalizing
  validators in place," and the back-compat `pipeline.json` fixtures.

It is a **review gate** (high-confidence findings applied before the phase is marked done), like the
code-reviewer; it does **not** touch logic or simplify.

**Optimal positions** — spawn it where annotated/serializable models are *introduced* or their patterns
are *set* (not per task — wasted; not on model-free phases):

| Position | Why it's optimal |
|----------|------------------|
| **After 1a** | the domains union + `Knob`/`SearchSpace` value-models **set the patterns** every later phase copies — catching drift here stops it propagating. The single highest-leverage spot. |
| **After 1d** | batch-audit the rest of the Phase-1 model surface — `Scorer` ABC, `ScorerField = polymorphic_field(base=Scorer)`, `TuningSpec`'s custom (de)serializer, `Budget`/`Trial`/`EvaluationResult`, the `StrategyConfig` union — riding with the milestone review. |
| **At each phase adding a new polymorphic subclass** (2: `OptunaConfig`; 3: `ReferenceFreeScorer` + the `TuneSpec` marker + `InferredSearchSpace`; 4: `SupervisedScorer`/`CompositeScorer`) | each new serializable model must round-trip via the registry + follow the union/`polymorphic_field`/frozen conventions — the highest-risk drift points. |
| **Per operation-family wave in the annotations workstream** | that workstream **is** the annotation deliverable — the adherence agent is its primary reviewer (⊆ invariant, coverage, validator rules, back-compat) every wave. |

Skip it on phases that add **no** new annotated model (most of Phase 5's Dash callbacks — except a
light check on 6c's `tuning_spec.json` emission).

### 3.5 Within-Phase-1 strict sequencing (the gotchas the plan reviews surfaced)

- **Phase 0 must parameterize the `polymorphic_field` guard** (`_make_require_value(base)`, not a
  hard-coded `BaseOperation` assert) — otherwise 1d's `ScorerField` round-trip rejects `QCScorer`
  (a `Scorer`, not a `BaseOperation`).
- **`tune/__init__.py` is cumulative.** 1a writes it (doctest + search-space), 1b appends the
  strategy *configs*, 1c writes the **complete** file again (must keep 1b's `GridConfig`/
  `RandomConfig`/`StrategyConfig`), 1d appends the engine surface. Dropping 1b's configs turns the
  1d suite red at import — the single highest-risk slip in Phase 1.
- **Capture the grid golden fixture (Phase 0 Task D) WHILE `sweep` still exists** — 1d's byte-compat
  lock reads it after `sweep` is deleted.

### 3.6 Outlines → full plans (Phases 2–5 + annotations)

Each outline ends with a **"Review findings (address at full-planning)"** section. Before executing
an outline, **expand it into a full TDD plan** (à la 1a–1d) that resolves those findings, then
execute that plan with the same cadence. Do not execute an outline directly.

### 3.7 Why orchestrator + isolated fan-out, not an agent team

A persistent **agent team** (teammates coordinating live over a shared context/worktree) is the wrong
tool for this work:

- **Phase 1 is a strict sequence** — there is nothing to coordinate; team messaging is pure overhead.
- **The 2–5 fan-out is mostly-disjoint work over a *known, static* dependency DAG** with a small,
  enumerable set of shared seam files. No evolving requirement needs live renegotiation — the team's
  core value (continuous coordination) doesn't apply.
- **A shared worktree invites the failures this repo has already hit** — index-contention
  stage-deletions and decision-oscillation when teammates edit the same files. The remedy that worked
  was sole-committer + scoped commits + isolated work, which is exactly the orchestrator model.

So: **isolated parallel subagents (own worktrees) + one orchestrator that owns the seam files and
integrates one-at-a-time.** Reach for a team only if the work later becomes genuinely interdependent
with shifting scope (not the case for a fixed DAG + a deterministic integration step).

---

## 4. Verification strategy

Three tiers of gate, plus standing cross-cutting invariants.

### 4.1 Gate tiers

| Tier | When | What |
|------|------|------|
| **Per-task** | after each plan task | the exact `uv run pytest tests/…::test_… -v` the task names — must FAIL first, then PASS |
| **Per-phase** | after a phase's tasks | full phase suite + `mypy` + `ruff` + doctests (the phase's final "gate" task) |
| **Cross-cutting locks** | at the phase that introduces them, then forever | golden byte-compat · no-new-deps · sweep-deletion · cross-phase type consistency · GUI ledgers |

### 4.2 Cross-cutting invariants (must hold after the owning phase)

- **Grid byte-compat lock** (Phase 1d, Task 6): the tune grid (`enumerate_grid` → `build_pipeline`)
  reproduces the **op-combination set** of the frozen `tests/fixtures/tune/grid_golden_manifest.json`.
  (Equivalence, not literal manifest bytes — legacy `Pipeline_N` names vs tune-clone uuids differ.)
  Reconstructed via core `ImagePipeline.from_json`, so it stands **after `sweep` is deleted**. If it
  fails, **do not edit the golden** — it means `build_pipeline`/`enumerate_grid` changed observable
  output; investigate.
- **No new third-party deps in Phase 1.** `scikit-learn`/`joblib`/`pyarrow` are already deps. Verify
  `git diff pyproject.toml` shows **no** added runtime dependency through end of Phase 1. (Optuna is
  Phase 2, behind the `tune` extra, lazy-imported — assert `import phenotypic` and the grid/random
  paths trigger no `import optuna`.)
- **Sweep hard-cutover** (end of Phase 1d): `grep -rn "phenotypic.sweep\|from ..sweep\|import sweep"
  src/ tests/ docs/ pyproject.toml` returns nothing actionable; the **full suite is green without
  `sweep` importable**; the `phenotypic-sweep` script entry (if any) is removed from `pyproject.toml`.
- **Cross-phase type consistency** (1a→1d): the shared contract (`SearchSpace`/`Knob`/`Domain`;
  `SearchStrategy.suggest/register_result/is_exhausted`; `StrategyConfig.build(space, store)`;
  `Scorer.score_image/finalize/availability`; `Evaluator.evaluate(base,scorer,params,images) ->
  EvaluationResult(score,terms,n_images,failed)`; `build_pipeline(base,params)`;
  `TuningSpec(pipeline,search_space,scorer,evaluator,strategy,budget)`) is used identically across
  all sub-phases. `mypy src/phenotypic/tune` is the mechanical guard.
- **Doctests runnable** on `load_synth_yeast_plate()` — `pytest --doctest-modules src/phenotypic/tune`.
- **GUI ledgers (Phase 5 only):** every affordance in `gui/FEATURES.md` (with a real `Test ref`);
  every flow in `gui/WORKFLOWS.md` with a matching `_capture_<id>` **defined AND dispatched** in
  `scripts/capture_gui_tutorial_screenshots.py` + a tutorial page; regenerate + commit **all** PNGs.

### 4.3 Regression gate (after review/simplify fixes, and after the cutover)

```bash
# Phase-1 regression surface (the modules tune touches + the cutover blast radius):
uv run pytest tests/unit/tune tests/unit/tools_ tests/unit/core tests/unit/detect \
              tests/unit/enhance tests/unit/analysis tests/unit/gui -q
uv run mypy src/phenotypic/tune src/phenotypic/_execution src/phenotypic/tools_/typing_.py
uv run ruff check src/phenotypic/tune src/phenotypic/_execution
```

After the **sweep deletion**, additionally run the broad suite to catch any straggler import:
`uv run pytest -q` (or the project's CI-equivalent selection).

### 4.4 Green-gate commands by phase

| Phase | Green gate (run after the phase's tasks) |
|-------|------------------------------------------|
| **0** | `uv run pytest tests/unit/tools_ tests/unit/core tests/unit/util tests/unit/tune tests/unit/sweep tests/unit/gui -q` · `uv run mypy src/phenotypic/tools_/typing_.py src/phenotypic/_execution src/phenotypic/tune` · `uv run ruff check …` |
| **1a** | `uv run pytest tests/unit/tune -q` · `pytest --doctest-modules src/phenotypic/tune/__init__.py` · `mypy` · `ruff` |
| **1b** | `uv run pytest tests/unit/tune -q` · `mypy src/phenotypic/tune/_strategies` · `ruff` |
| **1c** | `uv run pytest tests/unit/tune -q` · `pytest --doctest-modules src/phenotypic/tune/_scoring/_qc_scorer.py` · `mypy` · `ruff` |
| **1d** | `uv run pytest tests/unit/tune -q` · `pytest --doctest-modules src/phenotypic/tune` · `mypy src/phenotypic/tune` · `ruff` · **+ §4.2 locks** (golden, no-deps, sweep-gone) |
| **2–5** | defined in each full plan when the outline is expanded; always: phase suite + `mypy` + `ruff` + the phase's new locks (e.g. "no `import optuna` on the Phase-1 path"; Phase-5 GUI ledgers) |

---

## 5. Phase-by-phase execution checklist

| Phase | Entry criteria | Execute via | Exit gate | Post-phase |
|-------|----------------|-------------|-----------|------------|
| **0** | dev env synced; `sweep` present | 1 phase agent (sequential) | §4.4 row 0 green | code-review agent |
| **1a** | Phase 0 green (registry + `polymorphic_field` + golden) | 1 phase agent (sequential) | §4.4 row 1a | **annotation-adherence** (pattern-set checkpoint) |
| **1b** | 1a green | 1 phase agent (sequential) | §4.4 row 1b | — |
| **1c** | 1b green | 1 phase agent (sequential) | §4.4 row 1c | — |
| **1d** | 1c green | 1 phase agent (sequential) | §4.4 row 1d **+ §4.2 locks** | **code-review + annotation-adherence + regression** (Phase-1 milestone; simplify deferred to the final phase) |
| **2 · 3 · annotations** | Phase 1 green; outline expanded to a full plan | **parallel — 1 isolated-worktree agent each** | each full plan's gate + new locks | code-review **+ annotation-adherence** per phase; orchestrator integrates one-at-a-time |
| **4** | Phase 2 merged | 1 isolated-worktree agent | full plan gate | code-review + annotation-adherence; integrate |
| **5** | 2 + 3 + 4 merged (6a/6b: just 2) | 1 isolated-worktree agent | full plan gate + **GUI ledgers** | code-review; integrate → **then the one final simplify + full regression** |

---

## 6. Critical prerequisites & risk register (from the plan reviews)

| Risk | Owning phase | Mitigation (verified in the plan) |
|------|--------------|-----------------------------------|
| `ScorerField` rejects `QCScorer` | 0 → 1d | Phase-0 guard must be `_make_require_value(base)`; symptom is a `ValidationError` about expecting a `BaseOperation` |
| `tune/__init__.py` drops 1b configs → 1d red at import | 1c | write the **cumulative** `__init__.py` (keeps `GridConfig`/`RandomConfig`/`StrategyConfig` + the 1a doctest) |
| Golden uncapturable after cutover | 0 (Task D) | capture **while `sweep` exists**; 1d reads the frozen JSON via core |
| `mypy` fails on `Callable` | 0 (Task A) | add `from typing import Callable` (string annotations are runtime-safe but mypy isn't) |
| Pipeline won't round-trip in `TuningSpec` | 1d | embed via custom `field_validator`/`field_serializer` delegating to `to_json`/`from_json` (plain pydantic fails on the abstract `ImageOperation`) |
| Outline executed as-is propagates wrong assumptions | 2–5 | expand to a full plan first; each outline's **"Review findings"** section lists the corrections (e.g. P2 channel-not-wired, P4 `dict` not `tuple`, P3 nested-key grammar rewrite, annotations `⊆`-validator-blindness) |

---

## 7. Definition of done

**Phase 1 (MVP):**
- `python -m phenotypic.tune spec.json -i ./plates -o ./out` runs grid/random over a calibration set,
  writes `deliverables/{best_pipeline,tuning_spec,param_importance}.json` + `trials.parquet`, and
  resumes by re-pointing `-o` at an existing run.
- `--strategy grid` reproduces the deleted sweep's grid (golden lock green).
- `sweep` is gone; the full suite + `mypy` + `ruff` + doctests are green; no new third-party dep.
- Phase-1 code has passed the code-review agent + the **annotation-adherence agent** + the per-phase
  regression gate. (The single simplify pass + full regression run **once, after the final feature
  phase**, not here.)

**Full feature** (Phases 2–5 + annotations): Optuna backend behind `[tune]` (lazy); `--auto-space`
inference + reference-free scoring behind its meta-validation gate; supervised + multi-objective
Pareto; the `/tune/` Dash co-pilot (FEATURES/WORKFLOWS/screenshot gates green); operation fields
annotated so inference reads real envelopes. Each ships only when its full-plan gate + locks pass.
After the **final** phase, the single `code-simplifier` pass + a full regression have run green.
