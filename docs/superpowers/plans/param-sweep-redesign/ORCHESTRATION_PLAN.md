# Tune Engine — Implementation Orchestration Plan

> The concrete, executable orchestration plan that operationalizes
> [`docs/superpowers/plans/param-sweep-redesign/EXECUTION.md`](../../../bigdata/exfab/anguy344/PhenoTypic/.worktrees/redesign/param-sweep/docs/superpowers/plans/param-sweep-redesign/EXECUTION.md).
> It adds the **git workflow** (per-task commits, per-phase pushes, one final PR) the user
> requested and the concrete dispatch/integration sequence. EXECUTION.md is the strategy;
> this is the runbook I will actually drive.

---

## Context

**Why.** The current `sweep` module (`src/phenotypic/sweep/`) does eager, in-memory Cartesian
grid expansion over hand-written `Sweep`/`Presence` specs and materializes every pipeline into a
manifest. It cannot do sample-efficient search, robust evaluation, importance screening, auto
search-space inference, supervised/reference-free scoring, or interactive curation. The
param-sweep-redesign bundle replaces it with a **tune engine** (`src/phenotypic/tune/`): one
shared ask-and-tell core behind a `SearchStrategy` protocol (grid/random now, Optuna later),
pluggable `Scorer`s, robust CV evaluation, RF/fANOVA importance, a serializable `TuningSpec`, a
`python -m phenotypic.tune` CLI, and a Dash co-pilot.

**What prompted it.** The design bundle (12 specs + 6 full phase plans + 4 outlines + an
annotations workstream) is complete and reviewed. The user wants this implemented **end-to-end
(Phase 0 → 5 + annotations)** with phased commits/pushes and **one final PR**.

**Intended outcome.** `sweep` deleted and replaced by a fully-tested `tune` package: grid/random
+ Optuna backends, auto-space inference, supervised + multi-objective Pareto scoring, the `/tune/`
Dash co-pilot, and operation fields annotated with validity + search envelopes — each phase
gated by tests + `mypy` + `ruff` + doctests + cross-cutting locks, integrated on the
`redesign/param-sweep` branch, and shipped as a single PR to `main`.

**Source of truth.** The **specs win** on behavioral contracts (signatures, fields, methods);
the **phase plans** give the concrete TDD tasks + file paths. Where a plan's *file layout* differs
from a spec's sketch (e.g. plans use `_scoring/`+`_evaluation/`; the spec sketch says
`_scorers/`+`_evaluator.py`), follow the **phase plan's concrete paths** — they are the executable
artifacts. If a plan and spec disagree on a *contract*, **stop and reconcile before coding**.

---

## 0a. Resolved design decisions (settled 2026-06-03, before execution)

The outlines defer hard decisions to "full-planning"; these were resolved up front. Deferred items
are tracked in `docs/superpowers/plans/param-sweep-redesign/DEFERRED-WORK.md`.

**Settled by the specs (no choice — implement as specified):**
- **Knob key form = position-index** (`1.detectors[0].ignore_zeros`) — master design §5 + all docs; the
  parent-identifier segment is position-index (stable vs. same-class dup; shifts on top-level reorder, OK).
- **Nested-vs-presence key grammar** — `[i]` list-indexing for nested + `__enabled__` dunder for presence
  disambiguates the 3-segment collision (search-space-inference §6).
- **Evaluator `*, channel` arg + `EvaluationResult.pruned`** (engine-arch §4.2); **ASHA rung ladder** =
  number of calibration plates, first rung `max(~6-plate floor, ~⅓ calib)`, geometric ×3, self-disables
  (robust-eval §7); **Optuna lazy import + tune re-export** (engine-arch §6 / optuna §10); **`float | dict`
  multi-objective widening** (engine-arch §3.1); **nested presence depth cap = 1** (search-space-inference §6);
  **⊆-invariant blind to validator-enforced bounds → apply-time backstop** (search-space-inference §3).

**Internal defaults adopted (documented; override anytime):** `study.db` canonical when Optuna installed
(journal = audit fallback); autonomy gate auto-OK only when `proposal.needs_review == False`; reference-free
meta-validation ρ≥0.7 enable / ≥0.8 unattended, fail-safe to `QCScorer`; Dash overlays via background compute
+ LRU cache, plates from `splits/calibration.json`; shortlist = top-5 + Pareto + gap-flagged; 6c edits
flat+presence only (nested read-only v1); IoU τ=0.5 greedy unique-match (per-grid-cell on `GridImage`);
`CompositeScorer` nests `list[Scorer]` via `polymorphic_field` with cycle detection.

**User decisions (2026-06-03):**
1. **Test data = synthetic only** — `load_synth_yeast_plate()` everywhere + a synthetic E2E run; CI hermetic.
2. **Sweep GUI deleted at the 1d cutover** along with the core `sweep` module.
3. **GT validation deferred** — *ship* `SupervisedScorer` + the reference-free meta-validation gate with a
   path-configured GT loader (`gt_masks_source: Path`) + TODO, but don't validate against real GT now
   (→ DEFERRED-WORK §1). v1 tests cover construction/round-trip/term-shape/availability only.
4. **Optuna store = Postgres-first for SLURM** — Postgres is a first-class backend for distributed SLURM
   array studies (SQLite-WAL unsafe on NFS/Lustre); SQLite-WAL stays the local single-node default. The
   **server tooling now exists**: `~/util/postgres_server/` (PostgreSQL 18.4, conda env `pg`, launched
   via `sbatch pgserver.sh`, address in `connection_info.txt`, password in `pgpassword.txt`, port 54399).
   Used to run Phase-2 Postgres integration tests; wiring + its docs are Phase 6.
5. **Supervised metrics v1 = minimal** — Dice/IoU (region) + count MAE only; partition/instance/boundary +
   the unverified AJI+/Mahalanobis/Boundary-F deferred (→ DEFERRED-WORK §2).
6. **Annotations v1 = `detect/` + `enhance/` only** — `refine/`/`grid/`/`correction/` deferred but flagged
   *needed soon* (→ DEFERRED-WORK §3); coverage gate advisory until ≥70%, then hard-gates.

---

## 0b. Checkpoint resolutions (settled 2026-06-04, after expanding the 2–5 + annotations outlines)

The five outlines were expanded into full TDD plans; the open questions went to the user (the mandatory
pre-fan-out checkpoint). **User decisions:**
1. **Annotations Stage-2 = hints + bare-scalar Field bounds + the two clean intervals.** Add `TuneSpec`
   hints everywhere; convert the unambiguous bare-scalar validators (positivity / ≥1) **and** the two
   clean intervals (`EnhanceFeatures.sigma_onf ∈[0.1,1.0]`, `cutoff ∈(0,1)`) to `Field`; **keep** all
   cross-field + normalizing validators. Guarded by the back-compat fixture corpus.
2. **Multi-objective representation = SIDECAR.** Keep `Trial.score`/`EvaluationResult.score` as `float`
   (a scalar projection); add `objectives: dict[str,float] | None = None` alongside; persist an
   `objectives_json` parquet column; Pareto/knee/NSGA-II/per-objective importance read the sidecar;
   multi-objective "best" = the knee-point. (Reinterprets §0a's "widen score to `float|dict`" — approved;
   it honors §0a's *goal* — "never mutate the scalar path" — more literally than a union would.)
3. **GUI launch affordance = copy-paste command card** — `/tune/` renders the exact
   `python -m phenotypic.tune …` command; no `LocalRunner` spawn from the GUI.
4. **SlurmExecutor = reuse the forward-CLI drip-feed submission/throttle layer + a fresh tune worker
   body** (no image-chunk sentinels — Optuna workers drain a shared budget).

**Adopted defaults (orchestrator; user may veto any):**
- **Phase 2:** fANOVA-vs-RF dispatch on store capability; sampler TPE (single)/NSGA-II (multi)/CMA-ES
  opt-in; freeze thresholds conservative but config-exposed (ε=0.10, trigger >6 params); ASHA⊥CV kept
  orthogonal; always also write `trials.parquet` beside `study.db`.
- **Phase 3:** `TuneSpec` fields `low/high/step/log/categories/tunable`; unbounded heuristic `[d/4, d·4]`
  + auto-log, `d≤0` excluded; int rounds outward; `needs_review` True if any knob guessed *or* any
  inference-blind exclusion; `T|None` → infer over `T`; `--auto-space` file-only non-blocking; CLI
  `run`/`auto-space` subcommand split; reference-free = lean proxy set (grid/count QC + shape regularity
  + contrast + size-CV).
- **Phase 4:** knee = max-distance-to-chord; `deliverables/pareto/` = front parquet + per-objective best
  + knee as `best_pipeline.json`; multi-objective inferred from the scorer; grid/random + multi-objective
  scorer → reject at validation; GT loader = directory + image-stem match.
- **Phase 5:** one agent runs 6a→6b→6c; hermetic `run_tune_once` screenshots; overlays via
  interval-polled worker + disk LRU; gap-flag relative >0.15; 3s poll + manual refresh; `study_db_path`
  helper added to `_io_constants` (orchestrator seam); Pareto panel feature-flagged off single-objective.
- **Annotations:** coverage denominator = numeric tunable fields in `detect/`+`enhance/` only
  (FilamentousFungi auto-params marked `tunable=False`); `⊆` apply-time backstop = a wrapped
  `pydantic.ValidationError` at `build_pipeline` (knob-key + op-class prefixed), no new exception type.

**Refined fan-out ordering:** Phase 3 ships its `TuneSpec` marker + inference core FIRST (the keystone
that unblocks the annotations workstream); **Phase 2 ∥ Phase 3** (isolated worktrees, limited seam
overlap = `tune/__init__.py` + `__main__.py`, orchestrator-reconciled); annotations starts once the
marker lands; then **Phase 4** (needs Phase 2's NSGA-II + `StudyStore` Protocol); then **Phase 5**
(needs 2+3+4*). Big phases are **chunked into multiple sub-agent dispatches** (Phase 1d showed a single
agent overruns one context) — e.g. Phase 2 as A–C / D–F / G–H, Phase 3 as marker+inference / nested+
auto-space / reference-free — integrated incrementally.

---

## 0. Git workflow (commits, pushes, PR) — applies to every phase

| Granularity | Action | When |
|-------------|--------|------|
| **Per task** | `git commit` (scoped paths) | After each TDD task goes red→green and `ruff --fix` passes. The phase agent commits inside its worktree; messages end with the `Co-Authored-By: Claude Opus 4.8` trailer. |
| **Per phase** | `git push origin redesign/param-sweep` | After the phase's **green gate** (§7.4) passes **and** the code-review / annotation-adherence fixes are applied + re-verified. |
| **End of feature** | open **one PR** → `main` | After Phase 5 + the single `code-simplifier` pass + full regression are green. PR body ends with the `🤖 Generated with Claude Code` trailer. |

**Branch.** Integration branch is **`redesign/param-sweep`**. This worktree is currently a
**detached HEAD at `7d61e729`** (verified) — so **Step 0 of execution is**
`git switch -c redesign/param-sweep` (commits on a detached HEAD are unreferenced and lost).
Never implement on `main`. Remote is `origin git@github.com:exfab/PhenoTypic.git`; set upstream on
first push (`git push -u origin redesign/param-sweep`).

**Sole-committer rule for the fan-out.** Phases 0–1 are sequential and run directly on
`redesign/param-sweep` (the phase agent commits per task there). Phases 2–5 + annotations fan out,
each agent in its **own isolated git worktree** (`isolation: "worktree"`). Those agents commit in
their worktree; the **orchestrator is the sole committer on `redesign/param-sweep`**, merging each
returned worktree back **one at a time** with **scoped paths** and resolving every shared seam-file
edit centrally. **Never `git add -A` during a fan-out** (the documented index-contention /
stage-deletion trap in this repo).

---

## 1. Orchestration model

Act as **orchestrator**: dispatch one subagent per phase, review what it returns, integrate.
Implement inline only when a step is trivial (<30 lines, one file).

- **One subagent owns an entire phase** (a `phase-N-*.md` plan), running that plan's TDD tasks in
  order in its own context and returning only when the phase **green gate** passes. Brief it like a
  colleague: the plan file, the relevant spec sections, the constraints, and *"stop and report if
  the green gate can't be reached — don't improvise past a blocker."*
- **`0 → 1a → 1b → 1c → 1d` is a strict sequence** — one agent per sub-phase, in order (each imports
  the previous; `tune/__init__.py` is cumulative). Do **not** parallelize within Phase 1.
- **`2 / 3 / annotations` fan out** from a green Phase 1 (Wave A, parallel, isolated worktrees) →
  then **Phase 4** (needs Phase 2's NSGA-II sampler) → then **Phase 5** (needs 2+3+4*) →
  then **Phase 6** (documentation, final).
- **Orchestrator owns the seam files** (§5). Parallel agents own only their **disjoint new** modules.
- **Model policy:** every **writing/editing** subagent — phase implementers, the doc-writing agent
  (Phase 6), `pydoc-writer`, and the end-of-feature `code-simplifier` — is dispatched with
  **`model: "opus"`** (per user). Read-only agents (`Explore`, `feature-dev:code-reviewer`,
  `implementation-test-reviewer`, the annotation-adherence reviewer, outline-expansion `Plan` agents)
  may use defaults, but Opus is fine for them too when the budget allows.

**Seam files (orchestrator-owned, never edited live by two agents):**
`src/phenotypic/tune/__init__.py` · `pyproject.toml` (the **`tune` extra** = `optuna>=4.0`,
`sqlalchemy>=2.0`, `psycopg[binary]>=3.1` — Phase 2; + a `phenotypic-tune` console script) ·
the registry submodule list in
`src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py:592` ·
`src/phenotypic/tools_/typing_.py` (`polymorphic_field`) · `TuningSpec.strategy` widening ·
the `EvaluationResult`/`Trial` multi-objective (`float | dict`) widening.
(`scikit-learn`/`joblib`/`pyarrow` are already core deps; only the `tune` extra adds runtime deps,
and only at Phase 2 — Phase 1 stays dep-free.)

---

## 2. Pre-flight (once, before Phase 0)

```bash
# In this worktree (NOT the default checkout):
git switch -c redesign/param-sweep          # establish the integration branch (detached HEAD now)
uv sync --group dev                          # pytest, mypy, ruff
uv run pytest -q tests/unit/tune || true     # expect: dir doesn't exist yet (Phase 0 creates it)
uv run pytest -q tests/unit/sweep            # confirm sweep is GREEN and present (golden capture needs it)
```

Confirm `scikit-learn`, `joblib`, `pyarrow` are already in `pyproject.toml` (they are — **no new
runtime dep through end of Phase 1**; Optuna arrives in Phase 2 behind the `[tune]` extra).

---

## 3. Stage A — Phase 0: Prerequisites (sequential, on `redesign/param-sweep`)

**Plan:** `phase-0-prerequisites.md`. **Goal:** four zero-new-dep enablers that unblock Phase 1.

| Task | Test (must FAIL first) | Implements |
|------|------------------------|------------|
| **A** | `tests/unit/tools_/test_polymorphic_field.py` (`test_guard_accepts_base_subclass_instance` +3) | Parameterize the guard: `_make_require_value(base)` factory replacing the hard-coded `isinstance(value, BaseOperation)` at `tools_/typing_.py:257`; add `polymorphic_field(base, *, marker)` (with `OperationField = polymorphic_field(base=BaseOperation)` as the alias); `from typing import Callable`. **This is the highest-leverage fix** — without it `ScorerField = polymorphic_field(base=Scorer)` rejects a `Scorer` (which is `BaseModel`+`ABC`, **not** a `BaseOperation`). |
| **B** | `tests/unit/core/test_registry_finds_tune.py` (+1) | Append `"phenotypic.tune"` to the submodule list in `_serializable_pipeline.py:592–601`; create stub `src/phenotypic/tune/__init__.py`. |
| **C** | `tests/unit/util/test_local_executor.py` (`test_local_executor_maps_in_order` +3) | New `src/phenotypic/_execution/{__init__,_protocol,_local}.py` — `Executor` Protocol + `LocalExecutor(n_jobs)` `run(work, items) -> list[R]`, joblib-backed (no new dep). |
| **D** | `tests/unit/tune/test_grid_golden_manifest.py` (`test_golden_exists_and_is_stable` +1) | `scripts/capture_grid_golden_manifest.py` + frozen `tests/fixtures/tune/grid_golden_manifest.json` (6 pipelines). **Capture WHILE `sweep` still exists** — read-only after the 1d cutover. |
| **E** | (regression only) | Full Phase-0 suite green. |

**Green gate (§7.4 row 0):**
```bash
uv run pytest tests/unit/tools_ tests/unit/core tests/unit/util tests/unit/tune tests/unit/sweep tests/unit/gui -q
uv run mypy src/phenotypic/tools_/typing_.py src/phenotypic/_execution src/phenotypic/tune
uv run ruff check src/phenotypic/_execution src/phenotypic/tune src/phenotypic/tools_/typing_.py
```
**Post-phase:** `feature-dev:code-reviewer` over the diff → apply high-confidence fixes → re-run gate
→ **commit per task already done; push** `redesign/param-sweep`.

---

## 4. Stage B — Phase 1 MVP: `0 → 1a → 1b → 1c → 1d` (strict sequence)

**The cumulative `tune/__init__.py` is the #1 slip risk.** 1a writes it (doctest + search-space);
1b **appends** `GridConfig`/`RandomConfig`/`StrategyConfig`; 1c rewrites the **complete** file
(must keep 1b's configs); 1d appends the engine surface. Dropping 1b's configs turns 1d red at import.

### 4.1 Phase 1a — Search space (`phase-1a-search-space.md`)
- **New:** `tune/_search_space/_domains.py` (`Categorical`/`IntRange`/`FloatRange`/`Fixed`, each
  `frozen=True`, discriminated union `Field(discriminator="kind")`); `tune/_search_space/_space.py`
  (`Knob`, `SearchSpace`); cumulative `tune/__init__.py` + doctest.
- **Tasks:** `test_domains.py` → `test_search_space.py` → doctest/export.
- **Gate (§7.4 row 1a):** `pytest tests/unit/tune -q` · `pytest --doctest-modules
  src/phenotypic/tune/__init__.py` · `mypy src/phenotypic/tune` · `ruff`.
- **Post:** **annotation-adherence agent** (§6.2) — *pattern-set checkpoint, the single
  highest-leverage spot*; the domains/`Knob` value-models set the conventions every later phase copies.

### 4.2 Phase 1b — Strategies (`phase-1b-strategies.md`)
- **New:** `tune/_strategies/{_pruning,_enumerate,_protocol,_grid,_random,_config}.py`
  (`PruningChannel`+`NoOpChannel`; `grid_values`/`enumerate_grid` honoring `conditional_on`;
  `SearchStrategy` protocol; `GridStrategy`/`RandomStrategy`; `StrategyConfig` ABC + `GridConfig`/
  `RandomConfig` + union). Append the **configs** to `tune/__init__.py`.
- **Tasks:** pruning → enumerate → strategies → strategy-config.
- **Gate (§7.4 row 1b):** `pytest tests/unit/tune -q` · `mypy …/_strategies` · `ruff`.

### 4.3 Phase 1c — Scoring + evaluation (`phase-1c-scoring-evaluation.md`)
- **New:** `tune/_scoring/_scorer.py` (`Scorer(BaseModel, ABC)`: `score_image`/`finalize`=mean/
  `availability`); `tune/_scoring/_qc_scorer.py` (`_threshold_anchored = exp(-ln2·m/f)` + `QCScorer`
  wrapping `ExpectedVsDetectedCount`, **path-configured** so it round-trips); `tune/_evaluation/
  _builder.py` (`build_pipeline(base, params)`: clone + overlay + drop-disabled, fresh reconstruction);
  `tune/_evaluation/_evaluator.py` (`_robust_aggregate` = median−λ·IQR, λ=0.5; `EvaluationResult`;
  `Evaluator`). Rewrite the **complete cumulative** `tune/__init__.py`.
- **Tasks:** scorer → qc-scorer → builder → evaluator → end-to-end integration (synth plate 96
  objects → Count=1.0).
- **Gate (§7.4 row 1c):** `pytest tests/unit/tune -q` · `pytest --doctest-modules
  …/_qc_scorer.py` · `mypy` · `ruff`.
- **Post:** **annotation-adherence agent** (before 1d).

### 4.4 Phase 1d — Engine + CLI + cutover (`phase-1d-engine-cli.md`) — MVP milestone
- **New:** `tune/_study_store.py` (`Trial` frozen + `StudyStore` parquet I/O); `tune/_spec.py`
  (`Budget` + `TuningSpec` with **embedded pipeline** via custom `field_validator`/`field_serializer`
  delegating to `to_json`/`from_json`, and `scorer: ScorerField = polymorphic_field(base=Scorer)`);
  `tune/_engine.py` (`TuningEngine.optimize` ask-and-tell + resume fast-forward); `tune/_screening.py`
  (`compute_param_importance` RF+permutation, one-hot categorical aggregation); tune path helpers in
  `tools_/_io_constants.py`; `tune/_tune_cli/_run.py` + `tune/__main__.py` (argparse `-i/-o`, default
  `./<input>_tune/`, resume if `trials.parquet` exists); `scripts/migrate_sweep_manifest.py`. Append
  engine exports to `tune/__init__.py`.
- **Tasks:** study-store → tuning-spec → engine → param-importance → cli → **byte-compat lock** →
  **migrate + delete sweep**.
- **Gate (§7.4 row 1d) + cross-cutting locks (§7.2):** `pytest tests/unit/tune -q` ·
  `pytest --doctest-modules src/phenotypic/tune` · `mypy src/phenotypic/tune` · `ruff` · **grid
  byte-compat lock green** (op-combination-set equivalence vs the frozen golden, reconstructed via
  core `ImagePipeline.from_json` — **do not edit the golden if it fails**) · **no new dep** ·
  **sweep gone** (`grep -rn "phenotypic.sweep\|from ..sweep\|import sweep" src/ tests/ docs/
  pyproject.toml` returns nothing actionable; remove any `phenotypic-sweep` script entry; drop the
  `sweep` import + `__all__` entry from `src/phenotypic/__init__.py`; delete `gui/sweep/` +
  `tests/unit/sweep/` + sweep CLI/GUI integration tests; **scrub the `docs/` sweep references** —
  replace `docs/source/how_to/pages/parameter_sweeps.md`'s `python -m phenotypic.sweep` body with a
  short `python -m phenotypic.tune` stub (full tune how-to lands in Phase 6) and remove README sweep
  mentions, so the `docs/` grep is clean).
- **Post (milestone review):** **code-review + annotation-adherence** (batch-audit the Phase-1 model
  surface: `Scorer` ABC, `ScorerField`, `TuningSpec` (de)serializer, `Budget`/`Trial`/
  `EvaluationResult`, the `StrategyConfig` union) **+ Phase-1 regression** (§7.3). **Simplify is
  deferred to the final phase.** Then **push**.

---

## 5. Stage C — Phases 2–5 + annotations (expand-then-execute, DAG fan-out)

**§3.6 rule (mandatory):** each of `phase-2/3/4/5-*.md` and `workstream-operation-annotations.md` is
an **outline**, not a TDD plan. **Before dispatching an agent, expand the outline into a full TDD
plan** (à la 1a–1d) that resolves its **"Review findings (address at full-planning)"** section, then
execute that plan with the same per-task-commit cadence. *Do not execute an outline directly.*
Expansion is an orchestrator task (use a `Plan`/`feature-dev:code-architect` agent per outline).

> **🛑 USER CHECKPOINT (mandatory, user-requested 2026-06-03):** after expanding the Phase 2–5 +
> annotations outlines into full TDD plans — and **before dispatching ANY parallel implementation
> agent** — **STOP and ask the user.** Surface every open question / ambiguity / design fork that
> surfaced during the expansions (via `AskUserQuestion`), plus invite any other clarifications. Only
> after the user responds does the Wave-A fan-out begin. This gate sits between "Phase 1 green +
> outlines expanded" and the first `isolation: "worktree"` dispatch.

**Each fan-out phase runs in its own isolated worktree; the orchestrator integrates one-at-a-time
(scoped paths), resolves seam files, runs the phase gate + review + regression, then pushes.**

### Wave A (parallel, isolated worktrees): Phase 2 · Phase 3 · annotations

**Phase 2 — Optuna backend** (`phase-2-optuna-backend.md`). Optuna behind the **`[tune]` extra,
lazy-imported**; `OptunaStrategy`/`OptunaConfig` (`kind="optuna"`), ASHA pruning, `study.db`,
fANOVA importance, two-round screening freeze, `SlurmExecutor`. **Decisions are settled in §0a** —
implement (don't re-debate): Evaluator `*, channel` + `EvaluationResult.pruned`; ASHA rung ladder =
calibration-plate count (robust-eval §7); `StudyStore` extracted to a Protocol + Optuna impl +
resume awareness; widen `TuningSpec.strategy` to `polymorphic_field(base=StrategyConfig)` (seam —
orchestrator); `study.db` canonical when Optuna installed. **Store = Postgres-first for SLURM**
(user decision 4): Postgres is a first-class backend selected by `storage_url`; SQLite-WAL stays the
local single-node default; SLURM/NFS with SQLite = monitor-only. The expanded plan adds a
**`storage_url` config** (on `OptunaConfig` + a `--storage-url` CLI flag + a
`PHENOTYPIC_TUNE_STORAGE_URL` env fallback) and the **`tune` extra** = `optuna` + `sqlalchemy` +
`psycopg[binary]`. **Postgres URL scheme:** use `postgresql+psycopg://USER:PW@HOST:54399/DB`
(psycopg3 driver, matching PostgreSQL 18.4); document a `read_pg_connection_info()` helper that parses
`connection_info.txt` + `pgpassword.txt` into that URL.
**Testing while building (user-provided server `~/util/postgres_server/`):** unit tests use a local
SQLite RDB in `tmp_path` (hermetic). Postgres-backed integration tests are **gated** (skip unless
`PHENOTYPIC_TEST_PG_URL` is set / `@pytest.mark.postgres`). To exercise the distributed path on the
HPCC: `sbatch ~/util/postgres_server/pgserver.sh` → `squeue` until running → read node+port from
`connection_info.txt` + password from `pgpassword.txt` → `createdb` a test DB → export
`PHENOTYPIC_TEST_PG_URL=postgresql+psycopg://anguy344:<pw>@<node>:54399/<db>` → run the gated tests →
`scancel`. **Lazy-import lock:** `import phenotypic` + grid/random paths trigger **no** `import optuna`.
**Post:** code-review + **annotation-adherence** (new `OptunaConfig` subclass round-trips via registry).

**Phase 3 — Auto-space + reference-free** (`phase-3-autospace-reference-free.md`).
`infer_search_space(pipeline) -> InferredSearchSpace` (Tier-1 `TuneSpec` overrides + Tier-2 type
heuristics), `--auto-space` CLI, the **`TuneSpec` `@Annotated` marker** (hard dep for the annotations
workstream), nested-op overlay grammar in the builder, `ReferenceFreeScorer` with a meta-validation
gate. **Decisions settled in §0a** — implement directly: knob key = position-index; nested-vs-presence
grammar = `[i]` indexing + `__enabled__` dunder; depth cap 1; gate is `meta_validate(gt_images, grid)`
with a cached flag (ρ≥0.7/0.8), **not** zero-arg `availability()`; autonomy gate auto-OK iff
`needs_review==False`. **GT validation deferred** (decision 3): build the gate's loader + the abstain
machinery, but don't validate against real GT now (DEFERRED-WORK §1). Still resolve at expansion: the
Tier-2 heuristic edges (`⊆`-validator-blindness backstop, int outward rounding, `T | None` unions).
**Post:** code-review + **annotation-adherence** (`ReferenceFreeScorer` + `TuneSpec` marker +
`InferredSearchSpace`).

**Annotations workstream** (`workstream-operation-annotations.md`). **v1 scope = `detect/` + `enhance/`
only** (user decision 6); `refine/`/`grid/`/`correction/` are deferred but flagged *needed soon*
(DEFERRED-WORK §3). Per family: add **search hints** (`TuneSpec(lo, hi, log=)`, pure metadata, zero
behavior change) **first**, then tighten **validity bounds** (`Field(ge=, le=)`) **second** (guarded).
Enforce the **`⊆` invariant** (`TuneSpec[low,high] ⊆ [ge,le]`) + an apply-time backstop for
validator-enforced bounds (which `model_fields` metadata can't see); coverage gate **advisory until
≥70%** of the annotated families' numeric fields are covered, then **hard-gates**. Migration rule:
convert validator→`Field` **only** for a bare scalar bound; keep normalizing/conditional validators;
back-compat `pipeline.json` fixtures still load. **Hard dep:** the Phase-3 `TuneSpec` marker.
**Reviewer:** the **annotation-adherence agent is its primary reviewer every wave.** Lands incrementally.

### Then Phase 4 — Supervised + multi-objective (`phase-4-supervised-multiobjective.md`)
Needs Phase 2's NSGA-II sampler. **Metrics v1 = minimal** (user decision 5): `SupervisedScorer` ships
**Dice/IoU (region) + count MAE (counting) only**, IoU τ=0.5 greedy unique-match / per-grid-cell on
`GridImage`; `availability()` reports the runnable tier. Partition (ARI/VI), instance (PQ/SEG/AJI),
boundary (Hausdorff/NSD) + the unverified AJI+/Mahalanobis/Boundary-F are **deferred** (DEFERRED-WORK
§2). **GT validation deferred** (user decision 3): build the path-configured GT loader
(`gt_masks_source: Path`, like `QCScorer`'s metadata path, round-trippable) + a TODO; v1 tests cover
construction/round-trip/term-shape/availability, **not** numeric correctness vs. real GT. Also build
`CompositeScorer` (nests `list[Scorer]` via `polymorphic_field`, cycle detection) + true Pareto.
**Multi-objective widening** (settled §0a): widen `Scorer.finalize`/`EvaluationResult.score`/
`Trial.score` to `float | dict[str,float]` (seam — orchestrator); add a JSON `objectives_json` parquet
column; single-objective keeps scalar + `best()` unchanged. **Post:** code-review +
**annotation-adherence**; integrate.

### Then Phase 5 — Dash co-pilot (`phase-5-dash-copilot.md`)
6a monitor (live `study.db` WAL read, objective curve, Pareto knee, importance bars, generalization-gap
flag) → 6b curate (shortlist + winner pick + detection overlays → `best_pipeline.json`) → 6c space-edit
(`pipeline.json` → `InferredSearchSpace` forms reusing `_param_forms` → `tuning_spec.json`). GUI
**never re-optimizes** — delegates to the CLI. 6a/6b can ship on Phase 2 alone (feature-flag the Pareto
pieces); 6c needs Phase 3. **GUI is CI-gated (§7.2):** every affordance in `gui/FEATURES.md` (real
`Test ref`); every flow in `gui/WORKFLOWS.md` with a matching `_capture_<id>` **defined AND dispatched**
in `scripts/capture_gui_tutorial_screenshots.py` + a tutorial page; **regenerate + commit ALL PNGs**
(do not cherry-pick the collateral churn). Light annotation-adherence check on 6c's `tuning_spec.json`
emission only. **Post:** code-review; integrate → **then the one final simplify (§6.3) + full
regression** → **then Phase 6 (documentation)** → **open the PR**.

### Then Phase 6 — Documentation (final phase, on `redesign/param-sweep`)
**Goal:** the user-facing docs for the whole tune engine, authored against the *final, simplified* code
(after the §6.3 simplify pass). Sphinx + MyST, Diataxis.

**Agent-team workflow (user-directed 2026-06-04):** spawn a **writer + fact-checker + editor/reviewer
agent team** (`TeamCreate`) and run the **Python** subsection and the **GUI** subsection each through the
combo: **writer** drafts → **fact-checker** verifies every claim against the actual code/CLI/specs (run
the commands; check signatures/flags/outputs) → **editor/reviewer** polishes for clarity, structure, and
house style. All writing agents are Opus; `pydoc-writer` fills any docstring gaps.

**Structure — a dedicated "Tuning" how-to section** (its own section under
`docs/source/how_to/`, for now), with **two subsubsections**:
- **`### Python interface`** — `python -m phenotypic.tune run spec.json -i … -o …` + the `TuningSpec`
  Python API (pipeline/search-space/scorer/evaluator/strategy/budget), `--auto-space`, `--n-trials`,
  `--screen`, the `deliverables/` outputs (incl. `pareto/`), resume semantics. Runnable, doctest-style.
- **`### GUI interface`** — the `/tune/` Dash co-pilot (monitor → curate → space-edit); the
  copy-paste-command launch affordance. **Screenshots captured via Playwright** (drive the hub, snap each
  panel) **AND an automated capture script** mirroring `scripts/capture_gui_tutorial_screenshots.py`
  (a `capture_tune_docs_screenshots.py` or new `_capture_*` entries) so the shots regenerate
  reproducibly — boot a hermetic tune run over the synthetic dataset, then capture; commit the PNGs.
- **The four scoring objectives (the "four strategies" — user's headline framing, the four `Scorer`
  types):**
  - **`QCScorer`** — *no ground truth, statistical*: expected-vs-detected colony **count** check (wraps
    `ExpectedVsDetectedCount`).
  - **`ReferenceFreeScorer`** — *no ground truth, segmentation*: fixed-normalized proxies (shape
    regularity, contrast, size-CV) behind the meta-validation gate.
  - **`SupervisedScorer`** — *with ground truth*: Dice/IoU + count MAE.
  - **`CompositeScorer`** — combine the above (weighted blend or multi-objective Pareto).
  For each: when to use it, what it requires (GT masks / metadata path / nothing), how to configure it
  in `TuningSpec`. (The **search strategies** grid/random/Optuna-TPE/CMA-ES/GP/NSGA-II are the
  *optimizer* — a separate axis — covered under the Python/CLI interface via `--strategy`.)
- **`tune_distributed_hpcc.md` — the Postgres-on-clusters guide (user's explicit ask):** *why* Postgres
  (SQLite-WAL unsafe on NFS/Lustre where SLURM array jobs share state); *how to launch* the user-space
  Postgres Slurm job (`~/util/postgres_server/` — `sbatch pgserver.sh`, conda env `pg`, PG 18.4, port
  54399, `pgdata/` on `/bigdata`); *read the address* from `connection_info.txt`+`pgpassword.txt`; *wire
  it in* — `--storage-url postgresql+psycopg://USER:PW@NODE:54399/DB` / `OptunaConfig.storage_url` /
  `PHENOTYPIC_TUNE_STORAGE_URL` + `read_pg_connection_info()`. Mirror `slurm_pipelines.md` style.
- **Replaces** `parameter_sweeps.md` (remove from the `how_to/index.rst` toctree).
- **API reference:** `cli_reference.rst` Tune section; autodoc/autosummary over `phenotypic.tune`.
- **README.md:** a "Hyperparameter Tuning" section (replacing sweep mentions) + a `phenotypic.tune`
  module-table row noting the `tune` extra.
- **Gate:** `make -C docs html` clean (no new-page warnings); doctests pass; the tune screenshot capture
  script runs + PNGs committed; **no `phenotypic.sweep` left in `docs/`/README**.
- **Post:** editor/reviewer + a `feature-dev:code-reviewer` over the docs diff → **open the PR**.

---

## 6. Review cadence

### 6.1 Integration review gate (first, every phase)
**Read the diff yourself** and **re-run the phase green gate** — never trust the agent's summary. For
a fan-out phase this happens as you merge its isolated worktree back.

### 6.2 Per-phase review agents
- **`feature-dev:code-reviewer`** (or `implementation-test-reviewer`) over the phase diff — bugs, edge
  cases, test/impl alignment. Apply high-confidence fixes → re-run gate + regression.
- **Annotation-adherence agent** (specialized, convention-only — what `mypy` can't check): annotated
  class-level fields, no `__init__`, keyword-only, `field_validator` normalization; closed sets as
  `Enum | Literal[...]` (+ alignment test) never bare `str`; `polymorphic_field(base=)`/`OperationField`/
  `NdArrayField`/discriminated-union/`frozen=True` used correctly **and round-tripping via the registry**;
  Google `Args:` docstrings feeding `model_json_schema()`; doctests on `load_synth_yeast_plate()`. Run it
  **after 1a** (pattern-set), **after 1d** (Phase-1 model surface), at **each phase adding a polymorphic
  subclass** (2/3/4), and **every annotations wave**. It's a review gate, not a logic/simplify pass. Skip
  it on model-free phases (most Phase-5 callbacks).

### 6.3 The one end-of-feature simplify
**Once, after Phase 5 / before Phase 6** (not per phase): a single **`code-simplifier`** (Opus) over the
whole `tune/` (+ `gui/tune/`) surface → apply → **full regression (§7.3)**. Deferring avoids
re-simplifying code that later phases reshape (`StudyStore` Protocol extraction, the `float | dict`
widening). A simplify that breaks a test is reverted, not shipped. **Then Phase 6 documents the final,
simplified code** (docs authored after simplify so they describe the shipped surface).

### 6.4 Per-phase OQ checkpoint (user-requested 2026-06-04)
**After each phase/chunk returns, if it surfaced any genuine open question / design fork, STOP and ask
the user** (`AskUserQuestion`) before integrating/proceeding — don't silently default it. (Minor
implementation notes/caveats that need no user decision don't count; use judgment.)

### 6.5 Annotations bounds fact-check (user-requested 2026-06-04)
For the **annotations workstream**, after the agent proposes the `TuneSpec`/`Field` bounds, **spawn a
`fact-checker` subagent** to verify each proposed bound against peer-reviewed literature / reputable
sources (microbiology + image-processing) for reasonableness. Resolution:
- **Literature-supported** → keep the bound as proposed.
- **Not found in the literature** → reason it through, and **keep it SOFT for now** (a `TuneSpec`
  search hint and/or a loose `Field`, NOT a tight hard `Field` validity bound) + add a
  `# TODO: review bound (unverified vs literature)` comment at the field.
This pairs with §0b's "validity-loose vs search-tight" stance: unverified envelopes guide *search*
(`TuneSpec`) without constraining *validity* (`Field`).

### 6.6 Parallel test execution (user-requested 2026-06-04; corrected after an OOM crash)
Run pytest with **`-n 8`** (`pytest-xdist`, already a dev dep) wherever the tests are parallel-safe —
i.e. the unit suites (`tests/unit/**`, `tmp_path`-isolated) and `--doctest-modules`.
**Never use `-n auto`** — it reads `os.cpu_count()` = the **physical HPCC node's** core count (64+),
NOT the Slurm cpuset you're allocated, so it massively oversubscribes and the OOM-killer crashes the
session. **Hard-cap at `-n 8`** (matches a typical allocation). Use `-n0`/serial for tests sharing
fixed external state (the gated Postgres `@pytest.mark.postgres` tests on one DB; any fixed-port
Playwright/e2e lane). Brief every phase agent to use `-n 8` in its per-task + green-gate runs.
`mypy`/`ruff` are unaffected.

---

## 7. Verification

### 7.1 Gate tiers
- **Per-task:** the exact `uv run pytest …::test_… -v` the task names — FAIL first, then PASS.
- **Per-phase:** full phase suite + `mypy` + `ruff` + doctests (the phase's final gate task).
- **Cross-cutting locks:** introduced at their owning phase, then enforced forever (§7.2).

### 7.2 Cross-cutting invariants (must hold after the owning phase)
- **Grid byte-compat lock** (1d): tune grid (`enumerate_grid`→`build_pipeline`) reproduces the
  **op-combination set** of the frozen golden (equivalence, not literal bytes — legacy `Pipeline_N`
  names vs tune-clone uuids differ); reconstructed via core `from_json` so it stands **after sweep is
  deleted**. If it fails, investigate `build_pipeline`/`enumerate_grid` — **don't edit the golden.**
- **No new third-party deps through Phase 1** — `git diff pyproject.toml` shows zero added runtime dep.
- **Optuna lazy-import lock** (Phase 2): `import phenotypic` + grid/random paths trigger no `import optuna`.
- **Sweep hard-cutover** (end 1d): grep clean; full suite green without `sweep` importable.
- **Cross-phase type consistency** (1a→1d): the shared contract (`SearchSpace`/`Knob`/`Domain`;
  `SearchStrategy.suggest/register_result/is_exhausted`; `StrategyConfig.build`; `Scorer.score_image/
  finalize/availability`; `Evaluator.evaluate -> EvaluationResult`; `build_pipeline`; `TuningSpec`) is
  used identically everywhere — `mypy src/phenotypic/tune` is the mechanical guard.
- **Doctests runnable** on `load_synth_yeast_plate()` (`pytest --doctest-modules src/phenotypic/tune`).
- **GUI ledgers** (Phase 5): FEATURES.md + WORKFLOWS.md round-trip + all PNGs regenerated/committed.
- **Docs build clean** (Phase 6): `make -C docs html` builds without new-page warnings; the tune
  how-to + HPCC-Postgres pages exist and are in the toctree; **no `phenotypic.sweep` string anywhere in
  `docs/` or `README.md`**; the Postgres `storage_url` wiring is documented.

### 7.3 Regression gate (after each phase's review/simplify fixes; broad sweep after cutover)
```bash
uv run pytest tests/unit/tune tests/unit/tools_ tests/unit/core tests/unit/detect \
              tests/unit/enhance tests/unit/analysis tests/unit/gui -q
uv run mypy src/phenotypic/tune src/phenotypic/_execution src/phenotypic/tools_/typing_.py
uv run ruff check src/phenotypic/tune src/phenotypic/_execution
# after the sweep deletion, also the broad suite to catch stragglers:
uv run pytest -q
```

### 7.4 Green-gate commands by phase
| Phase | Gate |
|-------|------|
| **0** | `pytest tools_ core util tune sweep gui -q` · `mypy typing_.py _execution tune` · `ruff` |
| **1a** | `pytest tests/unit/tune -q` · `--doctest-modules …/__init__.py` · `mypy` · `ruff` |
| **1b** | `pytest tests/unit/tune -q` · `mypy …/_strategies` · `ruff` |
| **1c** | `pytest tests/unit/tune -q` · `--doctest-modules …/_qc_scorer.py` · `mypy` · `ruff` |
| **1d** | `pytest tests/unit/tune -q` · `--doctest-modules src/phenotypic/tune` · `mypy …/tune` · `ruff` · **+ §7.2 locks** |
| **2–5** | defined in each expanded full plan; always phase suite + `mypy` + `ruff` + new locks (Optuna lazy-import; Postgres integration tests gated on `PHENOTYPIC_TEST_PG_URL`; Phase-5 GUI ledgers) |
| **6** | `make -C docs html` clean · doctests pass · no `phenotypic.sweep` in `docs/`/`README.md` · tune + HPCC-Postgres pages in the toctree |

### 7.5 End-to-end acceptance (after the final phase, before the PR)
```bash
# MVP behavior (synthetic input — a small dir of load_synth_yeast_plate() renders + a synthetic spec.json):
uv run python -m phenotypic.tune spec.json -i ./synth_plates -o ./out      # grid/random run
ls out/deliverables/{best_pipeline,tuning_spec,param_importance}.json out/trials.parquet
uv run python -m phenotypic.tune spec.json -i ./synth_plates -o ./out      # re-point -o → resumes
uv run python -m phenotypic.tune spec.json -i ./synth_plates -o ./out2 --strategy grid   # golden-lock parity
# Optuna behind the extra (local SQLite):
uv sync --group dev --extra tune && uv run python -m phenotypic.tune spec.json -i ./synth_plates -o ./out3 --strategy tpe --n-trials 50
# Distributed path against the user's Postgres (HPCC): sbatch ~/util/postgres_server/pgserver.sh,
# read connection_info.txt, then:
uv run python -m phenotypic.tune spec.json -i ./synth_plates -o ./out4 --strategy tpe --n-trials 50 \
   --storage-url postgresql+psycopg://anguy344:$(cat ~/util/postgres_server/pgpassword.txt)@<node>:54399/<db>
# Docs build + full quality bar:
make -C docs html
uv run pytest -q && uv run mypy src/phenotypic/tune && uv run ruff check src/phenotypic/tune
```
Then **open the PR** → `main` from `redesign/param-sweep` (summary of phases, gates green, sweep
removed, screenshots regenerated, docs built; `🤖 Generated with Claude Code` trailer).

---

## 8. Risk register
| Risk | Owning phase | Mitigation |
|------|--------------|------------|
| `ScorerField` rejects `QCScorer` | 0 → 1d | Phase-0 guard **must** be `_make_require_value(base)` (replace the `isinstance(value, BaseOperation)` at `typing_.py:257`); symptom = `ValidationError` expecting a `BaseOperation`. |
| `tune/__init__.py` drops 1b configs → 1d red at import | 1c | write the **cumulative** `__init__` (keep `GridConfig`/`RandomConfig`/`StrategyConfig` + the 1a doctest). |
| Golden uncapturable after cutover | 0 (Task D) | capture **while sweep exists**; 1d reads the frozen JSON via core. |
| `mypy` fails on `Callable` | 0 (Task A) | add `from typing import Callable`. |
| Pipeline won't round-trip in `TuningSpec` | 1d | embed via custom `field_validator`/`field_serializer` → `to_json`/`from_json` (plain pydantic fails on abstract `ImageOperation`). |
| Two parallel agents edit a seam file live | 2–5 | orchestrator sole-committer; merge isolated worktrees one-at-a-time; scoped paths; never `git add -A`. |
| Outline executed as-is propagates wrong assumptions | 2–5 | expand to a full plan first; resolve each outline's "Review findings" (P2 channel-not-wired; P3 nested-key grammar + non-zero-arg gate; P4 `dict` not `tuple`; annotations `⊆`-validator-blindness). |
| `study.db` SQLite-WAL on NFS/SLURM | 2, 5, 6 | local-single-node = full write-back, SLURM/NFS = monitor-only; **Postgres-first** for distributed (user's `~/util/postgres_server/`); doc the wiring in Phase 6. |
| Optuna picks psycopg2 (not installed) | 2 | the `tune` extra installs `psycopg[binary]` (v3); use the **`postgresql+psycopg://`** URL scheme, not bare `postgresql://` (which defaults to psycopg2). |
| Postgres tests block hermetic CI | 2 | Postgres integration tests are **gated** (`PHENOTYPIC_TEST_PG_URL` / `@pytest.mark.postgres`); default suite uses local SQLite — CI stays green without a DB. |

---

## 9. Definition of done
- **Phase 1 (MVP):** `python -m phenotypic.tune spec.json -i ./plates -o ./out` runs grid/random over a
  calibration set, writes `deliverables/{best_pipeline,tuning_spec,param_importance}.json` +
  `trials.parquet`, resumes by re-pointing `-o`; `--strategy grid` reproduces the deleted sweep's grid
  (golden lock green); sweep gone; suite + `mypy` + `ruff` + doctests green; no new dep; code-review +
  annotation-adherence + regression passed.
- **Full feature (2–5 + annotations):** Optuna behind the `tune` extra (lazy), **Postgres-first store
  for SLURM** (psycopg3 `storage_url`, validated against the user's `~/util/postgres_server/`);
  `--auto-space` inference + reference-free scoring behind its meta-validation gate; supervised
  (Dice/IoU + count MAE) + multi-objective Pareto; the `/tune/` Dash co-pilot (FEATURES/WORKFLOWS/
  screenshot gates green); `detect/`+`enhance/` fields annotated. Each ships only when its expanded
  full-plan gate + locks pass. Deferred scope recorded in `DEFERRED-WORK.md`.
- **Phase 6 (docs):** tune-engine how-to + the **HPCC/Postgres distributed-tuning guide** (why
  Postgres, launching the Slurm pgserver, reading `connection_info.txt`, wiring the `storage_url`);
  `cli_reference` + autodoc for `phenotypic.tune`; README updated; `make -C docs html` clean; no
  `phenotypic.sweep` left in `docs/`/README.
- **Final:** the single `code-simplifier` pass + a full regression green **before** Phase 6; everything
  on `redesign/param-sweep`; **one PR opened to `main`.**
