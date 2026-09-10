# Optuna Integration (`OptunaStrategy`)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
Deep dive on **master §4** (the `SearchStrategy` Protocol, Optuna backend) and
**decision D3**: how Optuna provides TPE/CMA-ES/NSGA-II sampling, ASHA pruning, and a
persistent resumable study behind the thin `SearchStrategy` seam — as an **optional
extra**, never core.

- **Status:** Design settled (pre-implementation). **Phase 2** (the Optuna backend
  lands after the Phase-1 dependency-free engine core).
- **Maps to:** master §4 (`SearchStrategy`), D3, §6 (shared study / drivers), §7
  (multi-objective), §8 (study storage), §10 (dependency policy), §12 (resume), §14.
  Backs [`screening-importance.md`](screening-importance.md)'s fANOVA
  (`get_param_importances`) and supplies the ASHA pruner that
  [`robust-evaluation.md`](robust-evaluation.md) §7 reports into. Consumes the
  `SearchStrategy` Protocol + the domains from
  [`search-space-inference.md`](search-space-inference.md).

---

## 1. Purpose and where it fits

`OptunaStrategy` is one implementation of the `SearchStrategy` Protocol
(`suggest()` / `register_result()` / `is_exhausted()`). It wraps an Optuna study to
provide sample-efficient sampling (TPE default), multi-fidelity pruning (ASHA),
multi-objective search (NSGA-II), parameter importance (fANOVA), and a **persistent,
resumable, concurrently-writable** study. The homegrown `GridStrategy` /
`RandomStrategy` (Phase 1) implement the same Protocol with zero dependencies; Optuna is
the **default once installed**, and the seam stays open for a future `AxStrategy`.

---

## 2. What master §4 / D3 / §10 lock (documented, not re-litigated)

Optuna is the default backend behind the thin Protocol; it lives in an **optional
`tune` extra** (`uv sync --extra tune`), **never core**; Grid + Random + the importance
fallback are dependency-free; the seam stays open for `AxStrategy`. `optuna` is
pure-Python / cross-platform (no Windows exclusion). Promoting it to core later (so TPE
is the out-of-box default) is a one-line change.

---

## 3. Control model — ask-and-tell

`OptunaStrategy` drives Optuna through its **ask-and-tell** API, not `study.optimize()`,
so the **engine drives the loop** and a human (CLI/Dash) and an agent (MCP) can interleave
against the **same study** (D6):

```
suggest():           trial = study.ask(); params = _materialize(trial); stash trial; return params (+ channel, §6)
register_result():   study.tell(stashed_trial, score)        # or state=PRUNED / FAIL
is_exhausted():      count(study) >= n_trials                 # §8
```

**Conditional materialization.** `_materialize` walks the `SearchSpace` knobs **in
dependency order**: unconditional knobs and presence `__enabled__` knobs first, then each
conditional child **only if** its `conditional_on` parent value matches what was sampled
this trial (define-by-run). An inactive child is simply **absent** from the trial's param
dict — which Optuna's samplers handle natively (TPE conditions on presence).

---

## 4. Domain → Optuna distribution mapping

| `SearchSpace` domain | Optuna call |
|----------------------|-------------|
| `Categorical(choices)` | `trial.suggest_categorical(key, list(choices))` |
| `IntRange(low, high, step, log)` | `trial.suggest_int(key, low, high, step=step, log=log)` |
| `FloatRange(low, high, log)` | `trial.suggest_float(key, low, high, log=log)` |
| `Fixed(value)` | **not suggested** — injected as a constant into the params dict (never a trial dimension) |
| `conditional_on=(…)` | the `suggest_*` call is **skipped** unless the parent value matches (§3) |

**Guard:** Optuna forbids `suggest_int(step≠1, log=True)`; inference/strategy normalises
such a knob to `step=1` (log) or drops `log` (stepped), with a logged note.

---

## 5. Samplers

- **TPE — the default.** Handles our **mixed categorical/conditional** space and is
  sample-efficient; Optuna's default. Seeded for reproducibility (§8).
- **CMA-ES — continuous-dominant / focused round.** `CmaEsSampler` optimises the
  continuous knobs and **falls back to Optuna's independent sampling for
  categoricals/conditionals** (built-in behaviour). The engine **warns**: *"CMA-ES suits
  continuous-dominant spaces; for categorical-heavy spaces prefer TPE, or run CMA-ES in
  the post-screening focused round once categoricals are frozen."* — pointing at its real
  niche (the screening focused round, where the categoricals are `Fixed`).
- **GP** — `GPSampler` behind the same seam for low-dim continuous with expensive evals
  (optional, exposed but not advertised in the CLI roster).
- **NSGA-II — multi-objective, auto-selected** when the `Scorer` returns a dict (§9).

`--strategy {grid,random,tpe,cmaes}` selects the backend (master §6); `grid`/`random` are
the homegrown Phase-1 strategies.

---

## 6. Pruning — ASHA via a generic channel

The **default pruner is ASHA** (Optuna's asynchronous `SuccessiveHalvingPruner`), not
`HyperbandPruner`: ASHA makes prune decisions **asynchronously** without waiting for a
rung to fill — the right fit for our parallel joblib/SLURM execution. Its config is
**derived from the Evaluator's rung ladder** (robust-eval §7) so the two cannot disagree:
`min_resource` = first-rung plates (~6), `reduction_factor` = 3, the resource/`step` =
**number of calibration plates**. Pruning is **opt-in** and the **explore round runs
unpruned** (screening §4, to keep fANOVA's importance sample unbiased).

**The generic pruning channel (keeps the Evaluator Optuna-free).** The Evaluator must
report intermediate scores and check "should I stop?" mid-evaluation, but it must not
import Optuna (it ships in Phase 1; pruning is Phase 2). So `suggest()` returns, alongside
the params, a small per-trial **channel**:

```python
class PruningChannel(Protocol):
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...
```

- `OptunaStrategy` backs it with the live trial (`report → trial.report(value, step)`,
  `should_prune → trial.should_prune()`).
- `GridStrategy` / `RandomStrategy` supply a **no-op channel** (`should_prune` always
  `False`).
- A pruned outcome flows back through `register_result(..., pruned=True)` →
  `study.tell(trial, state=TrialState.PRUNED)`.

This is a **passthrough**, so ASHA's cross-trial comparisons at each `step` are exactly
Optuna's — identical pruning accuracy, with the Evaluator depending only on the two-method
Protocol. Phase 1 ships with the no-op; Phase 2 swaps in the Optuna-backed channel with
**zero Evaluator changes**. **Multi-objective disables pruning** (Optuna pruners are
single-objective, §9).

---

## 7. Persistence & concurrency

- **Storage.** A local run defaults to run-local **`study.db`** via Optuna
  `RDBStorage` (SQLite/WAL). A Slurm run instead defaults to one absolute,
  run-local `journal://` file with shared-filesystem locking. Operators may
  explicitly select a supported external RDB such as PostgreSQL. **Distributed
  SQLite is rejected** rather than treated as shared storage (master §8).
- **Distributed ask-and-tell over the shared study.** Every worker (joblib task / SLURM
  array task) — and every driver (engine, human-via-Dash, agent-via-MCP) — opens the
  **same study by name + storage URL**, then `ask`s a trial, evaluates it (the pruning
  channel reports to **its own** trial), and `tell`s the result, independently. This is
  Optuna's standard distributed pattern: uniform across local + SLURM, pruning-across-
  processes works natively, and it **is** the multi-driver shared study (D6).
- **Per-worker strategy state.** Each worker holds its own `OptunaStrategy` bound to the
  shared study, stashing **one in-flight trial** between `suggest()`/`register_result()`.
  The Protocol does **not** require multi-trial/thread-safe strategies — concurrency comes
  from running **one instance per worker**, each serial `ask→evaluate→tell` (image-level
  parallelism within a trial stays in the batch runner).
- **External RDB option.** Heavy parallel Slurm writes plus live human writes may
  use PostgreSQL — a **storage-URL config change**, not a rewrite (the seam).

---

## 8. Budget, failure, resume & reproducibility

- **Budget.** `is_exhausted()` counts **completed + pruned** trials in the shared study
  against `n_trials` — so the user gets *n real evaluations*, not *n attempts*. **Failed**
  trials don't consume the budget, but a global **max-failures cap** prevents a
  pathological loop. Optional wall-clock / no-improvement-in-K stop conditions compose
  (master §12). Minor parallel overshoot (≤ n_workers) is tolerated.
- **Dead-worker handling.** Optuna `RDBStorage` **heartbeat** detects a worker that died
  mid-trial (orphaned "running" trial). Because a node death is an **infra fault, not the
  candidate's** (consistent with robust-eval's infra ≠ candidate-bad principle), the
  trial's params are **re-enqueued once** (retry). Candidate-*caused* eval exceptions
  still → `FAIL` per robust-eval §10 (deterministic — retrying would just fail again).
- **Resume.** Re-invoking re-opens the study by name+storage; **persisted trials load
  exactly** and new trials continue from the reconstructed sampler state (master §12).
- **Reproducibility (honest guarantee, refines master §12).** A fixed sampler seed gives
  **bit-identical** results only for **sequential, single-worker** runs (`--deterministic`
  mode). **Default parallel runs are reproducible *in distribution*** (same seed →
  statistically-equivalent ensemble), because distributed workers race on trial order.
  `--deterministic` is **opt-in** (you don't sacrifice parallelism by default; it's for
  debugging / publication-grade reproduction).

---

## 9. Multi-objective

A dict-returning `Scorer` (or `--multi-objective`) creates a multi-objective study with
`directions=["minimize"] * n` — every objective is a normalized cost, lower-is-better
(robust-eval §5) — sampled by **NSGA-II** (or multi-objective TPE). The Pareto front is
`study.best_trials`; the report draws the trade-off curve for a knee-point pick (master
§7). **Pruning is disabled** for multi-objective studies (Optuna pruners are
single-objective); the stability/dispersion axis can instead be exposed as its own
objective (robust-eval §9).

---

## 10. Dependency boundary & the `AxStrategy` seam

- `import optuna` happens **lazily inside the `OptunaStrategy` module**, never at package
  import. Requesting `--strategy tpe`/`cmaes` (or fANOVA importance) without the extra →
  a clear, actionable error: *"Optuna is required for this strategy; install
  `phenotypic[tune]` (`uv sync --extra tune`)."*
- The `SearchStrategy` Protocol is the only contract; a future **`AxStrategy`** (BoTorch)
  drops in behind the same seam without touching the engine, the Evaluator, or the
  scorers. fANOVA importance uses Optuna when present and the RF-permutation fallback
  otherwise (screening §4).

---

## 11. Testing

- **Control model** — `suggest()` materializes a conditional space (inactive children
  absent); `register_result()` tells the right stashed trial; round-trip on a tiny study.
- **Domain mapping** — each domain → the correct `suggest_*` call; `Fixed` injected as a
  constant; the `step≠1 ∧ log` guard fires.
- **Pruning channel** — Optuna-backed channel reports + prunes; the no-op channel never
  prunes; a pruned outcome tells `PRUNED`; the Evaluator imports no Optuna (assert via a
  dependency check on the Phase-1 path).
- **Persistence/resume** — a study resumes from its configured backend (local
  SQLite, a shared journal file, or an external RDB) with persisted trials intact;
  `--deterministic` (sequential) reproduces bit-identically under a fixed seed; a parallel
  run reproduces the *best-score distribution* across seeds (statistical, not exact).
- **Budget/failure** — `is_exhausted()` counts completed+pruned; failures don't consume
  budget but hit the max-failures cap; a simulated dead-worker (orphaned running trial) is
  re-enqueued once.
- **Multi-objective** — a dict Scorer yields a Pareto front; pruning is disabled.
- **Dependency boundary** — `--strategy tpe` without the extra raises the actionable
  install error; `grid`/`random` work without Optuna installed.

Fixed seeds throughout; Optuna-requiring tests are skipped when the extra is absent.

---

## 12. Resolved choices / open questions

**Resolved:**

1. **Control model** — ask-and-tell (engine drives; conditional materialization).
2. **Samplers** — TPE default; CMA-ES native-fallback + warning; GP optional; NSGA-II
   auto for multi-objective.
3. **Pruning** — ASHA (`SuccessiveHalvingPruner`) via the generic channel; config from
   the Evaluator ladder; explore round unpruned; off for multi-objective.
4. **Concurrency** — distributed ask-and-tell over a shared journal file or
   supported external RDB; distributed SQLite is rejected; per-worker
   one-in-flight-trial strategy.
5. **Budget** — count completed+pruned vs `n_trials` + a max-failures cap.
6. **Dead-worker** — heartbeat → retry once (infra); candidate errors → `FAIL`.
7. **Reproducibility** — bit-identical only for `--deterministic` (sequential); parallel
   reproducible-in-distribution.
8. **Dependency** — lazy import behind the `tune` extra; actionable error; `AxStrategy`
   seam preserved.

**Still open (planning / empirical):**

- The **max-failures cap** default and the retry count for dead-worker trials.
- ASHA `min_resource` / `reduction_factor` final defaults (shared with robust-eval §7,
  master §14).
- Whether to promote `optuna` to core (making TPE the out-of-box default) post-Phase-2.
- The threshold for selecting external PostgreSQL instead of the shared journal backend.
