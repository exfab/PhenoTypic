# Tune Engine — Interface & ABC Architecture

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
The **reusable abstract-class / Protocol layer** that the per-aspect specs plug into and
the implementation plans build on — designed to fit the existing `phenotypic.abc_`
pydantic backbone (`BaseOperation`, the class registry, `OperationField`,
`model_json_schema`).

- **Status:** Design settled (pre-implementation). Defines the interfaces; the plans
  implement them.
- **Maps to:** master §4 (the five components + Protocols), §9 (shared batch runner), §11
  (file layout). Realises the contracts in [`qc-objective-mapping.md`](qc-objective-mapping.md)
  (Scorer), [`robust-evaluation.md`](robust-evaluation.md) (Evaluator + PruningChannel),
  [`search-space-inference.md`](search-space-inference.md) (domains + proposal),
  [`optuna-integration.md`](optuna-integration.md) (SearchStrategy), and
  [`screening-importance.md`](screening-importance.md).

---

## 1. Purpose & reusability goals

Two faces of reusability drive every choice here:

1. **Swappability** — drop in a new `SearchStrategy` (Grid→Random→Optuna→Ax) or a new
   `Scorer` (QC/Supervised/ReferenceFree/Composite) *without touching the engine or the
   Evaluator*.
2. **Serialization reuse** — `tuning_spec.json` round-trips the whole spec through the
   **existing** pydantic/registry machinery (`from_json`-style), and exposes
   `model_json_schema()` as the MCP's machine-readable contract — without a parallel
   serialization stack.

---

## 2. The three tiers

| Tier | Kind | Members | Why |
|------|------|---------|-----|
| 1 | **Pydantic ABCs** (config + algorithm) | `Scorer` (+ subclasses), `Evaluator` | serializable config, `model_json_schema`, polymorphic via the registry — like `BaseOperation`/`SetAnalyzer` |
| 2 | **Protocols** (runtime / stateful seams) | `SearchStrategy`, `PruningChannel` | hold live state (Optuna study/trial); structural typing → fakes/future impls "just work" |
| 3 | **Pydantic value-models** | domains, `Knob`/`Excluded`/`SearchSpace`/`InferredSearchSpace`, `StrategyConfig`, `Budget`, `TuningSpec` | the data the optimizer + serialization carry |

---

## 3. Tier 1 — pydantic ABCs

### 3.1 `Scorer`

```python
class Scorer(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # All emitted terms are [0,1] higher-is-better (robust-eval §5: Scorer owns
    # normalization, fixed/threshold-anchored, never min-max-over-trials).
    @abstractmethod
    def score_image(self, image, result) -> dict[str, float]: ...

    def finalize(
        self, aggregated_terms: Mapping[str, float],
        per_image_results: Sequence[dict], full_measurements,
    ) -> float | dict[str, float]:
        """Default = weighted mean of aggregated terms (per-image-only scorers)."""
        return _weighted_mean(aggregated_terms, self._term_weights())

    def availability(self, ctx: "DataContext") -> "ScorerAvailability":
        """Which terms are active / whether to abstain (qc §6 graceful degradation)."""
        ...
```

- **Two-phase** (qc §3): `score_image` → per-image terms; the Evaluator robust-aggregates;
  `finalize` adds batch-only terms (e.g. the ICC/MAD panel) and fuses. The **default
  `finalize`** covers per-image-only scorers (`SupervisedScorer`); `QCScorer` overrides it.
- **Normalization contract** (robust-eval §5): terms are `[0,1]` higher-is-better; the
  Evaluator *validates* the range, never transforms. A `dict`-returning `finalize` is the
  multi-objective path (master §7).
- **`availability`** expresses the qc §6 matrix (Count-only / full panel / abstain) so the
  Evaluator/engine degrade predictably.
- **Subclasses** `QCScorer`, `SupervisedScorer`, `ReferenceFreeScorer`, `CompositeScorer`,
  polymorphic-deserialized via the registry (§6). **Reuse:** `QCScorer` holds
  `QualityCheck` analyzers as `PolymorphicField` fields (it *calls* them, qc §1);
  `CompositeScorer` holds `list[Scorer]` + weights, recursively.

### 3.2 `Evaluator`

```python
class Evaluator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    stability_weight: float = 0.5          # λ (robust-eval §4)
    cv_group: str | None = None            # auto-inferred from QC grouping
    stratify_by: list[str] | None = None
    # ... held-out, rung ladder, caching config ...

    def evaluate(self, params: Mapping[str, Any], *,
                 channel: "PruningChannel") -> "EvaluationResult": ...
```

The robustness algorithm is **fixed** (one implementation) → a model, not an ABC. `evaluate`
runs the uniform 3-step loop (robust-eval §3): build pipeline from `params` → `score_image`
per calibration image (via the `Executor`, §7) → robust-aggregate each term
(`median − λ·IQR`) → `scorer.finalize`. Returns `EvaluationResult` (objective,
per-image/per-fold breakdown, held-out, diagnostics; a pydantic value-model).

---

## 4. Tier 2 — Protocols + configs

### 4.1 `SearchStrategy` (Protocol) + `StrategyConfig`

```python
class SearchStrategy(Protocol):
    def suggest(self) -> tuple[Mapping[str, Any], "PruningChannel"]: ...
    def register_result(self, params: Mapping[str, Any],
                        result: "EvaluationResult", *, pruned: bool = False) -> None: ...
    def is_exhausted(self) -> bool: ...
```

Runtime-stateful (Optuna study/trial; Grid cursor; Random RNG) → a **Protocol**, so a test
fake or a future `AxStrategy` conforms structurally. Per-worker: **one in-flight trial**
(optuna §7); concurrency = one instance per worker.

Construction is config-driven, and **the config knows how to build its strategy** (a factory
method per subclass — no central dispatch):

```python
class StrategyConfig(BaseModel, ABC):           # PolymorphicField-serialized (§6)
    seed: int = 0
    @abstractmethod
    def build(self, space: SearchSpace, store: "StudyStore") -> SearchStrategy: ...

class GridConfig(StrategyConfig): ...           # → GridStrategy
class RandomConfig(StrategyConfig): n_trials: int ...   # → RandomStrategy
class OptunaConfig(StrategyConfig):             # → OptunaStrategy (tune extra)
    sampler: Literal["tpe", "cmaes"] = "tpe"; n_trials: int; prune: bool = False ...
```

`GridConfig`/`RandomConfig` ship in Phase 1 (zero-dep); `OptunaConfig.build` lazy-imports
Optuna and raises the actionable extra error if absent (optuna §10).

### 4.2 `PruningChannel` (Protocol)

```python
class PruningChannel(Protocol):
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...
```

`NoOpChannel` (Grid/Random; `should_prune` always `False`) ships in Phase 1; the
Optuna-backed channel (live trial) in Phase 2. Keeps the Evaluator **Optuna-free**
(robust-eval §7, optuna §6).

---

## 5. Tier 3 — value-models

- **Domains** — `Categorical`/`IntRange`/`FloatRange`/`Fixed` as a **frozen pydantic
  discriminated union** (`Field(discriminator="kind")`); a *closed* set, so a union (not the
  registry) is right. *(Refines the Plan-1 draft, which sketched dataclasses — switch to
  pydantic for uniform serialization + schema.)*
- **Proposal** — `Knob` (key/domain/source/needs_review/description/conditional_on),
  `Excluded` (key/reason), `SearchSpace` (knobs), `InferredSearchSpace` (knobs + excluded +
  summary props + `to_search_space()`), frozen pydantic models (search-space §7).
- **`Budget`** — `n_trials` (counts completed+pruned) / `wall_clock` / `max_failures`
  (optuna §8).
- **`TuningSpec`** — the `tuning_spec.json` model (§6).

---

## 6. Polymorphic serialization

`OperationField` is **generalized into a type-agnostic `PolymorphicField`** (in
`tools_/typing_.py`): it serializes any pydantic model to `{"class": <name>, "params":
<model_dump>}` and reconstructs via the existing
`SerializablePipeline._find_class_in_phenotypic` registry. `OperationField` becomes a thin
alias; `Scorer`/`CompositeScorer` nesting and `StrategyConfig` reuse the same field. A
scorer/strategy extends **exactly like an operation**: export a subclass → it round-trips.

The whole spec is one pydantic model:

```python
class TuningSpec(BaseModel):
    search_space: SearchSpace
    scorer: Scorer                  # PolymorphicField — any subclass
    evaluator: Evaluator
    strategy: StrategyConfig        # PolymorphicField — any subclass
    budget: Budget
```

`TuningSpec.model_validate_json()` / `.model_dump_json()` round-trips everything through one
mechanism; `TuningSpec.model_json_schema()` is the MCP contract (deferred MCP doc) and the
CLI/Dash form schema.

---

## 7. The shared `Executor` seam (master §9/§11)

```python
class Executor(Protocol):
    def run(self, work: Callable[[Item], R], items: Sequence[Item]) -> list[R]: ...
```

`LocalExecutor` (joblib) + `SlurmExecutor` (array) are **extracted from `_sweep_cli` into a
shared `_execution` module** both `_sweep_cli` and `_tune_cli` import. `sweep` injects a
work-fn that saves each measurement + manifest; the tune `Evaluator` injects one that scores
+ aggregates (with the per-image cache + the pruning channel). **Two parallelism levels:**
the `Executor` parallelizes *images within a call*; tune's **distributed ask-and-tell**
(optuna §7) parallelizes *trials across workers* — so in distributed mode the Evaluator runs
images serially within its worker (a config flag) to avoid double-parallelizing.

---

## 8. `TuningEngine`

The orchestrator the CLI/MCP/Dash wrap (master §4). Holds the `TuningSpec`, an `Executor`,
and a `StudyStore`; drives the ask-and-tell loop:

```python
study = engine.create_study(spec, executor, store)
while not strategy.is_exhausted():
    params, channel = strategy.suggest()
    result          = evaluator.evaluate(params, channel=channel)
    strategy.register_result(params, result, pruned=result.pruned)
return study.best()        # or Pareto front (multi-objective)
```

In distributed mode this loop *is* the per-worker body (each worker holds its own strategy
bound to the shared study). Owns the budget, convergence stop, screening hand-off
(screening §3), and reporting.

---

## 9. The two extension seams (the reusability payoff)

- **New strategy:** implement the `SearchStrategy` Protocol + a `StrategyConfig` subclass
  with a `build()`; the engine is untouched (it only calls `suggest`/`register_result`/
  `is_exhausted`).
- **New scorer:** subclass `Scorer`, export it (registry finds it); the engine + Evaluator
  are untouched (they call `score_image`/`finalize`/`availability`).

The **Evaluator is Scorer-agnostic** and the **engine is Strategy-agnostic** — the two
seams that keep the system open.

---

## 10. Package layout

```
src/phenotypic/tune/
  __init__.py            # public: TuneSpec/TuningSpec, infer_search_space, Scorer,
                         #   domains, build helpers
  _search_space/         # domains (pydantic union), proposal, _tune_spec, _infer
  _scorers/              # _base.py (Scorer ABC), _qc.py, _supervised.py,
                         #   _reference_free.py, _composite.py
  _strategies/           # _protocol.py (SearchStrategy), _pruning.py (PruningChannel),
                         #   _config.py (StrategyConfig + subclasses), _grid.py,
                         #   _random.py, _optuna.py (lazy)
  _evaluator.py          # Evaluator + EvaluationResult
  _engine.py             # TuningEngine
  _study_store.py        # StudyStore (Optuna SQLite | homegrown journal fallback)
  _spec.py               # TuningSpec, Budget
  _screening.py          # importance (fANOVA | RF-permutation fallback) + freezing
  _tune_cli/             # mirrors _sweep_cli; shares _execution
src/phenotypic/_execution/   # Executor Protocol + Local/Slurm (shared with sweep)
src/phenotypic/tools_/typing_.py   # PolymorphicField (OperationField becomes an alias)
```

---

## 11. Naming conventions

Explicit, matching the project (`.apply()` for ops, `.analyze()` for analyzers):
`Scorer.score_image` / `finalize` / `availability`; `Evaluator.evaluate`;
`SearchStrategy.suggest` / `register_result` / `is_exhausted`; `StrategyConfig.build`;
`Executor.run`; `TuningEngine.create_study` / `optimize`. No generic `run()`/`process()` on
domain types.

---

## 12. Mapping to the phasing

- **Phase 1 (zero-dep):** `Scorer` ABC + **Count-only `QCScorer`**; `SearchStrategy`
  Protocol + `GridStrategy`/`RandomStrategy` + `GridConfig`/`RandomConfig`;
  `PruningChannel` + `NoOpChannel`; `Evaluator` (CV-only MVP); `TuningEngine`;
  `PolymorphicField`; domains/proposal/`TuningSpec`; the extracted `Executor`.
- **Phase 2:** `OptunaStrategy`/`OptunaConfig` (tune extra); Optuna-backed `PruningChannel`;
  ASHA; `_study_store` SQLite; fANOVA in `_screening`.
- **Later:** `SupervisedScorer`/`ReferenceFreeScorer`/`CompositeScorer`; MCP (deferred);
  Dash co-pilot.

---

## 13. Testing the interface layer

- **Protocol conformance** — `GridStrategy`/`RandomStrategy` (and a fake) satisfy
  `SearchStrategy`; `NoOpChannel` satisfies `PruningChannel` (runtime-checkable Protocols).
- **Registry round-trip** — `TuningSpec.model_validate_json(spec.model_dump_json())` is
  identity for a spec with a `QCScorer` + `CompositeScorer` nesting + a `GridConfig`;
  `PolymorphicField` reconstructs the concrete subclass.
- **Schema** — `TuningSpec.model_json_schema()` is well-formed (the MCP/CLI/Dash contract).
- **Swap tests** — the engine drives a fake strategy + a fake scorer unchanged; the
  Evaluator scores with a stub scorer.
- **`OperationField` back-compat** — existing operation round-trips stay green after the
  `PolymorphicField` generalization (the alias preserves behavior).

---

## 14. Resolved choices / open questions

**Resolved:** layered tiers (§2); `Scorer`/`Evaluator` pydantic ABCs (§3);
`SearchStrategy`/`PruningChannel` Protocols + `StrategyConfig.build()` factory (§4);
pydantic domains/proposal/`TuningSpec` (§5); `PolymorphicField` generalizing `OperationField`
(§6); extracted shared `Executor` (§7); the two extension seams (§9); the package layout
(§10).

**Open / for the plans:**

- The `_execution` module's exact home (a new top-level `_execution` vs under `_cli`).
- `StudyStore`'s homegrown (no-Optuna) journal fallback shape (Phase 1 persistence without
  Optuna).
- Whether `Evaluator` ever needs to be an ABC (only if a non-robustness evaluation variant
  appears — none planned).
- Plan-1 update: switch domains/proposal from `dataclasses` to pydantic value-models to
  match §5.
