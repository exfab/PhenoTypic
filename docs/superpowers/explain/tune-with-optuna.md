# How tuning works in `phenotypic.tune` (with Optuna)

A comprehensive walk-through of the math behind hyperparameter tuning: how
candidate parameters are *decided*, how they are *scored*, which ones are
*found important*, and how the whole thing forms one ask-and-tell loop.

> All file:line references point at `src/phenotypic/tune/…`. Everything in the
> module follows one sign convention: **bounded cost in `[0,1]` (`0` = perfect,
> `1` = worst), everything is minimized** (`_MINIMIZE = "minimize"`,
> `_strategies/_optuna_support.py`). Each scorer emits its *natural* per-term
> value and declares a `Sense`; the base `Scorer.score_image` orients it into
> cost via `to_cost` (`_scoring/_orient.py`).

---

## 0. The big picture

Tuning is an **ask-and-tell optimization loop**:

1. A **strategy** (Optuna / random / grid) proposes a parameter combo.
2. An **evaluator** builds a pipeline with those parameters and scores it on a
   calibration set of images.
3. A **scorer** turns segmentation quality into a scalar (or a vector for
   multi-objective).
4. The result is *told* back to the strategy so the next proposal is
   better-informed.

The orchestrator is `TuningEngine.optimize()` (`_engine.py:49`). Per trial:

```
strategy.suggest()  →  evaluator.evaluate(...)  →  store.append(Trial)  →
strategy.register_result(...)  →  bump budget counters (n_trials, max_failures)
```

Resume is handled by either fast-forwarding a deterministic strategy
(replay `suggest()` N times) or by Optuna's own RDB storage restoring the
sampler state.

---

## 1. How a parameter becomes a tunable dimension

Before any sampling, the search space is *inferred* from the pipeline's pydantic
operation fields in `infer_search_space` (`_search_space/_infer.py:685`). Each
field becomes a **Knob** (a tunable dimension) or an **Excluded** record. Two
tiers decide the domain:

### Tier 1 — explicit `TuneSpec` (`_infer.py:325`)
If a field carries a `TuneSpec`, it wins outright: explicit `low/high/step/log`
or `categories`. A subset assertion enforces `TuneSpec ⊆ Field-bounds`, so you
can never tune outside the operation's own pydantic validation.

### Tier 2 — type heuristics (`_infer.py:457`)

| Field type | Inferred domain |
|---|---|
| `bool` | `Categorical(True, False)` |
| `Literal[...]` / `Enum` | `Categorical(choices)` |
| numeric **with** `ge/le` bounds | `IntRange` / `FloatRange(low, high)` |
| numeric **unbounded**, default `d` | window `[d/4, d·4]` (`_DEFAULT_UNBOUNDED_FACTOR = 4`) |
| `str` / `Path` / `ndarray` | **Excluded** (open set / not scalar-tunable) |

**Log-scale auto-trip** is the key heuristic. A range is flagged `log=True`
when it spans enough orders of magnitude:

- unbounded heuristic: `high/low > 10` (`_LOG_SPAN_THRESHOLD`)
- explicit bounds: `high/low ≥ 100` (`_BOUNDED_LOG_SPAN_THRESHOLD`)

Non-positive defaults (`d ≤ 0`) are excluded because `[d/4, d·4]` is
meaningless; unbounded knobs are stamped `needs_review=True`. Inference recurses
exactly **one level** into operation-valued fields.

### Domains → Optuna suggest calls (`_strategies/_optuna.py:316`)

```python
Categorical → trial.suggest_categorical(key, choices)
IntRange    → trial.suggest_int(key, low, high, step, log)
FloatRange  → trial.suggest_float(key, low, high, step, log)
Fixed       → injected as a constant (never a trial dimension)
```

Two guards exist because Optuna forbids `step ≠ 1` under `log=True`: an
`IntRange` with a step normalizes to `step=1`; a stepped `FloatRange` drops its
step. **Conditional knobs** (define-by-run) are only suggested when their parent
values match (`knob.is_active(chosen)`, `_search_space/_space.py:92`) — an op's
sub-parameters only enter the space when that op is actually selected.

---

## 2. How the *next* parameters are decided — the samplers

The sampler is chosen in `_make_sampler` (`_optuna.py:248`); all are seeded:

```python
multi-objective → NSGAIISampler(seed)   # auto, overrides any choice
"tpe"   → TPESampler(seed)               # default
"cmaes" → CmaEsSampler(seed)
"gp"    → GPSampler(seed)
"nsga2" → NSGAIISampler(seed)
```

### TPE (the default) — the actual math

TPE = **Tree-structured Parzen Estimator**. Instead of modeling the objective
`p(y|x)` like a Gaussian process, it models the *inputs given the outcome*
`p(x|y)` and inverts via Bayes. After `n_startup_trials` random trials, it sorts
observed trials by score and splits them at a quantile γ into:

- `l(x)` = density of parameter values among the **good** trials (top γ)
- `g(x)` = density among the **rest**

Each density is a Parzen-window (kernel density) estimate. The next candidate
maximizes the **Expected Improvement**, which for TPE reduces to maximizing the
density ratio:

```
EI(x)  ∝  l(x) / g(x)
```

Intuition: propose values that were *common among good trials and rare among
bad ones*. Because each dimension gets its own KDE and conditional dimensions
form a tree, TPE handles the mixed categorical/conditional pipeline space
natively — which is why it is the default.

### The others
- **CMA-ES** — evolution strategy adapting a covariance matrix over a
  continuous space; strong once the space is continuous-dominant (e.g. a
  post-screening focused round); falls back to independent sampling for
  categoricals.
- **GP** — Gaussian-process Bayesian optimization for low-dimensional,
  expensive evaluations.
- **NSGA-II** — genetic multi-objective sampler, auto-selected whenever there
  are ≥ 2 objectives.

### Random and Grid (no learning)
- **`RandomStrategy`** (`_strategies/_random.py`) — seeded i.i.d. sampling.
  Continuous log floats sample as `exp(uniform(log low, log high))`;
  `register_result` is a no-op.
- **`GridStrategy`** (`_strategies/_grid.py`) — exhaustive conditional Cartesian
  product via `enumerate_grid` (`_strategies/_enumerate.py:31`). Conditional
  knobs only enter the product when active. **Raises on a continuous
  `FloatRange`** (not enumerable).

---

## 3. How one proposal is scored — rung ladder + robust aggregation

This is the heart of `Evaluator.evaluate` (`_evaluation/_evaluator.py:248`), and
where "accuracy over speed" becomes real math.

### Step A — fidelity ladder (ASHA rungs)
Rather than always scoring every plate, images are scored in **growing blocks**
(`_rung_sizes`, `_evaluator.py:222`):

```
first rung = max(rung_floor=6, ceil(n / rung_factor=3))
each next  = previous × rung_factor
last rung  = all n images
```

If the set is too small to yield `min_rungs = 2` distinct rungs, the ladder
self-disables to a single full-fidelity pass. Each image is scored **once**
(memoized across rungs).

### Step B — robust per-term aggregation
After each rung, every scoring term's per-image **cost** values are reduced not
by a plain mean but by a **spread-penalized median**, then clamped to `[0,1]`
(`_robust_aggregate`, `_evaluator.py:55`):

```
term = clamp01( median(b₁ … bₖ) + λ · IQR(b₁ … bₖ) )
```

with `λ = stability_weight = 0.5` and `IQR = Q75 − Q25`
(`_aggregate_math.py:28`). The `+λ·IQR` term **penalizes parameters that work
inconsistently across plates** — a config that is brilliant on two plates and
terrible on three (high IQR) is dragged toward worse cost and loses to a steady
one. The clamp matters: `median + λ·IQR` can reach `~1+λ` on an unstable-and-bad
term, so clamping keeps every per-child cost in `[0,1]` (the invariant the
Tchebycheff composite asserts).

### Step C — finalize to the objective
`scorer.finalize(terms)` collapses the term dict to the scalar Optuna minimizes
(or a dict for multi-objective; see §6). The shared projection is
`mean(objectives.values())` (`_scoring/_scorer.py: project_objectives_to_scalar`).

### Step D — pruning decision (between rungs only)

```python
channel.report(running_score, scored)
if rung_index < len(rungs) - 1 and channel.should_prune():
    return EvaluationResult(pruned=True, ...)   # partial aggregate kept
```

The Optuna pruner is **Successive Halving / ASHA** (`_optuna.py:218`):

```python
SuccessiveHalvingPruner(min_resource=rung_floor, reduction_factor=rung_factor)
```

At each rung, trials are ranked by their interim value; only the **top
`1/reduction_factor`** (top third by default) survive to the next rung, the rest
are killed (under `direction=minimize` the ASHA pruner keeps the **lowest-cost**
third). A clearly-losing config dies after 6 plates instead of wasting the
full set. Pruning is **disabled during explore rounds** (keeps the importance
sample unbiased) and **disabled for multi-objective** (Optuna pruners are
single-objective only). Grid/Random use a `NoOpChannel` that never prunes.

### Failure taxonomy (`_evaluator.py:283`)
- Candidate won't **build** → hard `failed=True`, score floored to `1.0`.
- **One** image raises → that image contributes the worst term (`1.0`) to
  *every* term and the loop continues (failures honestly drag the aggregate
  **up** (toward worse cost), `_aggregate`, `_evaluator.py:385`).
- **All** images raise → whole-candidate `failed=True`.

### Two diagnostic flags
- **`gap`** (`_per_trial_dispersion`, `_evaluator.py:76`) = relative IQR of the
  *primary* term computed on the **goodness-equivalent `1 − cost`**, with the
  denominator floored at `_GAP_EPS ≈ 0.02` — a cheap instability/overfit flag
  (not a held-out gap). Computing it on `1 − cost` moves the divide-by-zero
  singularity to the harmless *bad-cost* end so a near-perfect candidate
  (cost ≈ 0) does not explode.
- **`suspicious`** (`_is_suspicious`, `_evaluator.py:112`) =
  `score ≤ 0.3 AND Count ≥ 0.7` (cost) — catches the gaming signature where a
  pipeline has low *cost* (looks good) *because* it under-detects (high Count
  cost). A missing `Count` term defaults to best cost (`0.0`), so absent-Count
  candidates are never flagged.

---

## 4. The scoring strategies (the objective itself)

Every scorer returns named per-image terms, each emitted as a **natural** value
and oriented to **cost ∈ [0,1]** by the base `score_image` (a
`Sense.HIGHER_BETTER` term like Dice is complemented `1 − value`; a
`Sense.LOWER_BETTER` divergence passes through). There are four.

### A. SupervisedScorer — ground truth available (`_scoring/_supervised.py`)
Modality-tiered.

**Mask tier** — region overlap. For each matched predicted/GT object pair,
compute either Dice or IoU (exactly one — they rank identically since
`Dice = 2·IoU/(1+IoU)`):

```
Dice = 2·|A ∩ B| / (|A| + |B|)
IoU  =   |A ∩ B| / |A ∪ B|
```

then **macro-average** over pairs into the term `"Region"`. Two empty masks →
1.0; matched-vs-empty (a false positive or a missed object) → 0.0. These are the
natural Dice values; the base orients them to cost (`1 − Dice`), so two empty
masks → cost `0.0` (perfect) and a missed object → cost `1.0`.

Matching (`_scoring/_matching.py`) is **greedy by descending IoU**:
- *grid* path: each object is assigned to the grid cell it most overlaps, then
  paired within cell (gutter objects stay unmatched);
- *iou_greedy* path: global greedy with threshold τ. At **τ = 0.5 the assignment
  is provably one-to-one** (no object can exceed 0.5 IoU with two disjoint
  counterparts).

Binary GT is split into per-cell connected-component instances on a
**geometric, detection-independent** grid map — so under-segmentation is scored
honestly, not hidden behind the prediction's own labels.

**Count tier** — when only counts exist, reuses `ExpectedVsDetectedCount` and
the threshold-anchored fold (below); term `"CountMAE"`.

### B. ReferenceFreeScorer — no GT, proxy signals (`_scoring/_reference_free_scorer.py`)
Four **fixed-normalized** [0,1] terms (fixed normalization avoids the "Böck
trap", where the optimum migrates as grid endpoints change):

| Term | Formula |
|---|---|
| `ShapeRegularity` | `mean(Solidity, Circularity, 1 − Eccentricity)` |
| `Contrast` | Otsu between-class ratio `η = σ²_B / σ²_T`, with `σ²_B = w(1−w)(μ_fg − μ_bg)²` |
| `SizeCV` | `1/(1 + CV)`, `CV = σ/μ` (ddof=1) within replicate groups |
| `Count` (optional) | threshold-anchored count fold (same as QC) |

Crucially, it is **gated by meta-validation**: the proxy is only trusted if its
Spearman rank correlation with GT clears `ρ ≥ 0.7` (`_ENABLE_RHO`), and
`ρ ≥ 0.8` for unattended auto-tuning. Otherwise it abstains and the engine
degrades to QC.

### C. QCScorer — count-only (`_scoring/_qc_scorer.py`)
The count divergence `metric = |detected − expected| / expected` is the
scorer's **natural** value (a loss, `Sense.LOWER_BETTER`). Because it is
unbounded, the scorer supplies an **anchor** (`_term_anchor` → the check's
`fail_threshold`), and the base `to_cost` folds it via the threshold-anchored
transform `1 − exp(−ln2 · metric / fail_threshold)`:

```
cost(metric) = 1 − exp( −ln2 · metric / fail_threshold )
```

So metric 0 → cost 0.0 (perfect), metric = fail_threshold → exactly 0.5,
metric → ∞ → cost 1.0 (worst). Averaged across `groupby` units → term
`"Count"`. (In the shipped roster every scorer keeps its internal `[0,1]` fold
— OQ1=A — so `_term_anchor` returns `None` and `to_cost` is identity/complement;
the anchor branch is the contract for future raw-loss scorers.)

### D. CompositeScorer — blend multiple scorers (`_scoring/_composite.py`)
Each child owns a namespaced prefix `s0.`, `s1.`, … Children are finalized to
per-child **cost** scalars `bᵢ ∈ [0,1]`, then combined over the **study-global
active set** (children available study-wide; a study-wide abstainer is dropped
from *both* the `max` and the normalizer):

- **augmented Tchebycheff** (default, `blend="tchebycheff"`) with utopia point
  `z*ᵢ = −ε` and augmentation `ρ`:

  ```
  Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε)  +  ρ · Σᵢ wᵢ·bᵢ           (minimize)
  T_norm = Tᵨ(b) / Tᵨ(1…1)                            ∈ (0, 1]
  ```

  The `max` makes it **conjunctive** (worst axis dominates — all objectives
  must be good); the `ρ·Σ` augmentation upgrades minimizers from *weakly* to
  *properly* Pareto optimal. `_UTOPIA_EPS = 1e-3`, `rho = 0.05` (defaults the
  user never sets — §6.4). The normalizer is the **study-global** constant
  `Tᵨ(1…1)`, so the `[0,1]` rescale is argmin-preserving **across** trials.
- **weighted arithmetic mean** (opt-out, `blend="weighted_mean"`)
  `Σ wᵢbᵢ / Σ wᵢ` — *compensatory*: a strong axis offsets a weak one. Cannot
  reach non-convex-front compromises; that is why it is not the default.

**Never** a geometric mean of cost: `0` is the product's annihilator, so one
perfect axis (cost 0) would zero the product and dominate — the opposite of
the conjunctive property it has on goodness. It is removed from the live path.

`weights` are now **blend-dependent** — Tchebycheff per-axis weights under
`tchebycheff`, arithmetic weights under `weighted_mean` (a behavior change:
today *setting* `weights` switched to the compensatory mean). It rejects cyclic
nesting at construction, and in `multi_objective=True` mode returns the
per-child cost dict (NSGA-II, `directions=["minimize"]*n`; the abstainer floor
flips `0.0 → 1.0`).

### Summary table

| Scorer | Term(s) | Range | Formula |
|---|---|---|---|
| Supervised (mask) | `Region` | [0,1]↓ (cost) | `1 − Dice/IoU` (macro-avg over matched pairs) |
| Supervised (count) | `CountMAE` | [0,1]↓ (cost) | `1 − exp(−ln2·metric/thr)` |
| Reference-free | `ShapeRegularity`, `Contrast`, `SizeCV`, `Count`? | [0,1]↓ (cost) | see table above (complemented to cost) |
| QC | `Count` | [0,1]↓ (cost) | `1 − exp(−ln2·metric/thr)` |
| Composite | child blend | [0,1]↓ (cost) | augmented Tchebycheff (or weighted mean) / multi-objective dict |

---

## 5. Which parameters are "found important"

Computed *after* the study in `compute_param_importance_report`
(`_screening.py:85`), with **capability dispatch** (never a type check).

### fANOVA path (preferred)
If Optuna's `store.param_importances()` returns values, it uses **functional
ANOVA**: fit a random forest as a surrogate of `score = f(params)`, then
**decompose the surrogate's prediction variance** into per-parameter (and
interaction) contributions. A parameter's importance = the fraction of total
objective variance explained by varying it. This **accounts for interactions**
(`interactions_estimated=True`).

### RF-permutation fallback (`_screening.py:132`)
Used for per-objective requests or when no native model exists:

```python
RandomForestRegressor(n_estimators=200).fit(params, score)
permutation_importance(forest, X, y, n_repeats=10)
```

Permutation importance = the **drop in model accuracy when one parameter's
values are randomly shuffled**. A big drop ⇒ an important parameter.
Categoricals are one-hot encoded and summed back to their original key; this is
**main-effect only** (`interactions_estimated=False`). Returns `{}` with < 2
usable trials.

So a knob is "important" if changing it moves the objective a lot — measured
either by variance decomposition (fANOVA, with interactions) or by accuracy
degradation under shuffling (permutation, main effects).

---

## 6. Multiple objectives — Pareto math

When the scorer is multi-objective, directions become `["minimize"] * n`
(`_multi_objective.py:91`), NSGA-II is forced, and pruning is off.

**Dominance** (`_study/_pareto.py:54`): `a` dominates `b` iff `a` is **≤** `b` on
**every** axis AND strictly **<** on **at least one**. The **Pareto front** is
all non-dominated trials (ties deduplicated).

**Picking one config — the knee point** (`_study/_pareto.py:115`): draw the
chord between the lexicographic extremes `lo = min(vectors)` and
`hi = max(vectors)`; the recommended trial is the one with **maximum
perpendicular distance to that chord** — the elbow where you stop gaining on one
objective without sacrificing another (the extremes/chord are direction-agnostic;
under minimize the knee is still the elbow of the cost front). Exact for 2
objectives; heuristic for ≥ 3.

---

## 7. Generalization — the held-out check

A held-out split is reserved deterministically (`_evaluation/_split.py`) by a
3-tier policy: skip if fewer than `min_heldout_plates` plates; hold out a
**whole group** if ≥ 2 groups exist (strongest cross-batch test); else hold out
`ceil(held_out_fraction · n)` within-group. Those two thresholds are
**`HeldOutConfig` defaults** (`min_heldout_plates = 6`, `held_out_fraction =
0.2`, `_evaluation/_held_out.py`), not constants of `_split.py` — a caller can
override them. The seed is folded from a **content hash** (SHA-256) of sorted
plate names, so the same dataset always yields the same split regardless of
resume order.

The overfit gate (`compute_generalization_gap`, `_evaluation/_generalization.py:58`)
adopts the **standard loss-space generalization gap** — `gap = test − train`,
positive = overfit. Because cost *is* a loss, this is direction-correct by
construction (no custom sign flip):

```
abs_gap = heldout_cost − cal_cost          # positive = overfit
rel_gap = abs_gap / max(1 − cal_cost, _GAP_EPS)   # on the goodness-equivalent
flag  ⟺  rel_gap > rel_margin  AND  abs_gap > abs_margin
```

The relative term divides by the **goodness-equivalent `1 − cal_cost`** (with
`_GAP_EPS ≈ 0.02`) so a near-perfect calibration (cost ≈ 0) does not explode.
The principled blow-up-free upgrade — relative *overtuning* normalized by the
*achievable* test improvement (`> 1` ⇒ all gains lost; Schneider, Bischl &
Feurer, 2025) — needs incumbent/default tracking we don't have and is a
deferred v2 upgrade.

The margins are `HeldOutConfig` defaults (`gap_margin_relative = 0.15`,
`gap_margin_absolute = 0.05`, `_evaluation/_held_out.py`). It is report-only —
the winner never changes.

---

## 8. The data loop (ASCII)

```
                            ┌──────────────────────────────────────────────────────────┐
                            │                    TuningEngine.optimize()                  │
                            └──────────────────────────────────────────────────────────┘
                                                       │
   ┌───────────────────────────────────────────────────────────────────────────────────────────┐
   │                                                                                               │
   │   ┌──────────────────────┐   one-time, before loop                                            │
   │   │  infer_search_space  │   pipeline fields → Knobs (Tier-1 TuneSpec / Tier-2 heuristics)    │
   │   │  → SearchSpace        │   numeric: [d/4, d·4], log if span>10×/100×                        │
   │   └──────────┬───────────┘                                                                    │
   │              ▼                                                                                 │
   │   ┌──────────────────────┐                                                                     │
   │   │  Strategy (TPE/CMAES/ │◀──────────────────────── tell: register_result(params,result) ──┐  │
   │   │   GP/NSGA2/rand/grid) │                                                                  │  │
   │   └──────────┬───────────┘                                                                  │  │
   │              │ suggest():                                                                    │  │
   │              │   • ask study (TPE maximizes EI ∝ l(x)/g(x))                                   │  │
   │              │   • materialize active knobs → params                                         │  │
   │              │   • hand back PruningChannel (ASHA)                                            │  │
   │              ▼                                                                                │  │
   │   ┌──────────────────────────────── Evaluator.evaluate(params) ───────────────────────────┐ │  │
   │   │  build_pipeline(base, params)   ── fails to build? → FAIL (cost 1.0) ─────────────────┼─┘  │
   │   │           │                                                                            │    │
   │   │           ▼     rung ladder: sizes = max(6, ⌈n/3⌉), ×3, … , n                           │    │
   │   │   ┌───────────────┐                                                                    │    │
   │   │   │  for each rung │───────────────────────────────────────────┐                       │    │
   │   │   │   block of     │                                            │                       │    │
   │   │   │   images:      │   ┌──────────────────────────────────┐     │                       │    │
   │   │   │   measure +    │──▶│  Scorer.score_image → cost[0,1]  │     │                       │    │
   │   │   │   score once   │   │  supervised: 1−Dice/IoU (matched)│     │                       │    │
   │   │   │   (memoized)   │   │  ref-free:   shape/contrast/CV   │     │                       │    │
   │   │   └───────┬────────┘   │  qc:         1−exp(-ln2·m/thr)   │     │                       │    │
   │   │           │            │  composite:  aug. Tchebycheff    │     │                       │    │
   │   │           ▼            └──────────────────────────────────┘     │                       │    │
   │   │   robust-aggregate per term:  clamp01(median + λ·IQR)  (λ=0.5)  │                       │    │
   │   │           │                                                     │                       │    │
   │   │           ▼            scorer.finalize → running scalar         │                       │    │
   │   │   channel.report(score, scored)                                 │                       │    │
   │   │           │                                                     │                       │    │
   │   │   between rungs only: should_prune()? ──yes──▶ PRUNED (partial)─┘                       │    │
   │   │           │ no → next rung                                                              │    │
   │   │           ▼ (after final rung)                                                          │    │
   │   │   EvaluationResult{score, terms, objectives?, gap, suspicious, failed, pruned}          │    │
   │   └────────────────────────────────────────┬───────────────────────────────────────────────┘    │
   │                                             │                                                  │
   │                                             ▼                                                  │
   │                              store.append(Trial)  ──────────────────────────────────────────┘  │
   │                              budget: number+=1; failures+= failed                               │
   │                                             │                                                   │
   │                            is_exhausted() / n_trials / max_failures? ──no──▶ loop back up        │
   └─────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                  │ yes
                                                  ▼
              store.best()  ──▶  param-importance (fANOVA | RF-permutation)
                                  multi-objective: Pareto front → knee point
                                  held-out re-eval → generalization gap (report-only)
```

---

## 9. Takeaways

- **Decided**: search-space inference picks which fields become knobs (with
  log/bounds heuristics); the sampler — TPE by default — proposes values by
  maximizing the good/bad density ratio `l(x)/g(x)`.
- **Scored**: candidates run on a fidelity ladder with a stability-penalized
  robust aggregate (`median + λ·IQR`, clamped), so consistency across plates is
  rewarded and cheap losers are pruned via ASHA.
- **Objective**: one of four [0,1]-cost scoring strategies (supervised,
  reference-free, QC, composite).
- **Important**: read out post-hoc via fANOVA (variance decomposition, with
  interactions) or RF-permutation (accuracy drop, main effects only).
- **Trusted**: multi-objective runs surface a Pareto knee point, and a held-out
  generalization gap flags overfit before you trust the winner.
- **Combined**: the single-objective `CompositeScorer` uses an **augmented
  Tchebycheff** scalarization (conjunctive, worst-axis-dominant) over the
  study-global active set, replacing the old geometric mean — it can reach
  non-convex-front compromises a weighted sum cannot (Steuer & Choo, 1983;
  Miettinen, 1998).

### Key files
| File | Role |
|---|---|
| `_engine.py` | ask-and-tell orchestrator (`optimize`) |
| `_search_space/_infer.py` | pipeline → knobs (the two-tier inference) |
| `_strategies/_optuna.py` | sampler selection, suggest, pruning channel, tell |
| `_evaluation/_evaluator.py` | rung ladder + robust aggregate + finalize |
| `_evaluation/_aggregate_math.py` | `median/IQR`, eps-floored relative ratio |
| `_scoring/_orient.py` | `Sense`, `to_cost`, `clamp01` (the orientation boundary) |
| `_scoring/_supervised.py`, `_reference_free_scorer.py`, `_qc_scorer.py` | the per-modality scorers (natural terms, base-oriented to cost) |
| `_scoring/_composite.py` | the composite blend (augmented Tchebycheff / weighted mean) |
| `_scoring/_matching.py` | greedy-IoU / grid object matching |
| `_screening.py` | parameter importance (fANOVA vs RF-permutation) |
| `_multi_objective.py`, `_study/_pareto.py` | Pareto dominance + knee point |
| `_evaluation/_split.py`, `_generalization.py` | held-out split + overfit gate |

---

## References

The cost convention and composite math draw on:

- Steuer, R. E., & Choo, E.-U. (1983). *An interactive weighted Tchebycheff
  procedure for multiple objective programming.* Mathematical Programming,
  26(3), 326–344. https://doi.org/10.1007/BF02591870 — weighted Tchebycheff
  reaches every Pareto point; augmentation gives *proper* Pareto optimality.
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization.* Kluwer.
  https://doi.org/10.1007/978-1-4615-5563-6 — reachability and proper efficiency
  of (augmented) Tchebycheff scalarization.
- Carrell, A. M., Mallinar, N., Lucas, J., & Nakkiran, P. (2022). *The
  calibration generalization gap.* arXiv.
  https://doi.org/10.48550/arXiv.2210.01964 — generalization gap as
  `|Test − Train|` error; our loss-space `heldout_cost − cal_cost` is the same
  quantity.
- Schneider, L., Bischl, B., & Feurer, M. (2025). *Overtuning in hyperparameter
  optimization.* arXiv. https://doi.org/10.48550/arXiv.2506.19540 — the
  relative-overtuning normalization deferred to v2.
