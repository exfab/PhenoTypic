# Tune module: minimize bounded "cost" with augmented Tchebycheff

- **Status:** Draft (for review)
- **Date:** 2026-06-09
- **Scope:** `src/phenotypic/tune/**`, plus the QC direction flag in
  `src/phenotypic/analysis/abc_/_quality_check.py`, the GUI tune surface
  (`src/phenotypic/gui/tune/**`), and the explainer
  `docs/superpowers/explain/tune-with-optuna.md`.
- **Decision recorded:** adopt Option 1 ("bounded cost in `[0,1]`, optimizer
  minimizes") as the internal convention, and replace the geometric-mean
  composite with an **augmented Tchebycheff** scalarization.

---

## 1. Summary

Today the tuner normalizes every objective to a bounded `[0,1]` score where
**higher is better** and Optuna **maximizes**. This document specifies flipping
the internal convention to bounded **cost** in `[0,1]` where **lower is
better** (`0` = perfect, `1` = worst) and Optuna **minimizes**, and replacing the
single-objective composite combiner (currently a geometric mean of per-child
goodness) with an **augmented Tchebycheff** scalarization.

Two coupled changes:

1. **Pole flip + direction declaration.** Generalize the QC `_HIGHER_IS_BAD`
   flag into a scorer/metric-level *sense* declaration so each scorer emits its
   **natural, intuitive** value (a divergence stays a divergence; Dice stays
   Dice) and one framework boundary orients it into bounded cost. The
   optimizer minimizes.
2. **Composite combiner.** The single-objective `CompositeScorer` blends
   per-child cost with augmented Tchebycheff
   `Tᵨ(b) = maxᵢ wᵢ(bᵢ − z*ᵢ) + ρ·Σᵢ wᵢ(bᵢ − z*ᵢ)`, minimized, with a
   strictly-dominating utopia point `z*ᵢ = −ε`.

The per-term math is a pure reflection of today's, so for the linear/order paths
the **winner is provably unchanged**. The composite change is a **deliberate
scoring-semantics change** (not accuracy-neutral) and is gated by a regression
baseline.

---

## 2. Motivation, goals, non-goals

### Motivation
- The measurement/QC ecosystem is overwhelmingly **loss-shaped**: QC checks are
  `_HIGHER_IS_BAD=True` divergences; the raw scorer signals (count divergence,
  relative MAD, max modZ) are losses. Today every scorer hand-rolls a flip into
  higher-is-better `[0,1]` (`_threshold_anchored`, `1−Dice`, fold-and-flip). A
  cost-native convention lets loss metrics integrate **without any flip**, and
  consolidates the four hand-rolled flips into **one** framework boundary.
- A "minimize a loss/distance-from-ideal" framing makes the **augmented
  Tchebycheff** combiner natural, which (unlike the weighted sum, and unlike the
  geometric mean) can reach Pareto-optimal compromises on **non-convex** regions
  of the front (§9).

### Goals
- One internal convention: bounded cost `[0,1]`, minimize.
- New scorers/checks declare a *sense* once and emit natural values; the
  framework orients them. Reuse the QC `_HIGHER_IS_BAD` + `fail_threshold`
  contract verbatim.
- Single-objective composite uses augmented Tchebycheff with a fixed utopia
  point.
- Behavioral equivalence for every linear/order path (per-term aggregate,
  arithmetic-mean finalize, Pareto domination); the only intended behavior change
  is the composite combiner.

### Non-goals
- Changing the rung ladder / pruning policy, the search-space inference, the
  sampler selection, or the held-out generalization split.
- Replacing the multi-objective NSGA-II path with a scalarization (Tchebycheff is
  the **single-objective composite** combiner only; the `multi_objective=True`
  path still emits a per-child vector to NSGA-II/Pareto).
- Migrating raw, *unbounded* losses (Option 2 from the design discussion). The
  bounded `[0,1]` normalization is retained.

---

## 3. Current state — the invariant and where it lives

`src/phenotypic/tune/CLAUDE.md`: *"Higher-is-better everywhere ... the single
`_MAXIMIZE` literal."* The invariant is asserted across seven layers:

| # | Layer | File · symbol | Assumption |
|---|-------|---------------|------------|
| 1 | Optimizer direction | `_strategies/_optuna_support.py:97` `_MAXIMIZE`; `:115` `study_objective_kwargs`; `_multi_objective.py:113` `objective_directions` → `["maximize"]*n` | `direction="maximize"` |
| 2 | Best selection | `_study/_optuna_store.py:254`, `_study_store.py:102` `max(valid, key=t.score)`; `_study/_optuna_store.py:300` `best_trials` | highest score wins |
| 3 | Scorer contract | `_scoring/_scorer.py` `score_image` "higher = better"; `finalize`; `project_objectives_to_scalar` "0.0 is the worst score" | terms higher-better, 0 worst |
| 4 | Per-scorer flips | `_qc_scorer.py:23` `_threshold_anchored`; `_supervised.py` Dice/IoU **direct** (no `1−Dice`) + count fold; `_reference_free_scorer.py:94,360` hand-rolled `1/(1+d)` / `1−e` flips; `_composite.py:319` `_geometric_mean` | normalize *to* higher-better |
| 5 | Evaluator math | `_evaluation/_evaluator.py:23` `_WORST_TERM=0.0`; `:202` `failure_score=0.0`; `:53` `median − λ·IQR`; `:104` `_is_suspicious` (`score>=floor`); `:68` `_per_trial_dispersion` | 0 worst, subtract spread penalty, high score = good |
| 5b | **Generalization gap** | `_evaluation/_generalization.py:100` overfit gap `cal − heldout`; `:266` `calibration_stability=winner.gap`; shared `_aggregate_math._relative` (`_GAP_EPS=1e-12`) | overfit = `cal > heldout`; div-by-central-tendency |
| 6 | Pareto / screening | `_study/_pareto.py` `_dominates` (`>=`/`>`), `_vector` fill `0.0`; `_screening_freeze.py` six sites: `sorted(...reverse=True)`, `max(...,key=score)`, `float("-inf")`, `focused<explore` | higher-better domination, −inf sentinel |
| 6b | **Pruner direction** | `_strategies/_optuna.py` ASHA `SuccessiveHalvingPruner`; `_pruning.py` `channel.report(score)`; `_evaluator.py:314` reports `running_score` | Optuna pruner reads `study.direction` |
| 7 | GUI + docs | `gui/tune/_winner.py`, `_study_read.py` (`running_best max`, `shortlist reverse=True`, `GAP_FLAG_THRESHOLD=0.15`, y-axis "score"), `_run_root.py:46` `["maximize","maximize"]`, `_callbacks.py`; `explain/tune-with-optuna.md`; `tune/CLAUDE.md` | display/sort by max |

The reusable precedent: `analysis/abc_/_quality_check.py:88` `_HIGHER_IS_BAD`;
`tools_/_qc_recipe/_runner.py:349` already orients via
`(-metric if higher_is_bad else metric)`.

> **Line-number caveat.** All `file:line` refs in this doc are from the **main
> branch** and are off by 1–2 in the worktree (and `_screening_freeze.py` refs
> are materially stale). Re-resolve every cited line in the worktree before
> editing. Layers **5b** and **6b** were added after the migration review — they
> were absent from the original inventory.

---

## 4. Target design — the cost convention

**Definition.** Every per-term score is a **cost** `b ∈ [0,1]`, where `b=0` is
perfect and `b=1` is the worst. The optimizer minimizes. `inf`/degenerate inputs
floor to `b=1.0` (bounded worst).

**Equivalence (linear/order paths).** With `b = 1 − s`:
- `median(b) = 1 − median(s)`, `IQR(b) = IQR(s)` ⇒ the per-term aggregate
  `median(b) + λ·IQR(b) = 1 − (median(s) − λ·IQR(s))`.
- `mean(b) = 1 − mean(s)`.
- Pareto domination is order-based; `≤`/`<` is the exact reflection of `≥`/`>`.

So minimizing the cost aggregate yields the **identical winner** to today's
maximize for every path except the composite combiner (§5), which is changed on
purpose.

---

## 5. Target design — direction declaration + orientation boundary

### 5.1 Sense declaration (generalize `_HIGHER_IS_BAD`)
Introduce one shared sense concept used by both QC checks and scorers. A QC check
already declares it: `_HIGHER_IS_BAD=True` ⟺ the metric is a loss (lower-better
objective). For scorers, declare the sense of the terms they emit:

```python
class Scorer(BaseModel, ABC):
    # Sense of this scorer's natural per-term values (v1: uniform per scorer).
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER   # cost-native default
```

`Sense` is a closed value set: per `CLAUDE.md`, define it as an enum paired with
a `Literal` alias plus an alignment test (`set(get_args(SenseLiteral)) ==
{m.value for m in Sense}`). Mapping: `_HIGHER_IS_BAD=True` ↔ `Sense.LOWER_BETTER`.
**v1 scope (OQ8):** sense is a single per-scorer `ClassVar` (uniform across that
scorer's terms). A mixed-sense, per-term map is deferred/YAGNI — no shipped scorer
needs it.

The default is `LOWER_BETTER` (cost-native, **decided**): a new scorer that
emits a raw loss is zero-config; a goodness-emitting scorer must declare
`HIGHER_BETTER`. Because the default *assumes* cost, the migration cannot leave
today's goodness-emitting scorers un-annotated — each must be tagged in the same
change that rewrites its emission (see §7 Phase 1 and the §11 decision).

### 5.2 The orientation boundary — a base-class template method (OQ2)

Orientation is a **per-scorer responsibility**, not a flat Evaluator-boundary
pass. The base `Scorer` becomes a template method: a scorer produces its natural
per-term values in `_score_terms`, declares its sense once, and the base wraps
each term into cost via the single shared `to_cost` helper. This is what makes the
composite math unambiguous (§6.3): a child emits **already-cost** terms, so the
per-child reduction and the Tchebycheff combiner both operate on cost.

```python
def to_cost(value: float, *, sense: Sense, anchor: float | None) -> float:
    """Natural value -> cost in [0,1] (0 perfect, 1 worst)."""
    if anchor is None:                      # value already bounded in [0,1]
        return value if sense is Sense.LOWER_BETTER else 1.0 - value
    if not math.isfinite(value):
        return 1.0                          # inf divergence -> worst
    goodness = math.exp(-math.log(2.0) * value / anchor)   # 1 at 0, .5 at anchor
    return (1.0 - goodness) if sense is Sense.LOWER_BETTER else goodness

class Scorer(BaseModel, ABC):
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER

    @abstractmethod
    def _score_terms(self, image, measurements) -> dict[str, float]:
        """This scorer's natural per-term values (its own convention)."""

    def _term_anchor(self, term: str) -> float | None:
        return None                         # None => value already in [0,1]

    def score_image(self, image, measurements) -> dict[str, float]:
        # ONE orientation point; sense/anchor are local to the scorer that owns them.
        return {
            t: to_cost(v, sense=self._TERM_SENSE, anchor=self._term_anchor(t))
            for t, v in self._score_terms(image, measurements).items()
        }
```

The four `to_cost` cases: **bounded lower-better** → identity; **bounded
higher-better** (Dice, IoU, ICC, solidity) → `1 − value`; **unbounded
lower-better** (count divergence) → threshold-anchored complement (`anchor` =
the check's `fail_threshold`); **unbounded higher-better** (rare) → `goodness`.
A QC check's `(_HIGHER_IS_BAD, fail_threshold)` maps onto `(sense, anchor)`
exactly. **`CompositeScorer` overrides `score_image` (the merge), not
`_score_terms`** — its children already returned cost, so it must not re-orient.
Because `_score_terms` is `@abstractmethod`, `CompositeScorer` must still provide
a stub (`_score_terms` that raises `NotImplementedError`) to stay instantiable —
or the base declares `_score_terms` non-abstract with a `NotImplementedError`
body. Note this in Phase 3.

### 5.3 New-scorer authoring contract (canonical — must be documented in 3 places)

This is the source-of-truth contract for adding a tuning objective. **Per §7
Phase 5 it must be reproduced in: (1) the `Scorer` base-class docstring
(`_scoring/_scorer.py`), (2) `src/phenotypic/tune/CLAUDE.md`, and (3) the
contributor guide `docs/source/contrib_guide/contributing.rst`.** Keep the three
copies in sync with this section.

To add a `Scorer`:
1. **Subclass `Scorer`** and implement `_score_terms(image, measurements) ->
   dict[str, float]` returning your **natural** per-term values — do **not** flip
   or normalize by hand.
2. **Declare `_TERM_SENSE`** (`LOWER_BETTER` if larger = worse, the default;
   `HIGHER_BETTER` if larger = better, e.g. Dice/ICC).
3. **Supply an anchor only if a term is unbounded** — override `_term_anchor` to
   return the half-cost scale (for a QC-backed term, its check's
   `fail_threshold`). Bounded `[0,1]` terms need nothing.
4. **Do not add scalarization parameters.** `ε`, `ρ`, normalization, and default
   weights are framework-derived (§6.6); a scorer never sets them.
5. **Register** — re-export from `tune/__init__.py` and the class registry, or
   the GUI and `from_json` cannot see it.

The framework then orients (`to_cost`), robust-aggregates, reduces per child, and
combines (augmented Tchebycheff) — the author writes none of that. Existing
scorers are migrated to this shape in §7 **Phase 1** (rename their `score_image`
body to `_score_terms`, keep their internal fold, declare
`_TERM_SENSE = HIGHER_BETTER`); Phase 0 ships only the machinery (see §11).

---

## 6. Target design — augmented Tchebycheff composite

### 6.1 Formula
For per-child cost scalars `bᵢ ∈ [0,1]`, positive weights `wᵢ` (the existing
`weights` dict; missing → `1.0`), utopia point `z*ᵢ = −ε`, and augmentation
coefficient `ρ`:

```
dᵢ = wᵢ · (bᵢ − z*ᵢ) = wᵢ · (bᵢ + ε)          # all dᵢ > 0  (invariant: z*ᵢ < 0 ≤ bᵢ)
Tᵨ(b) = max_i dᵢ  +  ρ · Σ_i wᵢ·bᵢ            # minimize
```

> **Formula precision (lit review).** The **canonical** augmented weighted
> Tchebycheff (Steuer & Choo 1983; ParEGO, Knowles 2006) is
> `maxᵢ wᵢ|bᵢ − z*ᵢ| + ρ·Σᵢ wᵢ·bᵢ`. Two points:
> 1. The **augmentation term is `ρ·Σ wᵢ·bᵢ`** (raw weighted L1), as written
>    above. An earlier draft used `ρ·Σ wᵢ(bᵢ − z*ᵢ)`; that differs only by the
>    decision-independent constant `ρ·Σ wᵢ·ε`, so it is argmin-equivalent, but
>    the canonical form is used here to match the cited literature exactly.
> 2. The `max` term drops the absolute value **only because** the invariant
>    `z*ᵢ = −ε < 0 ≤ bᵢ ≤ 1` makes every `bᵢ − z*ᵢ > 0`. This must be
>    **asserted in code** — a future "tighten the reference to the ideal `0`"
>    change would silently make the unsigned form reward overshoot. The upper
>    bound `bᵢ ≤ 1` is **not automatic**: the robust aggregate is unclamped and
>    can reach `~1+λ`, so it holds only because Phase 1 clamps the aggregated cost
>    to `[0,1]` (B1). The assert guards that clamp staying in place.

- **Utopia `z*ᵢ = −ε`** (e.g. `ε = 1e-3`): a **strictly-dominating** reference
  point. Tchebycheff's reach-every-Pareto-point property requires the reference
  to strictly dominate (be unachievable for) the whole front; since child cost
  is achievable down to `0`, `z* = 0` is only *weakly* dominating on a perfect
  axis, so we shift to `−ε`. Static constant — no per-trial state, no estimation,
  no resume hazard (see §9 for why the running-ideal estimation problem does not
  apply here). `ε` is **not scale-free**: the weights that realize a given Pareto
  point scale as `wᵢ ~ 1/(bᵢ + ε)`, so `ε` must be small *relative to the `[0,1]`
  cost scale* (too large flattens weight differences, biasing toward the L1
  term like a large `ρ`). At `ε = 1e-3` the shift is ~0.1% of full range.
- **Augmentation `ρ·Σ dᵢ`** upgrades minimizers from *weakly* Pareto optimal
  (plain Tchebycheff) to **properly** Pareto optimal, eliminating
  weakly-dominated solutions. Default `ρ = 0.05`, exposed as a `CompositeScorer`
  field. ρ-tuning tension (§11): too small → numerically weak solutions; too
  large → cuts off properly-efficient points with extreme trade-offs.

### 6.2 Normalize to `[0,1]` for downstream consumers
`Tᵨ` ranges over `[ε(1+ρn), (1+ε)(1+ρn)]`, not `[0,1]`. Several consumers assume
bounded `[0,1]` cost (the `failure_score=1.0` floor, the `_is_suspicious`
thresholds). Normalize by the theoretical max:

```
T_norm = Tᵨ(b) / Tᵨ(1...1)      # in (0, 1]
```

> **Correction (lit review + OQ3) — the normalizer must be a study-global
> constant.** Dividing by `Tᵨ(1…1)` is argmin-preserving **only within one
> trial's objective**. `Tᵨ(1…1) = (1+ε)(maxᵢ wᵢ + ρ Σᵢ wᵢ)` depends on the active
> set and the weights, so if the active set varied per trial the normalizer would
> be monotone within a trial but **not across trials**, changing the cross-trial
> winner. Resolution (OQ3): both the `max` numerator **and** the normalizer are
> computed over the **study-global active set** — the children available
> *study-wide* (`availability() == True`). That set is fixed for the whole study,
> so the normalizer is constant, and numerator/denominator stay consistent (§6.3).

### 6.3 Where it plugs in
`_composite.py:203 finalize`:
- `multi_objective=True` → unchanged in shape: return the per-child cost
  dict; the abstainer floor flips `0.0 → 1.0` (worst cost) at `:240`; the
  vector feeds NSGA-II with `directions=["minimize"]*n`.
- single-objective with `weights` → augmented Tchebycheff with those weights as
  the **Tchebycheff weights** (conjunctive). See §6.5 — this is a semantics
  change from today, where setting `weights` switched the blend to a
  *compensatory* arithmetic mean.
- single-objective default → augmented Tchebycheff with uniform weights
  (**replaces** `_geometric_mean`, `:319`).

Introduce a `blend: CompositeBlend` selector (`Literal["tchebycheff",
"weighted_mean"]`, default `"tchebycheff"`). The geometric-mean-of-cost path
is **never** exposed: it inverts the conjunctive property (one perfect axis
zeroes the product and dominates), the documented trap this design avoids.

> **Abstainer handling (migration review + OQ3) — the active-set rule.** Under
> Tchebycheff `max`, flooring an abstaining child to cost `1.0` (the worst) would
> make its `dᵢ = wᵢ(1+ε)` **the maximum term for every candidate**, pinning the
> composite near its ceiling and destroying discrimination on the *available*
> axes. Abstention is a property of the run/data (e.g. `SupervisedScorer` without
> GT), so this would silently flatten an entire study.
>
> **Resolution (OQ3) — one study-global active set for both numerator and
> denominator.** Define the active set once as the children that are available
> **study-wide** (`availability()`); a study-wide abstainer is simply **not an
> objective** and is dropped from *both* the `max` and the normalizer roster
> (keeping them consistent — see §6.2). **Per-image** abstention (a study-wide
> available child that returns `{}` for one plate) is *not* a max-composition
> issue: it just yields fewer samples for that term in the robust aggregate, and
> the child still produces a scalar from the plates it did score. Edge cases:
> empty active set → composite is unavailable (engine degrades) and, defensively,
> the composite returns cost `1.0` (guard the `max([])`).
>
> **Plumbing (SF3):** the active set must be **pinned once at study start**, not
> recomputed inside `finalize` — `finalize(terms)` has no access to it, and
> `ReferenceFreeScorer.availability()` reads a *run-local* `_meta_validated`
> `PrivateAttr` that is `False` until `meta_validate()` runs. So the engine
> computes the active set after meta-validation and threads it to the composite
> (a `finalize` signature / stored-state change), so the `max` and the normalizer
> use the same fixed set for every trial.
>
> The **multi-objective** NSGA-II path is separate and keeps the abstainer floor
> `0.0 → 1.0` at `:240` (it needs a fixed-length value vector); only the
> single-objective Tchebycheff path uses the active-set rule. Keep the two paths
> distinct.

### 6.4 Choosing `ε` and `ρ` (and the harm of mistuning)

Both are **small, fixed constants chosen relative to the `[0,1]` cost scale**,
not tuned per run. Proposed defaults: `ε = 1e-3`, `ρ = 0.05`. Their jobs are
distinct and their failure modes are opposite-ended. **These are author-side
constants, never required of users** — the derivation routes below justify the
defaults; they are not operations a user performs (see §6.6 for the
zero-parameter user surface).

**`ρ` — the augmentation coefficient (`ρ·Σ wᵢ·bᵢ`).**
- *What it does.* It makes the Tchebycheff norm strongly monotone, which upgrades
  minimizers from merely *weakly* Pareto optimal to *properly* Pareto optimal —
  it is the term that breaks ties on the binding (worst) axis by preferring the
  candidate that is also better on the non-binding axes. Crucially, `ρ` sets the
  **trade-off bound** of the properly-efficient solutions it admits: it permits
  trade-off rates on the order of `1/ρ` and *excludes* solutions whose marginal
  trade-off exceeds that bound (Steuer & Choo, 1983; Dächert et al., 2012).
- *How it's typically derived.* Three established routes, in increasing rigor:
  1. **By convention on normalized objectives.** With objectives normalized to a
     common `[0,1]` scale, the field-standard value is `ρ = 0.05` (ParEGO,
     Knowles 2006; reused widely in surrogate MOO, e.g. Rojas-Gonzalez et al.
     2018, which likewise normalizes first). This is the default we adopt. The
     informal justification is a separation of scales — the `max` term spans
     `≈ max wᵢ` while the augmentation spans `ρ·Σ wᵢ`, so `ρ` is kept small
     enough that the `max` (the conjunctive L∞ character) still dominates yet
     large enough to break weak ties at float precision; the practical window is
     `ρ ∈ [1e-4, 0.1]`.
  2. **From a target trade-off bound (the principled derivation).** Because `ρ`
     controls the admissible trade-off `~1/ρ`, you can *invert* it: pick the
     maximum trade-off ratio `M` you are willing to accept between objectives
     and set `ρ ≈ 1/M`. `ρ = 0.05` ⇒ `M ≈ 20:1`, i.e. it rejects only
     near-pathological compromises (a candidate buying 20 normalized units on one
     objective per 1 unit lost on another) while keeping every reasonable
     balance. This "given a desired trade-off bound, the augmentation parameters
     are determined" derivation is exactly the contribution of Dächert, Gorski &
     Klamroth (2012).
  3. **Adaptive / problem-dependent.** Dächert et al. (2012) further derive,
     per problem instance, the **largest** `ρ` for which *every* non-dominated
     point is still reachable — maximizing numerical conditioning without losing
     any Pareto point. This is overkill for a fixed-weight composite, but it is
     the rigorous answer to "what is the best `ρ` for *this* front" and is worth
     citing as the upper-bound principle behind the `[1e-4, 0.1]` window.

  For this design we adopt route 1 (`ρ = 0.05`) with route 2 as the
  interpretation, and confirm with a small sweep (`{0.01, 0.05, 0.1}`) that the
  winner stays balanced.
- *Harm of mistuning.*
  - **Too small (→0):** reverts to plain Tchebycheff. Minimizers can be only
    weakly Pareto optimal, so the composite may pick a candidate tied on its
    worst axis but **strictly worse on every other objective** — it can no longer
    distinguish "balanced" from "wasteful" among equal-worst-axis candidates. At
    the limit the term underflows and contributes nothing.
  - **Too large:** the weighted-sum-like `Σ` term begins to dominate the `max`,
    so the combiner **degrades toward a weighted sum** and loses the non-convex
    reachability that motivated Tchebycheff in the first place (§9). It then
    favors extreme single-objective "supported" solutions over balanced
    compromises — the opposite of the conjunctive intent, and exactly the
    phenotyping failure we are trying to avoid (a pipeline that nails one metric
    and tanks the rest).

**`ε` — the utopia shift (`z*ᵢ = −ε`).**
- *What it does.* It pushes the reference point strictly below the achievable
  front so the reachability guarantee holds and the unsigned `max` is valid
  (`bᵢ − z*ᵢ = bᵢ + ε > 0`). It is **not** a tuning knob for quality — it is a
  numerical safety margin. The standard requirement (confirmed across the
  reference-point literature) is that the reference be a **utopia point strictly
  dominated by every feasible objective vector** — `z*ᵢ < min bᵢ` for all `i`.
- *How it's typically derived.* In the **general** case you first **estimate the
  ideal point** `z*ᵢ = minₓ fᵢ(x)` — the per-objective optimum, in practice the
  best value observed so far (this is the running-best estimate, and estimating
  the ideal/nadir is itself a recognized sub-problem; Deb et al. 2010) — then
  subtract a small offset to get a strictly-dominating utopia point. The offset
  is conventionally a **small fraction of the objective range** (a few percent),
  i.e. `z**ᵢ = z*ᵢ − ε·(nadirᵢ − z*ᵢ)`. Two things make this trivial here:
  1. **The ideal is known a priori, not estimated.** Because cost is
     normalized to `[0,1]` with `0` = perfect, `z*ᵢ = 0` exactly — so we skip the
     whole running-ideal estimation (and its determinism/resume hazards, §9) and
     only need the offset, giving `z*ᵢ = 0 − ε = −ε`.
  2. **The offset is set from weight conditioning, not quality.** The weight that
     realizes a target front point is `wᵢ ∝ 1/(bᵢ − z*ᵢ) = 1/(bᵢ + ε)`, so `ε`
     **caps the weight dynamic range at `1/ε`**. Choosing `ε = 1e-3` (~0.1% of
     the `[0,1]` range) bounds the realizer at `≤ 1000×` — well-conditioned —
     while staying far smaller than any cost gap worth resolving. That is the
     principled lower/upper bracket; no sweep is needed.
- *Harm of mistuning.*
  - **Too small (→0):** on a perfect axis (`bᵢ = 0` is achievable, e.g. an exact
    count match) the reference stops strictly dominating, so reachability degrades
    at the boundary; the axis contributes `0` to the `max` (a perfect axis drops
    out), and the realizing weight `wᵢ ~ 1/(bᵢ + ε)` **diverges** as `ε → 0`
    (division by zero at `ε = 0`).
  - **Too large:** `ε` is argmin-neutral under *uniform* weights (a constant
    shift inside the `max`), but with **non-uniform** weights the per-axis term
    `wᵢ·ε` perturbs which axis wins the `max`, biasing the result; and a large
    `ε` flattens the weight realizer `1/(bᵢ + ε)`, eroding the combiner's ability
    to steer toward specific compromises (again drifting toward equal-weight L1).
    It also inflates the §6.2 normalizer `(1+ε)(…)`, compressing the normalized
    `[0,1]` scale and miscalibrating downstream thresholds.

**Summary of the asymmetry.** `ρ` is the real quality dial (too small → weak
Pareto ties; too large → weighted-sum behavior); `ε` is a safety margin that
should simply be "small and positive." Both are documented module constants
(`_UTOPIA_EPS`, default `rho`), with `ρ` exposed as a `CompositeScorer` field for
the rare advanced user, `ε` kept internal.

### 6.5 Blend selection — keep `weighted_mean` as an opt-out (decided)

`tchebycheff` (conjunctive: worst-axis-dominant) and `weighted_mean`
(compensatory: a strong axis offsets a weak one) encode genuinely different user
intents:
- **Conjunctive** fits the usual phenotyping case where objectives are
  *complementary and all required* (you need good count **and** good shape **and**
  good contrast) — so it is the **default**.
- **Compensatory** fits the rarer case where objectives are *substitutable* and
  the user wants best average quality, tolerating one weak axis.

**Decision: keep `weighted_mean` as an explicit `blend="weighted_mean"`
opt-out.** Rationale: it already exists, costs almost nothing to retain, and
removing it would strand the legitimate compensatory use case. Two consequences
to document loudly:
1. **`weights` is now blend-dependent.** Under `tchebycheff`, `weights` are the
   Tchebycheff per-axis weights; under `weighted_mean`, they are arithmetic
   weights. Same field, different meaning by `blend`.
2. **Behavior change from today.** Today, *setting* `weights` silently switches
   the composite from geometric mean (conjunctive) to weighted arithmetic mean
   (compensatory). After this change, setting `weights` keeps the **conjunctive**
   Tchebycheff semantics; a user who wants the old compensatory behavior must set
   `blend="weighted_mean"` explicitly. Call this out in the migration/release
   notes — it is the most likely silent surprise for existing tuning configs.

`weighted_mean` shares the weighted-sum convex-hull limitation (§9): it cannot
reach non-convex-front compromises. That is acceptable for an explicitly-chosen
compensatory blend, but it is a reason it is **not** the default.

### 6.6 User-facing surface — zero exposed scalarization parameters (decided)

**Principle: a user configuring a tuning run sets *no* scalarization parameters.**
`ε`, `ρ`, the reference point, and the per-axis normalization are all derived by
the framework. This is a deliberate design goal — the §6.4 derivations are
*author-side justifications for the fixed defaults*, not operations a user
performs. The reasoning, grounded in the literature:

1. **`ε` and `ρ` are scale-coupled constants, and normalization removes the
   scale.** Once every objective is in `[0,1]`, `ρ = 0.05` is the field-standard
   constant (Knowles, 2006) and `ε = 1e-3` is a numerical margin (§6.4). Neither
   is problem-dependent, so neither is exposed: `ε` is an internal `Final`, `ρ`
   an advanced-only `CompositeScorer` field with a default the user need never
   touch.

2. **Per-axis normalization needs no user input (OQ1=A).** Getting normalization
   right is the step that actually matters: Marler & Arora (2010) show that, for a
   fixed weight set, *rescaling an objective changes which Pareto point you
   obtain* — so without normalization, weights conflate importance with units.
   Under OQ1=A this is already handled, because **every shipped scorer already
   folds its raw signal to a bounded `[0,1]` value internally** (`QCScorer` via
   `_threshold_anchored` on the count check's `fail_threshold`; `ReferenceFreeScorer`
   via `_bounded_inverse`/`_clamp01`; `SupervisedScorer` via Dice/IoU + the folded
   count tier). So at the orientation boundary every term is already `[0,1]` and
   `to_cost` is the **identity/complement** (`_term_anchor` returns `None`); no
   per-scorer anchor field is added. The `to_cost` `anchor` branch (and the
   `fail_threshold`-as-anchor story) is the contract for **future raw-loss
   scorers**, not a refit of the current `SupervisedScorer`/`ReferenceFreeScorer`
   (which have no top-level `fail_threshold` — it lives on their nested
   `count_check`). Either way, normalization requires **zero new user input**.

3. **Weights default to uniform (no-preference).** True relative importance
   cannot be conjured from a pipeline config that carries no preference signal, so
   the principled default is equal weights on the normalized axes. **v1 ships
   uniform weights only (OQ5 deferred);** an advanced user who genuinely needs to
   reweight sets the `weights` dict directly. A coarse-importance preset
   (`low|med|high` / "primary objective") is **deferred to v2** — it would add a
   preset→weight mapping decision and GUI surface (OQ6) for speculative demand.

**Why not a lower-parameter method instead** (so the choice is on the record):
- **Weighted sum** removes `ρ`/reference but cannot reach non-convex-front points
  (Das & Dennis, 1997) and is *more* normalization-sensitive (Marler & Arora,
  2010) — it does not escape the scale problem and is strictly weaker for the
  all-objectives-required goal.
- **PBI** (MOEA/D; Zhang & Li, 2007) replaces `ρ` with a penalty `θ` the
  literature finds *harder* to set (no single `θ` works across problems;
  Mohammadi et al., 2015) — more burden, not less.

So augmented Tchebycheff + `[0,1]` normalization + fixed `ρ`/`ε` + uniform default
weights is already near the minimum-parameter frontier; the win is in **not
exposing** the parameters, achieved by auto-normalizing from the existing
`fail_threshold` anchors.

---

## 7. Migration plan (phased)

The per-term reflection makes a coordinated cutover safe, but the work is phased
so each phase is independently testable. Phase 0 is additive/behavior-preserving;
**Phases 1 and 2 are a single atomic cutover** (the direction-aware ASHA pruner
couples them — see Phase 1); Phase 3 is the deliberate composite change. The
reflection equivalence holds for the linear/order paths only once both 1 and 2
land together.

### Phase 0 — direction declaration + orientation boundary (additive)
- Add `Sense` enum + `Literal` alias + alignment test (`tools_/typing_.py`).
- Add `to_cost` (new `_scoring/_orient.py`), with unit tests, but **do not
  invoke it on the live scoring path yet**.
- Lift `_HIGHER_IS_BAD` ↔ `Sense` mapping; keep QC checks unchanged.
- **No optimizer or scorer change yet.** Because the `Sense` default is
  `LOWER_BETTER`, the boundary cannot be activated while scorers still emit
  goodness (it would mis-orient them). Phase 0 ships only the machinery (enum,
  alias, `to_cost`, mapping); scorer annotation + activation happen atomically
  per scorer in Phase 1.

### Phase 1 — flip the Evaluator math (`_evaluation/**`)
- **BLOCKER (B1) — clamp the aggregated cost to `[0,1]`.** `_robust_aggregate`
  (`_evaluator.py:65`) is `median − λ·IQR`, **unclamped**: goodness ranges
  `[−λ, 1]` (≈`[−0.5, 1]`), so the reflected cost `median + λ·IQR` ranges
  `[0, 1+λ]` (≈`[0, 1.5]`) and **can exceed 1.0**. Today only `_geometric_mean`
  clamps (bottom-only, `_composite.py:338`); the mean/`_weighted_mean` paths do
  not. Left unclamped this breaks the §6.1 invariant/assert (`0 ≤ bᵢ ≤ 1`), the
  §6.2 `T_norm ≤ 1`, and the `_FAILURE_COST = 1.0` floor. **Fix:** clamp the
  robust-aggregated cost to `[0,1]` (in/after `_robust_aggregate`). The clamp is
  monotone and only bites on *terrible* terms (cost > 1, i.e. unstable+bad), so it
  is **winner-preserving** — the §4 winner-equivalence still holds (it is a
  winner-level, not bit-level, guarantee).
- `_robust_aggregate`: `median − λ·IQR → median + λ·IQR`, **then clamp to
  `[0,1]`** (`:53`, `:65`).
- `_WORST_TERM: 0.0 → 1.0`; `failure_score: 0.0 → 1.0`. With the B1 clamp the
  per-child cost and the normalized composite are genuinely `≤ 1`, so `1.0` is a
  valid worst-floor (no separate `_FAILURE_COST` constant needed).
- `_is_suspicious` (`:104`): reflect **every** constant (OQ9), not just the
  comparison: `score <= (1 − suspicious_score_floor)` **and** `count_cost >=
  (1 − suspicious_count_floor)`; flip the **`Count` default** `terms.get("Count",
  1.0) → 0.0` (a missing Count term must default to *faithful = best cost*, not
  worst); reflect the two floor fields. Enumerate the full `_evaluator.py` flip
  set in the checklist: `_WORST_TERM 0.0→1.0`, `failure_score 0.0→1.0`,
  `_is_suspicious` Count default `1.0→0.0` + both floors, and the `_aggregate`
  worst-term pad. Update docstrings.
- **BLOCKER — second direction-sensitive gap: `compute_generalization_gap`
  (`_evaluation/_generalization.py:100`).** The overfit detector computes
  `absolute_drop = cal_score − heldout_score` and flags `drop > margin`. Under
  goodness, overfit = `cal > heldout` (positive drop). Under **cost** the sign
  inverts: a genuinely overfit winner has *higher* held-out cost, so
  `cal_cost − heldout_cost` is **negative** and the gate **never fires** —
  the overfit detector silently dies (it writes a wrong `generalization.json`
  rather than erroring). Flip to `heldout − cal` and re-examine the margin
  comparisons. **This file is in neither the original §3 inventory nor any
  phase** — added here (layer 5b). Update its doctest (`:91`).
- **PITFALL — relative-IQR blowup at BOTH `_relative` call sites (OQ4 resolved).**
  The shared `_aggregate_math._relative(x, central)` floors the denominator at
  `_GAP_EPS=1e-12`. Under cost a great candidate has central tendency `≈ 0`, so
  the relative value explodes. This hits **(a)** `_per_trial_dispersion`
  (`_evaluator.py:68`, the `gap` signal) **and (b)** `compute_generalization_gap`
  (`_generalization.py:101`). **Resolution — do both:** (i) compute the relative
  quantity on the **goodness-equivalent (`1 − cost`)** so a good candidate's
  central tendency is `≈ 1`, not `≈ 0` — this is reflection-clean and keeps the
  calibrated `GAP_FLAG_THRESHOLD=0.15` valid (the flip moves the singularity to
  the harmless *bad* end); **and** (ii) raise the existing too-small denominator
  floor `_GAP_EPS` from `1e-12` to a **meaningful constant `≈ 0.02`** (a few
  percent of the `[0,1]` scale) as a defensive cap for the residual bad-end case.
  Keep the floor small enough that it does not materially shift the gap for normal
  candidates (so `0.15` still holds). Note `_GAP_EPS=1e-12` *is* already a
  stability term — just far too small to do anything. One shared-helper change
  fixes both callers. Consumers of `gap` to re-confirm unchanged: the GUI
  `GAP_FLAG_THRESHOLD=0.15` and the data-poor fallback
  `calibration_stability=winner.gap` (`_generalization.py:266`).
- **COUPLING — Phase 1 and Phase 2 must land together (pruner).** The ASHA
  `SuccessiveHalvingPruner` (`_strategies/_optuna.py`) is **direction-aware** —
  it reads `study.direction`. The Evaluator reports `running_score` to it
  (`:314`). Flipping the reported value (Phase 1) without flipping the study
  direction (Phase 2), or vice-versa, breaks pruning. So "Phases 0–1 are
  behavior-preserving" holds only if pruning is disabled between them; otherwise
  Phase 1 + Phase 2 are a single atomic cutover.
- Scorers rewritten to emit natural values + declare sense (QCScorer →
  raw count divergence + `Sense.LOWER_BETTER` + anchor=`fail_threshold`;
  SupervisedScorer → Dice/IoU `Sense.HIGHER_BETTER`; ReferenceFreeScorer shape
  signals `Sense.HIGHER_BETTER`). Per-child `finalize` mean stays (reflection-clean).
- Note `_reference_free_scorer.py:166,216` `_last_rho` (`float("-inf")`) is a
  **diagnostic** Spearman ρ (higher = better correlation), not the objective;
  it stays higher-better. Review, expected unaffected.

### Phase 2 — flip optimizer direction + study persistence (STICKY)
- `_strategies/_optuna_support.py`: `_MAXIMIZE → _MINIMIZE = "minimize"`;
  `study_objective_kwargs` → `{"direction": "minimize"}` /
  `{"directions": ["minimize"]*n}`.
- `_multi_objective.py:113` `objective_directions` → `["minimize"]*n`.
- Best selection: `_study/_optuna_store.py:254`, `_study_store.py:102`
  `max → min`.
- **Study persistence — HARD CUTOVER via study-name bump (decided; OQ7).**
  ⚠️ **Verified hazard (optuna 4.9.0):** `create_study(load_if_exists=True,
  direction="minimize")` against an existing `maximize` study **does NOT raise —
  it silently loads the old study and keeps `direction = MAXIMIZE`.** (An earlier
  draft of this spec claimed it raises; that is **false**, tested empirically.) So
  without intervention, re-running into a storage that holds the pre-cutover
  `"tune"` study would silently **maximize cost** → pick the *worst* pipeline, no
  error. We do **not** ship a converter. The mechanism is:
  1. **Bump the study name** — the single `_STUDY_NAME` constant
     (`_tune_cli/_run.py:70`, today `"tune"`) → e.g. `"tune_cost_v1"`. New code
     only ever opens the new-name study, so a pre-cutover `"tune"` study is
     **never reopened** — the silent-maximize hazard is impossible *by
     construction*, not contingent on a guard. Old-convention resume becomes
     impossible (the intended cutover); a re-run creates a fresh new-name study
     beside the inert old one. **Every reader (CLI, worker, GUI monitor) must use
     the one constant**, not re-spell `"tune"`. Known desync sites to fix in this
     phase: `gui/tune/_run_root.py:38` `_DEFAULT_STUDY_NAME = "tune"` (a fallback
     that would silently miss the bumped study) and the hardcoded
     `study_name="tune"` in the `gui/tune/_winner.py:72` doctest. Import
     `_STUDY_NAME` rather than re-spelling it.
  2. **Stamp `tune_convention`** (`study.set_user_attr`, e.g. `"minimize-cost-v1"`)
     for observability / future cutovers.
  3. **Friendly detector (UX, not correctness):** if a legacy `"tune"` study is
     present in the storage, log/raise an actionable note ("a pre-cutover study
     exists here; it cannot be resumed under the cost convention — starting
     fresh"). Correctness is already guaranteed by the name bump, so this is
     purely a better message.
  - **Consequence (accepted):** in-flight and completed pre-cutover studies
    **cannot be resumed**; they must be re-run. Chosen over a converter, which
    would carry permanent reflection-correctness risk (every stored field's
    `1 − value` must stay exactly right across schema evolution) for the one-time
    benefit of resuming old runs. Old `pheno_terms`/`pheno_objectives` are not
    interpreted; cross-study comparison with pre-cutover runs is invalid (document
    in release notes / `tune/CLAUDE.md`).

### Phase 3 — augmented Tchebycheff composite (`_scoring/_composite.py`)
- Add `rho: float = 0.05` and `blend: CompositeBlend = "tchebycheff"` fields;
  `_UTOPIA_EPS: Final = 1e-3`.
- Implement `_tchebycheff(child_scalars)` (§6.1) + `[0,1]` normalization (§6.2).
- `finalize` (`:203`) routes to Tchebycheff (default/weighted) or
  `_weighted_mean` (opt-out); remove `_geometric_mean` from the live path.
- Abstainer floor `0.0 → 1.0` (`:240`); update doctests.

### Phase 4 — Pareto + screening + GUI
- `_study/_pareto.py:53` `_dominates`: `>=`/`>` → `<=`/`<`; `:50` `_vector` fill
  `0.0 → 1.0`; knee-point chord math is direction-agnostic (verify).
- `_screening_freeze.py` — **six** score-direction sites (the original draft
  listed three and mis-cited them): the two `freeze_value` / `_warm_started_store`
  `sorted(..., key=score, reverse=True)`, the `_genuinely_focused_best`
  `max(fresh, key=score)`, the winner-across-rounds `max(union, key=score)`, the
  two `float("-inf")` sentinels, and the `_resolve_winner` `focused_score <
  explore_score` recovery test. Flip all: `reverse=True→False`, `max→min`,
  `-inf→+inf`, `<→>`, and `_apply_focused_penalty` adds instead of subtracts.
  `_screening.py` importance sorts are over **importances** (variance attribution
  — sign-independent, unchanged).
- GUI: `gui/tune/_winner.py` (doctest `score=0.9` → low cost);
  `_study_read.py` (`running_best max→min`, `shortlist reverse=True→False`,
  `GAP_FLAG_THRESHOLD=0.15` re-review after the gap fix, y-axis label "score");
  `_run_root.py:46` `_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS
  ["maximize","maximize"] → ["minimize","minimize"]`; `_callbacks.py`. Relabel
  "score" semantics (lower = better). Follow `gui/FEATURES.md` + `WORKFLOWS.md`
  gates if any affordance text changes.

### Phase 5 — docs + tests
- **New-scorer authoring contract — propagate §5.3 to all three surfaces (OQ2).**
  The §5.3 contract is the source of truth; reproduce it (kept in sync) in:
  1. the **`Scorer` base-class docstring** (`src/phenotypic/tune/_scoring/_scorer.py`)
     — the `_score_terms` / `_TERM_SENSE` / `_term_anchor` template-method contract
     and the "emit natural values, declare sense, no scalarization params" rules;
  2. **`src/phenotypic/tune/CLAUDE.md`** — a short "Adding a Scorer" subsection
     plus the convention flip below;
  3. the contributor guide **`docs/source/contrib_guide/contributing.rst`** — a
     walkthrough for adding a tuning objective.
  A docs check should fail if these drift (mirror the existing CLAUDE.md/FEATURES
  gating philosophy).
- `tune/CLAUDE.md`: "Higher-is-better everywhere" → "Cost everywhere
  (lower-is-better, minimize); the single `_MINIMIZE` literal."
- `docs/superpowers/explain/tune-with-optuna.md` (+ `.graph.md`): rewrite the
  scorer/aggregate/Pareto/composite math sections (CLAUDE.md mandates this in the
  same change).
- Tests (§10).

---

## 8. Migration costs

| Layer | Effort | Risk | Notes |
|-------|--------|------|-------|
| Phase 0 orient boundary | S | Low | Additive; enum + 1 function + tests |
| Phase 1 evaluator + scorers | M–L | **High** | Reflection provable for the core, but it must also fix the inverted overfit detector (`_generalization.py`), both `_relative` blowups, and `_is_suspicious`; couples atomically with Phase 2 via the pruner |
| Phase 2 direction + persistence | M | **High** | Sticky: Optuna study direction is immutable; guard must precede `create_study`; **hard cutover** — pre-cutover studies error out and must be re-run (no converter) |
| Phase 3 Tchebycheff composite | M | Med | The combiner is ~10 lines, but it must exclude abstainers from the `max`, use a study-global normalizer, and ship a **non-convex regression test** + an abstainer-masking test |
| Phase 4 Pareto/screening/GUI | M | Med | Mechanical flips; GUI relabeling + FEATURES/WORKFLOWS gates |
| Phase 5 docs/tests | M | Low | Explainer rewrite is required, not optional |

**Stickiness / reversibility.** Phase 0 and Phases 3–4 are reversible. Phases 1–2
(the atomic cutover) are sticky: they change the persisted study `direction` and
the *meaning* of stored `score`/`objectives`/`terms`. With the **hard cutover**
(decided), pre-cutover studies are refused by the startup guard and must be
re-run; there is no in-place SLURM resume across the convention boundary. This is
a one-time disruption accepted in exchange for never carrying converter
reflection-correctness risk.

**Accuracy cost.** Zero for Phases 0–2 and 4 (provable reflection equivalence).
Phase 3 (composite) is a **deliberate** change: augmented Tchebycheff selects
different multi-criteria compromises than the geometric mean (it can reach
non-convex-front points the geometric mean cannot). This is an intended upgrade,
not a regression, but it changes which candidate wins on composite objectives, so
it requires a baseline snapshot and reviewer sign-off, not a silent swap.

---

## 9. Accuracy & theory notes (literature-audited)

- **Reflection equivalence** (median/IQR/mean): elementary; verified.
- **Geometric-mean-of-cost is the trap**: the geometric mean is a conjunctive
  aggregation function (Beliakov, Pradera & Calvo, 2007) and does not commute
  with `s → 1−s`; because `0` is the product's annihilator, feeding *cost*
  into it makes one perfect axis (cost 0) zero the product and dominate — the
  opposite of the conjunctive "all axes must be good" property it has on
  *goodness*. Avoided by never exposing it (§6.3).
- **Weighted sum** reaches only convex-hull ("supported") Pareto points; concave
  regions are unreachable for any weights (Das & Dennis, 1997; Geoffrion, 1968).
- **Weighted Tchebycheff** can reach *every* Pareto-optimal point for some weight
  (Steuer & Choo, 1983; Miettinen, 1998), but plain-Tchebycheff minimizers are in
  general only **weakly** Pareto optimal — hence the **augmentation** (`ρ` term),
  which yields **properly** Pareto-optimal points (Steuer & Choo, 1983;
  Miettinen, 1998; Engau, 2017). **Scope:** reachability is a statement over
  *varying* weight vectors. The single-objective `CompositeScorer` uses **one**
  weight vector per run, so it selects **one** compromise point (which *may* lie
  in a non-convex region a weighted sum could never reach) — it does **not** trace
  the front. Tracing the front is the `multi_objective=True` NSGA-II path's job.
- **Reference point**: reachability requires only a **strictly dominating**
  (unachievable, lower-bounding) reference, not the tight ideal (Bauß &
  Stiglmayr, 2023; Tripp, 2025) — which is why `z*ᵢ = −ε` is both sufficient and
  state-free here. The "estimated running ideal" problem (a study-global `z*`
  updated from best-so-far values, which would break per-candidate score purity,
  deterministic resume, and cross-worker independence) **does not arise**, because
  the bounded `[0,1]` normalization gives a known static lower bound.
- **Power-mean note**: `p ≤ 0` means (geometric/harmonic) require strictly
  positive arguments; cost can be `0`. Augmented Tchebycheff (an L∞-family,
  `p → ∞` flavor) has no positivity hazard — another reason to prefer it over any
  geometric/harmonic cost blend.

---

## 10. Testing strategy

- **Reflection-equivalence regression**: a fixed synthetic study (seeded) run
  under the old maximize code and the new minimize code must select the
  **identical winner** for single-term, arithmetic-mean, and Pareto paths
  (assert on chosen params, tolerance 0.0 where deterministic).
- **Composite is intentionally different**: snapshot baseline of the geometric-
  mean composite winner; assert the Tchebycheff winner and document the delta.
- **Non-convex reachability test**: a small synthetic 2-objective front with a
  concave region; assert augmented Tchebycheff selects a knee point that a
  weighted sum (and `1 − geomean`) cannot, and that the ρ term removes a
  weakly-dominated point.
- **Orientation boundary**: table test over the four (bounded?, sense) cases incl.
  `inf → 1.0`; assert QC `(_HIGHER_IS_BAD, fail_threshold)` maps to
  `(sense, anchor)`.
- **`gap` pitfall guard**: a near-perfect candidate (median cost ≈ 0) must not
  produce an exploding/NaN `gap` — assert for **both** `_per_trial_dispersion`
  and `compute_generalization_gap`.
- **Overfit-gap sign**: a synthetic winner that is better on calibration than
  held-out must be **flagged** under cost (currently no test asserts the gap
  *sign*; this is how the inverted detector would ship undetected). Invert the
  `test_generalization.py` flag assertions.
- **Pruner inversion**: `test_optuna_pruning.py`'s end-to-end test seeds good
  trials at `1.0` and a bad one at `0.0` and asserts PRUNED — under minimize this
  must invert (good = low cost, bad = high cost) or it silently prunes the
  *best* candidate.
- **Abstainer masking**: a single-objective composite with one abstaining child
  must still discriminate among candidates on the present axes (regression for
  the §6.3 fix).
- **Cost clamp (B1)**: a high-variance term (`median + λ·IQR > 1`) must yield a
  per-child cost and `T_norm` in `[0,1]` (assert the §6.1 `0 ≤ bᵢ ≤ 1` invariant
  holds and the assert does not fire); confirm the clamp is winner-preserving
  against the unclamped reflection on a non-boundary winner.
- **Persistence (hard cutover) — silent-maximize regression**: in a storage that
  already holds a `maximize` study named `"tune"`, the new store must **not**
  reopen it as a maximization of cost (the verified optuna `load_if_exists`
  hazard). Assert the new run opens the **bumped** study name (`"tune_cost_v1"`,
  `direction=minimize`), leaves the legacy study inert, and the friendly detector
  fires; assert a fresh `minimize` study carries the `tune_convention` attr.
- **`ρ` / `ε` sensitivity**: assert ρ→0 admits a weakly-dominated winner that
  ρ=0.05 rejects; assert a large ρ drifts toward the weighted-sum winner.
- **Enum/Literal alignment** test for `Sense` and `CompositeBlend`.

---

## 11. Pitfalls & open decisions

**Pitfalls** (ordered by risk after the migration review)
1. **Inverted overfit detector** (§7 Phase 1, `_generalization.py:100`) —
   **highest risk.** The migration originally missed this file entirely; the gap
   sign inverts under cost and the detector silently stops flagging real
   overfit, writing a wrong `generalization.json` with no error and no test
   catching it.
2. **`_relative` blowup at both call sites** (§7 Phase 1; **OQ4 resolved**) —
   `_per_trial_dispersion` *and* `compute_generalization_gap` divide by a central
   tendency that → 0 for good candidates. Fix: compute on `1 − cost` **and** raise
   `_GAP_EPS` 1e-12 → ~0.02.
3. **Unclamped aggregate → cost > 1** (§7 Phase 1, B1) — `_robust_aggregate` is
   unclamped (`median ± λ·IQR`), so reflected cost reaches `~1+λ`, breaking the
   `bᵢ ∈ [0,1]` assert, `T_norm ≤ 1`, and the `1.0` worst-floor. Fix: clamp the
   aggregated cost to `[0,1]` (winner-preserving).
4. **Abstainer masks all axes under Tchebycheff `max`** (§6.3) — a data-dependent
   abstention can flatten an entire study.
5. **Pruner ↔ direction coupling** (§7 Phase 1) — Phase 1 and Phase 2 are atomic
   unless pruning is disabled between them.
6. **Composite normalization across trials** (§6.2) — must use a study-global
   constant roster, else the cross-trial winner can change.
7. **Geometric-mean literal port** (§9) — never expose geomean-of-cost.
8. **Silent direction-mismatch load** (§7 Phase 2; **verified**) — optuna
   `create_study(load_if_exists=True)` does **not** reject a `maximize` study when
   asked for `minimize`; it silently keeps `maximize`, so reusing a pre-cutover
   study would maximize cost (pick the worst pipeline) with no error. Fixed by
   bumping `_STUDY_NAME` (collision-impossible), not by relying on Optuna.
9. **`_is_suspicious` reflection** — both halves invert; off-by-reflection
   silently disables the gaming flag.

**Decisions made** (2026-06-09)
- **Persistence → HARD CUTOVER** (§7 Phase 2): version bump + guard, no
  converter; pre-cutover studies error out and must be re-run.
- **`ε` / `ρ` defaults → `1e-3` / `0.05`** (§6.4): `ρ` is the quality dial (sweep
  `{0.01, 0.05, 0.1}` to confirm); `ε` is a fixed numerical safety margin.
- **Keep `_weighted_mean` as a `blend="weighted_mean"` opt-out** (§6.5);
  `tchebycheff` is the default; `weights` semantics are now blend-dependent.
- **`Sense` default → `LOWER_BETTER`** (cost-native, §5.1). A new scorer that
  emits a raw loss needs no annotation; a goodness-emitting scorer must declare
  `HIGHER_BETTER` explicitly. **Migration implication:** an un-annotated scorer is
  assumed to emit cost, so **every existing scorer must be annotated in the
  same change that rewrites its emission** (Phase 1) — Phase 0 cannot "land dark"
  by defaulting all scorers, because today's scorers still emit goodness and would
  be mis-oriented. Phase 0 therefore ships only the machinery; Phase 1 flips
  emission + annotation atomically per scorer.
- **Terminology → `cost`** (was "badness"): normalized cost ∈ [0,1], minimize;
  `to_cost`, `_MINIMIZE`. Aligns with `_HIGHER_IS_BAD` (higher metric = higher
  cost). (Spec filename keeps its original slug; content uses "cost".)
- **OQ1 = A — keep internal folds.** Existing scorers already emit bounded `[0,1]`
  values; they declare `_TERM_SENSE = HIGHER_BETTER` and `to_cost` complements
  (`1 − value`). The `fail_threshold`-as-anchor path is for *future* raw-loss
  scorers only (§6.6 corrected).
- **OQ2 — orientation is a base-class template method** (§5.2): scorers implement
  `_score_terms` + declare `_TERM_SENSE` (+ optional `_term_anchor`); base
  `score_image` wraps via `to_cost`; composite merges already-cost terms;
  Tchebycheff runs on per-child cost means (each ∈ [0,1]). Contract documented in
  §5.3 and propagated to 3 surfaces (§7 Phase 5).
- **OQ3 — single study-global active set** (§6.2/§6.3): the `max` numerator and
  the normalizer both use the children available study-wide; per-image abstention
  is a robust-aggregate sampling matter; empty active set → cost `1.0`.
- **OQ4 — gap fix** (§7 Phase 1): compute relative dispersion on `1 − cost` **and**
  raise `_GAP_EPS` 1e-12 → ~0.02 (both `_relative` callers).
- **OQ8 — `Sense` is uniform per scorer for v1** (§5.1); per-term mixed sense is
  YAGNI/deferred.
- **OQ9 — full `_is_suspicious`/`_evaluator.py` reflection list** enumerated in
  §7 Phase 1 (incl. the `Count` default `1.0→0.0`).
- **Zero exposed scalarization parameters** (§6.6): `ε`/`ρ`/reference are
  framework constants; normalization needs no input (terms are already `[0,1]`);
  weights default to uniform. Users set no scalarization parameters.

- **OQ5 — coarse importance lever → DEFERRED to v2** (§6.6): v1 ships uniform
  weights only; advanced users set `weights` directly.
- **OQ6 — GUI/`FEATURES.md` coverage → scoped into Phase 4**: cost relabel +
  `FEATURES.md` row (and `WORKFLOWS.md`/screenshots if affordance text changes);
  the read-only monitor reuses the friendly legacy-study detector; **no new
  control** (OQ5 deferred).
- **OQ7 — persistence → study-name bump (§7 Phase 2).** ⚠️ The guard **is**
  load-bearing for correctness: optuna `load_if_exists=True` silently loads a
  mismatched-direction study (verified, 4.9.0). Bumping `_STUDY_NAME`
  (`"tune"` → `"tune_cost_v1"`) makes the silent-maximize hazard impossible by
  construction; `tune_convention` stamp + legacy detector are UX on top.

**Still open:** none — all design decisions resolved.

---

## 12. References

- Bauß, J., & Stiglmayr, M. (2023). *Augmenting bi-objective branch and bound by
  scalarization-based information.* arXiv. https://doi.org/10.48550/arxiv.2301.11974
- Beliakov, G., Pradera, A., & Calvo, T. (2007). *Aggregation functions: A guide
  for practitioners.* Springer. (Geometric mean as a conjunctive aggregation
  function.) https://doi.org/10.1007/978-3-540-73721-6
- Dächert, K., Gorski, J., & Klamroth, K. (2012). An augmented weighted
  Tchebycheff method with adaptively chosen parameters for discrete bicriteria
  optimization problems. *Computers & Operations Research, 39*(12), 2929–2943.
  (Deriving `ρ` from a desired trade-off bound; the adaptive, largest-feasible-`ρ`
  rule.) https://doi.org/10.1016/j.cor.2012.02.021
- Das, I., & Dennis, J. E. (1997). A closer look at drawbacks of minimizing
  weighted sums of objectives for Pareto set generation in multicriteria
  optimization problems. *Structural Optimization, 14*(1), 63–69.
  https://doi.org/10.1007/BF01197559
- Deb, K., Miettinen, K., & Chaudhuri, S. (2010). Toward an estimation of nadir
  objective vector using a hybrid of evolutionary and local search approaches.
  *IEEE Transactions on Evolutionary Computation, 14*(6), 821–841. (Estimating
  the ideal/nadir reference is itself a sub-problem.)
  https://doi.org/10.1109/tevc.2010.2041667
- Engau, A. (2017). Proper efficiency and tradeoffs in multiple criteria and
  stochastic optimization. *Mathematics of Operations Research, 42*(1), 119–134.
  https://doi.org/10.1287/moor.2016.0796
- Geoffrion, A. M. (1968). Proper efficiency and the theory of vector
  maximization. *Journal of Mathematical Analysis and Applications, 22*(3),
  618–630. https://doi.org/10.1016/0022-247X(68)90201-1
- Marler, R. T., & Arora, J. S. (2010). The weighted sum method for
  multi-objective optimization: New insights. *Structural and Multidisciplinary
  Optimization, 41*(6), 853–862. (Objective scaling changes the Pareto point a
  given weight set yields → normalize before weighting.)
  https://doi.org/10.1007/s00158-009-0460-7
- Mohammadi, A., Omidvar, M. N., & Li, X. (2015). Sensitivity analysis of
  Penalty-based Boundary Intersection on aggregation-based EMO algorithms. *IEEE
  Congress on Evolutionary Computation (CEC)*, 2891–2898. (No single PBI penalty
  `θ` works across problems.) https://doi.org/10.1109/CEC.2015.7257248
- Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line landscape
  approximation for expensive multiobjective optimization problems. *IEEE
  Transactions on Evolutionary Computation, 10*(1), 50–66. (Origin of the
  augmented Tchebycheff `ρ = 0.05` convention in surrogate MOO.)
  https://doi.org/10.1109/TEVC.2005.851274
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization.* Kluwer.
  https://doi.org/10.1007/978-1-4615-5563-6
- Rojas-Gonzalez, S., Jalali, H., & Van Nieuwenhuyse, I. (2018). A
  stochastic-kriging-based multiobjective simulation optimization algorithm.
  *Proceedings of the Winter Simulation Conference*, 2155–2166. (Reproduces the
  ParEGO augmented Tchebycheff with `ρ = 0.05` and `[0,1]` normalization.)
  https://doi.org/10.1109/WSC.2018.8632322
- Steuer, R. E., & Choo, E.-U. (1983). An interactive weighted Tchebycheff
  procedure for multiple objective programming. *Mathematical Programming,
  26*(3), 326–344. https://doi.org/10.1007/BF02591870
- Tripp, A. (2025). *Chebyshev scalarization explained.*
  https://www.austintripp.ca/blog/2025-05-12-chebyshev-scalarization/
- Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary algorithm
  based on decomposition. *IEEE Transactions on Evolutionary Computation, 11*(6),
  712–731. (Weighted-sum / Tchebycheff / PBI decomposition; PBI penalty `θ`.)
  https://doi.org/10.1109/TEVC.2007.892759

*Sourcing note:* Miettinen (1998), Steuer & Choo (1983), Das & Dennis (1997), and
Geoffrion (1968) were confirmed via citing literature and standard secondary
sources, not retrieved full-text. Verify directly before formal citation.
