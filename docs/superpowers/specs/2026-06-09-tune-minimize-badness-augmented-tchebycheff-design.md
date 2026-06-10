# Tune module: minimize bounded "badness" with augmented Tchebycheff

- **Status:** Draft (for review)
- **Date:** 2026-06-09
- **Scope:** `src/phenotypic/tune/**`, plus the QC direction flag in
  `src/phenotypic/analysis/abc_/_quality_check.py`, the GUI tune surface
  (`src/phenotypic/gui/tune/**`), and the explainer
  `docs/superpowers/explain/tune-with-optuna.md`.
- **Decision recorded:** adopt Option 1 ("bounded badness in `[0,1]`, optimizer
  minimizes") as the internal convention, and replace the geometric-mean
  composite with an **augmented Tchebycheff** scalarization.

---

## 1. Summary

Today the tuner normalizes every objective to a bounded `[0,1]` score where
**higher is better** and Optuna **maximizes**. This document specifies flipping
the internal convention to bounded **badness** in `[0,1]` where **lower is
better** (`0` = perfect, `1` = worst) and Optuna **minimizes**, and replacing the
single-objective composite combiner (currently a geometric mean of per-child
goodness) with an **augmented Tchebycheff** scalarization.

Two coupled changes:

1. **Pole flip + direction declaration.** Generalize the QC `_HIGHER_IS_BAD`
   flag into a scorer/metric-level *sense* declaration so each scorer emits its
   **natural, intuitive** value (a divergence stays a divergence; Dice stays
   Dice) and one framework boundary orients it into bounded badness. The
   optimizer minimizes.
2. **Composite combiner.** The single-objective `CompositeScorer` blends
   per-child badness with augmented Tchebycheff
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
  badness-native convention lets loss metrics integrate **without any flip**, and
  consolidates the four hand-rolled flips into **one** framework boundary.
- A "minimize a loss/distance-from-ideal" framing makes the **augmented
  Tchebycheff** combiner natural, which (unlike the weighted sum, and unlike the
  geometric mean) can reach Pareto-optimal compromises on **non-convex** regions
  of the front (§9).

### Goals
- One internal convention: bounded badness `[0,1]`, minimize.
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
| 4 | Per-scorer flips | `_qc_scorer.py:23` `_threshold_anchored`; `_supervised.py` `1−Dice`/fold; `_reference_free_scorer.py:326`; `_composite.py:319` `_geometric_mean` | normalize *to* higher-better |
| 5 | Evaluator math | `_evaluation/_evaluator.py:23` `_WORST_TERM=0.0`; `:202` `failure_score=0.0`; `:53` `median − λ·IQR`; `:104` `_is_suspicious` (`score>=floor`); `:68` `_per_trial_dispersion` | 0 worst, subtract spread penalty, high score = good |
| 6 | Pareto / screening | `_study/_pareto.py:53` `_dominates` (`>=`/`>`), `:50` `_vector` fill `0.0`; `_screening_freeze.py:177,366,433,446,450,477` `max`/`reverse=True`/`float("-inf")` | higher-better domination, −inf sentinel |
| 7 | GUI + docs | `gui/tune/_winner.py`, `_study_read.py`, `_callbacks.py`; `docs/superpowers/explain/tune-with-optuna.md`; `tune/CLAUDE.md` | display/sort by max |

The reusable precedent: `analysis/abc_/_quality_check.py:88` `_HIGHER_IS_BAD`;
`tools_/_qc_recipe/_runner.py:349` already orients via
`(-metric if higher_is_bad else metric)`.

---

## 4. Target design — the badness convention

**Definition.** Every per-term score is a **badness** `b ∈ [0,1]`, where `b=0` is
perfect and `b=1` is the worst. The optimizer minimizes. `inf`/degenerate inputs
floor to `b=1.0` (bounded worst).

**Equivalence (linear/order paths).** With `b = 1 − s`:
- `median(b) = 1 − median(s)`, `IQR(b) = IQR(s)` ⇒ the per-term aggregate
  `median(b) + λ·IQR(b) = 1 − (median(s) − λ·IQR(s))`.
- `mean(b) = 1 − mean(s)`.
- Pareto domination is order-based; `≤`/`<` is the exact reflection of `≥`/`>`.

So minimizing the badness aggregate yields the **identical winner** to today's
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
    # Sense of the natural values returned by score_image (per term, or uniform).
    _TERM_SENSE: ClassVar[Sense] = Sense.LOWER_BETTER   # badness-native default
```

`Sense` is a closed value set: per `CLAUDE.md`, define it as an enum paired with
a `Literal` alias plus an alignment test (`set(get_args(SenseLiteral)) ==
{m.value for m in Sense}`). Mapping: `_HIGHER_IS_BAD=True` ↔ `Sense.LOWER_BETTER`.

### 5.2 The orientation boundary
One function converts a scorer's **natural** term value into bounded badness.
Applied at the Evaluator boundary (`_score_one_image`, `_evaluator.py:345`) so
every consumer downstream sees badness only:

```python
def to_badness(value: float, *, sense: Sense, anchor: float | None) -> float:
    """Natural value -> badness in [0,1] (0 perfect, 1 worst)."""
    if anchor is None:                      # value already bounded in [0,1]
        return value if sense is Sense.LOWER_BETTER else 1.0 - value
    if not math.isfinite(value):
        return 1.0                          # inf divergence -> worst
    goodness = math.exp(-math.log(2.0) * value / anchor)   # 1 at 0, .5 at anchor
    return (1.0 - goodness) if sense is Sense.LOWER_BETTER else goodness
```

- **Bounded, lower-better** (1−Dice already in `[0,1]`, etc.): identity.
- **Bounded, higher-better** (Dice, IoU, ICC, solidity): `1 − value`.
- **Unbounded, lower-better** (count divergence, relative MAD): the
  threshold-anchored complement; `anchor` is the check's `fail_threshold`.
- **Unbounded, higher-better** (rare): `goodness` directly.

The QC check's `(_HIGHER_IS_BAD, fail_threshold)` pair maps onto `(sense,
anchor)` exactly, so a QC-backed scorer needs **no** bespoke flip.

### 5.3 New-module authoring contract
A new scorer/check declares **sense**, supplies an **anchor** only if its natural
value is unbounded, returns its **natural** value, and **registers** (re-export
from `tune/__init__.py` + class registry, else GUI/`from_json` cannot see it).
Existing scorers are rewritten to emit natural values and declare sense (§7,
Phase 0).

---

## 6. Target design — augmented Tchebycheff composite

### 6.1 Formula
For per-child badness scalars `bᵢ ∈ [0,1]`, positive weights `wᵢ` (the existing
`weights` dict; missing → `1.0`), utopia point `z*ᵢ = −ε`, and augmentation
coefficient `ρ`:

```
dᵢ = wᵢ · (bᵢ − z*ᵢ) = wᵢ · (bᵢ + ε)          # all dᵢ > 0
Tᵨ(b) = max_i dᵢ  +  ρ · Σ_i dᵢ               # minimize
```

- **Utopia `z*ᵢ = −ε`** (e.g. `ε = 1e-3`): a **strictly-dominating** reference
  point. Tchebycheff's reach-every-Pareto-point property requires the reference
  to strictly dominate (be unachievable for) the whole front; since child badness
  is achievable down to `0`, `z* = 0` is only *weakly* dominating on a perfect
  axis, so we shift to `−ε`. Static constant — no per-trial state, no estimation,
  no resume hazard (see §9 for why the running-ideal estimation problem does not
  apply here).
- **Augmentation `ρ·Σ dᵢ`** upgrades minimizers from *weakly* Pareto optimal
  (plain Tchebycheff) to **properly** Pareto optimal, eliminating
  weakly-dominated solutions. Default `ρ = 0.05`, exposed as a `CompositeScorer`
  field. ρ-tuning tension (§11): too small → numerically weak solutions; too
  large → cuts off properly-efficient points with extreme trade-offs.

### 6.2 Normalize to `[0,1]` for downstream consumers
`Tᵨ` ranges over `[ε(1+ρn), (1+ε)(1+ρn)]`, not `[0,1]`. Several consumers assume
bounded `[0,1]` badness (the `failure_score=1.0` floor, the `_is_suspicious`
thresholds). Normalize by the theoretical max (a strictly monotonic transform, so
the **argmin / winner is unchanged**):

```
T_norm = Tᵨ(b) / Tᵨ(1...1)      # in (0, 1]
```

### 6.3 Where it plugs in
`_composite.py:203 finalize`:
- `multi_objective=True` → unchanged in shape: return the per-child badness
  dict; the abstainer floor flips `0.0 → 1.0` (worst badness) at `:240`; the
  vector feeds NSGA-II with `directions=["minimize"]*n`.
- single-objective with `weights` → augmented Tchebycheff with those weights
  (replaces `_weighted_mean` as the compensatory-vs-conjunctive choice; keep
  `_weighted_mean` available as an explicit `blend="weighted_mean"` opt-out).
- single-objective default → augmented Tchebycheff with uniform weights
  (**replaces** `_geometric_mean`, `:319`).

Introduce a `blend: CompositeBlend` selector (`Literal["tchebycheff",
"weighted_mean"]`, default `"tchebycheff"`). The geometric-mean-of-badness path
is **never** exposed: it inverts the conjunctive property (one perfect axis
zeroes the product and dominates), the documented trap this design avoids.

---

## 7. Migration plan (phased)

The per-term reflection makes a coordinated cutover safe, but the work is phased
so each phase is independently testable. Phases 0–1 are behavior-preserving;
Phase 2 is the sticky persistence cutover; Phase 3 is the deliberate composite
change.

### Phase 0 — direction declaration + orientation boundary (additive)
- Add `Sense` enum + `Literal` alias + alignment test (`tools_/typing_.py`).
- Add `to_badness` (new `_scoring/_orient.py`).
- Lift `_HIGHER_IS_BAD` ↔ `Sense` mapping; keep QC checks unchanged.
- **No optimizer change yet**; boundary is wired but configured to reproduce
  current behavior (orient to goodness) behind a flag, or landed dark.

### Phase 1 — flip the Evaluator math (`_evaluation/_evaluator.py`)
- `_WORST_TERM: 0.0 → 1.0`; `failure_score: 0.0 → 1.0` (and define a finite
  `_FAILURE_BADNESS` ≥ achievable composite max if the composite can exceed 1
  before normalization — but §6.2 normalization keeps it `≤ 1`, so `1.0` holds).
- `_robust_aggregate`: `median − λ·IQR → median + λ·IQR` (`:53`, `:65`).
- `_is_suspicious` (`:104`): reflect — `score <= (1 − suspicious_score_floor)`
  **and** `count_badness >= (1 − suspicious_count_floor)`. Rename floors or
  reflect internally; update docstrings.
- **PITFALL — `_per_trial_dispersion` (`:68`)**: it returns the *relative* IQR
  `iqr / max(|median|, eps)`. Under badness a great candidate has
  `median_b ≈ 0`, so the denominator collapses and `gap` explodes. Compute the
  gap on the **goodness-equivalent** (`1 − b`) or against a fixed denominator;
  do **not** let it divide by a near-zero badness median. This does not change
  the optimum but its calibration (and any threshold on it) must be reviewed.
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
- **Study persistence migration.** Optuna stores `direction` in the study;
  reopening a `maximize` study as `minimize` raises. Provide:
  1. A **version bump** of the study/spec schema (a `tune_convention`/version
     tag), and
  2. A startup **guard** that detects an old-direction study and refuses with an
     actionable message, and
  3. An optional **one-shot converter** that creates a new `minimize` study and
     re-adds trials with `score → 1 − score`, `objectives/terms → 1 − value`
     (valid because all stored values are bounded `[0,1]`).
  Stored `pheno_terms`/`pheno_objectives` change meaning; cross-study comparison
  with pre-migration runs is invalid (document).

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
- `_screening_freeze.py`: `reverse=True → reverse=False` (`:178,367,477`);
  `max(fresh, key=score) → min` (`:433`); `float("-inf") → float("inf")`
  (`:446,450`); `_resolve_winner` comparison `focused_score < explore_score`
  inverts to `>` (`:456`); `_apply_focused_penalty` (`:410`) adds instead of
  subtracts. `_screening.py:116,211` importance sorts are over **importances**
  (unchanged, higher = more important).
- GUI `gui/tune/_winner.py` (doctest `score=0.9` → low badness), `_study_read.py`,
  `_callbacks.py`: sort/`max → min`; relabel "score" semantics (lower = better);
  update tooltips/badges. Follow `gui/FEATURES.md` + `WORKFLOWS.md` gates if any
  affordance text changes.

### Phase 5 — docs + tests
- `tune/CLAUDE.md`: "Higher-is-better everywhere" → "Badness everywhere
  (minimize); the single `_MINIMIZE` literal."
- `docs/superpowers/explain/tune-with-optuna.md` (+ `.graph.md`): rewrite the
  scorer/aggregate/Pareto/composite math sections (CLAUDE.md mandates this in the
  same change).
- Tests (§10).

---

## 8. Migration costs

| Layer | Effort | Risk | Notes |
|-------|--------|------|-------|
| Phase 0 orient boundary | S | Low | Additive; enum + 1 function + tests |
| Phase 1 evaluator + scorers | M | Med | Reflection is provable; `gap` pitfall + `_is_suspicious` reflection need care |
| Phase 2 direction + persistence | M | **High** | Sticky: Optuna study direction is immutable; resume of old studies needs converter or hard cutover |
| Phase 3 Tchebycheff composite | S–M | Med | ~10-line combiner; the cost is the selector field, normalization, ρ default, and a **non-convex regression test** |
| Phase 4 Pareto/screening/GUI | M | Med | Mechanical flips; GUI relabeling + FEATURES/WORKFLOWS gates |
| Phase 5 docs/tests | M | Low | Explainer rewrite is required, not optional |

**Stickiness / reversibility.** Phases 0–1 and 3–4 are reversible. Phase 2 is
sticky: it changes the persisted study `direction` and the *meaning* of stored
`score`/`objectives`/`terms`. Distributed SLURM resume of a pre-migration study
is the highest-risk item; the converter + guard mitigate it but pre-migration
runs cannot be silently reopened.

**Accuracy cost.** Zero for Phases 0–2 and 4 (provable reflection equivalence).
Phase 3 (composite) is a **deliberate** change: augmented Tchebycheff selects
different multi-criteria compromises than the geometric mean (it can reach
non-convex-front points the geometric mean cannot). This is an intended upgrade,
not a regression, but it changes which candidate wins on composite objectives, so
it requires a baseline snapshot and reviewer sign-off, not a silent swap.

---

## 9. Accuracy & theory notes (literature-audited)

- **Reflection equivalence** (median/IQR/mean): elementary; verified.
- **Geometric-mean-of-badness is the trap**: the geometric mean does not commute
  with `s → 1−s`; feeding badness into it makes one perfect axis (badness 0) zero
  the product and dominate — the opposite of the conjunctive "all axes must be
  good" property. Avoided by never exposing it (§6.3).
- **Weighted sum** reaches only convex-hull ("supported") Pareto points; concave
  regions are unreachable for any weights (Das & Dennis, 1997; Geoffrion, 1968).
- **Weighted Tchebycheff** can reach *every* Pareto-optimal point for some weight
  (Steuer & Choo, 1983; Miettinen, 1998), but plain-Tchebycheff minimizers are in
  general only **weakly** Pareto optimal — hence the **augmentation** (`ρ` term),
  which yields **properly** Pareto-optimal points (Steuer & Choo, 1983;
  Miettinen, 1998; Engau, 2017).
- **Reference point**: reachability requires only a **strictly dominating**
  (unachievable, lower-bounding) reference, not the tight ideal (Bauß &
  Stiglmayr, 2023; Tripp, 2025) — which is why `z*ᵢ = −ε` is both sufficient and
  state-free here. The "estimated running ideal" problem (a study-global `z*`
  updated from best-so-far values, which would break per-candidate score purity,
  deterministic resume, and cross-worker independence) **does not arise**, because
  the bounded `[0,1]` normalization gives a known static lower bound.
- **Power-mean note**: `p ≤ 0` means (geometric/harmonic) require strictly
  positive arguments; badness can be `0`. Augmented Tchebycheff (an L∞-family,
  `p → ∞` flavor) has no positivity hazard — another reason to prefer it over any
  geometric/harmonic badness blend.

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
- **`gap` pitfall guard**: a near-perfect candidate (median badness ≈ 0) must not
  produce an exploding/NaN `gap`.
- **Persistence**: converter round-trips a maximize study to minimize with
  `1 − value`; guard rejects an unconverted old study with a clear error.
- **Enum/Literal alignment** test for `Sense` and `CompositeBlend`.

---

## 11. Pitfalls & open decisions

**Pitfalls**
1. **`_per_trial_dispersion` relative-IQR blowup** under badness (§7 Phase 1) —
   the sharpest non-obvious trap.
2. **Geometric-mean literal port** (§9) — never expose geomean-of-badness.
3. **Study-direction immutability** (§7 Phase 2) — resume conflict on old studies.
4. **`_is_suspicious` reflection** — the count "under-detection" half and the
   "great score" half both invert; off-by-reflection here silently disables the
   gaming flag.
5. **Composite normalization** (§6.2) — without it, `failure_score=1.0` and the
   `_is_suspicious` `[0,1]` thresholds are miscalibrated against the raw
   Tchebycheff range.

**Open decisions**
- `ε` (utopia shift) and `ρ` (augmentation) defaults: proposed `1e-3` and `0.05`;
  ρ warrants a short sensitivity check.
- Persistence: hard cutover (version bump + guard, no old-study resume) vs. ship
  the converter. Recommend converter + guard.
- Keep `_weighted_mean` as a `blend` opt-out, or drop it? Recommend keep
  (compensatory blending is a legitimate user choice).
- Whether `Sense` default is `LOWER_BETTER` (badness-native) — proposed — or
  `HIGHER_BETTER` (matches today's emitted goodness pre-rewrite). Tied to whether
  Phase 0 lands dark.

---

## 12. References

- Bauß, J., & Stiglmayr, M. (2023). *Augmenting bi-objective branch and bound by
  scalarization-based information.* arXiv. https://doi.org/10.48550/arxiv.2301.11974
- Das, I., & Dennis, J. E. (1997). A closer look at drawbacks of minimizing
  weighted sums of objectives for Pareto set generation in multicriteria
  optimization problems. *Structural Optimization, 14*(1), 63–69.
  https://doi.org/10.1007/BF01197559
- Engau, A. (2017). Proper efficiency and tradeoffs in multiple criteria and
  stochastic optimization. *Mathematics of Operations Research, 42*(1), 119–134.
  https://doi.org/10.1287/moor.2016.0796
- Geoffrion, A. M. (1968). Proper efficiency and the theory of vector
  maximization. *Journal of Mathematical Analysis and Applications, 22*(3),
  618–630. https://doi.org/10.1016/0022-247X(68)90201-1
- Miettinen, K. (1998). *Nonlinear Multiobjective Optimization.* Kluwer.
  https://doi.org/10.1007/978-1-4615-5563-6
- Steuer, R. E., & Choo, E.-U. (1983). An interactive weighted Tchebycheff
  procedure for multiple objective programming. *Mathematical Programming,
  26*(3), 326–344. https://doi.org/10.1007/BF02591870
- Tripp, A. (2025). *Chebyshev scalarization explained.*
  https://www.austintripp.ca/blog/2025-05-12-chebyshev-scalarization/

*Sourcing note:* Miettinen (1998), Steuer & Choo (1983), Das & Dennis (1997), and
Geoffrion (1968) were confirmed via citing literature and standard secondary
sources, not retrieved full-text. Verify directly before formal citation.
