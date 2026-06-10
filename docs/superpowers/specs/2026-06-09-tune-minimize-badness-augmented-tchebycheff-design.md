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
>    change would silently make the unsigned form reward overshoot.

- **Utopia `z*ᵢ = −ε`** (e.g. `ε = 1e-3`): a **strictly-dominating** reference
  point. Tchebycheff's reach-every-Pareto-point property requires the reference
  to strictly dominate (be unachievable for) the whole front; since child badness
  is achievable down to `0`, `z* = 0` is only *weakly* dominating on a perfect
  axis, so we shift to `−ε`. Static constant — no per-trial state, no estimation,
  no resume hazard (see §9 for why the running-ideal estimation problem does not
  apply here). `ε` is **not scale-free**: the weights that realize a given Pareto
  point scale as `wᵢ ~ 1/(bᵢ + ε)`, so `ε` must be small *relative to the `[0,1]`
  badness scale* (too large flattens weight differences, biasing toward the L1
  term like a large `ρ`). At `ε = 1e-3` the shift is ~0.1% of full range.
- **Augmentation `ρ·Σ dᵢ`** upgrades minimizers from *weakly* Pareto optimal
  (plain Tchebycheff) to **properly** Pareto optimal, eliminating
  weakly-dominated solutions. Default `ρ = 0.05`, exposed as a `CompositeScorer`
  field. ρ-tuning tension (§11): too small → numerically weak solutions; too
  large → cuts off properly-efficient points with extreme trade-offs.

### 6.2 Normalize to `[0,1]` for downstream consumers
`Tᵨ` ranges over `[ε(1+ρn), (1+ε)(1+ρn)]`, not `[0,1]`. Several consumers assume
bounded `[0,1]` badness (the `failure_score=1.0` floor, the `_is_suspicious`
thresholds). Normalize by the theoretical max:

```
T_norm = Tᵨ(b) / Tᵨ(1...1)      # in (0, 1]
```

> **Correction (lit review) — the normalizer must be a study-global constant.**
> Dividing by `Tᵨ(1…1)` is argmin-preserving **only within one trial's
> objective**. But `Tᵨ(1…1) = (1+ε)(maxᵢ wᵢ + ρ Σᵢ wᵢ)` depends on the active
> term set `n` and the weights — and the active set **varies per trial** when a
> child abstains (§6.3). A per-trial-varying normalizer is monotone *within* a
> trial but **not** monotone *across* trials, so it can change the **cross-trial
> winner** that Optuna selects — contradicting "winner unchanged." The
> normalizer must therefore be computed against the **full, fixed term roster**
> (treat an abstaining term at its worst, do not drop it from the denominator),
> i.e. a constant for the whole study. This ties directly to the abstainer
> handling in §6.3.

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

> **New pitfall (migration review) — an abstaining child masks all axes under
> `max`.** Today an abstaining child is floored to `0.0` and a geometric mean
> simply ignores a perfect-looking axis. Under Tchebycheff `max`, flooring an
> abstainer to badness `1.0` (the worst) makes its `dᵢ = wᵢ(1+ε)` **the maximum
> term for every candidate**, pinning the composite near its ceiling and
> destroying discrimination on the *available* axes. Abstention is a property of
> the run/data (e.g. a `SupervisedScorer` with missing GT masks), so this would
> silently flatten an entire study. **Fix:** the single-objective Tchebycheff
> path must **exclude abstaining children from the `max`** (compute over present
> terms only) rather than floor-then-max — while §6.2's normalizer still uses the
> full roster for cross-trial comparability. (The `0.0 → 1.0` floor at `:240`
> remains correct for the *multi-objective* NSGA-II vector, which needs a
> fixed-length vector; the two paths handle abstention differently and the doc
> must keep them distinct.)

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
- Add `to_badness` (new `_scoring/_orient.py`).
- Lift `_HIGHER_IS_BAD` ↔ `Sense` mapping; keep QC checks unchanged.
- **No optimizer change yet**; boundary is wired but configured to reproduce
  current behavior (orient to goodness) behind a flag, or landed dark.

### Phase 1 — flip the Evaluator math (`_evaluation/**`)
- `_WORST_TERM: 0.0 → 1.0`; `failure_score: 0.0 → 1.0` (and define a finite
  `_FAILURE_BADNESS` ≥ achievable composite max if the composite can exceed 1
  before normalization — but §6.2 normalization keeps it `≤ 1`, so `1.0` holds).
- `_robust_aggregate`: `median − λ·IQR → median + λ·IQR` (`:53`, `:65`).
- `_is_suspicious` (`:104`): reflect — `score <= (1 − suspicious_score_floor)`
  **and** `count_badness >= (1 − suspicious_count_floor)`. Rename floors or
  reflect internally; update docstrings.
- **BLOCKER — second direction-sensitive gap: `compute_generalization_gap`
  (`_evaluation/_generalization.py:100`).** The overfit detector computes
  `absolute_drop = cal_score − heldout_score` and flags `drop > margin`. Under
  goodness, overfit = `cal > heldout` (positive drop). Under **badness** the sign
  inverts: a genuinely overfit winner has *higher* held-out badness, so
  `cal_badness − heldout_badness` is **negative** and the gate **never fires** —
  the overfit detector silently dies (it writes a wrong `generalization.json`
  rather than erroring). Flip to `heldout − cal` and re-examine the margin
  comparisons. **This file is in neither the original §3 inventory nor any
  phase** — added here (layer 5b). Update its doctest (`:91`).
- **PITFALL — relative-IQR blowup at BOTH `_relative` call sites.** The shared
  `_aggregate_math._relative(x, central)` floors the denominator at
  `_GAP_EPS=1e-12`. Under badness a great candidate has central tendency `≈ 0`,
  so the relative value explodes. This hits **(a)** `_per_trial_dispersion`
  (`_evaluator.py:68`, the `gap` signal) **and (b)** `compute_generalization_gap`
  (`_generalization.py:101`). Fix **once** at the shared helper or at both
  callers: compute on the goodness-equivalent (`1 − b`) or against a fixed
  denominator; never divide by a near-zero badness central tendency. Consumers of
  `gap` to re-review: the GUI `GAP_FLAG_THRESHOLD=0.15` and the data-poor
  fallback `calibration_stability=winner.gap` (`_generalization.py:266`).
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
- **Study persistence migration.** Optuna stores `direction` in the study;
  reopening a `maximize` study as `minimize` raises. Provide:
  1. A **version bump** of the study/spec schema (a `tune_convention`/version
     tag), and
  2. A startup **guard** that detects an old-direction study and refuses with an
     actionable message. **It must run inside `OptunaStudyStore.__init__`
     *before* the `create_study(..., load_if_exists=True)` call (`:80`)** — that
     call itself raises on a direction mismatch, so a guard placed after it never
     executes. Read the persisted direction via `optuna.load_study` (or storage
     introspection) first.
  3. An optional **one-shot converter** that creates a new `minimize` study and
     re-adds trials (via `add_trial`, already used at `:187`; FAIL trials carry
     `value=None`) with `score → 1 − score`, `objectives/terms → 1 − value`
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
- `_screening_freeze.py` — **six** score-direction sites (the original draft
  listed three and mis-cited them): the two `freeze_value` / `_warm_started_store`
  `sorted(..., key=score, reverse=True)`, the `_genuinely_focused_best`
  `max(fresh, key=score)`, the winner-across-rounds `max(union, key=score)`, the
  two `float("-inf")` sentinels, and the `_resolve_winner` `focused_score <
  explore_score` recovery test. Flip all: `reverse=True→False`, `max→min`,
  `-inf→+inf`, `<→>`, and `_apply_focused_penalty` adds instead of subtracts.
  `_screening.py` importance sorts are over **importances** (variance attribution
  — sign-independent, unchanged).
- GUI: `gui/tune/_winner.py` (doctest `score=0.9` → low badness);
  `_study_read.py` (`running_best max→min`, `shortlist reverse=True→False`,
  `GAP_FLAG_THRESHOLD=0.15` re-review after the gap fix, y-axis label "score");
  `_run_root.py:46` `_MULTI_OBJECTIVE_PLACEHOLDER_DIRECTIONS
  ["maximize","maximize"] → ["minimize","minimize"]`; `_callbacks.py`. Relabel
  "score" semantics (lower = better). Follow `gui/FEATURES.md` + `WORKFLOWS.md`
  gates if any affordance text changes.

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
| Phase 1 evaluator + scorers | M–L | **High** | Reflection provable for the core, but it must also fix the inverted overfit detector (`_generalization.py`), both `_relative` blowups, and `_is_suspicious`; couples atomically with Phase 2 via the pruner |
| Phase 2 direction + persistence | M | **High** | Sticky: Optuna study direction is immutable; guard must precede `create_study`; resume of old studies needs converter or hard cutover |
| Phase 3 Tchebycheff composite | M | Med | The combiner is ~10 lines, but it must exclude abstainers from the `max`, use a study-global normalizer, and ship a **non-convex regression test** + an abstainer-masking test |
| Phase 4 Pareto/screening/GUI | M | Med | Mechanical flips; GUI relabeling + FEATURES/WORKFLOWS gates |
| Phase 5 docs/tests | M | Low | Explainer rewrite is required, not optional |

**Stickiness / reversibility.** Phase 0 and Phases 3–4 are reversible. Phases 1–2
(the atomic cutover) are sticky: they change the persisted study `direction` and
the *meaning* of stored
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
- **Geometric-mean-of-badness is the trap**: the geometric mean is a conjunctive
  aggregation function (Beliakov, Pradera & Calvo, 2007) and does not commute
  with `s → 1−s`; because `0` is the product's annihilator, feeding *badness*
  into it makes one perfect axis (badness 0) zero the product and dominate — the
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
  produce an exploding/NaN `gap` — assert for **both** `_per_trial_dispersion`
  and `compute_generalization_gap`.
- **Overfit-gap sign**: a synthetic winner that is better on calibration than
  held-out must be **flagged** under badness (currently no test asserts the gap
  *sign*; this is how the inverted detector would ship undetected). Invert the
  `test_generalization.py` flag assertions.
- **Pruner inversion**: `test_optuna_pruning.py`'s end-to-end test seeds good
  trials at `1.0` and a bad one at `0.0` and asserts PRUNED — under minimize this
  must invert (good = low badness, bad = high badness) or it silently prunes the
  *best* candidate.
- **Abstainer masking**: a single-objective composite with one abstaining child
  must still discriminate among candidates on the present axes (regression for
  the §6.3 fix).
- **Persistence**: converter round-trips a maximize study to minimize with
  `1 − value`; guard rejects an unconverted old study with a clear error.
- **Enum/Literal alignment** test for `Sense` and `CompositeBlend`.

---

## 11. Pitfalls & open decisions

**Pitfalls** (ordered by risk after the migration review)
1. **Inverted overfit detector** (§7 Phase 1, `_generalization.py:100`) —
   **highest risk.** The migration originally missed this file entirely; the gap
   sign inverts under badness and the detector silently stops flagging real
   overfit, writing a wrong `generalization.json` with no error and no test
   catching it.
2. **`_relative` blowup at both call sites** (§7 Phase 1) — `_per_trial_dispersion`
   *and* `compute_generalization_gap` divide by a central tendency that → 0 for
   good candidates.
3. **Abstainer masks all axes under Tchebycheff `max`** (§6.3) — a data-dependent
   abstention can flatten an entire study.
4. **Pruner ↔ direction coupling** (§7 Phase 1) — Phase 1 and Phase 2 are atomic
   unless pruning is disabled between them.
5. **Composite normalization across trials** (§6.2) — must use a study-global
   constant roster, else the cross-trial winner can change.
6. **Geometric-mean literal port** (§9) — never expose geomean-of-badness.
7. **Study-direction immutability** (§7 Phase 2) — guard must precede
   `create_study`; resume conflict on old studies.
8. **`_is_suspicious` reflection** — both halves invert; off-by-reflection
   silently disables the gaming flag.

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
- Beliakov, G., Pradera, A., & Calvo, T. (2007). *Aggregation functions: A guide
  for practitioners.* Springer. (Geometric mean as a conjunctive aggregation
  function.) https://doi.org/10.1007/978-3-540-73721-6
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
- Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line landscape
  approximation for expensive multiobjective optimization problems. *IEEE
  Transactions on Evolutionary Computation, 10*(1), 50–66. (Origin of the
  augmented Tchebycheff `ρ = 0.05` convention in surrogate MOO.)
  https://doi.org/10.1109/TEVC.2005.851274
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
