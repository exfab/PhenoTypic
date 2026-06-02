# QC-as-Objective Mapping — the `QCScorer`

**Purpose & scope.** This document specifies how PhenoTypic's existing
quality-control (QC) checks are mapped into a tuning **objective** — the
`QCScorer`. The QC system's normal job is *diagnostic* (flag suspicious plates
after a run); the `QCScorer` repurposes the *same* signals to score a candidate
parameter set during tuning: **the best parameters are the ones that produce the
most plausible, internally-consistent, reproducible plates** — with no
ground-truth annotation.

**Pointer to the master spec.** Companion to
[`2026-06-01-parameter-tuning-engine-design.md`](2026-06-01-parameter-tuning-engine-design.md),
elaborating **§4** (the `Scorer` protocol) and decision **D1** (`QCScorer` is the
**primary Phase-1 default** objective; `ReferenceFreeScorer` is gated). It is the
no-ground-truth sibling of [`supervised-scorers.md`](supervised-scorers.md) and
the canonical home for the count/grid signals that
[`reference-free-segmentation-metrics.md`](reference-free-segmentation-metrics.md)
§C.6 also references. Its weights and knobs are tuned by that doc's
**meta-validation gate** (§E).

> **Design principle: reuse, don't re-implement.** The `QCScorer` *calls* the
> existing `phenotypic.analysis` `QualityCheck` classes and reads their outputs.
> It adds a normalization + reduction + fusion layer on top; it never
> re-implements a metric.

---

## 1. What the QC checks already give us

Every check subclasses `QualityCheck` (`analysis/abc_/_quality_check.py`) and
exposes a uniform contract that is *most of the way* to a `Scorer`:

- **A continuous, directional metric** `QC_<name>_Metric` in the check's own
  units.
- **A class-level `_HIGHER_IS_BAD` flag** — which is *exactly the inverse of the
  master spec's `Scorer.higher_is_better` contract*. Direction is declared, per
  check, already.
- **Calibrated `warn_threshold` / `fail_threshold`** in metric units (the QC
  authors' "this is getting bad" / "this is bad" levels).
- **A per-group `summary()`** → `qc_worst_metric`, `qc_n_flagged`,
  `qc_n_members`, `qc_status` per `groupby` key.
- Per-row `Flag` / `Status` (`pass`/`warn`/`fail`) — these drive **GUI
  curation**, *not* the objective. The objective uses the continuous metric.

So the `QCScorer` is **not new metrics**. It is three layers over the existing
checks: a fixed **normalizer** (metric → bounded higher-is-better term), a
**reduction** (per-group terms → one term per check), and a **fusion** (terms →
one score), reusing `_HIGHER_IS_BAD` for direction and the thresholds for scale.

### The six v1 checks

| Check (class) | `name` | Metric | Range | Direction | warn / fail | Needs |
|---|---|---|---|---|---|---|
| `ExpectedVsDetectedCount` | **Count** | `\|detected−expected\| / expected` | `[0,∞)`, **`inf`** if no layout match | higher-bad | 0.05 / 0.10 | per-plate layout metadata |
| `TukeyOutlierFraction` | **Tukey** | fraction outside Tukey fences (`k=1.5`) | **`[0,1]`** | higher-bad | 0.10 / 0.25 | ≥4 members/bin |
| `ReplicateAgreement` | **SE** | `\|SE\| / \|mean\|` (relative SE) | `[0,∞)` | higher-bad | 0.10 / 0.20 | ≥2 replicates/bin |
| `RelativeMAD` | **MAD** | `MAD / \|median\|` (relative MAD) | `[0,∞)` | higher-bad | 0.10 / 0.20 | ≥2 replicates/bin |
| `MaxModifiedZScore` | **ZMax** | `max 0.6745·\|x−median\| / MAD` | `[0,∞)`, Z-units | higher-bad | 3.5 / 5.0 | ≥2 members/bin |
| `ICC` | **ICC** | ICC(2,1) two-way absolute-agreement | **`(−,1]`, can be negative** | **lower-bad** | class defaults (`fail ≤ warn`) | complete subject×rater matrix |

`NaN` metrics are emitted for under-powered / degenerate / missing-axis bins. The
base class maps `NaN → "pass"` so degenerate bins never *gate curation* — **this
is a trap for an objective** (see §5).

---

## 2. The replicate-structure insight (what makes this tractable)

**In PhenoTypic's arrayed design each grid position is a unique strain (one
colony), and technical replicates are *between* plates, not within one.** That
single fact reshapes the mapping:

- The replicate/reliability checks (**SE, MAD, ZMax, Tukey, ICC**) compare the
  *same strain/position across replicate plates*. Within one plate each strain is
  size-1 → under-powered → `NaN`. So **they are inherently batch-level
  (cross-plate)**, grouped by strain/position with replicates = plates.
- The **per-image tier collapses to Count alone** — detected-vs-expected colonies
  on that plate vs. its layout. (A within-plate *size-distribution* outlier check
  is technically per-image, but it is a weak/ambiguous signal here — different
  strains legitimately differ in size — so it is deferred.)

This yields a clean two-tier structure: **per-image = Count**, **batch = the
reliability panel.**

---

## 3. Architecture — a two-phase `Scorer`

Because Count is per-image but the panel is batch-level, `QCScorer` cannot be a
pure per-image `score(image)`. The `Scorer`/`Evaluator` protocol gains a
**batch-finalize hook** (general — any batch objective, e.g. ICC, needs it):

```
score_image(image, result)  -> { "Count": t_count, ... }   # per-plate terms
        carried by the Evaluator, robust-aggregated across plates ⇒ Count_agg
finalize(per_image_results, full_measurements) -> float     # the QCScore
        runs the batch panel once over all plates, reduces + fuses
```

- `score_image` runs `ExpectedVsDetectedCount` against *that plate's* layout and
  returns the normalized **Count** term. The Evaluator aggregates Count across
  the calibration plates with its robustness operator (`median − λ·dispersion`,
  master §4 D4).
- `finalize` runs the **batch panel** once across all calibration plates, reduces
  each check across strains (§4.3), builds the reliability composite, and fuses
  it with `Count_agg` (§4.4). Pruning early-stops on the per-image Count terms;
  the batch panel is computed only for survivors, at full fidelity.

---

## 4. The term pipeline

### 4.1 Run the existing check

Construct each `QualityCheck` from the run config (layout path, `groupby`,
`time_label` / `subject_label` / `rater_label`) and call `analyze()` to obtain
`QC_<name>_Metric` and the per-group `summary()`. Direction comes from
`_HIGHER_IS_BAD`; scale anchors come from `warn_threshold` / `fail_threshold`.

### 4.2 Normalize — threshold-anchored smooth map → `t ∈ [0,1]` (higher-is-better)

A **fixed, smooth, bounded** map per check, anchored by the calibrated fail
threshold so `t = 0.5` exactly at the "fail" level. Fixed/external (not min–max
over swept candidates) ⇒ Böck-safe. The **continuous metric** is used, never the
`pass/warn/fail` tiers (the optimizer needs a gradient, not a 3-step cliff).

- **Higher-is-bad, unbounded** (Count, SE, MAD, ZMax), metric `m ≥ 0`, fail `f`:

  ```
  t = exp(−ln2 · m / f)        # t=1 at m=0, t=0.5 at m=f, →0 as m→∞
  ```

  (A linear-clip `t = clip(1 − m/2f, 0, 1)` is the simpler alternative; the
  exp-decay is the default for its smooth tail.)

- **Tukey** (already `[0,1]`): same exp form with its own `f` (0.25), so the
  scale stays threshold-anchored and consistent with the others.

- **ICC** (lower-is-bad, `≤1`, can be negative): flip + clip onto `[0,1]`:

  ```
  t = clip( (ICC − ICC_floor) / (ICC_good − ICC_floor), 0, 1 )
  ```

  with `ICC_good ≈ warn` (→ `t≈1`) and `ICC_floor ≈ fail` (→ `t≈0`). A negative
  ICC (agreement worse than chance) falls below the floor → `t = 0`.

The exact functional form (exp vs linear vs the ICC anchors) is **gate-tunable**;
exp-decay anchored at `t(f)=0.5` is the default.

### 4.3 Special-value policy (per term)

| Value | Meaning | Objective treatment |
|---|---|---|
| `inf` (Count, no layout match) | plate has no metadata counterpart | `t = 0` (worst) |
| negative ICC | agreement worse than chance | clip to `t = 0` |
| `NaN` (under-powered / degenerate / missing axis) | **could not evaluate** | **excluded** from the reducer — *never* scored as a passing term; it instead drags the **coverage factor** (§4.4) |

The `NaN → exclude + coverage-penalty` rule is the heart of the anti-gaming
guard (§5): un-evaluable bins must not be free "passes."

### 4.4 Reduce — per check across its groups → one term per check

- **Count** has one group per plate ⇒ one per-image term; no within-image
  reduction. The Evaluator aggregates across plates.
- **Batch checks** produce one value per **strain/position** (that strain's
  reliability across its replicate plates). Collapse across strains with a
  **coverage-weighted, one-sided trimmed mean**:

  Let evaluable strains have normalized terms `t_s` and weights `w_s`
  (≈ replicate count); `E` = expected strains, `V` = evaluable (non-`NaN`,
  `≥ min_replicates`) strains.

  ```
  coverage = V / E
  μ_trim   = weighted_mean( t_s  for the upper (1−α) of strains by t_s )   # drop worst α
  T_check  = coverage · μ_trim
  ```

  Default **α = 0.10, one-sided (lower tail only)** — robust to a few
  biological/contaminated strains without chasing tail noise; size-weighting
  downweights flaky low-replicate strains. With `V < ~10` strains, `⌊αV⌋` rounds
  to 0 ⇒ it degrades gracefully to a plain coverage-weighted mean. **α and the
  reducer are gate-tunable** (the gate can switch to mean / quantile / CVaR).

### 4.5 Fuse — terms → one `QCScore`

Two stages, chosen to put **soft-AND on the gaming boundary** but **averaging
inside the noisy panel**:

1. **Reliability composite** (across the ~5 panel checks) — a weighted arithmetic
   mean, with **ZMax down-weighted** (spiky / gameable):

   ```
   R = Σ_c ω_c · T_c  /  Σ_c ω_c          over available panel checks c
   ```

2. **Final score** — a weighted **geometric mean** of Count and reliability, with
   per-term ε-floors and an explicit Count floor:

   ```
   QCScore = ( clip(Count_agg, ε, 1)^{w_C} · clip(R, ε, 1)^{w_R} )^{1/(w_C + w_R)}
   ```

The geometric mean means **no compensation across the count↔reliability
boundary**: great reliability cannot paper over a bad Count (each is a *necessary*
condition). The arithmetic mean *within* the panel means one spiky check (ZMax)
cannot veto the whole score. `w_C ≥ w_R` by default (Count is the anchor); all
weights, ε, and the Count floor are **gate-tunable**.

---

## 5. Anti-gaming (the load-bearing property)

The base class maps `NaN → "pass"` for *curation* (don't gate removal on
un-evaluable bins). For the *objective* this would be a hole: a candidate could
**win by detecting too few colonies** — fewer than 2 per strain ⇒ SE/MAD/ZMax
all `NaN` ⇒ "pass" ⇒ a great QC score on garbage. Four mechanisms close it,
together:

1. **Count is mandatory + two-sided** — `|detected − expected|/expected`
   penalizes under- *and* over-detection; it is the anti-gaming anchor.
2. **`NaN`-excluded + coverage factor** — un-evaluable strains lower
   `coverage = V/E`, so under-detection directly lowers every batch term.
3. **Soft-AND fusion** — the geometric mean drives `QCScore → 0` when `Count_agg`
   is low, regardless of reliability.
4. **Count floor** — an explicit cap below a minimum `Count_agg`.

A **gaming regression test** is mandatory: a synthetic "under-detect" candidate
must score *strictly lower* than a faithful one (§7).

---

## 6. Graceful degradation & ownership

**Availability matrix** — `QCScorer` auto-selects the terms the data supports:

| Available | Terms active |
|---|---|
| per-plate layout metadata | **Count** |
| ≥2 replicate plates of the same strains/positions | the **batch panel** (SE/MAD/ZMax/Tukey/ICC) |
| both | full `QCScorer` |
| neither | abstain (cannot run) — surface a clear error |

The common no-replicate case (one plate per condition) runs **Count-only**.

**Reuse & ownership (no duplication).**
- `QCScorer` is the canonical **no-GT** home for Count + the reliability panel.
- `supervised-scorers.md` owns the **ground-truth** versions (count modality,
  ICC against GT); `QCScorer` and `SupervisedScorer` share the count/grid checks
  by *calling* the same `analysis/` classes.
- `reference-free-segmentation-metrics.md` §C.6's count/grid priors should
  **call** `QCScorer`'s Count rather than re-derive it.

---

## 7. Honest limitation, testing, phasing

**Limitation.** `QCScorer` measures **plausible (Count) + reproducible
(panel)**, *not* **correct** (right boundaries). A systematically-biased-but-
consistent segmenter scores well; Count rewards the right *number*, not the right
*shape*. So it is the trustworthy Phase-1 default but a **proxy** — pair it with
supervised/reference-free shape terms when annotations exist, and let the
**meta-validation gate** (reference-free §E) certify it and tune its knobs.

**Testing.**
- *Unit:* normalizer maps (`t(f)=0.5`; `inf→0`; negative `ICC→0`; `NaN→excluded`);
  coverage-weighted trimmed-mean reducer (small-`V` fallback to mean;
  under-detection ⇒ coverage drop); hybrid fusion (low `Count_agg` vetoes; a ZMax
  spike does **not** veto); graceful degradation (Count-only path; abstain path).
- *Gaming regression:* an under-detecting candidate scores **strictly lower**.
- *Reuse:* assert `QCScorer` reads `QC_<name>_Metric` from the real
  `QualityCheck` instances (no metric re-implementation).
- *Integration:* on `load_synth_yeast_plate()` (+ a small replicate set),
  `QCScorer` ranks a known-good parameter set above a known-bad one and plugs
  into the Evaluator's two-phase loop.

**Phasing.**
1. **Phase 1** (with the engine core): **Count-only** `QCScorer` — needs no
   replicates, ships with the per-image Evaluator path.
2. **Phase 1+/2:** add the batch reliability panel + the `finalize` hook once the
   Evaluator's batch-finalize phase exists.
3. **Phase 3+:** the meta-validation gate tunes weights, α, reducer, normalizer
   form, ε, and the Count floor.

---

## 8. Open questions / gate-tunable knobs

- Normalizer functional form (exp-decay vs linear-clip) and the ICC anchors.
- Trim α (default 0.10) and whether to switch reducer (mean / quantile / CVaR)
  per the data's biological-noise rate — the gate decides empirically.
- Panel weights `ω_c` (esp. how far to down-weight ZMax; SE vs MAD are largely
  redundant — possibly keep only the robust MAD).
- Fusion weights `w_C / w_R`, ε-floor, and the explicit Count-floor level.
- Whether to add ICC at all in the first batch cut, given its fragility
  (complete-matrix requirement, NaN-heaviness) — it may be the last panel check
  enabled.
- The deferred within-plate size-sanity per-image signal (currently out of scope).
