# Robust Evaluation (the `Evaluator`)

Companion to the [parameter tuning engine design](2026-06-01-parameter-tuning-engine-design.md).
Deep dive on **master §4** (the `Evaluator` component) and **decision D4**: how a
candidate's parameters become a **generalization-aware** objective that resists
overfitting a single set of plates.

- **Status:** Design settled (pre-implementation). Core lands in **Phase 1**;
  pruning (the Optuna pruner) in **Phase 2**.
- **Maps to:** master §4 (`Evaluator`), D4 (calibration + stability + held-out),
  §7 (multi-objective), §8 (outputs), §12 (error handling / reproducibility), §14
  (open `λ`, low-fidelity questions). Composes with
  [`qc-objective-mapping.md`](qc-objective-mapping.md) §3 (the two-phase `Scorer`)
  and consumes candidates from [`search-space-inference.md`](search-space-inference.md).

---

## 1. Purpose and where it fits

The `Evaluator` is the **robustness layer**. Given one candidate's parameters, a
`Scorer`, and a calibration set, it produces a single objective (plus a per-image
breakdown) that reflects **how well the parameters generalise across the plate set**,
not how well they fit one lucky plate. It is the hub between the `SearchStrategy`
(which proposes params), the `Scorer` (which scores a result), and the existing
joblib/SLURM execution machinery (which runs the pipeline).

Its four jobs: (1) run the candidate pipeline across the calibration set; (2)
robust-aggregate the per-image scores with a stability penalty; (3) cross-validate
and validate the winner on held-out data; (4) feed multi-fidelity pruning. It owns
*robustness mechanics* — the `Scorer` owns *what a good result is*.

---

## 2. What master §4 / D4 lock (documented, not re-litigated)

Calibration set, metadata-stratified; `level − λ·dispersion` on a normalized
higher-is-better scale (`level` = median, `dispersion` = IQR); k-fold /
leave-one-plate-out for small counts; held-out validation of the winner; pruning
fidelity = number of calibration images. This doc resolves the *open* parts.

> **Terminology.** A calibration "image" is one plate (a `GridImage`); the per-image
> term is per-plate. "Plate" and "image" are used interchangeably below.

---

## 3. The uniform 3-step evaluation loop

The `Evaluator` runs **one** loop that serves both a per-image `Scorer` (e.g.
`SupervisedScorer`) and the two-phase `QCScorer` (qc §3) without special-casing:

```
for image in calibration_subset:
    result        = pipeline.measure(image, apply_post=False)     # clean, per-image (CLAUDE.md path)
    terms[image]  = scorer.score_image(image, result)             # dict of higher-is-better terms

aggregated_terms  = { t: median(terms[:, t]) − λ·IQR(terms[:, t]) for t in terms }   # §4
objective         = scorer.finalize(aggregated_terms, per_image_results, full_measurements)   # §scorer
```

1. **`score_image(image, result)`** → a dict of per-image **terms**, already
   normalized higher-is-better (§5). A per-image-only Scorer returns its metric
   terms; `QCScorer` returns `{"Count": t}`.
2. The **Evaluator robust-aggregates each term** across images (§4).
3. **`finalize(aggregated_terms, per_image_results, full_measurements)`** → the scalar
   objective. The **default `finalize`** is a weighted mean of `aggregated_terms`
   (covers every per-image-only Scorer); **`QCScorer` overrides** it to run the batch
   panel once and geometric-fuse using `aggregated_terms["Count"]` as `Count_agg`.

> **Reconciliation with qc §3.** That doc wrote `finalize(per_image_results,
> full_measurements)` but never said *how* `Count_agg` reaches the fusion. Adding
> `aggregated_terms` as `finalize`'s first argument closes exactly that gap: the
> Evaluator computes the robust per-term aggregates, the Scorer fuses them.
> **qc §3's signature is updated to match** (applied alongside this doc), so the two
> docs publish one `finalize` contract.

`pipeline.measure(image, apply_post=False)` matches the per-image CLI path; whether
post-processing is applied for *scoring* is a Scorer concern, and `finalize` receives
the merged measurement frame.

---

## 4. The stability penalty

Each term is aggregated as `agg = median(termᵢ) − λ·IQR(termᵢ)`, **clamped at 0**,
with `λ = 0.5` by default (CLI `--stability-weight`). The median rewards a high
central level; subtracting `λ·IQR` rewards *flat* optima over *sharp* ones (D4) — a
candidate whose per-plate scores swing wildly is penalised even if its median is high.

Two properties:

- **It is a per-image-term operator.** It applies to the terms `score_image`
  emits — every term for a per-image Scorer, **Count only** for `QCScorer`. The
  batch panel is **not** subject to it: the panel carries its *own* robustness (the
  coverage-weighted, one-sided trimmed mean across strains, qc §4.4), so applying an
  image-level dispersion penalty there would double-count. This **refines master D4**,
  which phrases the penalty on the overall objective; D4/§4 are annotated to match (the
  batch panel's robustness is the Scorer's own reducer, §9).
- **Small-`n` guard.** IQR is unstable on few points; below `min_stability_n` plates
  the penalty is dropped (`λ` effectively 0) and the report says so — analogous to the
  QC checks' `min_replicates`.

`λ = 0.5` is a conservative default, **not** a validated value (master §14): it needs
empirical calibration on real plates. `λ = 0` recovers pure-median selection.

---

## 5. Metric-normalization contract

**The `Scorer` owns normalization; the `Evaluator` never re-normalizes.** A Scorer
emits terms already on the common `[0, 1]` higher-is-better scale, using **fixed /
threshold-anchored** maps (qc §4.2) — *never* min–max over the tested candidates,
which would make the argmax grid-dependent (the **Böck trap**, master §2). The
Evaluator only **validates** that emitted terms fall in the declared range (a cheap
guard) and **aggregates** them; it performs no transformation.

This reconciles master §4's "the Evaluator normalizes each metric": the Scorer still
*declares* `higher_is_better` + range/normalizer (for reporting and the Evaluator's
validation) but *applies* it itself. **Master §4's wording is updated to match** (applied
alongside this doc — normalization ownership moves to the Scorer). Keeping normalization
in the Scorer keeps the Evaluator metric-agnostic and keeps Böck-safety where the domain
knowledge lives.

---

## 6. Cross-validation and the replicate structure

Naïve leave-one-plate-out or random k-fold would scatter the replicate plates of a
strain across folds, leaving each strain under-powered (→ `NaN` → coverage collapse)
and silently gutting the batch panel. So CV is **group-aware**.

**Two distinct keys** (which may overload one column):

- **`--cv-group`** — the *keep-together* unit (a replicate batch / experiment). Its
  plates never split across folds, so the panel stays computable within each fold.
  **Auto-inferred** from the experimental/batch Metadata tag the QC panel already
  groups by (so the CV unit and the panel's replicate unit can't desync); explicit
  override allowed. **Label overloading is permitted** — one Metadata column may serve
  as `cv-group`, `stratify-by`, *and* the panel's replicate key at once.
- **`--stratify-by`** — the *balance-across* variable (agar / timepoint / density).
  Its distribution is balanced across folds **best-effort under the group constraint**
  (`StratifiedGroupKFold` is approximate — with very few groups the balance can be
  markedly skewed). Defaults to `None` (over-stratifying a tiny dataset creates empty
  folds), so the approximation is acceptable.

The default scheme is **leave-one-group-out** (or grouped-stratified k-fold), with
held-out reserving a whole group. CV engages for **small counts** (D4); larger sets
use a single stratified calibration split. Two degeneracies:

- **Single group** → degrade to a plate-level calibration/held-out split with the
  panel computed once (it cannot be cross-validated; the report says so).
- **Per-image-only Scorer** (Count-only `QCScorer`, `SupervisedScorer`) has no batch
  dependency, so finer **plate-level folds** are safe and give more folds —
  CV granularity is effectively **Scorer-aware**.

---

## 7. Multi-fidelity pruning (and incremental evaluation)

**Fidelity = the number of calibration plates scored on the per-image term.** The
Evaluator scores that term on a growing, **metadata-stratified** subset (so even the
first rung spans the strata) and calls `trial.report(value=running_agg, step=n_plates)`
(Optuna's positional order is `report(value, step)`) after each rung, so a candidate
with bad early Count is pruned before the full set is spent.

- **Rung ladder (conservative, stratified).** First rung = `max(~6-plate floor,
  ~⅓ of calibration)`; geometric ×3 growth; self-disables when the set is too small
  for ≥2 rungs. This guards the Mikkola failure mode (unreliable low-fidelity making
  pruning *worse* than none, master §2) — never prune on a few unrepresentative plates.
  Floor / fraction / factor are tunable (master §14).
- **The batch panel is never pruned.** It needs the whole replicate structure, so it
  runs once, at full fidelity, in `finalize`, only for **survivors** (qc §3). A
  panel-only Scorer (no per-image term) is unprunable — pruning disables cleanly.
- **Ownership split.** The Evaluator owns the fidelity axis and the `trial.report()`
  calls; the **pruner** itself (ASHA / Hyperband) is Optuna's and is specced in
  [`optuna-integration.md`](optuna-integration.md) (planned). Pruning is **opt-in**
  and absent in Phase 1.

### Pruning and CV compose independently

Pruning and cross-validation operate on **separate axes**, so they never interfere:

- Pruning runs a **single progressive pass** over a stratified, **Count-only** subset,
  *independent of the CV fold structure*. The reported `running_agg` is the running
  per-image `median − λ·IQR` of the Count term — a lower-fidelity estimate of the **same
  quantity** the final objective's `Count_agg` uses, so Optuna's `step`-keyed pruner
  always compares like with like.
- Rung subsets **need not be group-intact**: only the per-plate Count term is scored per
  rung, and Count has no replicate dependency, so splitting a replicate group across the
  fidelity ladder is harmless. (The batch panel — which *does* need intact groups — never
  runs on the ladder; it runs only for survivors in `finalize`.)
- **CV runs at full fidelity, for survivors only.** A candidate that survives the ladder
  is then evaluated across the group-aware folds (§6) for its final objective + stability;
  the pruner never sees the fold structure.

### Incremental evaluation & caching (what makes the ladder cheap)

The ladder only saves compute if rung 2 does **not** re-run rung 1's plates. The
Evaluator memoizes the **per-image `measure()` frame** keyed on
`(canonical-param-hash, image-id)`, where `param-hash` hashes the *fully-serialized*
pipeline JSON (the same canonical form as `best_pipeline.json`, so no param field is
silently missed → no stale reuse). Consequences:

- **Intra-trial** (primary): rung N runs only the *new* plates; overlapping CV folds
  reuse evaluations.
- **Cross-trial** (bonus): identical candidates dedup (Optuna can re-suggest; grid
  never does).
- **Lightweight & bounded:** caches measurement *frames*, not image arrays (master
  §8 disk policy); cleared after a trial except retained survivors.
- **SLURM:** disk-backed — and the existing CLI already writes per-image measurement
  parquets, so the cache reuses that store rather than inventing one.
- **Determinism precondition.** The key assumes `measure()` is a pure function of
  `(params, image)`. Under the project's fixed seeds this holds for deterministic ops;
  for a **stochastic** op the resolved per-image seed is folded into the key, so it
  simply never earns a spurious cross-trial hit (covered by a caching test, §14).

---

## 8. Overfitting / held-out guard

The guard is **adaptive**, because arrayed experiments are often data-poor (few
groups) and the held-out should be a whole group (so the panel runs on it with
replicates intact):

- **When data allows** — reserve a **pristine held-out group up front** (before any
  tuning; grouped + stratified). After optimization + screening pick the winner, run
  the **full Scorer (incl. panel)** on the held-out group and report the
  **generalization gap**. Flag loudly when held-out underperforms calibration beyond a
  **tunable margin** — a *relative* ratio with an *absolute* floor — as a **report
  flag only**; it never silently alters the winner.
- **When too data-poor to reserve** without starving calibration — skip the held-out
  and report the **group-aware CV estimate** as the generalisation signal, with an
  explicit "*no untouched held-out — CV is the best available estimate*" warning.

With exactly **one group**, any reserved held-out *plates* are *within-group* (same
experiment) — a weaker guarantee than a held-out *group*. The report labels it a
**within-group held-out**, and the "no untouched (cross-group) held-out" caveat applies:
"untouched" in §8/§12 specifically means a held-out *group*, not merely held-out plates.

The two roles stay distinct: **CV** (within calibration) informs selection/stability
*during* tuning; **held-out** is the *final, untouched* check on the chosen winner.
The CV estimate is optimistically biased for the winner (the folds informed
selection), which is exactly why the pristine held-out matters when affordable.

**Phasing:** Phase 1 may ship the **CV-only MVP** (a strict subset of the adaptive
logic) and add held-out reservation later.

---

## 9. The three aggregation scopes (synthesis)

Three different aggregations compose here; keeping them straight prevents
double-counting:

| Scope | Who | Over what | Operator |
|-------|-----|-----------|----------|
| **Within-pass, per-image** | Evaluator (§4) | per-image terms across calibration plates | `median − λ·IQR` (stability penalty) |
| **Scorer-internal, cross-strain** | Scorer `finalize` (qc §4.4–4.5) | batch-panel values across strains | coverage-weighted trimmed mean → geometric fusion |
| **Across resampling** | Evaluator (§6, §8) | folds / held-out | CV estimate + generalization-gap flag |

The per-image stability penalty hits only the per-image terms (Count); the panel's
robustness is internal to the Scorer; generalization is the CV/held-out layer. They
operate on disjoint inputs, so none double-counts another.

**Multi-objective seam (master §7).** The Evaluator returns the scalar
`level − λ·dispersion` *and* can surface the `(level, dispersion)` pair, so
`--multi-objective` can put stability on a Pareto axis (level vs dispersion) instead
of hard-scalarising via `λ`. For a two-phase Scorer this exposes the **per-image term's**
level/dispersion (e.g. Count's stability), *not* the fused `QCScore`'s — the fusion is
the Scorer's, and surfacing fused-score stability would need multiple `finalize` passes
(out of scope). One output shape serves both paths.

---

## 10. Failure & degenerate-trial handling

The principle: *measured-bad* (ran, scored poorly) **teaches** the surrogate;
*errored* (couldn't measure) must **not poison** it (master §12).

| Case | Treatment |
|------|-----------|
| Invalid params / pipeline won't construct | `TrialState.FAIL` immediately — no scoring |
| **All** images error | `TrialState.FAIL` — the candidate could not be measured at all (Optuna excludes it from the surrogate) |
| Per-image **exception** in an otherwise-working trial | that image → **worst term (0)** + logged (the existing per-image failure log); aggregate continues — honestly drags the median and closes the "crash a bad plate to dodge it" hole |
| Per-image **degenerate-but-ran** (0 objects, `NaN` metric) | scored by the metric naturally (Count error → ~0); a definite low score the surrogate learns from — **never** FAIL |

**Boundary notes.** "All images error" is the *post-loop* observation that every
per-image result came from the exception path (count of exception-terms == `n_images`) —
distinct from per-image `NaN` metrics. An op that **returns** a degenerate sentinel
(empty / all-`NaN` frame) *without raising* counts as "ran" → a low score, not FAIL; the
Scorer must tolerate an empty frame (a divide-by-zero there would itself raise and wrongly
flip the classification to "errored").

Failed trials always emit best-so-far and never halt the study (master §12).

---

## 11. Determinism & reproducibility

A single master seed lives in the study; the Evaluator derives independent sub-seeds
(`numpy.random.SeedSequence.spawn`) for the calibration/held-out split, the CV fold
assignment, and the rung-subset ordering — each reproducible yet independent. The
**resolved splits are persisted in the study**, so resume continues with the *same*
split rather than re-randomising (master §12: "resume reproduces an identical study").
Splits are a function of `(seed, dataset-identity)`: adding plates changes them, which
the report flags as "a different study."

---

## 12. Graceful degradation (the Evaluator's availability matrix)

The Evaluator degrades predictably rather than emitting garbage:

| Data available | Behaviour |
|----------------|-----------|
| `< min_stability_n` plates | drop the stability penalty (IQR unstable); report notes it |
| single `cv-group` | no leave-one-group-out → plate-level calibration/held-out split (a *within-group* held-out — weaker than cross-group, flagged as such, §8); panel computed once |
| `< 2` groups for held-out | CV-only generalisation estimate + "no untouched held-out" warning (§8) |
| too few plates for ≥2 rungs | pruning self-disables (§7) |
| Scorer can't run on the data at all | abstain with a clear error (defers to the Scorer's own availability matrix, qc §6) |

---

## 13. Output & reporting

The Evaluator returns `{objective, per_image_breakdown, per_fold, held_out,
diagnostics}`. These feed `trials.parquet` (per-trial params + scores + per-image
breakdown) and `tuning_report.html` (master §8): objective-vs-trial, calibration vs
held-out, per-image stability, and the **generalization-gap flag** surfaced loudly.
The `(level, dispersion)` pair is recorded per trial for the optional Pareto view.

---

## 14. Testing

- **Aggregation math** — `median − λ·IQR`, the 0-clamp, and the small-`n` guard;
  per-term placement (penalty applies to per-image terms, not the panel).
- **Loop composition** — the default `finalize` (weighted mean) and the `QCScorer`
  override both drive off `aggregated_terms`; `Count_agg` reaches the fusion.
- **Normalization guard** — the Evaluator validates the Scorer's declared range and
  does not transform; an out-of-range term is flagged.
- **Group-aware CV** — replicate plates of a strain stay in one fold (panel
  computable); single-group degrades to a plate split; `--cv-group` auto-infers from
  the QC grouping; label overloading works.
- **Pruning** — rung reporting via `trial.report()`; the panel runs only for
  survivors; incremental cache hit means rung N runs only the new plates; cache key
  changes with any param field (no stale reuse).
- **Failure taxonomy** — invalid-params → FAIL; all-errored → FAIL; per-image
  exception → 0-term with the aggregate continuing; degenerate-but-ran → low score, not
  FAIL.
- **Determinism** — same `(seed, dataset)` reproduces identical split/folds/rungs;
  resume reuses the persisted split.
- **Held-out** — gap flag fires when held-out underperforms beyond the margin;
  data-poor path falls back to the CV estimate + warning.

Fixed seeds throughout (project reproducibility requirement).

---

## 15. Resolved choices / open questions

**Resolved (recorded so they aren't re-litigated):**

1. **Eval↔Scorer contract** — uniform 3-step loop; `finalize(aggregated_terms,
   per_image_results, full_measurements)`; default = weighted mean, `QCScorer`
   overrides.
2. **Stability penalty** — per-term `median − λ·IQR`, clamp at 0, `λ = 0.5` default,
   small-`n` guard; per-image-term operator (panel excluded).
3. **Normalization** — Scorer owns it (fixed/threshold-anchored, Böck-safe); Evaluator
   validates, never transforms.
4. **CV** — group-aware (`--cv-group` auto-inferred + overloadable, `--stratify-by`
   default `None`); single-group + per-image-only degradations.
5. **Pruning** — fidelity = stratified progressive plate count on the per-image term;
   conservative ×3 ladder; panel never pruned; opt-in / Optuna-gated; incremental
   per-image-frame cache.
6. **Held-out** — adaptive (pristine group when affordable, CV-estimate + warning when
   not); Phase-1 CV-only MVP.
7. **Failure** — invalid/all-errored → FAIL; per-image exception → 0-term; degenerate →
   low score.
8. **Determinism** — `SeedSequence`-derived sub-seeds, splits persisted, dataset-identity
   dependent.

**Still open (planning / empirical):**

- Default **`λ`** (0.5) — needs empirical calibration on real plates (master §14).
- **First-rung** floor/fraction/factor and low-fidelity representativeness (master §14).
- The held-out **gap-margin** value (relative ratio + absolute floor).
- `min_stability_n` for the small-`n` IQR guard.
