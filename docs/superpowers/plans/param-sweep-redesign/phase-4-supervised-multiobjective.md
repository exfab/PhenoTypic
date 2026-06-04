# Tune Engine — Phase 4: Supervised + Multi-Objective (Structured Outline)

> **Status: OUTLINE.** A structured task map, not a full TDD plan. Expand into bite-sized TDD
> tasks before implementing.

**Goal:** Use ground truth when it exists, and optimize for more than one objective at once.
Add a `SupervisedScorer` (GT-based segmentation/detection metrics), a `CompositeScorer`
(weighted/blended terms), and **multi-objective** optimization with Pareto reporting
(`--multi-objective`, the `pareto/` deliverables, NSGA-II from Phase 2).

**Maps to:** `supervised-scorers.md` (whole doc); `qc-objective-mapping.md` §multi-objective;
master §7 (multi-objective); `reference-free-segmentation-metrics.md` §E (supervised doubles as
the meta-validation reference signal).

**Depends on:** Phase 1 (`Scorer`/`Evaluator`/`Trial`/`StudyStore`), Phase 2 (NSGA-II sampler).
Phase 3 optional.

---

## Scope — what it adds / changes

| Phase-1 piece | Phase-4 change |
|---------------|----------------|
| `Scorer.score_image → dict[str,float]` | `SupervisedScorer` + `CompositeScorer` subclasses; `score_image` still returns named terms |
| `Evaluator` single-objective `finalize → float` | optional **vector** objective (multi-objective): `finalize` may return a tuple; `EvaluationResult.score` widens to `float | tuple[float,...]` |
| `Trial.score: float` | `Trial` gains a multi-objective vector + Pareto membership; `StudyStore.best()` → `pareto_front()` |
| best-pipeline pick | Pareto front + knee-point selection; `pareto/` deliverables |

## Key components (interfaces — bodies TBD)

- **`SupervisedScorer`** (`_scoring/_supervised.py`) — needs a **GT source** (annotation masks /
  layout). Modality-tiered composite of region/overlap (Dice/IoU/Tversky), boundary
  (HD95/NSD), instance/detection (PQ/AJI/SEG/AP), counting/localization (MAE/CCC/grid-cell F1),
  partition (ARI/VI) — supervised-scorers §families. The **matching-free ARI/VI guard** for
  touching colonies. `availability()` → GT present.
- **`CompositeScorer`** (`_scoring/_composite.py`) — wraps N scorers (`polymorphic_field`
  list), merges their terms, `finalize` weights/blends (hybrid geometric fusion, qc §);
  optionally emits a **vector** for multi-objective.
- **multi-objective plumbing** — `EvaluationResult`/`Trial` carry the objective vector;
  `Evaluator` aggregates per-objective; the engine selects NSGA-II (Phase 2) when the scorer is
  multi-objective; `StudyStore.pareto_front()` + knee-point.
- **GT loader** — resolve annotation masks/labels per calibration image (path-configured, like
  the `QCScorer`'s metadata, so it round-trips).

## Task breakdown (high-level)

1. **GT source abstraction** — a path-configured, round-trippable annotation provider (mirror
   the `ExpectedVsDetectedCount` metadata-path contract).
2. **`SupervisedScorer`** — start with one robust family (e.g. region+counting), add the
   instance/boundary families; the matching-free ARI/VI guard.
3. **`CompositeScorer`** — term-merge + weighted/geometric `finalize`; serialization (nested
   scorers via `polymorphic_field(base=Scorer)`).
4. **Multi-objective `Evaluator`/`Trial`/`EvaluationResult`** — vector scores; back-compat with
   the single-objective path (a 1-vector reduces to today's float).
5. **Pareto in `StudyStore`** — `pareto_front()`, knee-point, `pareto/` deliverables (best per
   objective + the front).
6. **CLI `--multi-objective`** — selects NSGA-II, writes `pareto/`, the report surfaces the
   front.

## Deferred / out of scope
- The Dash Pareto-front *visual* + curation → Phase 5 (this phase produces the `pareto/` data).
- Active-learning / human-in-the-loop GT acquisition → out of scope.

## Review findings (address at full-planning)

Opus plan-review flagged these — fix when expanding to TDD:

- **Multi-objective return is `dict[str, float]`, NOT `tuple`.** Master §7 / engine-arch §5 specify a dict-returning `finalize`; the named keys carry which objective is which (for `pareto/`, report axes, and Optuna `directions=`). Also: **Phase-1c narrowed** `Scorer.finalize`/`EvaluationResult.score`/`Trial.score` to `float`, so Phase 4 *re-widens* three `frozen=True` types + an ABC method to `float | dict[str, float]`. State that explicitly.
- **Blast radius is bigger than the 4 listed seams.** `StudyStore.to_dataframe`/`to_parquet`/`from_parquet` persist `score` as a **scalar parquet column** (a vector needs a JSON `objectives_json` column); `compute_param_importance` does `[t.score for t]` (breaks on vectors); `StudyStore.best()`/engine `optimize`/`run_tuning` return-types change. Back-compat rule: single-objective keeps the scalar column + `best()`; the vector path **adds** parallel structures, never mutating the scalar ones.
- **`SupervisedScorer` must REUSE the existing `analysis/` count/grid/ICC checks** (the same ones `QCScorer` wraps), not re-implement them — a stated design rule (supervised-scorers §5). It *adds* only the GT-requiring families.
- **Modality-tiered graceful degradation is a deliverable, not a footnote:** `availability()` reports *which tier* is runnable given the GT modality present (mirror `QCScorer`'s availability matrix); the composite assembles only that tier's terms.
- Name the **matching strategy** (per-grid-cell default on `GridImage`; IoU>0.5 unique-match mask fallback); pin τ/symmetrization/ICC-form/aggregation. **Metric caveats:** AJI+/Mahalanobis/Boundary-F θ are unverified — implement only the verified subset; VI is unbounded & lower-is-better → negate+normalize; don't put both Dice and IoU in the panel.
- **Coupling:** Phase 4's `SupervisedScorer` is the reference signal Phase 3's meta-validation gate correlates against — "Phase 3 optional" hides that Phase 3's gate runs degraded until these metrics exist.

## Open questions for the full plan
- Single-objective stays the default; how does the spec/CLI signal multi-objective — a flag, or
  inferred from a `CompositeScorer` returning a vector?
- Which supervised metric family is the v1 default (modality-tiered composite vs. a single
  trustworthy metric)? supervised-scorers §"modality-tiered" recommends tiering — confirm the
  v1 subset against the project's real annotation availability.
