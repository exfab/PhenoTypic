# phenotypic.tune

Hyperparameter tuning for image pipelines: an **ask-and-tell loop** where a
strategy proposes a parameter combo, the evaluator builds and scores a pipeline
on a calibration set, and the result is told back to the strategy.

## Layout

- `_engine.py` — `TuningEngine.optimize()`, the orchestrator loop.
- `_search_space/` — infer tunable knobs from pipeline fields (`_infer.py`),
  domain types (`_domains.py`).
- `strategy/` — **public** namespace (`phenotypic.tune.strategy`): Optuna
  (TPE/CMA-ES/GP/NSGA-II), random, grid; ASHA pruning. Configs:
  `StrategyConfig`/`GridConfig`/`RandomConfig`/`OptunaConfig`/`SamplerKind`.
- `_evaluation/` — rung ladder + robust aggregate (`clamp01(median + λ·IQR)`),
  held-out split + generalization gap.
- `score/` — **public** namespace (`phenotypic.tune.score`): the four objectives
  — supervised (Dice/IoU), reference-free (shape/contrast/SizeCV), QC (count),
  composite. Classes: `Scorer`/`QCScorer`/`SupervisedScorer`/
  `ReferenceFreeScorer`/`CompositeScorer`/`GroundTruthMasks`/`CompositeBlend`.
- `_study/` — study stores + Pareto front / knee point.
- `_screening.py` — parameter importance (fANOVA vs RF-permutation).
- `_tune_cli/` — `python -m phenotypic.tune` entry (`run` + `auto-space`
  subcommands; `auto-space` infers a reviewable search space, file-only).

## Conventions

- **Cost everywhere (lower-is-better, minimize)**: every per-term and
  per-child value the optimizer sees is a bounded **cost** `∈ [0,1]` (`0` =
  perfect, `1` = worst); every objective (and every axis of a multi-objective
  study) **minimizes**. The single `_MINIMIZE` literal lives in
  `strategy/_optuna_support.py`. The word in code/docs/fields is **"cost"**
  (never "score" for the new quantity, never "badness"). The QC flag
  `_HIGHER_IS_BAD` is unchanged: `True` ⟺ the metric is a loss ⟺
  `Sense.LOWER_BETTER`.
- **Composite = augmented Tchebycheff**: the single-objective `CompositeScorer`
  blends per-child cost with `Tᵨ(b) = maxᵢ wᵢ(bᵢ + ε) + ρ·Σᵢ wᵢ·bᵢ`
  (utopia `z*ᵢ = −ε`, `_UTOPIA_EPS = 1e-3`, `rho = 0.05`), normalized to
  `[0,1]` over the **study-global active set**. `blend="weighted_mean"` is the
  compensatory opt-out; geometric-mean-of-cost is **never** exposed (one perfect
  axis would zero the product). `weights` are now blend-dependent (§6.5).
- **Study persistence is a hard cutover**: `_STUDY_NAME = "tune_cost_v1"` (was
  `"tune"`). Pre-cutover `"tune"` (maximize) studies are **never reopened** and
  cannot be resumed under the cost convention — re-run them. Cross-study
  comparison with pre-cutover runs is invalid. (Optuna `load_if_exists=True`
  silently keeps a mismatched direction — verified 4.9.0 — so correctness rests
  on the name bump, not a runtime guard.)
- **Closed value sets** (sampler kinds, etc.) are `Literal` aliases — see
  `strategy/_config.py` (`SamplerKind`), never bare `str`.
- **Optuna is lazy-imported** — the boundary stays importable without it.
- **Scorers & strategies are public sub-namespaces (hard cutover)**: import
  objectives from `phenotypic.tune.score` and optimizer configs from
  `phenotypic.tune.strategy` — they are **no longer** re-exported at the
  `phenotypic.tune` top level. Everything else (`TuningSpec`, `Evaluator`,
  `SearchSpace`/`Knob`, `TuningEngine`, study/screening/CLI symbols) stays at
  the top level. The inner `_*` modules of both packages remain private.

## Adding a Scorer

The full authoring contract is canonical in the `Scorer` base-class docstring
(`score/_scorer.py`) and the contributor guide
(`docs/source/contrib_guide/contributing.rst`) — read those rather than
duplicating them here. The non-obvious parts to keep in mind:

- Return **natural** per-term values from `_score_terms` and declare `_TERM_SENSE`;
  the base `score_image` orients each term via `to_cost` (`score/_orient.py`), so do
  **not** flip/normalize by hand. Override `_term_anchor` only for an unbounded term.
- **Never** add scalarization parameters — `ε`, `ρ`, normalization, and default
  weights are framework-derived. `CompositeScorer` overrides `score_image` (it
  merges already-cost children), never `_score_terms`.
- **Register or it's invisible**: re-export from `tune/score/__init__.py` and the
  class registry. `_find_class_in_phenotypic`
  (`_core/_pipeline_parts/_serializable_pipeline.py`) resolves serialized classes
  by bare name across `phenotypic.tune.score` / `phenotypic.tune.strategy`, so a new
  scorer just needs to be importable from that package.

## Math & logic doc — keep it in sync

The authoritative explainer for the tuning math (sampler EI, rung ladder,
robust aggregate, scorer formulas, importance, Pareto, generalization gap) is
[`docs/superpowers/explain/tune-with-optuna.md`](../../../docs/superpowers/explain/tune-with-optuna.md),
with a Claude-Desktop graph companion at `tune-with-optuna.graph.md`.

**When you change any math or control-flow logic in this module — a scoring
formula, the aggregation/pruning rule, a search-space heuristic, sampler
selection, the Pareto/knee or generalization-gap math, or the trial loop — update
that explainer (and its data-flow diagram) in the same change so the doc and its
`file:line` references stay accurate.**
