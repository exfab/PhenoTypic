# phenotypic.tune

Hyperparameter tuning for image pipelines: an **ask-and-tell loop** where a
strategy proposes a parameter combo, the evaluator builds and scores a pipeline
on a calibration set, and the result is told back to the strategy.

## Layout

- `_engine.py` — `TuningEngine.optimize()`, the orchestrator loop.
- `_search_space/` — infer tunable knobs from pipeline fields (`_infer.py`),
  domain types (`_domains.py`).
- `_strategies/` — Optuna (TPE/CMA-ES/GP/NSGA-II), random, grid; ASHA pruning.
- `_evaluation/` — rung ladder + robust aggregate (`median − λ·IQR`), held-out
  split + generalization gap.
- `_scoring/` — the four objectives: supervised (Dice/IoU), reference-free
  (shape/contrast/SizeCV), QC (count), composite.
- `_study/` — study stores + Pareto front / knee point.
- `_screening.py` — parameter importance (fANOVA vs RF-permutation).
- `_tune_cli/` — `python -m phenotypic.tune` entry.

## Conventions

- **Higher-is-better everywhere**: every objective (and every axis of a
  multi-objective study) maximizes; the single `_MAXIMIZE` literal lives in
  `_strategies/_optuna_support.py`.
- **Closed value sets** (sampler kinds, etc.) are `Literal` aliases — see
  `_strategies/_config.py` (`SamplerKind`), never bare `str`.
- **Optuna is lazy-imported** — the boundary stays importable without it.

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
