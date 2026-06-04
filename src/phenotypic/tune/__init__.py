"""Parameter-tuning engine — public API (in progress).

This package is built up phase-by-phase; each phase appends its public symbols
to ``__all__`` below. Phase 1a ships the hand-authorable **search space**: the
discriminated-union domains plus the ``Knob`` / ``SearchSpace`` containers.
Phase 1b adds the serializable **strategy configs**. Phase 1c adds the
**scoring** objective (``Scorer`` / ``QCScorer``) and the candidate
**evaluation** layer (``Evaluator`` / ``EvaluationResult`` / ``build_pipeline``).
Phase 1d closes the MVP with the runnable **engine**: ``Budget`` / ``Trial`` /
``StudyStore`` (the journal), ``TuningSpec`` (the embedded-pipeline recipe),
``TuningEngine`` (the ask-and-tell loop + resume), ``compute_param_importance``,
and ``run_tuning`` (the ``python -m phenotypic.tune`` orchestration).

Hand-author a search space and inspect it:

    >>> from phenotypic.tune import SearchSpace, Knob, FloatRange, Categorical
    >>> space = SearchSpace(knobs=(
    ...     Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
    ...     Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ... ))
    >>> space.keys()
    ['0.sigma', '1.ignore_zeros']
    >>> space.domain("0.sigma").high
    8.0
"""
from __future__ import annotations

# --- Phase 1d: engine + spec + study + screening + CLI ------------------------
from ._engine import TuningEngine

# --- Phase 1c: scoring + evaluation -------------------------------------------
from ._evaluation import EvaluationResult, Evaluator, build_pipeline

# --- Phase 1a: search space (domains + Knob/SearchSpace) ----------------------
from ._scoring import QCScorer, Scorer
from ._screening import compute_param_importance
from ._search_space import (
    Categorical,
    Domain,
    Excluded,
    Fixed,
    FloatRange,
    InferredSearchSpace,
    IntRange,
    Knob,
    SearchSpace,
    TuneSpec,
)
from ._spec import Budget, TuningSpec

# --- Phase 1b: strategy configs (serializable; build live SearchStrategy) ------
from ._strategies import GridConfig, RandomConfig, StrategyConfig
from ._study_store import StudyStore, Trial
from ._tune_cli import run_tuning

__all__ = [
    # Phase 1a: search space
    "Categorical",
    "IntRange",
    "FloatRange",
    "Fixed",
    "Domain",
    "Knob",
    "SearchSpace",
    # Phase 1b: strategy configs
    "StrategyConfig",
    "GridConfig",
    "RandomConfig",
    # Phase 1c: scoring
    "Scorer",
    "QCScorer",
    # Phase 1c: evaluation
    "Evaluator",
    "EvaluationResult",
    "build_pipeline",
    # Phase 1d: engine + spec + study + screening + CLI
    "TuningEngine",
    "TuningSpec",
    "Budget",
    "StudyStore",
    "Trial",
    "compute_param_importance",
    "run_tuning",
    # Phase 3: search-space inference
    "TuneSpec",
    "InferredSearchSpace",
    "Excluded",
]
