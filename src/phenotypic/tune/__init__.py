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
Phase 3 adds **search-space inference**: the ``TuneSpec`` per-field marker, the
reviewable ``InferredSearchSpace`` / ``Excluded`` proposal, and
``infer_search_space`` — which mines a pipeline's pydantic fields into a
generous, flagged-for-review candidate space.

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

Infer a candidate space from a configured pipeline:

    >>> from phenotypic import ImagePipeline
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import OtsuDetector
    >>> from phenotypic.tune import infer_search_space
    >>> pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    >>> proposal = infer_search_space(pipe)
    >>> next(k.domain.high for k in proposal.knobs if k.key == "0.sigma")
    5.0
    >>> proposal.needs_review  # every knob fully resolved (sigma via TuneSpec)
    False
"""
from __future__ import annotations

# --- Phase 1d: engine + spec + study + screening + CLI ------------------------
from ._engine import TuningEngine

# --- Phase 1c: scoring + evaluation -------------------------------------------
from ._evaluation import (
    EvaluationResult,
    Evaluator,
    HeldOutConfig,
    build_pipeline,
    infer_group_key,
)

# --- Phase 1a: search space (domains + Knob/SearchSpace) ----------------------
# --- Phase 4 chunk B: supervised + composite scorers + GT loader --------------
from ._scoring import (
    CompositeScorer,
    GroundTruthMasks,
    QCScorer,
    ReferenceFreeScorer,
    Scorer,
    SupervisedScorer,
)
from ._screening import (
    ImportanceMethod,
    ImportanceReport,
    compute_param_importance,
    compute_param_importance_report,
)
from ._screening_freeze import (
    ScreeningConfig,
    ScreeningController,
    ScreeningResult,
    build_reduced_space,
    count_free_params,
    freeze_value,
    screening_warmup_floor,
    select_params_to_freeze,
)
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
    infer_search_space,
)
from ._spec import Budget, TuningSpec

# --- Phase 1b: strategy configs (serializable; build live SearchStrategy) ------
# --- Phase 2 (chunk 2): OptunaConfig (registry round-trip; builds OptunaStrategy)
from ._strategies import (
    GridConfig,
    OptunaConfig,
    RandomConfig,
    SamplerKind,
    StrategyConfig,
)

# --- Phase 2 (chunk 1): StudyStore Protocol + concrete journal -----------------
# ``StudyStore`` remains the public name (back-compat alias for the concrete
# ``JournalStudyStore``); the Protocol of the same name lives in
# ``_study/_protocol.py`` and types the engine's store.
from ._study_store import JournalStudyStore, StudyStore, Trial
from ._tune_cli import run_auto_space, run_tuning

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
    # Phase 2 (chunk 2): Optuna backend config
    "OptunaConfig",
    "SamplerKind",
    # Phase 1c: scoring
    "Scorer",
    "QCScorer",
    # Phase 3 (chunk 3): reference-free objective + meta-validation gate
    "ReferenceFreeScorer",
    # Phase 4 chunk B: supervised + composite scorers + GT loader
    "GroundTruthMasks",
    "SupervisedScorer",
    "CompositeScorer",
    # Phase 1c: evaluation
    "Evaluator",
    "EvaluationResult",
    "build_pipeline",
    # Phase 4.5 part 1: robust-eval held-out config + group-key inference
    "HeldOutConfig",
    "infer_group_key",
    # Phase 1d: engine + spec + study + screening + CLI
    "TuningEngine",
    "TuningSpec",
    "Budget",
    "StudyStore",
    "JournalStudyStore",
    "Trial",
    "compute_param_importance",
    "compute_param_importance_report",
    "ImportanceReport",
    "ImportanceMethod",
    # Phase 2 (chunk 3): two-round screening freeze
    "ScreeningController",
    "ScreeningConfig",
    "ScreeningResult",
    "count_free_params",
    "screening_warmup_floor",
    "select_params_to_freeze",
    "freeze_value",
    "build_reduced_space",
    "run_tuning",
    # Phase 3: search-space inference
    "TuneSpec",
    "InferredSearchSpace",
    "Excluded",
    "infer_search_space",
    "run_auto_space",
]
