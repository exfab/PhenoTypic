"""4.8 — NSGA-II auto-selection + the grid/random multi-objective guard.

A multi-objective scorer (a ``CompositeScorer(multi_objective=True)``, inferred
via the scorer — plan §0b) makes an Optuna ``build`` auto-select **NSGA-II** with
``directions=["minimize"] * n`` over the scorer's stable objective-name order.
Pairing a **grid/random** strategy with a multi-objective scorer is rejected at
``TuningSpec`` construction (and again at run validation) with a clear,
actionable error: grid/random cannot do Pareto search — use ``--strategy nsga2``
/ an Optuna strategy. The NSGA-II body is ``skipif`` when the ``tune`` extra is
absent; the guard tests run dependency-free.
"""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Budget,
    Categorical,
    CompositeScorer,
    Evaluator,
    GridConfig,
    Knob,
    OptunaConfig,
    QCScorer,
    RandomConfig,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._multi_objective import (
    is_multi_objective,
    objective_directions,
    objective_names,
)

_OPTUNA = importlib.util.find_spec("optuna") is not None


def _qc() -> QCScorer:
    layout = pd.DataFrame(
        {"Metadata_ImageName": ["p"] * 96, "Object_Label": list(range(96))}
    )
    return QCScorer(check=ExpectedVsDetectedCount(
        metadata=layout, groupby=["Metadata_ImageName"]))


def _multi_scorer() -> CompositeScorer:
    return CompositeScorer(scorers=[_qc(), _qc()], multi_objective=True)


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _spec(*, scorer, strategy) -> TuningSpec:
    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=_space(),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=strategy,
        budget=Budget(n_trials=4),
    )


# ---------------------------------------------------------------------------
# Multi-objective inference + stable objective-name order
# ---------------------------------------------------------------------------


def test_multi_objective_inferred_from_composite_scorer():
    assert is_multi_objective(_multi_scorer()) is True
    assert is_multi_objective(_qc()) is False
    # A scalar composite is single-objective.
    assert is_multi_objective(
        CompositeScorer(scorers=[_qc(), _qc()], multi_objective=False)
    ) is False


def test_objective_name_order_is_stable_and_matches_handles():
    scorer = _multi_scorer()
    assert objective_names(scorer) == ["s0", "s1"]
    # Directions mirror the axis count, all minimize (cost convention §4).
    assert objective_directions(scorer) == ["minimize", "minimize"]


def test_single_objective_scorer_yields_no_directions():
    assert objective_directions(_qc()) is None


# ---------------------------------------------------------------------------
# Grid/random + multi-objective → rejected (the guard)
# ---------------------------------------------------------------------------


def test_grid_plus_multi_objective_rejected_at_spec_construction():
    with pytest.raises(ValueError) as excinfo:
        _spec(scorer=_multi_scorer(), strategy=GridConfig())
    msg = str(excinfo.value).lower()
    assert "multi-objective" in msg
    assert "nsga2" in msg or "optuna" in msg


def test_random_plus_multi_objective_rejected_at_spec_construction():
    with pytest.raises(ValueError) as excinfo:
        _spec(scorer=_multi_scorer(), strategy=RandomConfig(n_trials=4))
    assert "multi-objective" in str(excinfo.value).lower()


def test_grid_plus_single_objective_is_accepted():
    # The guard must not fire for the common single-objective grid run.
    spec = _spec(scorer=_qc(), strategy=GridConfig())
    assert isinstance(spec.strategy, GridConfig)


def test_optuna_plus_multi_objective_is_accepted():
    # An Optuna strategy is the sanctioned multi-objective optimizer.
    spec = _spec(scorer=_multi_scorer(), strategy=OptunaConfig(sampler="nsga2", n_trials=4))
    assert isinstance(spec.strategy, OptunaConfig)


def test_cli_strategy_override_to_grid_rejected_at_run_validation():
    # A --strategy grid override bypasses the model validator (model_copy skips
    # validators), so run_tuning re-asserts the guard.
    from phenotypic.tune._tune_cli._run import run_tuning

    spec = _spec(
        scorer=_multi_scorer(), strategy=OptunaConfig(sampler="nsga2", n_trials=4)
    )
    with pytest.raises(ValueError) as excinfo:
        run_tuning(spec, [], "/tmp/pht_guard_run", strategy="grid")
    msg = str(excinfo.value).lower()
    assert "multi-objective" in msg
    assert "nsga2" in msg or "optuna" in msg


# ---------------------------------------------------------------------------
# NSGA-II auto-selection (needs optuna)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_optuna_build_auto_selects_nsga2_when_multi_objective():
    import optuna

    config = OptunaConfig(sampler="tpe", n_trials=4)  # tpe overridden by directions
    strategy = config.build(_space(), None, directions=["maximize", "maximize"])
    # The multi-objective study forces NSGA-II regardless of the configured sampler.
    assert isinstance(strategy._study.sampler, optuna.samplers.NSGAIISampler)
    assert strategy._multi_objective is True


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_optuna_build_single_objective_keeps_configured_sampler():
    import optuna

    config = OptunaConfig(sampler="tpe", n_trials=4)
    strategy = config.build(_space(), None, directions=None)
    assert isinstance(strategy._study.sampler, optuna.samplers.TPESampler)
    assert strategy._multi_objective is False
