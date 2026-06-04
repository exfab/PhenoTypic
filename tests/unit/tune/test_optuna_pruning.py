"""``OptunaPruningChannel`` + the ASHA pruner derived from the rung ladder (E1).

The channel is a passthrough to the live Optuna trial; the study's pruner is a
``SuccessiveHalvingPruner`` whose ``min_resource``/``reduction_factor`` come from
the Evaluator's rung config so the two cannot disagree. ``prune=False`` and the
explore round never prune; a multi-objective study gets a ``NoOpChannel``.
"""
from __future__ import annotations

import importlib.util

import pytest

from phenotypic.tune import (
    Categorical,
    EvaluationResult,
    Evaluator,
    FloatRange,
    Knob,
    SearchSpace,
)
from phenotypic.tune._strategies._pruning import NoOpChannel

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _strategy(space: SearchSpace, **kw):
    from phenotypic.tune._strategies._optuna import OptunaStrategy

    kw.setdefault("sampler", "tpe")
    kw.setdefault("n_trials", 50)
    kw.setdefault("prune", True)
    kw.setdefault("seed", 0)
    kw.setdefault("storage_url", None)
    kw.setdefault("store", None)
    return OptunaStrategy(space, **kw)


def _cat_space() -> SearchSpace:
    return SearchSpace(knobs=(Knob(key="c", domain=Categorical(choices=(1, 2))),))


def test_report_forwards_to_trial():
    space = _cat_space()
    strat = _strategy(space)
    _params, channel = strat.suggest()
    channel.report(0.5, 6)  # must not raise
    # The in-flight Trial does not expose intermediate_values; read the frozen
    # trial from the study's storage to confirm the report landed.
    frozen = strat._study.get_trials(deepcopy=False)[strat._stashed.number]
    assert frozen.intermediate_values.get(6) == 0.5


def test_should_prune_delegates_to_trial():
    space = _cat_space()
    strat = _strategy(space)
    _params, channel = strat.suggest()
    # First trial, nothing to compare against → should not prune.
    channel.report(1.0, 6)
    assert channel.should_prune() is False


def test_pruner_derived_from_rung_config():
    import optuna

    space = _cat_space()
    strat = _strategy(space, rung_floor=6, rung_factor=3)
    pruner = strat._study.pruner
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert pruner._min_resource == 6
    assert pruner._reduction_factor == 3


def test_pruner_matches_evaluator_default_ladder():
    import optuna

    ev = Evaluator()  # rung_floor=6, rung_factor=3
    space = _cat_space()
    strat = _strategy(space, rung_floor=ev.rung_floor, rung_factor=ev.rung_factor)
    pruner = strat._study.pruner
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert pruner._min_resource == ev.rung_floor
    assert pruner._reduction_factor == ev.rung_factor


def test_prune_disabled_never_prunes():
    space = _cat_space()
    strat = _strategy(space, prune=False)
    _params, channel = strat.suggest()
    # With pruning off the channel reports a value but never prunes.
    channel.report(0.0, 6)
    assert channel.should_prune() is False


def test_explore_round_channel_never_prunes():
    # The explore round passes explore=True to suggest() → a NoOpChannel even
    # when prune is enabled (keeps fANOVA's importance sample unbiased).
    space = _cat_space()
    strat = _strategy(space, prune=True)
    _params, channel = strat.suggest(explore=True)
    assert isinstance(channel, NoOpChannel)


def test_multi_objective_gets_noop_channel():
    # Optuna pruners are single-objective → a multi-objective study disables
    # pruning by handing back a NoOpChannel.
    space = _cat_space()
    strat = _strategy(space, prune=True, directions=["maximize", "maximize"])
    _params, channel = strat.suggest()
    assert isinstance(channel, NoOpChannel)


def test_bad_candidate_pruned_end_to_end(tmp_path):
    """A deliberately-bad candidate reporting a low rung-1 value is PRUNED.

    Seed the study with several good trials at the first rung, then a clearly
    inferior one; the ASHA pruner should mark the inferior trial PRUNED at the
    first rung. Exercised through the channel passthrough (no engine needed).
    """
    import optuna

    space = SearchSpace(knobs=(
        Knob(key="f", domain=FloatRange(low=0.0, high=1.0)),
    ))
    strat = _strategy(space, sampler="tpe", n_trials=50, rung_floor=1, rung_factor=2)

    # A handful of strong trials reporting high values at the first rung. They
    # must check should_prune() (as the real Evaluator does between rungs) so
    # ASHA registers their values into the rung's competing pool.
    for _ in range(6):
        params, channel = strat.suggest()
        channel.report(1.0, 1)
        channel.should_prune()
        strat.register_result(params, EvaluationResult(
            score=1.0, terms={"t": 1.0}, n_images=2,
        ))

    # A clearly-inferior trial: reports a low value at the first rung.
    params, channel = strat.suggest()
    channel.report(0.0, 1)
    pruned = channel.should_prune()
    assert pruned is True
    strat.register_result(
        params,
        EvaluationResult(score=0.0, terms={"t": 0.0}, n_images=1, pruned=True),
        pruned=True,
    )
    states = [t.state for t in strat._study.get_trials(deepcopy=False)]
    assert optuna.trial.TrialState.PRUNED in states
