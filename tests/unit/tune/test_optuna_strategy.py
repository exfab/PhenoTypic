"""``OptunaStrategy`` — ask-and-tell over an Optuna study (D3).

All tests require the live ``tune`` extra (``skipif`` when absent). They cover
domain materialization, define-by-run conditional children, the
``suggest_int(step≠1, log=True)`` guard, ``register_result``→``study.tell``
state mapping, the ``(params, channel)`` suggest contract, ``is_exhausted``
counting, sampler selection, and ``SearchStrategy`` protocol conformance.
"""
from __future__ import annotations

import importlib.util

import pytest

from phenotypic.tune import (
    Categorical,
    EvaluationResult,
    Fixed,
    FloatRange,
    IntRange,
    Knob,
    OptunaConfig,
    SearchSpace,
)
from phenotypic.tune._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV

_OPTUNA = importlib.util.find_spec("optuna") is not None
pytestmark = pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")


def _strategy(space: SearchSpace, **kw):
    from phenotypic.tune._strategies._optuna import OptunaStrategy

    kw.setdefault("sampler", "tpe")
    kw.setdefault("n_trials", 5)
    kw.setdefault("prune", False)
    kw.setdefault("seed", 0)
    kw.setdefault("storage_url", None)
    kw.setdefault("store", None)
    return OptunaStrategy(space, **kw)


def _result(score: float = 1.0, *, failed: bool = False, pruned: bool = False):
    return EvaluationResult(
        score=score, terms={"t": score}, n_images=1, failed=failed, pruned=pruned
    )


# ---------------------------------------------------------------------------
# Domain materialization
# ---------------------------------------------------------------------------


def test_categorical_materializes_choice():
    space = SearchSpace(knobs=(
        Knob(key="0.c", domain=Categorical(choices=("a", "b", "c"))),
    ))
    strat = _strategy(space)
    params, _channel = strat.suggest()
    assert params["0.c"] in ("a", "b", "c")


def test_int_range_materializes_with_step_and_log():
    space = SearchSpace(knobs=(
        Knob(key="0.i", domain=IntRange(low=2, high=64, step=2)),
        Knob(key="0.j", domain=IntRange(low=1, high=1000, log=True)),
    ))
    strat = _strategy(space)
    params, _ = strat.suggest()
    assert 2 <= params["0.i"] <= 64 and params["0.i"] % 2 == 0
    assert 1 <= params["0.j"] <= 1000


def test_float_range_materializes_log():
    space = SearchSpace(knobs=(
        Knob(key="0.f", domain=FloatRange(low=1e-4, high=1.0, log=True)),
    ))
    strat = _strategy(space)
    params, _ = strat.suggest()
    assert 1e-4 <= params["0.f"] <= 1.0


def test_fixed_injected_constant_not_a_trial_dim():
    import optuna

    space = SearchSpace(knobs=(
        Knob(key="0.fixed", domain=Fixed(value=42)),
        Knob(key="0.c", domain=Categorical(choices=("x", "y"))),
    ))
    strat = _strategy(space)
    params, _ = strat.suggest()
    assert params["0.fixed"] == 42
    # The Fixed knob must NOT be a sampled trial dimension.
    trial = strat._stashed
    assert isinstance(trial, optuna.trial.Trial)
    assert "0.fixed" not in trial.params
    assert "0.c" in trial.params


# ---------------------------------------------------------------------------
# Define-by-run conditional children
# ---------------------------------------------------------------------------


def test_conditional_child_absent_when_parent_inactive():
    # Parent presence knob first, then a child gated on it being True.
    space = SearchSpace(knobs=(
        Knob(key="0.p", domain=Categorical(choices=(False,))),
        Knob(
            key="0.child",
            domain=FloatRange(low=0.0, high=1.0),
            conditional_on=(("0.p", True),),
        ),
    ))
    strat = _strategy(space)
    params, _ = strat.suggest()
    assert params["0.p"] is False
    assert "0.child" not in params


def test_conditional_child_present_when_parent_active():
    space = SearchSpace(knobs=(
        Knob(key="0.p", domain=Categorical(choices=(True,))),
        Knob(
            key="0.child",
            domain=FloatRange(low=0.0, high=1.0),
            conditional_on=(("0.p", True),),
        ),
    ))
    strat = _strategy(space)
    params, _ = strat.suggest()
    assert params["0.p"] is True
    assert "0.child" in params


# ---------------------------------------------------------------------------
# step≠1 ∧ log guard (Optuna forbids it)
# ---------------------------------------------------------------------------


def test_int_step_and_log_guard_normalizes(caplog):
    import logging

    space = SearchSpace(knobs=(
        Knob(key="0.i", domain=IntRange(low=1, high=128, step=4, log=True)),
    ))
    strat = _strategy(space)
    with caplog.at_level(logging.WARNING):
        params, _ = strat.suggest()
    # Must not raise; a value within range is produced.
    assert 1 <= params["0.i"] <= 128
    # A logged note about the normalization.
    assert any("log" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# register_result → study.tell state mapping
# ---------------------------------------------------------------------------


def test_register_result_tells_complete():
    import optuna

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space)
    params, _ = strat.suggest()
    strat.register_result(params, _result(0.75))
    trials = strat._study.get_trials(deepcopy=False)
    assert trials[0].state == optuna.trial.TrialState.COMPLETE
    assert trials[0].value == 0.75


def test_register_result_pruned_tells_pruned():
    import optuna

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space)
    params, _ = strat.suggest()
    strat.register_result(params, _result(0.2, pruned=True), pruned=True)
    trials = strat._study.get_trials(deepcopy=False)
    assert trials[0].state == optuna.trial.TrialState.PRUNED


def test_register_result_failed_tells_fail():
    import optuna

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space)
    params, _ = strat.suggest()
    strat.register_result(params, _result(0.0, failed=True))
    trials = strat._study.get_trials(deepcopy=False)
    assert trials[0].state == optuna.trial.TrialState.FAIL


# ---------------------------------------------------------------------------
# suggest() returns (params, OptunaPruningChannel)
# ---------------------------------------------------------------------------


def test_suggest_returns_optuna_channel_when_pruning():
    from phenotypic.tune._strategies._optuna import OptunaPruningChannel

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, prune=True)
    _params, channel = strat.suggest()
    assert isinstance(channel, OptunaPruningChannel)


# ---------------------------------------------------------------------------
# is_exhausted counts completed + pruned vs n_trials
# ---------------------------------------------------------------------------


def test_is_exhausted_counts_completed_and_pruned():
    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, n_trials=2)
    assert not strat.is_exhausted()
    p1, _ = strat.suggest()
    strat.register_result(p1, _result(1.0))
    assert not strat.is_exhausted()
    p2, _ = strat.suggest()
    strat.register_result(p2, _result(0.5, pruned=True), pruned=True)
    # completed (1) + pruned (1) == n_trials (2) → exhausted.
    assert strat.is_exhausted()


def test_failed_trials_do_not_count_toward_budget():
    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, n_trials=1)
    p1, _ = strat.suggest()
    strat.register_result(p1, _result(0.0, failed=True))
    # A failed trial does not consume the budget (§8).
    assert not strat.is_exhausted()


# ---------------------------------------------------------------------------
# Sampler selection
# ---------------------------------------------------------------------------


def test_tpe_default_sampler():
    import optuna

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, sampler="tpe")
    assert isinstance(strat._study.sampler, optuna.samplers.TPESampler)


def test_cmaes_explicit_sampler():
    import optuna

    space = SearchSpace(knobs=(
        Knob(key="0.f", domain=FloatRange(low=0.0, high=1.0)),
    ))
    strat = _strategy(space, sampler="cmaes")
    assert isinstance(strat._study.sampler, optuna.samplers.CmaEsSampler)


def test_multi_objective_selects_nsga2():
    import optuna

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, directions=["maximize", "maximize"])
    assert isinstance(strat._study.sampler, optuna.samplers.NSGAIISampler)
    assert len(strat._study.directions) == 2


# ---------------------------------------------------------------------------
# Protocol conformance by calling
# ---------------------------------------------------------------------------


def test_conforms_to_search_strategy_by_calling():
    from phenotypic.tune._strategies._protocol import SearchStrategy

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    strat = _strategy(space, n_trials=1)
    assert isinstance(strat, SearchStrategy)
    params, channel = strat.suggest()
    assert hasattr(channel, "report") and hasattr(channel, "should_prune")
    strat.register_result(params, _result(1.0))
    assert isinstance(strat.is_exhausted(), bool)


# ---------------------------------------------------------------------------
# OptunaConfig.build (moved from test_optuna_config.py — needs OptunaStrategy)
# ---------------------------------------------------------------------------


def test_config_build_constructs_strategy():
    from phenotypic.tune._strategies._optuna import OptunaStrategy

    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    cfg = OptunaConfig(n_trials=5)
    strat = cfg.build(space, store=None)
    assert isinstance(strat, OptunaStrategy)


def test_config_build_resolves_storage_url_from_env(monkeypatch, tmp_path):
    db = tmp_path / "env.db"
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, f"sqlite:///{db}")
    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    cfg = OptunaConfig(n_trials=3, storage_url=None)
    strat = cfg.build(space, store=None)
    assert strat._storage_url == f"sqlite:///{db}"


def test_config_build_explicit_storage_url_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv(PHENOTYPIC_TUNE_STORAGE_URL_ENV, "sqlite:///env.db")
    db = tmp_path / "explicit.db"
    space = SearchSpace(knobs=(Knob(key="0.c", domain=Categorical(choices=(1, 2))),))
    cfg = OptunaConfig(n_trials=3, storage_url=f"sqlite:///{db}")
    strat = cfg.build(space, store=None)
    assert strat._storage_url == f"sqlite:///{db}"
