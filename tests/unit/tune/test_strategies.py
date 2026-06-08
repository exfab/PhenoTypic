from __future__ import annotations

from phenotypic.tune import Categorical, FloatRange, Knob, SearchSpace
from phenotypic.tune._strategies import (
    GridStrategy,
    NoOpChannel,
    RandomStrategy,
    SearchStrategy,
)


def _conditional_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="0.sigma", domain=FloatRange(low=0.5, high=5.0),
             conditional_on=(("0.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def _grid_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.__enabled__", domain=Categorical(choices=(True, False))),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def test_strategies_satisfy_protocol():
    assert isinstance(GridStrategy(_grid_space()), SearchStrategy)
    assert isinstance(RandomStrategy(_conditional_space(), n_trials=3, seed=0),
                      SearchStrategy)


def test_grid_exhausts_after_enumeration():
    strat = GridStrategy(_grid_space())
    seen = []
    while not strat.is_exhausted():
        params, channel = strat.suggest()
        assert isinstance(channel, NoOpChannel)
        strat.register_result(params, result=None)  # grid ignores results
        seen.append(params)
    assert len(seen) == 6  # the conditional Cartesian product


def test_random_respects_conditionals_and_seed():
    a = RandomStrategy(_conditional_space(), n_trials=20, seed=42)
    seq_a = []
    while not a.is_exhausted():
        p, _ = a.suggest()
        a.register_result(p, result=None)
        seq_a.append(p)
        # sigma present iff blur enabled
        assert ("0.sigma" in p) == (p["0.__enabled__"] is True)
    assert len(seq_a) == 20

    b = RandomStrategy(_conditional_space(), n_trials=20, seed=42)
    seq_b = []
    while not b.is_exhausted():
        p, _ = b.suggest()
        b.register_result(p, result=None)
        seq_b.append(p)
    assert seq_a == seq_b  # seeded determinism


def test_random_samples_stepped_float_from_quantized_values():
    space = SearchSpace(knobs=(
        Knob(key="0.f", domain=FloatRange(low=0.0, high=1.0, step=0.25)),
    ))
    strat = RandomStrategy(space, n_trials=30, seed=8)
    allowed = set(FloatRange(low=0.0, high=1.0, step=0.25).values())
    while not strat.is_exhausted():
        params, _ = strat.suggest()
        assert params["0.f"] in allowed


def test_grid_rejects_floatrange():
    import pytest
    with pytest.raises(ValueError):
        GridStrategy(_conditional_space())  # has a FloatRange knob
