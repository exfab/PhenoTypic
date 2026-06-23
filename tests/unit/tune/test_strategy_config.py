from __future__ import annotations

from phenotypic.tune import (
    Categorical,
    Knob,
    SearchSpace,
)
from phenotypic.tune.strategy import (
    GridConfig,
    RandomConfig,
)
from phenotypic.tune.strategy import GridStrategy, RandomStrategy


def _space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.a", domain=Categorical(choices=(1, 2))),
    ))


def test_grid_config_builds_grid_strategy():
    strat = GridConfig().build(_space(), store=None)
    assert isinstance(strat, GridStrategy)


def test_random_config_builds_random_strategy():
    cfg = RandomConfig(n_trials=7, seed=3)
    strat = cfg.build(_space(), store=None)
    assert isinstance(strat, RandomStrategy)
    assert strat._n == 7


def test_config_roundtrips_via_discriminator():
    from pydantic import TypeAdapter

    from phenotypic.tune.strategy._config import StrategyConfigUnion

    adapter = TypeAdapter(StrategyConfigUnion)
    cfg = RandomConfig(n_trials=5, seed=1)
    back = adapter.validate_json(adapter.dump_json(cfg))
    assert back == cfg
