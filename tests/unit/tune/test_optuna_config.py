"""``OptunaConfig`` — the serializable Phase-2 strategy config.

Construction + dump must stay Optuna-free (the lazy-import boundary): only
``build`` resolves the dependency. The Optuna-requiring assertions are
``skipif``'d when the extra is absent.
"""
from __future__ import annotations

import sys
from typing import get_args

import pytest

from phenotypic.tune import OptunaConfig


def test_optuna_samplers_roster_matches_sampler_kind_literal():
    # The runtime ``OPTUNA_SAMPLERS`` frozenset and the ``STRATEGY_CHOICES`` CLI
    # tuple are derived from ``SamplerKind`` — assert they cannot drift apart.
    from phenotypic.tune._strategies._config import (
        OPTUNA_SAMPLERS,
        STRATEGY_CHOICES,
        SamplerKind,
    )

    assert set(get_args(SamplerKind)) == OPTUNA_SAMPLERS
    assert STRATEGY_CHOICES == ("grid", "random", *get_args(SamplerKind))


def test_kind_and_defaults():
    cfg = OptunaConfig(n_trials=10)
    assert cfg.kind == "optuna"
    assert cfg.sampler == "tpe"
    assert cfg.n_trials == 10
    assert cfg.prune is False
    assert cfg.seed == 0
    assert cfg.storage_url is None


def test_sampler_closed_literal_rejects_junk():
    from pydantic import ValidationError

    for ok in ("tpe", "cmaes", "gp", "nsga2"):
        assert OptunaConfig(n_trials=1, sampler=ok).sampler == ok
    with pytest.raises(ValidationError):
        OptunaConfig(n_trials=1, sampler="not-a-sampler")


def test_construction_and_dump_do_not_import_optuna():
    # Building the config and serializing it must NOT pull optuna in — that only
    # happens inside build(). Snapshot sys.modules around the calls.
    sys.modules.pop("optuna", None)
    cfg = OptunaConfig(n_trials=4, sampler="cmaes", prune=True, seed=7)
    _ = cfg.model_dump_json()
    assert "optuna" not in sys.modules, "OptunaConfig must not import optuna"


def test_round_trips_via_registry():
    # The polymorphic class registry round-trips the concrete OptunaConfig
    # through TuningSpec.strategy's polymorphic_field machinery.
    from phenotypic.tune._spec import StrategyConfigField
    from pydantic import TypeAdapter

    adapter = TypeAdapter(StrategyConfigField)
    cfg = OptunaConfig(n_trials=12, sampler="nsga2", prune=True, seed=3)
    dumped = adapter.dump_json(cfg)
    assert b'"class"' in dumped and b"OptunaConfig" in dumped
    back = adapter.validate_json(dumped)
    assert isinstance(back, OptunaConfig)
    assert back == cfg


# NOTE: ``OptunaConfig.build`` tests (env-resolution, strategy construction) live
# in test_optuna_strategy.py — they require the live ``OptunaStrategy`` (D3) and
# are ``skipif``'d when the extra is absent.
