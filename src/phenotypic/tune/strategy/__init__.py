"""Search strategies — the public ``phenotypic.tune.strategy`` namespace.

The serializable optimizer configs a :class:`~phenotypic.tune.TuningSpec` carries:
:class:`StrategyConfig` (the abstract base) plus :class:`GridConfig`,
:class:`RandomConfig`, and :class:`OptunaConfig`, each of which ``build()``s a
live :class:`SearchStrategy`. The inner ``_*`` modules stay private; import the
configs from this package.
"""
from __future__ import annotations

from ._config import (
    GridConfig,
    OptunaConfig,
    RandomConfig,
    SamplerKind,
    StrategyConfig,
    StrategyConfigUnion,
)
from ._grid import GridStrategy
from ._protocol import SearchStrategy
from ._pruning import NoOpChannel, PruningChannel
from ._random import RandomStrategy

__all__ = [
    "SearchStrategy",
    "PruningChannel",
    "NoOpChannel",
    "GridStrategy",
    "RandomStrategy",
    "StrategyConfig",
    "GridConfig",
    "RandomConfig",
    "OptunaConfig",
    "SamplerKind",
    "StrategyConfigUnion",
]
