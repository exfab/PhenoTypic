"""Serializable strategy configs; each builds its live SearchStrategy.

These are a closed set in Phase 1 (grid/random) → a discriminated union.
Phase 2 adds ``OptunaConfig``; the polymorphic-field path (engine-architecture
§6) lets the open Scorer/Strategy sets extend, but the in-spec config field uses
this union for the built-in kinds.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .._search_space import SearchSpace
from ._grid import GridStrategy
from ._protocol import SearchStrategy
from ._random import RandomStrategy

#: Closed set of strategy-config discriminator tags (reused by the union).
StrategyKind = Literal["grid", "random"]


class StrategyConfig(BaseModel):
    """Abstract, serializable config that builds its live ``SearchStrategy``.

    A frozen value-model; concrete subclasses (``GridConfig`` / ``RandomConfig``)
    carry a ``kind`` discriminator and implement ``build``. ``StrategyConfig``
    itself cannot be instantiated (it has an abstract method).

    Args:
        seed: The RNG seed forwarded to seeded strategies; defaults to ``0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int = 0

    @abstractmethod
    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy:
        """Construct the live strategy for ``space``.

        Args:
            space: The search space the built strategy operates over.
            store: The study store (Phase 1d's ``StudyStore``); accepted for a
                uniform factory signature but unused by the zero-dependency
                grid/random strategies.
        """
        ...


class GridConfig(StrategyConfig):
    """Builds a ``GridStrategy`` (exhaustive conditional-Cartesian enumeration).

    Args:
        kind: The discriminator tag; always ``"grid"``.
    """

    kind: Literal["grid"] = "grid"

    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy:
        return GridStrategy(space)


class RandomConfig(StrategyConfig):
    """Builds a ``RandomStrategy`` (seeded random sampling).

    Args:
        kind: The discriminator tag; always ``"random"``.
        n_trials: The number of configurations to sample before exhaustion.
    """

    kind: Literal["random"] = "random"
    n_trials: int

    def build(self, space: SearchSpace, store: Optional[Any]) -> SearchStrategy:
        return RandomStrategy(space, n_trials=self.n_trials, seed=self.seed)


#: Discriminated union of the built-in (Phase 1) strategy configs.
StrategyConfigUnion = Annotated[
    Union[GridConfig, RandomConfig], Field(discriminator="kind")
]
