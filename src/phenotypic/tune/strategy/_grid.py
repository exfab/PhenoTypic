"""GridStrategy — exhaustive enumeration (one trial per combo)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._search_space import SearchSpace
from ._enumerate import enumerate_grid
from ._pruning import NoOpChannel, PruningChannel


class GridStrategy:
    """Walks the full conditional Cartesian product of the SearchSpace.

    The degenerate "strategy" that reproduces exhaustive grid search. Raises
    on a continuous ``FloatRange`` knob (not enumerable).

    Args:
        space: The search space to enumerate; every knob must have an
            enumerable domain (``Categorical`` / ``IntRange`` / ``Fixed``).
    """

    def __init__(self, space: SearchSpace) -> None:
        self._combos = enumerate_grid(space)  # raises on FloatRange
        self._cursor = 0

    def suggest(self) -> tuple[Mapping[str, Any], PruningChannel]:
        # defensive copy: callers must not mutate the stored combo
        params = dict(self._combos[self._cursor])
        self._cursor += 1
        return params, NoOpChannel()

    def register_result(
        self, params: Mapping[str, Any], result: Any, *, pruned: bool = False
    ) -> None:
        return None  # grid does not learn

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._combos)
