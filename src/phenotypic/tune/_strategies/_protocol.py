"""The SearchStrategy seam (engine-architecture.md §4)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ._pruning import PruningChannel


@runtime_checkable
class SearchStrategy(Protocol):
    """The swappable optimizer seam: ask for params, tell results, stop on exhaustion.

    Runtime-stateful (Grid cursor / Random RNG / Optuna study) → a Protocol so a
    test fake or a future strategy conforms structurally. ``result`` is typed
    ``Any`` in Phase 1b; Phase 1c narrows it to ``EvaluationResult``.
    """

    def suggest(self) -> tuple[Mapping[str, Any], PruningChannel]: ...

    def register_result(
        self, params: Mapping[str, Any], result: Any, *, pruned: bool = False
    ) -> None: ...

    def is_exhausted(self) -> bool: ...
