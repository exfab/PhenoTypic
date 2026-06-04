"""The pruning channel: how the Evaluator reports + checks early-stop.

Phase 1 ships only the no-op (Grid/Random never prune). Phase 2 adds an
Optuna-trial-backed channel. Keeping this a Protocol keeps the Evaluator
Optuna-free (robust-evaluation.md §7, optuna-integration.md §6).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PruningChannel(Protocol):
    """Reports interim values and answers whether the trial should early-stop."""

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


class NoOpChannel:
    """Never prunes; ``report`` is a no-op. Used by Grid/Random strategies."""

    def report(self, value: float, step: int) -> None:
        return None

    def should_prune(self) -> bool:
        return False
