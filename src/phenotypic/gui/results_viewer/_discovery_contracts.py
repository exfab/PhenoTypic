"""Thread-safe cancellation and progress contracts for Results discovery."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

DiscoveryPhase = Literal[
    "classifying",
    "inventory",
    "measurements",
    "indexing",
    "verifying",
    "complete",
]


class OutputDiscoveryCancelledError(RuntimeError):
    """Raised when a caller cancels an in-progress output discovery."""


@dataclass
class OutputDiscoveryCancellation:
    """Thread-safe cooperative cancellation handle owned by the caller."""

    _event: threading.Event = field(
        default_factory=threading.Event,
        init=False,
        repr=False,
    )

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """Raise the public cancellation exception when requested."""
        if self.cancelled:
            raise OutputDiscoveryCancelledError("Output discovery was cancelled.")


@dataclass(frozen=True)
class OutputDiscoveryProgress:
    """One phase update emitted during output discovery.

    Phases and completed counts are monotonic within an attempt. A stable-read
    retry increments ``attempt`` and restarts at ``classifying`` so clients can
    reset per-attempt progress without mistaking the reset for a regression.
    """

    phase: DiscoveryPhase
    detail: str
    attempt: int = 1
    completed: int | None = None
    total: int | None = None
    cache_hit: bool = False


OutputDiscoveryProgressCallback = Callable[[OutputDiscoveryProgress], None]


def report_discovery_progress(
    callback: OutputDiscoveryProgressCallback | None,
    *,
    phase: DiscoveryPhase,
    detail: str,
    completed: int | None = None,
    total: int | None = None,
    cache_hit: bool = False,
) -> None:
    """Emit a progress update when the caller supplied a callback."""
    if callback is None:
        return
    callback(
        OutputDiscoveryProgress(
            phase=phase,
            detail=detail,
            completed=completed,
            total=total,
            cache_hit=cache_hit,
        )
    )
