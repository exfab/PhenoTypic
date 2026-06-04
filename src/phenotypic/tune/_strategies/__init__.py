"""Search strategies (private)."""
from __future__ import annotations

from ._pruning import NoOpChannel, PruningChannel

__all__ = ["PruningChannel", "NoOpChannel"]
