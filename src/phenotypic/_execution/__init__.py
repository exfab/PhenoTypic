"""Shared execution primitives (Local now; Slurm in tune Phase 2)."""
from __future__ import annotations

from ._local import LocalExecutor
from ._protocol import Executor

__all__ = ["Executor", "LocalExecutor"]
