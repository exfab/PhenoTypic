"""Shared execution primitives (Local + distributed-tune Slurm fleet)."""
from __future__ import annotations

from ._local import LocalExecutor
from ._protocol import Executor
from ._slurm import SlurmExecutor

__all__ = ["Executor", "LocalExecutor", "SlurmExecutor"]
