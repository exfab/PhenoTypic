"""The Executor seam: run a work-fn over items in parallel.

A low-level parallel-map primitive shared by callers that need to fan a
pure function over many inputs (the tuning Evaluator over calibration
images). Orchestration (saving, logging, scoring) lives in the caller's
injected ``work`` function.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence, TypeVar, runtime_checkable

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class Executor(Protocol):
    """Runs ``work(item)`` for every item, returning results in input order."""

    def run(self, work: Callable[[T], R], items: Sequence[T]) -> list[R]: ...
