"""Local (joblib) Executor."""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class LocalExecutor:
    """Map ``work`` over ``items`` with joblib, preserving input order.

    Args:
        n_jobs: Worker count (``-1`` = all cores). Default ``-1``.

    Examples:
        >>> from phenotypic._execution import LocalExecutor
        >>> LocalExecutor(n_jobs=1).run(lambda x: x + 1, [1, 2, 3])
        [2, 3, 4]
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs

    def run(self, work: Callable[[T], R], items: Sequence[T]) -> list[R]:
        if not items:
            return []
        from joblib import Parallel, delayed

        return list(
            Parallel(n_jobs=self.n_jobs)(delayed(work)(item) for item in items)
        )
