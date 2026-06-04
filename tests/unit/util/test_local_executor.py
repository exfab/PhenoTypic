from __future__ import annotations

from phenotypic._execution import Executor, LocalExecutor


def _square(x: int) -> int:
    return x * x


def test_local_executor_maps_in_order():
    ex = LocalExecutor(n_jobs=1)
    assert ex.run(_square, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_local_executor_parallel_results_ordered():
    ex = LocalExecutor(n_jobs=2)
    assert ex.run(_square, list(range(10))) == [i * i for i in range(10)]


def test_local_executor_empty():
    assert LocalExecutor(n_jobs=1).run(_square, []) == []


def test_local_executor_satisfies_protocol():
    assert isinstance(LocalExecutor(), Executor)
