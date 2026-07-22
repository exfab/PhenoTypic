from __future__ import annotations

import pytest

from phenotypic._execution import Executor, LocalExecutor
from phenotypic.sdk_.exceptions_ import OperationFailedError


def _square(x: int) -> int:
    return x * x


def _raise_operation_failed_error(_: object) -> None:
    raise OperationFailedError(
        operation="MeasureSize",
        image_name="plate_001",
        err_type=ValueError,
        message="bad bins",
    )


def test_local_executor_maps_in_order():
    ex = LocalExecutor(n_jobs=1)
    assert ex.run(_square, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_local_executor_parallel_results_ordered():
    ex = LocalExecutor(n_jobs=2)
    assert ex.run(_square, list(range(10))) == [i * i for i in range(10)]


def test_local_executor_propagates_operation_failure_context():
    ex = LocalExecutor(n_jobs=2)

    with pytest.raises(OperationFailedError) as error:
        ex.run(_raise_operation_failed_error, [None])

    assert error.value.operation == "MeasureSize"
    assert error.value.image_name == "plate_001"
    assert error.value.err_type is ValueError
    assert error.value.message == "bad bins"


def test_local_executor_empty():
    assert LocalExecutor(n_jobs=1).run(_square, []) == []


def test_local_executor_satisfies_protocol():
    assert isinstance(LocalExecutor(), Executor)
