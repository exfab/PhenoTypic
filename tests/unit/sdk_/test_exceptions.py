"""Regression tests for exception transport across process boundaries."""

from __future__ import annotations

import pickle

from phenotypic.sdk_.exceptions_ import OperationFailedError


def test_operation_failed_error_pickle_round_trip_preserves_context() -> None:
    """Pickling must retain every constructor argument and the rendered message."""
    error = OperationFailedError(
        operation="MeasureSize",
        image_name="plate_001",
        err_type=ValueError,
        message="bad bins",
    )

    restored = pickle.loads(pickle.dumps(error))

    assert restored.operation == "MeasureSize"
    assert restored.image_name == "plate_001"
    assert restored.err_type is ValueError
    assert restored.message == "bad bins"
    assert str(restored) == str(error)
