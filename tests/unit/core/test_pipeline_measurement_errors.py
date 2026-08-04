"""Regression tests for pipeline-step exception context."""

from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureSize
from phenotypic.sdk_.exceptions_ import OperationFailedError


class _ContextRequiredError(Exception):
    """Test exception whose constructor cannot accept one message argument."""

    def __init__(self, context: str, message: str) -> None:
        super().__init__(f"{context}: {message}")


def test_operation_error_wraps_exceptions_with_nonstandard_constructors() -> None:
    """Operation context must not reconstruct an arbitrary exception type."""
    pipeline = ImagePipeline(ops=[BlurGauss(sigma=1.0)])
    image = load_synth_yeast_plate()

    def fail_after_operation(*_args: object) -> None:
        raise _ContextRequiredError("callback", "boom")

    with pytest.raises(RuntimeError, match=r"\[BlurGauss\].*key='BlurGauss'") as error:
        pipeline._run_operations(image, on_op_complete=fail_after_operation)

    assert isinstance(error.value.__cause__, _ContextRequiredError)


def test_measurement_error_wraps_exceptions_with_nonstandard_constructors() -> None:
    """Pipeline context must not reconstruct an arbitrary exception type."""
    pipeline = ImagePipeline(meas=[MeasureSize()])
    image = load_synth_yeast_plate()
    image.objmap[:] = 0

    with pytest.raises(RuntimeError, match=r"\[MeasureSize\].*key='MeasureSize'") as error:
        pipeline.measure(image)

    assert isinstance(error.value.__cause__, OperationFailedError)
