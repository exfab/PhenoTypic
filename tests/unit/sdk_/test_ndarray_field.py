"""Tests for the ``NdArrayField`` pydantic-friendly array annotation."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from phenotypic.tools_.typing_ import NdArrayField


class _KernelModel(BaseModel):
    """Throwaway model carrying an ``NdArrayField``.

    Args:
        kernel: A 2D convolution kernel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel: NdArrayField


class TestNdArrayFieldInput:
    def test_nested_list_input_becomes_ndarray(self) -> None:
        model = _KernelModel(kernel=[[1, 0], [0, 1]])
        assert isinstance(model.kernel, np.ndarray)
        np.testing.assert_array_equal(
            model.kernel, np.array([[1, 0], [0, 1]])
        )

    def test_ndarray_input_preserved(self) -> None:
        arr = np.array([[2.0, 3.0], [4.0, 5.0]])
        model = _KernelModel(kernel=arr)
        assert isinstance(model.kernel, np.ndarray)
        np.testing.assert_array_equal(model.kernel, arr)

    def test_flat_list_input_becomes_1d_ndarray(self) -> None:
        model = _KernelModel(kernel=[1, 2, 3])
        assert isinstance(model.kernel, np.ndarray)
        assert model.kernel.shape == (3,)


class TestNdArrayFieldSerialization:
    def test_model_dump_json_mode_yields_nested_list(self) -> None:
        model = _KernelModel(kernel=np.array([[1, 0], [0, 1]]))
        dumped = model.model_dump(mode="json")
        assert dumped["kernel"] == [[1, 0], [0, 1]]
        assert isinstance(dumped["kernel"], list)
        assert isinstance(dumped["kernel"][0], list)

    def test_round_trip_list_to_model_to_json(self) -> None:
        original = [[1, 2], [3, 4]]
        model = _KernelModel(kernel=original)
        dumped = model.model_dump(mode="json")
        revived = _KernelModel.model_validate(dumped)
        np.testing.assert_array_equal(
            revived.kernel, np.asarray(original)
        )

    def test_round_trip_ndarray_to_model_to_json(self) -> None:
        original = np.array([[5, 6], [7, 8]])
        model = _KernelModel(kernel=original)
        revived = _KernelModel.model_validate(
            model.model_dump(mode="json")
        )
        np.testing.assert_array_equal(revived.kernel, original)

    def test_model_dump_json_string_is_valid(self) -> None:
        model = _KernelModel(kernel=np.array([[1, 0], [0, 1]]))
        # model_dump_json must not raise on the ndarray field.
        payload = model.model_dump_json()
        assert "[[1,0],[0,1]]" in payload.replace(" ", "")


class TestNdArrayFieldJsonSchema:
    def test_schema_property_is_array_type(self) -> None:
        schema = _KernelModel.model_json_schema()
        kernel_schema = schema["properties"]["kernel"]
        assert kernel_schema["type"] == "array"
        assert kernel_schema["items"] == {}

    def test_schema_generation_does_not_raise(self) -> None:
        # An arbitrary numpy type would otherwise break schema gen;
        # WithJsonSchema must keep it safe.
        schema = _KernelModel.model_json_schema()
        assert "kernel" in schema["properties"]
