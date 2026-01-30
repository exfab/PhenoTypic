"""Tests for SweepSpec class."""

import pytest

from phenotypic.gui.explorer import SweepSpec
from phenotypic.gui.explorer._sweep_spec import (
    expand_sweep_combinations,
    count_sweep_combinations,
)


class TestSweepSpecBasics:
    """Test basic SweepSpec functionality."""

    def test_create_from_values(self):
        """Test creating SweepSpec with explicit values."""
        spec = SweepSpec("sigma", [1.0, 1.5, 2.0])

        assert spec.param == "sigma"
        assert spec.values == [1.0, 1.5, 2.0]
        assert spec.count == 3

    def test_create_from_range(self):
        """Test creating SweepSpec from numeric range."""
        spec = SweepSpec.from_range("sigma", 1.0, 2.0, 0.5)

        assert spec.param == "sigma"
        assert spec.values == [1.0, 1.5, 2.0]
        assert spec.count == 3

    def test_create_from_range_integer_step(self):
        """Test range with integer values."""
        spec = SweepSpec.from_range("threshold", 10, 50, 10)

        assert spec.values == [10, 20, 30, 40, 50]
        assert spec.count == 5

    def test_create_from_linspace(self):
        """Test creating SweepSpec with linear spacing."""
        spec = SweepSpec.from_linspace("sigma", 1.0, 3.0, 5)

        assert spec.param == "sigma"
        assert spec.count == 5
        assert spec.values[0] == pytest.approx(1.0)
        assert spec.values[-1] == pytest.approx(3.0)

    def test_create_from_logspace(self):
        """Test creating SweepSpec with logarithmic spacing."""
        spec = SweepSpec.from_logspace("threshold", 1, 3, 3)

        assert spec.count == 3
        assert spec.values[0] == pytest.approx(10.0)
        assert spec.values[1] == pytest.approx(100.0)
        assert spec.values[2] == pytest.approx(1000.0)

    def test_categorical_values(self):
        """Test SweepSpec with categorical string values."""
        spec = SweepSpec("shape", ["disk", "square", "diamond"])

        assert spec.param == "shape"
        assert spec.values == ["disk", "square", "diamond"]
        assert spec.count == 3

    def test_is_operation_sweep(self):
        """Test detection of operation sweep."""
        regular = SweepSpec("sigma", [1.0, 2.0])
        operation = SweepSpec("__operation__", ["op1", "op2"])

        assert regular.is_operation_sweep is False
        assert operation.is_operation_sweep is True


class TestSweepSpecSerialization:
    """Test SweepSpec serialization."""

    def test_to_dict_numeric(self):
        """Test serialization of numeric sweep."""
        spec = SweepSpec("sigma", [1.0, 1.5, 2.0])
        data = spec.to_dict()

        assert data["param"] == "sigma"
        assert data["values"] == [1.0, 1.5, 2.0]

    def test_from_dict_numeric(self):
        """Test deserialization of numeric sweep."""
        data = {"param": "sigma", "values": [1.0, 1.5, 2.0]}
        spec = SweepSpec.from_dict(data)

        assert spec.param == "sigma"
        assert spec.values == [1.0, 1.5, 2.0]

    def test_roundtrip_serialization(self):
        """Test serialize then deserialize preserves data."""
        original = SweepSpec.from_range("threshold", 10, 50, 10)
        restored = SweepSpec.from_dict(original.to_dict())

        assert restored.param == original.param
        assert restored.values == original.values


class TestSweepCombinations:
    """Test sweep combination expansion."""

    def test_single_sweep(self):
        """Test expanding a single sweep."""
        sweeps = [SweepSpec("sigma", [1.0, 2.0])]
        combos = expand_sweep_combinations(sweeps)

        assert len(combos) == 2
        assert combos[0] == {"sigma": 1.0}
        assert combos[1] == {"sigma": 2.0}

    def test_two_sweeps(self):
        """Test expanding two sweeps (cartesian product)."""
        sweeps = [
            SweepSpec("sigma", [1.0, 2.0]),
            SweepSpec("threshold", [50, 100]),
        ]
        combos = expand_sweep_combinations(sweeps)

        assert len(combos) == 4
        assert {"sigma": 1.0, "threshold": 50} in combos
        assert {"sigma": 1.0, "threshold": 100} in combos
        assert {"sigma": 2.0, "threshold": 50} in combos
        assert {"sigma": 2.0, "threshold": 100} in combos

    def test_three_sweeps(self):
        """Test expanding three sweeps."""
        sweeps = [
            SweepSpec("a", [1, 2]),
            SweepSpec("b", [10, 20]),
            SweepSpec("c", [100, 200]),
        ]
        combos = expand_sweep_combinations(sweeps)

        assert len(combos) == 8  # 2 * 2 * 2

    def test_empty_sweeps(self):
        """Test expanding empty sweep list."""
        combos = expand_sweep_combinations([])

        assert len(combos) == 1
        assert combos[0] == {}

    def test_count_combinations(self):
        """Test counting combinations without expansion."""
        sweeps = [
            SweepSpec("a", [1, 2, 3]),
            SweepSpec("b", [10, 20]),
            SweepSpec("c", [100, 200, 300, 400]),
        ]
        count = count_sweep_combinations(sweeps)

        assert count == 3 * 2 * 4  # 24

    def test_count_empty(self):
        """Test counting empty sweep list."""
        assert count_sweep_combinations([]) == 1


class TestSweepSpecRepr:
    """Test string representation."""

    def test_repr_short(self):
        """Test repr for short value list."""
        spec = SweepSpec("sigma", [1.0, 2.0])
        repr_str = repr(spec)

        assert "sigma" in repr_str
        assert "1.0" in repr_str
        assert "2.0" in repr_str

    def test_repr_long(self):
        """Test repr for long value list (truncated)."""
        spec = SweepSpec.from_range("sigma", 0, 10, 1)
        repr_str = repr(spec)

        assert "sigma" in repr_str
        assert "11 values" in repr_str
