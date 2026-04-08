"""Tests for phenotypic.sweep.Sweep and Fixed classes."""

import pytest

from phenotypic.sweep import Fixed, Sweep
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector


class TestFixed:
    """Tests for the Fixed sentinel wrapper."""

    def test_wraps_scalar(self):
        f = Fixed(42)
        assert f.value == 42

    def test_wraps_tuple(self):
        f = Fixed((1.0, 2.0))
        assert f.value == (1.0, 2.0)

    def test_wraps_list(self):
        f = Fixed([1, 2, 3])
        assert f.value == [1, 2, 3]

    def test_repr(self):
        f = Fixed("hello")
        assert "Fixed" in repr(f)
        assert "hello" in repr(f)


class TestSweep:
    """Tests for the Sweep class."""

    def test_tuple_classified_as_sweep(self):
        s = Sweep(GaussianBlur, sigma=(1.0, 2.0, 3.0))
        assert "sigma" in s.sweep_params
        assert s.sweep_params["sigma"] == [1.0, 2.0, 3.0]
        assert "sigma" not in s.fixed_params

    def test_scalar_classified_as_fixed(self):
        s = Sweep(GaussianBlur, truncate=4.0)
        assert "truncate" in s.fixed_params
        assert s.fixed_params["truncate"] == 4.0
        assert "truncate" not in s.sweep_params

    def test_list_classified_as_fixed(self):
        """Lists are fixed values (passed as-is), not swept."""
        s = Sweep(OtsuDetector, ignore_zeros=True)
        assert "ignore_zeros" in s.fixed_params
        assert s.fixed_params["ignore_zeros"] is True

    def test_fixed_wrapper_classified_as_fixed(self):
        s = Sweep(GaussianBlur, sigma=Fixed(1.5))
        assert "sigma" in s.fixed_params
        assert s.fixed_params["sigma"] == 1.5
        assert "sigma" not in s.sweep_params

    def test_fixed_wrapping_tuple_stays_fixed(self):
        """Fixed() wrapping a tuple should treat it as fixed, not swept."""
        s = Sweep(GaussianBlur, sigma=Fixed((1.0, 2.0)))
        assert "sigma" in s.fixed_params
        assert s.fixed_params["sigma"] == (1.0, 2.0)
        assert "sigma" not in s.sweep_params

    def test_mixed_params(self):
        s = Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0)
        assert s.sweep_params == {"sigma": [1.0, 2.0]}
        assert s.fixed_params == {"truncate": 4.0}

    def test_no_params_gives_empty(self):
        s = Sweep(GaussianBlur)
        assert s.sweep_params == {}
        assert s.fixed_params == {}

    def test_stores_operation_class(self):
        s = Sweep(GaussianBlur, sigma=(1.0,))
        assert s.operation_class is GaussianBlur

    def test_instance_raises_type_error(self):
        with pytest.raises(TypeError, match="class.*not an instance"):
            Sweep(GaussianBlur(), sigma=(1.0,))

    def test_invalid_param_name_raises_value_error(self):
        with pytest.raises(ValueError, match="no parameter 'nonexistent'"):
            Sweep(GaussianBlur, nonexistent=(1.0, 2.0))

    def test_repr_contains_class_name(self):
        s = Sweep(GaussianBlur, sigma=(1.0, 2.0))
        assert "GaussianBlur" in repr(s)

    def test_bool_scalar_classified_as_fixed(self):
        s = Sweep(OtsuDetector, ignore_zeros=False)
        assert "ignore_zeros" in s.fixed_params
        assert s.fixed_params["ignore_zeros"] is False

    def test_none_classified_as_fixed(self):
        """None is a scalar and should be classified as fixed."""
        s = Sweep(GaussianBlur, sigma=None)
        assert "sigma" in s.fixed_params
        assert s.fixed_params["sigma"] is None
