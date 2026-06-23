"""Unit tests for the cost-orientation primitives."""

import pytest

from phenotypic.tune.score._orient import Sense, clamp01, to_cost


class TestSense:
    def test_two_members_with_string_values(self):
        assert {s.value for s in Sense} == {"lower_better", "higher_better"}

    def test_str_enum_is_value_comparable(self):
        # str, Enum → member is its string value, robust for ClassVar use.
        assert Sense.LOWER_BETTER == "lower_better"


class TestToCostBounded:
    def test_lower_better_bounded_is_identity(self):
        # A value already in [0,1] that is a cost (lower better) passes through.
        assert to_cost(0.3, sense=Sense.LOWER_BETTER) == pytest.approx(0.3)

    def test_higher_better_bounded_is_complement(self):
        # A [0,1] goodness (Dice/IoU/ICC) is complemented to cost.
        assert to_cost(0.3, sense=Sense.HIGHER_BETTER) == pytest.approx(0.7)

    def test_perfect_goodness_maps_to_zero_cost(self):
        assert to_cost(1.0, sense=Sense.HIGHER_BETTER) == pytest.approx(0.0)

    def test_worst_goodness_maps_to_unit_cost(self):
        assert to_cost(0.0, sense=Sense.HIGHER_BETTER) == pytest.approx(1.0)


class TestToCostAnchored:
    def test_zero_divergence_is_zero_cost(self):
        assert to_cost(0.0, sense=Sense.LOWER_BETTER, anchor=0.1) == pytest.approx(0.0)

    def test_at_anchor_is_half_cost(self):
        assert to_cost(0.1, sense=Sense.LOWER_BETTER, anchor=0.1) == pytest.approx(0.5)

    def test_large_divergence_approaches_unit_cost(self):
        assert to_cost(10.0, sense=Sense.LOWER_BETTER, anchor=0.1) > 0.99

    def test_inf_divergence_is_worst(self):
        assert to_cost(float("inf"), sense=Sense.LOWER_BETTER, anchor=0.1) == 1.0

    def test_higher_better_unbounded_zero_is_worst(self):
        # higher-better unbounded: value 0 = worst → cost 1.0.
        assert to_cost(0.0, sense=Sense.HIGHER_BETTER, anchor=0.1) == pytest.approx(1.0)

    def test_higher_better_unbounded_at_anchor_is_half(self):
        assert to_cost(0.1, sense=Sense.HIGHER_BETTER, anchor=0.1) == pytest.approx(0.5)


class TestClamp01:
    @pytest.mark.parametrize(
        "value,expected", [(-0.5, 0.0), (0.0, 0.0), (0.3, 0.3), (1.0, 1.0), (1.5, 1.0)]
    )
    def test_clamps_to_unit_interval(self, value, expected):
        assert clamp01(value) == pytest.approx(expected)
