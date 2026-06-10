from __future__ import annotations

import pytest

from phenotypic.tune._evaluation._aggregate_math import (
    _GAP_EPS,
    _median_iqr,
    _relative,
)


def test_gap_eps_is_a_meaningful_floor():
    # Raised 1e-12 -> 0.02 so a near-zero denominator cannot explode the ratio.
    assert _GAP_EPS == pytest.approx(0.02)


def test_relative_floors_tiny_denominator_at_gap_eps():
    # numerator 0.01 / max(0.0, 0.02) = 0.5, not a blow-up.
    assert _relative(0.01, 0.0) == pytest.approx(0.5)


def test_relative_uses_true_denominator_above_floor():
    # denominator 0.5 > floor → 0.1 / 0.5 = 0.2 (floor does not bite).
    assert _relative(0.1, 0.5) == pytest.approx(0.2)
