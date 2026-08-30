"""The three decisions the measurement display makes, pinned.

Each of these is a judgement call the spec left open, so each is written down
here rather than left to whichever call site happens to read the code next:
one sequential ramp taken from the brand tokens, one number format for every
column, and an ink colour computed from the tint rather than chosen.
"""

from __future__ import annotations

import pytest

from phenotypic.gui._shared._measurement_tint import (
    MeasurementScale,
    format_measurement_value,
    sequential_tint,
)
from phenotypic.sdk_.viz.figures import SEQUENTIAL_COLORSCALE


# ---------------------------------------------------------------------------
# The ramp
# ---------------------------------------------------------------------------


def test_the_ramp_is_the_brand_sequential_scale_not_a_second_one() -> None:
    """A value must read the same on a card as on the heatmap tab.

    The endpoints are sampled rather than the constant re-spelled, so
    changing ``SEQUENTIAL_COLORSCALE`` moves the cards with it.
    """
    assert sequential_tint(1.0) == SEQUENTIAL_COLORSCALE[-1][1].lower()
    # The 0.0 stop is translucent navy; a card needs it flattened.
    assert sequential_tint(0.0).startswith("#")
    assert len(sequential_tint(0.0)) == 7


def test_the_ramp_passes_through_its_midpoint() -> None:
    assert sequential_tint(0.5) == SEQUENTIAL_COLORSCALE[1][1].lower()


def test_the_ramp_is_monotone_in_darkness() -> None:
    """Higher fraction, darker card -- otherwise the map reads backwards."""

    def _sum_channels(color: str) -> int:
        return sum(int(color[i : i + 2], 16) for i in (1, 3, 5))

    samples = [sequential_tint(index / 10) for index in range(11)]
    brightness = [_sum_channels(color) for color in samples]
    assert brightness == sorted(brightness, reverse=True)


@pytest.mark.parametrize("fraction", [-5.0, -0.01, 1.01, 42.0])
def test_the_ramp_clamps_rather_than_extrapolating(fraction: float) -> None:
    assert sequential_tint(fraction) in {
        sequential_tint(0.0),
        sequential_tint(1.0),
    }


# ---------------------------------------------------------------------------
# The number format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Integral values keep no decimal point: a grid of areas reads as
        # numbers, not as padded floats.
        (86770.0, "86770"),
        (770.0, "770"),
        (5.0, "5"),
        (0.0, "0"),
        # Everything else to four significant figures, one rule for every
        # column -- Shape_Area and a Delta-E share it.
        (0.535051, "0.5351"),
        (3.2, "3.200"),
        (-0.5, "-0.5000"),
        # Outside [1e-4, 1e6) the fixed form stops being readable.
        (1234567.5, "1.235e+06"),
        (1.2345e-9, "1.235e-09"),
    ],
)
def test_the_one_number_rule(value: float, expected: str) -> None:
    assert format_measurement_value(value) == expected


def test_a_non_finite_value_reads_as_not_available() -> None:
    """An empty mask divides by zero. That is a measured outcome."""
    assert format_measurement_value(float("nan")) == "n/a"
    assert format_measurement_value(float("inf")) == "n/a"


# ---------------------------------------------------------------------------
# The scale, and the ink it implies
# ---------------------------------------------------------------------------


def test_a_scale_spans_only_the_values_it_was_given() -> None:
    scale = MeasurementScale.over("Shape_Area", [100.0, 400.0, 200.0])
    assert scale is not None
    assert (scale.minimum, scale.maximum) == (100.0, 400.0)
    assert scale.fraction_of(100.0) == 0.0
    assert scale.fraction_of(400.0) == 1.0
    assert scale.fraction_of(250.0) == pytest.approx(0.5)


def test_a_scale_over_no_values_is_none() -> None:
    """Nothing to tint, and nothing to legend."""
    assert MeasurementScale.over("Shape_Area", []) is None
    assert MeasurementScale.over("Shape_Area", [float("nan")]) is None


def test_a_flat_scale_maps_everything_to_zero() -> None:
    """One shade is the honest rendering of "no variation here"."""
    scale = MeasurementScale.over("Shape_Area", [7.0, 7.0])
    assert scale is not None
    assert scale.fraction_of(7.0) == 0.0


def test_ink_flips_so_the_label_stays_readable_at_both_ends() -> None:
    """The ramp runs near-white to navy; one fixed ink fails at one end."""
    scale = MeasurementScale.over("Shape_Area", [0.0, 1.0])
    assert scale is not None
    low = scale.measurement_for(0.0)
    high = scale.measurement_for(1.0)
    assert low.tint == sequential_tint(0.0)
    assert high.tint == sequential_tint(1.0)
    assert low.ink != high.ink
