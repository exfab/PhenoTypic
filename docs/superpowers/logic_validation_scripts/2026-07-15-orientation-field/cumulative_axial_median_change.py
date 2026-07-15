"""Validate seam-safe cumulative axial-median change independently."""

from __future__ import annotations

import numpy as np


def axial_difference(outer: float, inner: float) -> float:
    """Return the signed axial difference in [-pi/2, pi/2]."""
    difference = outer - inner
    return 0.5 * float(
        np.arctan2(
            np.sin(2.0 * difference),
            np.cos(2.0 * difference),
        )
    )


def cumulative_axial_change(values: np.ndarray) -> np.ndarray:
    """Accumulate adjacent axial differences until evidence is missing."""
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    starts = np.flatnonzero(np.isfinite(values))
    if starts.size == 0:
        return output
    start = int(starts[0])
    output[start] = 0.0
    for index in range(start, values.size - 1):
        if not (
            np.isfinite(output[index])
            and np.isfinite(values[index])
            and np.isfinite(values[index + 1])
        ):
            break
        change = axial_difference(values[index + 1], values[index])
        if np.isclose(abs(change), 0.5 * np.pi, rtol=0.0, atol=1e-12):
            break
        output[index + 1] = output[index] + change
    return output


def validate_cumulative_axial_median_change() -> None:
    """Check rotation invariance, axial seams, gaps, and ambiguity."""
    degrees = np.radians

    straight = cumulative_axial_change(degrees([35.0, 35.0, 35.0, 35.0]))
    np.testing.assert_allclose(np.degrees(straight), [0.0, 0.0, 0.0, 0.0])

    seam = cumulative_axial_change(degrees([80.0, 89.0, -89.0, -80.0]))
    np.testing.assert_allclose(np.degrees(seam), [0.0, 9.0, 11.0, 20.0])

    known = cumulative_axial_change(degrees([0.0, 40.0, 80.0, -80.0, -40.0]))
    np.testing.assert_allclose(np.degrees(known), [0.0, 40.0, 80.0, 100.0, 140.0])

    rotated = cumulative_axial_change(
        degrees([0.0, 40.0, 80.0, -80.0, -40.0]) + degrees(27.0)
    )
    np.testing.assert_allclose(rotated, known, rtol=0.0, atol=1e-12)

    gapped = cumulative_axial_change(degrees([10.0, 20.0, np.nan, 40.0]))
    assert np.isfinite(gapped[:2]).all()
    assert np.isnan(gapped[2:]).all()

    ambiguous = cumulative_axial_change(degrees([0.0, 90.0, 80.0]))
    assert ambiguous[0] == 0.0
    assert np.isnan(ambiguous[1:]).all()


if __name__ == "__main__":
    validate_cumulative_axial_median_change()
    print("PASS: cumulative axial-median change is seam-safe and rotation invariant")
