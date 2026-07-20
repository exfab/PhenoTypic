"""Re-derive equal-crossing axial ring-consensus invariants independently."""

from __future__ import annotations

import numpy as np


def axial_consensus(degrees: np.ndarray) -> tuple[float, float]:
    """Return equal-sample axial consensus and doubled-angle resultant."""
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    cosine = float(np.mean(np.cos(2.0 * radians)))
    sine = float(np.mean(np.sin(2.0 * radians)))
    return (
        float(np.rad2deg(0.5 * np.arctan2(sine, cosine))),
        float(np.hypot(cosine, sine)),
    )


def contiguous_change(degrees: np.ndarray) -> np.ndarray:
    """Accumulate seam-safe axial change within each supported run."""
    radians = np.deg2rad(np.asarray(degrees, dtype=float))
    cumulative = np.full_like(radians, np.nan)
    previous: int | None = None
    for index, angle in enumerate(radians):
        if not np.isfinite(angle):
            previous = None
            continue
        if previous is None:
            cumulative[index] = 0.0
            previous = index
            continue
        difference = angle - radians[previous]
        step = 0.5 * np.arctan2(np.sin(2.0 * difference), np.cos(2.0 * difference))
        if np.isclose(abs(step), 0.5 * np.pi, atol=1e-12, rtol=0.0):
            previous = None
            continue
        cumulative[index] = cumulative[previous] + step
        previous = index
    return np.rad2deg(cumulative)


def verify_ced_literal_crossing_consensus() -> None:
    """Assert axial seam, duplication, ambiguity, and unwrapping controls."""
    consensus, resultant = axial_consensus(np.array([85.0, -85.0]))
    assert np.isclose(abs(consensus), 90.0)
    assert resultant > 0.98

    base = np.array([10.0, 20.0, 30.0])
    repeated = np.repeat(base, 12)
    base_consensus, base_resultant = axial_consensus(base)
    repeated_consensus, repeated_resultant = axial_consensus(repeated)
    assert np.isclose(base_consensus, repeated_consensus)
    assert np.isclose(base_resultant, repeated_resultant)

    _ambiguous_consensus, ambiguous_resultant = axial_consensus(
        np.array([0.0, 90.0])
    )
    assert ambiguous_resultant < 1e-12

    changes = contiguous_change(np.array([80.0, -85.0, -70.0]))
    assert np.allclose(changes, [0.0, 15.0, 30.0])

    gap_changes = contiguous_change(np.array([10.0, 20.0, np.nan, 70.0, 80.0]))
    assert np.allclose(gap_changes, [0.0, 10.0, np.nan, 0.0, 10.0], equal_nan=True)

    ambiguous_changes = contiguous_change(np.array([0.0, 90.0, 80.0, 70.0]))
    assert np.allclose(
        ambiguous_changes,
        [0.0, np.nan, 0.0, -10.0],
        equal_nan=True,
    )

    print(
        "PASS: equal-crossing axial consensus is seam-safe, invariant to "
        "uniform replication, gap-safe, and exact-90 ambiguity-aware"
    )


if __name__ == "__main__":
    verify_ced_literal_crossing_consensus()
