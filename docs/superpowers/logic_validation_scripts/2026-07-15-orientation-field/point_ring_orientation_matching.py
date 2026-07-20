"""Re-derive point-ring orientation matching invariants independently."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment


def axial_difference(outer_degrees: float, inner_degrees: float) -> float:
    """Return the seam-safe axial difference in degrees."""
    difference = np.deg2rad(outer_degrees - inner_degrees)
    return float(
        np.rad2deg(
            0.5
            * np.arctan2(
                np.sin(2.0 * difference),
                np.cos(2.0 * difference),
            )
        )
    )


def lift_against_prediction(
    wrapped_outer: float,
    parent_unwrapped: float,
    previous_step: float,
) -> tuple[float, float, float, bool]:
    """Lift an axial angle by 180 degrees against the parent's trend."""
    prediction = parent_unwrapped + previous_step
    centre_lift = int(round((prediction - wrapped_outer) / 180.0))
    candidates = wrapped_outer + 180.0 * (
        centre_lift + np.arange(-1, 2, dtype=float)
    )
    errors = np.abs(candidates - prediction)
    order = np.argsort(errors, kind="stable")
    ambiguous = bool(
        np.isclose(errors[order[0]], errors[order[1]], atol=1e-12, rtol=0.0)
    )
    chosen = float(candidates[order[0]])
    return (
        chosen,
        chosen - parent_unwrapped,
        chosen - prediction,
        ambiguous,
    )


def normalized_cost(
    distance: float,
    axial_delta: float,
    max_distance: float,
    max_axial_delta: float,
) -> float:
    """Return the prototype's dimensionless evidence cost."""
    return float(
        np.square(distance / max_distance)
        + np.square(abs(axial_delta) / max_axial_delta)
    )


def global_assignment(
    pair_costs: np.ndarray,
    *,
    unmatched_cost: float = 1.0,
    ambiguity_margin: float = 0.05,
) -> tuple[set[tuple[int, int]], set[int]]:
    """Solve the prototype's private-dummy assignment and ambiguity test."""
    n_outer, n_inner = pair_costs.shape
    forbidden = 1e6
    costs = np.full((n_outer, n_inner + n_outer), forbidden, dtype=float)
    costs[:, :n_inner] = pair_costs
    for row in range(n_outer):
        costs[row, n_inner + row] = unmatched_cost
    rows, cols = linear_sum_assignment(costs)
    base_total = float(costs[rows, cols].sum())
    tentative = [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if col < n_inner and costs[row, col] < unmatched_cost
    ]
    selected: set[tuple[int, int]] = set()
    ambiguous: set[int] = set()
    for row, col in tentative:
        alternatives = costs.copy()
        alternatives[row, col] = forbidden
        alt_rows, alt_cols = linear_sum_assignment(alternatives)
        alternative_total = float(alternatives[alt_rows, alt_cols].sum())
        if alternative_total - base_total <= ambiguity_margin:
            ambiguous.add(row)
        else:
            selected.add((row, col))
    return selected, ambiguous


def verify_unwrap_and_guards() -> None:
    """Verify period-pi lifting and both history-dependent guards."""
    lifted, step, residual, ambiguous = lift_against_prediction(-85.0, 80.0, 15.0)
    assert np.isclose(lifted, 95.0)
    assert np.isclose(step, 15.0)
    assert np.isclose(residual, 0.0)
    assert not ambiguous

    # The current endpoint is a modest -20-degree step from its parent, but it
    # reverses a prior +20-degree trend by 40 degrees and fails the residual guard.
    _lifted, guarded_step, guarded_residual, _ambiguous = lift_against_prediction(
        60.0,
        80.0,
        20.0,
    )
    assert abs(guarded_step) <= 60.0
    assert abs(guarded_residual) > 30.0

    # This lift is unambiguous but exceeds the separate 60-degree step guard.
    _lifted, guarded_step, _residual, _ambiguous = lift_against_prediction(
        70.0,
        0.0,
        0.0,
    )
    assert abs(guarded_step) > 60.0

    # A prediction exactly halfway between axial equivalents has no defensible
    # period-pi lift and is explicitly ambiguous.
    _lifted, _step, _residual, ambiguous = lift_against_prediction(
        90.0,
        0.0,
        0.0,
    )
    assert ambiguous


def verify_connectivity_and_no_restart() -> None:
    """Verify annular connectivity and strict previous-ring inheritance."""
    skeleton = np.zeros((11, 11), dtype=bool)
    skeleton[1:10, 2] = True
    skeleton[1:10, 8] = True
    rows, cols = np.indices(skeleton.shape)
    radius = np.hypot(rows - 5.0, cols - 5.0)
    corridor = skeleton & (radius >= 3.0) & (radius <= 5.1)
    labels, count = connected_components(
        corridor,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    assert count == 2
    assert labels[1, 2] == labels[4, 2] != 0
    assert labels[1, 2] != labels[1, 8]

    # Only a finite inherited state on the immediately previous ring can seed
    # the next state. A later raw crossing remains unsupported after a gap.
    raw_present = [True, True, False, True]
    inherited = np.full(4, np.nan)
    inherited[0] = 0.0
    proposed_steps = [np.nan, 10.0, np.nan, 5.0]
    for ring in range(1, len(raw_present)):
        if raw_present[ring] and np.isfinite(inherited[ring - 1]):
            inherited[ring] = inherited[ring - 1] + proposed_steps[ring]
    assert np.allclose(inherited[:2], [0.0, 10.0])
    assert np.isnan(inherited[2])
    assert np.isnan(inherited[3])


def verify_matching_policies() -> None:
    """Verify local ambiguity and the three parent-selection policies."""
    max_distance = 8.0 / np.cos(np.deg2rad(75.0))
    assert np.isclose(max_distance, 30.90962644125018)
    max_angle = 20.0

    # Two outer crossings prefer the same predecessor. The other predecessor
    # is excluded by the hard 20-degree axial-orientation gate.
    inner = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 40.0]])
    outer = np.array([[8.0, 0.0, 5.0], [8.0, 1.0, 6.0]])
    costs = np.full((outer.shape[0], inner.shape[0]), np.inf)
    for outer_index, (outer_row, outer_col, outer_axis) in enumerate(outer):
        for inner_index, (inner_row, inner_col, inner_axis) in enumerate(inner):
            distance = float(np.hypot(outer_row - inner_row, outer_col - inner_col))
            delta = axial_difference(outer_axis, inner_axis)
            if distance <= max_distance and abs(delta) <= max_angle:
                costs[outer_index, inner_index] = normalized_cost(
                    distance,
                    delta,
                    max_distance,
                    max_angle,
                )

    many_to_one = np.argmin(costs, axis=1)
    assert np.array_equal(many_to_one, [0, 0])

    outer_best = np.argmin(costs, axis=1)
    inner_best = np.argmin(costs, axis=0)
    reciprocal = [
        (outer_index, inner_index)
        for outer_index, inner_index in enumerate(outer_best)
        if np.isfinite(costs[outer_index, inner_index])
        and inner_best[inner_index] == outer_index
    ]
    assert reciprocal == [(0, 0)]

    selected, ambiguous = global_assignment(costs)
    assert selected == set()
    assert ambiguous == {0}

    # The 0.05 threshold is inclusive for both local and global alternatives.
    local_costs = np.array([0.10, 0.15, 0.150001])
    assert local_costs[1] - local_costs[0] <= 0.05
    assert local_costs[2] - local_costs[0] > 0.05

    close_global = np.array([[0.10, 0.12], [0.12, 0.10]])
    selected, ambiguous = global_assignment(close_global)
    assert selected == set()
    assert ambiguous == {0, 1}

    decisive_global = np.array([[0.10, 0.90], [0.90, 0.10]])
    selected, ambiguous = global_assignment(decisive_global)
    assert selected == {(0, 0), (1, 1)}
    assert not ambiguous


def verify_point_ring_orientation_matching() -> None:
    """Assert seam, policy, guard, connectivity, and inheritance invariants."""
    assert np.isclose(axial_difference(-85.0, 80.0), 15.0)
    assert np.isclose(axial_difference(190.0, 10.0), 0.0)
    assert np.isclose(abs(axial_difference(90.0, 0.0)), 90.0)
    verify_unwrap_and_guards()
    verify_connectivity_and_no_restart()
    verify_matching_policies()
    print(
        "PASS: point-ring seam handling, history guards, connectivity, "
        "matching policies, ambiguity, and strict inheritance are consistent"
    )


if __name__ == "__main__":
    verify_point_ring_orientation_matching()
