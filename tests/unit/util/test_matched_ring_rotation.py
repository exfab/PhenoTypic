"""Tests for nearby-sector matched cumulative ring rotation."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.util._matched_ring_rotation import (
    matched_ring_cumulative_rotation_profile,
    matched_tracks_to_ring_sector_values,
)


def _moving_corridor() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a four-ring fiber corridor that advances one sector per ring."""
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    orientation = np.full((4, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    sectors = np.arange(4)
    orientation[np.arange(4), sectors] = np.deg2rad([0.0, 10.0, 20.0, 30.0])
    resultant[np.arange(4), sectors] = 1.0
    return radii, orientation, resultant


def test_matched_profile_follows_moving_corridor_and_accumulates_rotation():
    radii, orientation, resultant = _moving_corridor()

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert np.array_equal(paths[:, 0], [0, 1, 2, 3])
    assert np.degrees(cumulative[:, 0]) == pytest.approx(
        [0.0, 10.0, 20.0, 30.0]
    )


def test_matched_profile_unwraps_axial_seam():
    radii = np.array([20.0, 28.0, 36.0, 44.0])
    orientation = np.full((4, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    sectors = np.array([8, 9, 10, 11])
    orientation[np.arange(4), sectors] = np.deg2rad(
        [80.0, -89.0, -78.0, -67.0]
    )
    resultant[np.arange(4), sectors] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert np.array_equal(paths[:, 8], sectors)
    assert np.degrees(cumulative[:, 8]) == pytest.approx(
        [0.0, 11.0, 22.0, 33.0]
    )


def test_matched_profile_uses_radial_position_predictor_between_candidates():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    sector_width = 2.0 * np.pi / 36.0
    radial_tilt = np.arctan(sector_width / np.log(radii[1] / radii[0]))
    previous = sector_width / 2.0 + radial_tilt
    orientation[0, 0] = previous
    resultant[0, 0] = 1.0
    orientation[1, 0] = previous
    orientation[1, 1] = previous + sector_width / 2.0
    resultant[1, :2] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert paths[1, 0] == 1
    assert np.degrees(cumulative[1, 0]) == pytest.approx(5.0)


def test_matched_profile_uses_orientation_continuity_between_candidates():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    previous = np.deg2rad(5.0)
    orientation[0, 0] = previous
    resultant[0, 0] = 1.0
    orientation[1, 0] = previous + np.deg2rad(20.0)
    orientation[1, 1] = previous
    resultant[1, :2] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert paths[1, 0] == 1
    assert np.degrees(cumulative[1, 0]) == pytest.approx(0.0)


def test_matched_profile_uses_reliability_to_break_equal_geometric_cost():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    half_sector = np.pi / 36.0
    radial_tilt = np.arctan(half_sector / np.log(radii[1] / radii[0]))
    previous = np.pi / 36.0 + radial_tilt
    orientation[0, 0] = previous
    resultant[0, 0] = 1.0
    orientation[1, :2] = previous
    resultant[1, 0] = 0.2
    resultant[1, 1] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert paths[1, 0] == 1
    assert cumulative[1, 0] == pytest.approx(0.0)


def test_matched_profile_terminates_near_tangential_predictor():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(85.0)
    resultant[0, 0] = 1.0
    orientation[1, 1] = np.deg2rad(85.0)
    resultant[1, 1] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert np.isnan(cumulative[1, 0])
    assert paths[1, 0] == -1


def test_matched_profile_terminates_when_prediction_exceeds_search_window():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(75.0)
    resultant[0, 0] = 1.0
    orientation[1, 2] = np.deg2rad(75.0)
    resultant[1, 2] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert np.isnan(cumulative[1, 0])
    assert paths[1, 0] == -1


def test_matched_profile_does_not_bridge_missing_ring():
    radii, orientation, resultant = _moving_corridor()
    orientation[1] = np.nan
    resultant[1] = np.nan

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert cumulative[0, 0] == pytest.approx(0.0)
    assert np.isnan(cumulative[1:, 0]).all()
    assert np.array_equal(paths[:, 0], [0, -1, -1, -1])


def test_matched_profile_can_bridge_missing_ring_without_filling_gap():
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(5.0)
    orientation[2, 0] = np.deg2rad(25.0)
    resultant[0, 0] = 1.0
    resultant[2, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )

    assert np.array_equal(paths[:, 0], [0, -1, 0])
    assert cumulative[0, 0] == pytest.approx(0.0)
    assert np.isnan(cumulative[1, 0])
    assert np.degrees(cumulative[2, 0]) == pytest.approx(20.0)


def test_gap_bridge_predictor_uses_complete_radial_interval():
    radii = np.array([20.0, 28.0, 44.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    sector_width = 2.0 * np.pi / 36.0
    predicted_step = np.deg2rad(8.0)
    radial_tilt = np.arctan(predicted_step / np.log(radii[2] / radii[0]))
    previous = sector_width / 2.0 + radial_tilt
    orientation[0, 0] = previous
    resultant[0, 0] = 1.0
    orientation[2, :2] = previous
    resultant[2, :2] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )

    assert paths[2, 0] == 1
    assert cumulative[2, 0] == pytest.approx(0.0)


def test_gap_bridge_does_not_skip_hard_axial_ambiguity():
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[:, 0] = np.deg2rad([5.0, -85.0, 15.0])
    resultant[:, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
    )

    assert np.array_equal(paths[:, 0], [0, -1, -1])
    assert np.isnan(cumulative[1:, 0]).all()


def test_matched_profile_can_restart_after_gap_with_zeroed_segment():
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(5.0)
    orientation[2, 0] = np.deg2rad(25.0)
    resultant[0, 0] = 1.0
    resultant[2, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_restarts=True,
    )

    assert np.array_equal(paths[:, 0], [0, -1, 0])
    assert cumulative[0, 0] == pytest.approx(0.0)
    assert np.isnan(cumulative[1, 0])
    assert cumulative[2, 0] == pytest.approx(0.0)


def test_gap_bridging_takes_precedence_when_both_relaxations_are_enabled():
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = np.deg2rad(5.0)
    orientation[2, 0] = np.deg2rad(25.0)
    resultant[0, 0] = 1.0
    resultant[2, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_gap_bridging=True,
        allow_restarts=True,
    )

    assert np.array_equal(paths[:, 0], [0, -1, 0])
    assert np.degrees(cumulative[2, 0]) == pytest.approx(20.0)


def test_restart_can_begin_after_hard_tangential_cutoff():
    radii = np.array([20.0, 28.0, 36.0])
    orientation = np.full((3, 36), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[:, 0] = np.deg2rad([85.0, 5.0, 15.0])
    resultant[:, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        allow_restarts=True,
    )

    assert np.array_equal(paths[:, 0], [0, 0, 0])
    assert np.degrees(cumulative[:, 0]) == pytest.approx([0.0, 0.0, 10.0])


def test_matched_profile_respects_maximum_sector_shift():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 12), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[0, 0] = 0.0
    resultant[0, 0] = 1.0
    orientation[1, 3] = np.deg2rad(15.0)
    resultant[1, 3] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
        max_sector_shift=2,
    )

    assert cumulative[0, 0] == pytest.approx(0.0)
    assert np.isnan(cumulative[1, 0])
    assert np.array_equal(paths[:, 0], [0, -1])


def test_matched_profile_blanks_orthogonal_ambiguous_step():
    radii = np.array([20.0, 28.0])
    orientation = np.full((2, 8), np.nan)
    resultant = np.full_like(orientation, np.nan)
    orientation[:, 0] = np.deg2rad([0.0, 90.0])
    resultant[:, 0] = 1.0

    cumulative, paths = matched_ring_cumulative_rotation_profile(
        radii,
        orientation,
        resultant,
    )

    assert cumulative[0, 0] == pytest.approx(0.0)
    assert np.isnan(cumulative[1, 0])
    assert np.array_equal(paths[:, 0], [0, -1])


def test_matched_tracks_project_strongest_collision_to_ring_sector():
    cumulative = np.array(
        [
            [0.0, 0.0, np.nan],
            [0.1, -0.3, 0.2],
        ]
    )
    paths = np.array(
        [
            [0, 1, -1],
            [2, 2, 1],
        ]
    )

    field = matched_tracks_to_ring_sector_values(cumulative, paths)

    assert field[0, 0] == pytest.approx(0.0)
    assert field[0, 1] == pytest.approx(0.0)
    assert np.isnan(field[0, 2])
    assert field[1, 1] == pytest.approx(0.2)
    assert field[1, 2] == pytest.approx(-0.3)


@pytest.mark.parametrize(
    "paths",
    [
        np.array([[0.5]]),
        np.array([[-2]]),
        np.array([[1]]),
    ],
)
def test_matched_tracks_reject_invalid_path_indices(paths):
    with pytest.raises(ValueError, match="path sector indices"):
        matched_tracks_to_ring_sector_values(np.array([[0.0]]), paths)


@pytest.mark.parametrize(
    ("radii", "orientation", "resultant", "message"),
    [
        (
            np.array([2.0, 1.0]),
            np.zeros((2, 1)),
            np.ones((2, 1)),
            "increasing",
        ),
        (
            np.array([1.0, 2.0]),
            np.zeros((2, 1)),
            np.ones((1, 2)),
            "must match",
        ),
        (
            np.array([1.0, 2.0]),
            np.zeros((2, 1)),
            np.full((2, 1), 1.1),
            r"in \[0, 1\]",
        ),
    ],
)
def test_matched_profile_rejects_invalid_inputs(
    radii,
    orientation,
    resultant,
    message,
):
    with pytest.raises(ValueError, match=message):
        matched_ring_cumulative_rotation_profile(
            radii,
            orientation,
            resultant,
        )


@pytest.mark.parametrize(
    "parameter",
    ["allow_gap_bridging", "allow_restarts"],
)
def test_matched_profile_rejects_non_boolean_relaxation_flags(parameter):
    radii = np.array([1.0, 2.0])
    orientation = np.zeros((2, 1))
    resultant = np.ones((2, 1))

    with pytest.raises(ValueError, match="must be a boolean"):
        matched_ring_cumulative_rotation_profile(
            radii,
            orientation,
            resultant,
            **{parameter: 1},
        )
