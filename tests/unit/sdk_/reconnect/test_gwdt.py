"""Tests for the APP2-derived grey-weighted distance helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from phenotypic.sdk_.reconnect._gwdt import app2_gwdt_cost, grey_weighted_distance


FIXTURE = (
    Path(__file__).parents[3] / "fixtures" / "reconnect" / "gwdt" / "app2_source.npz"
)


def _neighbors(connectivity: int) -> tuple[tuple[int, int, float], ...]:
    neighbors = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            offset = abs(row_offset) + abs(column_offset)
            if offset == 0 or (connectivity == 4 and offset != 1):
                continue
            neighbors.append((row_offset, column_offset, math.sqrt(offset)))
    return tuple(neighbors)


def _source_phase_oracle(
    image: np.ndarray,
    background: np.ndarray,
    connectivity: int,
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    distance = np.full(values.shape, np.inf)
    distance[background] = values[background]
    rows, columns = values.shape
    trial = np.zeros(values.shape, dtype=bool)
    edges = _neighbors(connectivity)
    for seed_row, seed_column in np.argwhere(background):
        for row_offset, column_offset, _ in edges:
            row = int(seed_row) + row_offset
            column = int(seed_column) + column_offset
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            if background[row, column] or trial[row, column]:
                continue
            minimum_row = int(seed_row)
            minimum_column = int(seed_column)
            if distance[minimum_row, minimum_column] > 0.0:
                for alive_row_offset, alive_column_offset, _ in edges:
                    alive_row = row + alive_row_offset
                    alive_column = column + alive_column_offset
                    if (
                        0 <= alive_row < rows
                        and 0 <= alive_column < columns
                        and background[alive_row, alive_column]
                        and distance[alive_row, alive_column]
                        < distance[minimum_row, minimum_column]
                    ):
                        minimum_row = alive_row
                        minimum_column = alive_column
            distance[row, column] = (
                distance[minimum_row, minimum_column] + values[row, column]
            )
            trial[row, column] = True

    for _ in range(values.size - 1):
        changed = False
        previous = distance.copy()
        for row in range(rows):
            for column in range(columns):
                if background[row, column]:
                    continue
                for row_offset, column_offset, length in edges:
                    source_row = row + row_offset
                    source_column = column + column_offset
                    if (
                        0 <= source_row < rows
                        and 0 <= source_column < columns
                    ):
                        candidate = (
                            previous[source_row, source_column]
                            + values[row, column] * length
                        )
                        if candidate < distance[row, column]:
                            distance[row, column] = candidate
                            changed = True
        if not changed:
            break
    return distance


def _float32_path_tolerance(image: np.ndarray) -> float:
    path_terms = max(1, image.size - 1)
    epsilon = np.finfo(np.float32).eps
    gamma = path_terms * epsilon / (1.0 - path_terms * epsilon)
    largest_path = path_terms * float(np.max(image)) * math.sqrt(2.0)
    return gamma * largest_path


@pytest.mark.parametrize("connectivity", [4, 8])
def test_gwdt_matches_source_generated_fixture(connectivity: int):
    with np.load(FIXTURE) as fixture:
        actual = grey_weighted_distance(
            fixture["image"],
            fixture["background"],
            connectivity=connectivity,
        )
        expected = fixture[f"source_standard_distance_{connectivity}"]

    np.testing.assert_array_equal(actual, expected)


def test_cost_matches_source_lookup_exactly():
    with np.load(FIXTURE) as fixture:
        distance = fixture["source_standard_distance_8"]
        expected = fixture["source_standard_cost_8"]
    actual = app2_gwdt_cost(distance)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("case_name", "image_key", "background_key"),
    [
        ("diagonal", "diagonal_image", "diagonal_background"),
        ("threshold", "threshold_image", "threshold_background"),
        ("all_background", "all_background_image", "all_background"),
        ("no_background", "no_background_image", "no_background"),
        (
            "post_frontier_diagonal",
            "post_frontier_diagonal_image",
            "post_frontier_diagonal_background",
        ),
    ],
)
@pytest.mark.parametrize("connectivity", [4, 8])
def test_expanded_source_cases_match_exactly(
    case_name: str,
    image_key: str,
    background_key: str,
    connectivity: int,
):
    with np.load(FIXTURE) as fixture:
        actual = grey_weighted_distance(
            fixture[image_key],
            fixture[background_key],
            connectivity=connectivity,
        )
        expected = fixture[f"source_{case_name}_distance_{connectivity}"]
    np.testing.assert_array_equal(actual, expected)


def test_fixture_is_load_bearing_for_source_versus_destination_intensity():
    with np.load(FIXTURE) as fixture:
        image = fixture["image"].astype(np.float64)
        expected = fixture["source_standard_distance_4"]
    background = image == 0
    source_weighted = np.full(image.shape, np.inf)
    source_weighted[background] = 0.0
    rows, columns = image.shape
    for _ in range(image.size - 1):
        previous = source_weighted.copy()
        for row in range(rows):
            for column in range(columns):
                if background[row, column]:
                    continue
                for row_offset, column_offset, length in _neighbors(4):
                    source_row = row + row_offset
                    source_column = column + column_offset
                    if 0 <= source_row < rows and 0 <= source_column < columns:
                        source_weighted[row, column] = min(
                            source_weighted[row, column],
                            previous[source_row, source_column]
                            + image[source_row, source_column] * length,
                        )

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(source_weighted, expected, rtol=0.0, atol=1e-12)


def test_documented_reference_initialization_fork_is_pinned():
    with np.load(FIXTURE) as fixture:
        image = fixture["diagonal_image"]
        background = fixture["diagonal_background"]
        source = fixture["source_diagonal_distance_8"]
    actual = grey_weighted_distance(image, background, connectivity=8)

    assert source[1, 1] == pytest.approx(1.0)
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_heap_matches_independent_two_phase_oracle(connectivity: int):
    image = np.array(
        [[0.0, 3.0, 5.0, 2.0], [2.0, 8.0, 1.0, 4.0], [6.0, 1.0, 7.0, 0.0]]
    )
    background = image == 0.0
    expected = _source_phase_oracle(image, background, connectivity)

    actual = grey_weighted_distance(image, background, connectivity=connectivity)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=_float32_path_tolerance(image),
    )


def test_one_dimensional_cumulative_sum():
    image = np.array([[0.0, 2.0, 3.0, 4.0]])
    actual = grey_weighted_distance(image, image == 0.0, connectivity=4)
    np.testing.assert_array_equal(actual, [[0.0, 2.0, 5.0, 9.0]])


def test_frontier_omits_diagonal_length_but_later_step_uses_it():
    image = np.ones((3, 3), dtype=np.float64)
    image[1, 1] = 0.0
    background = np.zeros_like(image, dtype=bool)
    background[1, 1] = True
    actual = grey_weighted_distance(image, background, connectivity=8)

    assert actual[1, 2] == pytest.approx(1.0)
    assert actual[0, 0] == pytest.approx(1.0)

    post_frontier = np.array(
        [[0.0, 100.0, 100.0], [100.0, 1.0, 100.0], [100.0, 100.0, 1.0]]
    )
    post_distance = grey_weighted_distance(
        post_frontier,
        post_frontier == 0.0,
        connectivity=8,
    )
    assert post_distance[2, 2] == pytest.approx(
        np.float32(1.0 + math.sqrt(2.0))
    )


def test_multiple_background_seeds_take_pointwise_minimum():
    image = np.ones((1, 5), dtype=np.float64)
    image[0, [0, 4]] = 0.0
    background = np.array([[True, False, False, False, True]])
    np.testing.assert_array_equal(
        grey_weighted_distance(image, background, connectivity=4),
        [[0.0, 1.0, 2.0, 1.0, 0.0]],
    )


def test_two_route_case_chooses_lower_cumulative_intensity():
    image = np.array([[1.0, 1.0, 1.0], [0.0, 10.0, 1.0]])
    background = np.zeros_like(image, dtype=bool)
    background[1, 0] = True
    actual = grey_weighted_distance(image, background, connectivity=8)
    assert actual[1, 2] == pytest.approx(1.0 + math.sqrt(2.0))


def test_intensity_and_added_seed_monotonicity():
    image = np.array([[0.0, 1.0, 2.0, 3.0]])
    background = np.array([[True, False, False, False]])
    baseline = grey_weighted_distance(image, background, connectivity=4)

    increased = image.copy()
    increased[0, 2] += 5.0
    assert np.all(
        grey_weighted_distance(increased, background, connectivity=4) >= baseline
    )

    more_seed_image = image.copy()
    more_seed_image[0, 3] = 0.0
    more_seeds = more_seed_image == 0.0
    more_seeds[0, 3] = True
    assert np.all(
        grey_weighted_distance(more_seed_image, more_seeds, connectivity=4) <= baseline
    )


def test_positive_scaling():
    image = np.array([[0.0, 1.5, 2.5], [3.0, 4.0, 5.0]])
    background = image == 0.0
    baseline = grey_weighted_distance(image, background, connectivity=8)
    scaled = grey_weighted_distance(7.0 * image, background, connectivity=8)
    np.testing.assert_allclose(scaled, 7.0 * baseline, rtol=3e-15, atol=3e-15)


@pytest.mark.parametrize(
    "transform",
    [np.transpose, np.fliplr, np.flipud, lambda array: np.rot90(array, 1)],
)
def test_grid_equivariance(transform):
    image = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0], [2.0, 6.0, 0.0]])
    background = image == 0.0
    baseline = grey_weighted_distance(image, background, connectivity=8)

    transformed = grey_weighted_distance(
        transform(image),
        transform(background),
        connectivity=8,
    )

    np.testing.assert_allclose(transformed, transform(baseline), rtol=0.0, atol=1e-14)


def test_four_connectivity_is_not_smaller_than_eight():
    image = np.array([[0.0, 2.0, 5.0], [3.0, 1.0, 7.0], [6.0, 4.0, 8.0]])
    background = image == 0.0
    distance_4 = grey_weighted_distance(image, background, connectivity=4)
    distance_8 = grey_weighted_distance(image, background, connectivity=8)
    assert np.all(distance_4 >= distance_8)


def test_thick_bar_has_center_depth_and_inverse_cost_preference():
    image = np.zeros((5, 7), dtype=np.float64)
    image[1:4, :] = 1.0
    background = image == 0.0
    distance = grey_weighted_distance(image, background, connectivity=4)
    cost = app2_gwdt_cost(distance)

    assert np.all(distance[2] > distance[1])
    assert np.all(cost[2] < cost[1])
    assert np.all(cost[1] < cost[0])


def test_all_background_preserves_source_intensity():
    image = np.array([[2.0, 3.0], [4.0, 5.0]])
    background = np.ones_like(image, dtype=bool)
    np.testing.assert_array_equal(
        grey_weighted_distance(image, background),
        image.astype(np.float32),
    )


def test_no_background_returns_source_sentinel():
    image = np.ones((2, 2))
    actual = grey_weighted_distance(image, np.zeros_like(image, dtype=bool))
    np.testing.assert_array_equal(actual, np.float32(1e20))


def test_cost_constant_map_is_one():
    np.testing.assert_array_equal(app2_gwdt_cost(np.full((2, 2), 3.0)), 1.0)


def test_cost_repairs_source_bounds_scan_for_strictly_increasing_map():
    distance = np.arange(1.0, 5.0).reshape(2, 2)
    expected = np.array([[22026.5, 85.1526], [3.03773, 1.0]])
    np.testing.assert_array_equal(app2_gwdt_cost(distance), expected)


def test_helpers_do_not_mutate_inputs():
    image = np.array([[0.0, 1.0], [2.0, 3.0]])
    background = image == 0.0
    original_image = image.copy()
    original_background = background.copy()
    distance = grey_weighted_distance(image, background)
    original_distance = distance.copy()
    app2_gwdt_cost(distance)

    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(background, original_background)
    np.testing.assert_array_equal(distance, original_distance)


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.empty((0, 2)), "empty"),
        (np.ones((2, 2, 1)), "two-dimensional"),
        (np.array([[0.0, -1.0]]), "nonnegative"),
        (np.array([[0.0, np.nan]]), "finite"),
        (np.array([[0.0, np.inf]]), "finite"),
    ],
)
def test_invalid_image_values_are_rejected(image: np.ndarray, match: str):
    background = np.zeros(image.shape, dtype=bool)
    with pytest.raises(ValueError, match=match):
        grey_weighted_distance(image, background)


@pytest.mark.parametrize(
    "image",
    [np.array([[False, True]]), np.array([[0 + 0j, 1 + 0j]]), np.array([["0", "1"]])],
)
def test_non_real_numeric_image_dtypes_are_rejected(image: np.ndarray):
    with pytest.raises(TypeError, match="real numeric dtype"):
        grey_weighted_distance(image, np.array([[True, False]]))


def test_invalid_background_is_rejected():
    image = np.ones((2, 2))
    with pytest.raises(TypeError, match="boolean"):
        grey_weighted_distance(image, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="same two-dimensional shape"):
        grey_weighted_distance(image, np.zeros((2, 3), dtype=bool))


@pytest.mark.parametrize("connectivity", [0, 1, 6, 16, "8", 4.0, 8.0, True])
def test_invalid_connectivity_is_rejected(connectivity):
    image = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="4 or 8"):
        grey_weighted_distance(image, image == 0.0, connectivity=connectivity)


def test_array_container_is_required():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        grey_weighted_distance([[0.0, 1.0]], np.array([[True, False]]))
    with pytest.raises(TypeError, match="numpy.ndarray"):
        grey_weighted_distance(np.array([[0.0, 1.0]]), [[True, False]])
    with pytest.raises(TypeError, match="numpy.ndarray"):
        app2_gwdt_cost([[0.0, 1.0]])
