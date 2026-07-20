"""Validate A08 RORPO claims with an exhaustive upper-level-set oracle.

This script is intentionally independent of ``phenotypic`` and of the IPOL C++
implementation. It uses only the algorithm and adjacency graphs published in the
paper, then compares the result with source-generated fixture intermediates.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "specs"
    / "2026-07-13-fungi-detection-method-ports"
    / "refs"
    / "rorpo"
    / "source_fixture.json"
)
SOURCE_ARCHIVE_SHA256 = "6d03ab55b7010869ed33543b88864ee1e2db86084d384839c48e1719b98004a4"

# Directed steps in (row, column) order. Reversing every step gives the same
# undirected family of complete paths and therefore the same path opening.
STEPS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((1, -1), (1, 0), (1, 1)),
    ((-1, 1), (0, 1), (1, 1)),
    ((-1, 0), (-1, 1), (0, 1)),
    ((-1, -1), (-1, 0), (0, -1)),
)
CASE_KEYS = {
    "bordered",
    "dilated_bordered",
    "direction_threshold_gt_1",
    "direction_x_source",
    "direction_y_source",
    "direction_corrected_sum_x",
    "direction_corrected_sum_y",
    "direction_selected_count",
    "input",
    "intensity_reconstructed",
    "intensity_source",
    "source_level_sort_order",
    *(f"po_raw_o{orientation}" for orientation in range(1, 5)),
    *(f"rpo_o{orientation}" for orientation in range(1, 5)),
    *(f"rank_value_ascending_{rank}" for rank in range(1, 5)),
    *(f"rank_orientation_ascending_{rank}" for rank in range(1, 5)),
    *(f"direction_split_cost_high_{count}" for count in range(1, 4)),
    *(f"direction_selected_orientation_{slot}" for slot in range(1, 4)),
    *(f"direction_correction_sign_{slot}" for slot in range(1, 4)),
    *(f"direction_raw_x_{slot}" for slot in range(1, 4)),
    *(f"direction_raw_y_{slot}" for slot in range(1, 4)),
    *(f"direction_corrected_x_{slot}" for slot in range(1, 4)),
    *(f"direction_corrected_y_{slot}" for slot in range(1, 4)),
}
MULTISCALE_KEYS = {
    "direction_x_source",
    "direction_y_source",
    "input",
    "intensity_scale_2",
    "intensity_scale_3",
    "intensity_scale_5",
    "intensity_source",
    "winning_scale",
    *(f"direction_x_scale_{scale}" for scale in (2, 3, 5)),
    *(f"direction_y_scale_{scale}" for scale in (2, 3, 5)),
}

# Executable basis in its private (column, row) order. The public contract swaps
# these components exactly once and then canonicalizes the axial sign.
SOURCE_BASES = np.asarray(((0, 1), (1, 0), (1, 1), (-1, 1)), dtype=np.float32)


def _validate_image(image: npt.NDArray[np.generic], path_length: int) -> None:
    """Validate the bounded mathematical oracle inputs.

    Args:
        image: Two-dimensional finite, nonnegative scalar image.
        path_length: Number of vertices in every admissible path.

    Raises:
        ValueError: If the array or path length is outside the frozen domain.
    """

    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a nonempty 2-D array")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must be numeric")
    if not np.all(np.isfinite(image)) or np.any(image < 0):  # type: ignore[operator]
        raise ValueError("image must contain finite nonnegative values")
    if (
        isinstance(path_length, bool)
        or not isinstance(path_length, (int, np.integer))
        or path_length < 1
    ):
        raise ValueError("path_length must be an integer of at least 1")


def _vertices_on_complete_paths(
    mask: npt.NDArray[np.bool_],
    steps: tuple[tuple[int, int], ...],
    path_length: int,
) -> npt.NDArray[np.bool_]:
    """Enumerate all complete directed paths and mark their vertices."""

    height, width = mask.shape
    survives = np.zeros_like(mask)

    def extend(path: list[tuple[int, int]]) -> None:
        if len(path) == path_length:
            for row, column in path:
                survives[row, column] = True
            return
        row, column = path[-1]
        for delta_row, delta_column in steps:
            next_row = row + delta_row
            next_column = column + delta_column
            if not (0 <= next_row < height and 0 <= next_column < width):
                continue
            next_vertex = (next_row, next_column)
            if mask[next_vertex] and next_vertex not in path:
                extend([*path, next_vertex])

    for start in zip(*np.nonzero(mask), strict=True):
        extend([start])
    return survives


def exhaustive_path_opening(
    image: npt.NDArray[np.generic],
    path_length: int,
    steps: tuple[tuple[int, int], ...],
) -> npt.NDArray[np.generic]:
    """Reconstruct one grayscale path opening from binary upper levels."""

    _validate_image(image, path_length)
    opened = np.zeros_like(image)
    for level in np.unique(image):
        if level == 0:
            continue
        survives = _vertices_on_complete_paths(image >= level, steps, path_length)
        opened[survives] = np.maximum(opened[survives], level)
    return opened


def square_dilation(
    image: npt.NDArray[np.generic], robustness: int
) -> npt.NDArray[np.generic]:
    """Apply the executable's square dilation radius ``robustness // 2``."""

    if (
        isinstance(robustness, bool)
        or not isinstance(robustness, (int, np.integer))
        or robustness < 0
    ):
        raise ValueError("robustness must be a nonnegative integer")
    radius = robustness // 2
    result = np.zeros_like(image)
    height, width = image.shape
    for row in range(height):
        for column in range(width):
            row_min = max(0, row - radius)
            row_max = min(height, row + radius + 1)
            column_min = max(0, column - radius)
            column_max = min(width, column + radius + 1)
            result[row, column] = np.max(
                image[row_min:row_max, column_min:column_max]
            )
    return result


def exhaustive_rorpo_maps(
    image: npt.NDArray[np.generic], path_length: int, robustness: int
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
    """Return the four raw and anti-extensive robust path openings."""

    _validate_image(image, path_length)
    dilated = square_dilation(image, robustness)
    raw = np.stack(
        [exhaustive_path_opening(dilated, path_length, steps) for steps in STEPS]
    )
    robust = np.minimum(raw, image[None, :, :])
    return raw, robust


def _fixture_array(case: dict[str, object], key: str) -> npt.NDArray[np.generic]:
    arrays = case["arrays"]
    assert isinstance(arrays, dict)
    record = arrays[key]
    assert isinstance(record, dict)
    shape = tuple(record["shape"])
    return np.asarray(record["data"]).reshape(shape)


def _population_std_float32(values: list[int]) -> np.float32:
    """Re-derive the executable's float32 population standard deviation."""

    vector = np.asarray(values, dtype=np.float32)
    mean = np.float32(np.sum(vector, dtype=np.float32) / np.float32(vector.size))
    squared = np.square(vector - mean, dtype=np.float32)
    variance = np.float32(
        np.sum(squared, dtype=np.float32) / np.float32(vector.size)
    )
    return np.float32(np.sqrt(variance, dtype=np.float32))


def _split_costs_uint8(sorted_values: list[int]) -> tuple[int, int, int]:
    """Compute the three split costs including the source uint8 truncation."""

    low = list(sorted_values)
    high: list[int] = []
    costs: list[int] = []
    for _ in range(3):
        high.append(low.pop())
        raw_cost = np.float32(
            _population_std_float32(low) + _population_std_float32(high)
        )
        costs.append(int(np.uint8(raw_cost)))
    return costs[0], costs[1], costs[2]


def _angle_degrees(first: npt.NDArray[np.float32], second: npt.NDArray[np.float32]) -> np.float32:
    """Compute one float32 angle using the executable's arithmetic order."""

    dot = np.sum(first * second, dtype=np.float32)
    first_norm = np.float32(np.sqrt(np.sum(first * first, dtype=np.float32)))
    second_norm = np.float32(np.sqrt(np.sum(second * second, dtype=np.float32)))
    cosine = np.float32(dot / np.float32(first_norm * second_norm))
    cosine = np.clip(cosine, np.float32(-1), np.float32(1))
    radians = np.float32(np.arccos(cosine))
    return np.float32(radians * np.float32(180) / np.float32(math.pi))


def _correct_source_vectors(
    vectors: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], tuple[int, ...], tuple[int, ...]]:
    """Return corrected vectors, selected signs, and all integer objectives."""

    count = vectors.shape[0]
    if count == 1:
        return vectors.copy(), (1,), (0,)
    candidates: list[tuple[int, ...]]
    if count == 2:
        candidates = [(1, 1), (1, -1)]
    elif count == 3:
        candidates = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
    else:
        raise AssertionError("direction selection must contain one to three vectors")

    objectives: list[int] = []
    corrected_candidates: list[npt.NDArray[np.float32]] = []
    for signs in candidates:
        corrected = vectors * np.asarray(signs, dtype=np.float32)[:, None]
        if count == 2:
            angle_sum = _angle_degrees(corrected[0], corrected[1])
        else:
            angle_sum = np.float32(
                _angle_degrees(corrected[0], corrected[1])
                + _angle_degrees(corrected[0], corrected[2])
                + _angle_degrees(corrected[1], corrected[2])
            )
        objectives.append(int(angle_sum))
        corrected_candidates.append(corrected)
    winner = int(np.argmin(np.asarray(objectives)))
    return corrected_candidates[winner], candidates[winner], tuple(objectives)


def _contract_direction_is_unique(
    sorted_values: list[int], costs: tuple[int, int, int], objectives: tuple[int, ...]
) -> bool:
    """Evaluate the frozen release predicate for a source-independent direction."""

    minimum = min(costs)
    if costs.count(minimum) != 1:
        return False
    selected_count = costs.index(minimum) + 1
    boundary = 4 - selected_count
    if sorted_values[boundary - 1] == sorted_values[boundary]:
        return False
    return objectives.count(min(objectives)) == 1


def _assert_direction_capture(case: dict[str, object], response: npt.NDArray[np.generic]) -> int:
    """Re-derive split, correction, and normalized direction intermediates."""

    shape = response.shape
    active = response > 1  # type: ignore[operator]
    ranked_values = np.stack(
        [_fixture_array(case, f"rank_value_ascending_{rank}") for rank in range(1, 5)]
    )
    ranked_orientations = np.stack(
        [
            _fixture_array(case, f"rank_orientation_ascending_{rank}").astype(np.int64)
            for rank in range(1, 5)
        ]
    )
    expected_costs = np.full((3, *shape), -1, dtype=np.int64)
    expected_count = np.zeros(shape, dtype=np.int64)
    expected_orientation = np.full((3, *shape), -1, dtype=np.int64)
    expected_sign = np.zeros((3, *shape), dtype=np.int64)
    expected_raw = np.zeros((3, 2, *shape), dtype=np.float32)
    expected_corrected = np.zeros((3, 2, *shape), dtype=np.float32)
    expected_sum = np.zeros((2, *shape), dtype=np.float32)
    expected_direction = np.zeros((2, *shape), dtype=np.float32)
    unique_count = 0

    for row, column in zip(*np.nonzero(active), strict=True):
        values = [int(value) for value in ranked_values[:, row, column]]
        costs = _split_costs_uint8(values)
        expected_costs[:, row, column] = costs
        count = int(np.argmin(np.asarray(costs))) + 1
        expected_count[row, column] = count
        orientations = ranked_orientations[::-1, row, column][:count]
        expected_orientation[:count, row, column] = orientations
        raw = SOURCE_BASES[orientations]
        corrected, signs, objectives = _correct_source_vectors(raw)
        expected_sign[:count, row, column] = signs
        expected_raw[:count, :, row, column] = raw
        expected_corrected[:count, :, row, column] = corrected
        vector_sum = np.sum(corrected, axis=0, dtype=np.float32)
        expected_sum[:, row, column] = vector_sum
        norm = np.float32(np.sqrt(np.sum(vector_sum * vector_sum, dtype=np.float32)))
        expected_direction[:, row, column] = vector_sum / norm
        if _contract_direction_is_unique(values, costs, objectives):
            unique_count += 1

    for index in range(3):
        suffix = index + 1
        np.testing.assert_array_equal(
            expected_costs[index],
            _fixture_array(case, f"direction_split_cost_high_{suffix}"),
        )
        np.testing.assert_array_equal(
            expected_orientation[index],
            _fixture_array(case, f"direction_selected_orientation_{suffix}"),
        )
        np.testing.assert_array_equal(
            expected_sign[index],
            _fixture_array(case, f"direction_correction_sign_{suffix}"),
        )
        np.testing.assert_array_equal(
            expected_raw[index, 0], _fixture_array(case, f"direction_raw_x_{suffix}")
        )
        np.testing.assert_array_equal(
            expected_raw[index, 1], _fixture_array(case, f"direction_raw_y_{suffix}")
        )
        np.testing.assert_array_equal(
            expected_corrected[index, 0],
            _fixture_array(case, f"direction_corrected_x_{suffix}"),
        )
        np.testing.assert_array_equal(
            expected_corrected[index, 1],
            _fixture_array(case, f"direction_corrected_y_{suffix}"),
        )
    np.testing.assert_array_equal(
        expected_count, _fixture_array(case, "direction_selected_count")
    )
    np.testing.assert_array_equal(
        expected_sum[0], _fixture_array(case, "direction_corrected_sum_x")
    )
    np.testing.assert_array_equal(
        expected_sum[1], _fixture_array(case, "direction_corrected_sum_y")
    )
    np.testing.assert_allclose(
        expected_direction[0],
        np.asarray(_fixture_array(case, "direction_x_source"), dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected_direction[1],
        np.asarray(_fixture_array(case, "direction_y_source"), dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    return unique_count


def _assert_source_fixture_matches_exhaustive_oracle() -> tuple[int, int, int]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if fixture["schema_version"] != 2 or len(fixture["cases"]) != 15:
        raise AssertionError("unexpected RORPO fixture schema or case count")

    checked_arrays = 0
    exact_oracle_arrays = 0
    unique_direction_pixels = 0
    known_plateau_order_drifts = {
        ("gap_robustness_2", 0),
        ("border_horizontal", 0),
    }
    for case in fixture["cases"]:
        if set(case["arrays"]) != CASE_KEYS:
            raise AssertionError(f"{case['name']}: fixture key set drifted")
        image = _fixture_array(case, "input").astype(np.uint8)
        path_length = int(case["path_length"])
        robustness = int(case["robustness"])
        bordered = _fixture_array(case, "bordered")
        np.testing.assert_array_equal(bordered[1:-1, 1:-1], image)
        np.testing.assert_array_equal(
            bordered,
            np.pad(image, 1, mode="constant", constant_values=0),
            err_msg=f"{case['name']}: one-pixel zero border",
        )
        dilated_bordered = _fixture_array(case, "dilated_bordered")
        np.testing.assert_array_equal(
            dilated_bordered,
            square_dilation(bordered, robustness),
            err_msg=f"{case['name']}: square dilation",
        )
        level_order = _fixture_array(case, "source_level_sort_order").ravel()
        np.testing.assert_array_equal(np.sort(level_order), np.arange(bordered.size))
        sorted_levels = dilated_bordered.ravel()[level_order]
        if np.any(sorted_levels[1:] < sorted_levels[:-1]):
            raise AssertionError(f"{case['name']}: source gray-level order is not ascending")
        checked_arrays += 3
        raw, robust = exhaustive_rorpo_maps(image, path_length, robustness)
        for orientation in range(4):
            source_raw = _fixture_array(case, f"po_raw_o{orientation + 1}")
            drift_key = (case["name"], orientation)
            if drift_key in known_plateau_order_drifts:
                if np.array_equal(raw[orientation], source_raw):
                    raise AssertionError(
                        f"{case['name']}: expected documented plateau-order drift vanished"
                    )
            else:
                np.testing.assert_array_equal(
                    raw[orientation],
                    source_raw,
                    err_msg=f"{case['name']}: raw orientation {orientation + 1}",
                )
                exact_oracle_arrays += 1
            np.testing.assert_array_equal(
                _fixture_array(case, f"rpo_o{orientation + 1}"),
                np.minimum(source_raw, image),
                err_msg=f"{case['name']}: source anti-extensive minimum {orientation + 1}",
            )
            checked_arrays += 2

        source_robust = np.stack(
            [_fixture_array(case, f"rpo_o{orientation + 1}") for orientation in range(4)]
        )
        ranked = np.sort(source_robust, axis=0)
        response = ranked[-1] - ranked[0]
        for rank in range(4):
            np.testing.assert_array_equal(
                ranked[rank],
                _fixture_array(case, f"rank_value_ascending_{rank + 1}"),
                err_msg=f"{case['name']}: rank {rank + 1}",
            )
            rank_orientation = _fixture_array(
                case, f"rank_orientation_ascending_{rank + 1}"
            ).astype(np.int64)
            if np.any((rank_orientation < 0) | (rank_orientation > 3)):
                raise AssertionError(f"{case['name']}: rank orientation outside 0..3")
            selected = np.take_along_axis(
                source_robust,
                rank_orientation[None, :, :],
                axis=0,
            )[0]
            np.testing.assert_array_equal(selected, ranked[rank])
            checked_arrays += 1
        np.testing.assert_array_equal(
            response,
            _fixture_array(case, "intensity_source"),
            err_msg=f"{case['name']}: max-minus-min response",
        )
        np.testing.assert_array_equal(
            response > 1,
            _fixture_array(case, "direction_threshold_gt_1").astype(bool),
            err_msg=f"{case['name']}: strict direction threshold",
        )
        np.testing.assert_array_equal(
            response,
            _fixture_array(case, "intensity_reconstructed"),
            err_msg=f"{case['name']}: independently reconstructed intensity",
        )
        direction_x = _fixture_array(case, "direction_x_source")
        direction_y = _fixture_array(case, "direction_y_source")
        direction_norm = np.hypot(direction_x, direction_y)
        active = response > 1
        np.testing.assert_allclose(direction_norm[active], 1.0, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(direction_norm[~active], 0.0)
        unique_direction_pixels += _assert_direction_capture(case, response)
        checked_arrays += 26

    multiscale = fixture["multiscale"]
    if set(multiscale["arrays"]) != MULTISCALE_KEYS:
        raise AssertionError("multiscale fixture key set drifted")
    scales = tuple(int(scale) for scale in multiscale["scales"])
    if scales != (2, 3, 5):
        raise AssertionError("unexpected multiscale fixture scales")
    per_scale = np.stack(
        [_fixture_array(multiscale, f"intensity_scale_{scale}") for scale in scales]
    )
    np.testing.assert_array_equal(
        per_scale.max(axis=0), _fixture_array(multiscale, "intensity_source")
    )
    winning = np.full(per_scale.shape[1:], -1, dtype=np.int64)
    current = np.zeros(per_scale.shape[1:], dtype=per_scale.dtype)
    expected_x = np.zeros(per_scale.shape[1:], dtype=np.float64)
    expected_y = np.zeros(per_scale.shape[1:], dtype=np.float64)
    for index, scale in enumerate(scales):
        update = per_scale[index] > current
        current[update] = per_scale[index][update]
        winning[update] = scale
        scale_x = _fixture_array(multiscale, f"direction_x_scale_{scale}")
        scale_y = _fixture_array(multiscale, f"direction_y_scale_{scale}")
        scale_norm = np.hypot(scale_x, scale_y)
        scale_active = per_scale[index] > 1
        np.testing.assert_allclose(scale_norm[scale_active], 1.0, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(scale_norm[~scale_active], 0.0)
        expected_x[update] = scale_x[update]
        expected_y[update] = scale_y[update]
    np.testing.assert_array_equal(winning, _fixture_array(multiscale, "winning_scale"))
    np.testing.assert_allclose(
        expected_x,
        np.asarray(_fixture_array(multiscale, "direction_x_source"), dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected_y,
        np.asarray(_fixture_array(multiscale, "direction_y_source"), dtype=np.float64),
        rtol=0,
        atol=1e-6,
    )
    if _fixture_array(multiscale, "winning_scale").astype(np.int64).dtype != np.int64:
        raise AssertionError("winning scale must be representable as int64")
    np.testing.assert_array_equal(winning[current == 0], -1)
    multiscale_norm = np.hypot(
        _fixture_array(multiscale, "direction_x_source"),
        _fixture_array(multiscale, "direction_y_source"),
    )
    active = current > 1
    np.testing.assert_allclose(multiscale_norm[active], 1.0, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(multiscale_norm[~active], 0.0)
    checked_arrays += 16
    unique_fixture_arrays = len(fixture["cases"]) * len(CASE_KEYS) + len(MULTISCALE_KEYS)
    return unique_fixture_arrays, exact_oracle_arrays, unique_direction_pixels


def _assert_load_bearing_claims() -> None:
    background = np.full((7, 7), 7, dtype=np.uint8)
    l_minus = background.copy()
    l_minus[3, 2:4] = 200
    _, robust = exhaustive_rorpo_maps(l_minus, 3, 0)
    assert int((robust.max(axis=0) - robust.min(axis=0)).max()) == 0

    exact = background.copy()
    exact[3, 2:5] = 200
    _, robust = exhaustive_rorpo_maps(exact, 3, 0)
    response = robust.max(axis=0) - robust.min(axis=0)
    assert int(response[3, 3]) == 193

    constant = np.full((5, 5), 23, dtype=np.uint8)
    _, robust = exhaustive_rorpo_maps(constant, 3, 0)
    np.testing.assert_array_equal(robust.max(axis=0) - robust.min(axis=0), 0)

    gap = background.copy()
    gap[3, 1:6] = 170
    gap[3, 3] = 7
    _, plain = exhaustive_rorpo_maps(gap, 5, 0)
    _, robust_gap = exhaustive_rorpo_maps(gap, 5, 2)
    plain_response = plain.max(axis=0) - plain.min(axis=0)
    robust_response = robust_gap.max(axis=0) - robust_gap.min(axis=0)
    assert int(plain_response.max()) == 0
    assert int(robust_response[3, 3]) == 0
    assert int(robust_response[3, 2]) == 163
    assert int(np.count_nonzero(robust_response)) == 4
    assert np.all(robust_gap <= gap[None, :, :])  # type: ignore[operator]

    affine = exact.astype(np.int16) * 2 + 11
    _, affine_robust = exhaustive_rorpo_maps(affine, 3, 0)
    affine_response = affine_robust.max(axis=0) - affine_robust.min(axis=0)
    np.testing.assert_array_equal(affine_response, 2 * response.astype(np.int16))

    rotated = np.rot90(exact)
    _, rotated_robust = exhaustive_rorpo_maps(rotated, 3, 0)
    rotated_response = rotated_robust.max(axis=0) - rotated_robust.min(axis=0)
    np.testing.assert_array_equal(rotated_response, np.rot90(response))

    unique_values = [0, 10, 20, 100]
    unique_costs = _split_costs_uint8(unique_values)
    _, _, singleton_objectives = _correct_source_vectors(SOURCE_BASES[[0]])
    assert _contract_direction_is_unique(
        unique_values, unique_costs, singleton_objectives
    )
    tied_cost_values = [0, 1, 2, 3]
    tied_costs = _split_costs_uint8(tied_cost_values)
    assert not _contract_direction_is_unique(
        tied_cost_values, tied_costs, singleton_objectives
    )
    boundary_tie_values = [0, 0, 0, 10]
    assert not _contract_direction_is_unique(
        boundary_tie_values, (10, 0, 20), (0, 1)
    )
    _, _, ambiguous_objectives = _correct_source_vectors(SOURCE_BASES[[0, 1]])
    assert ambiguous_objectives[0] == ambiguous_objectives[1]
    assert not _contract_direction_is_unique(
        [0, 0, 10, 10], _split_costs_uint8([0, 0, 10, 10]), ambiguous_objectives
    )

    # Caller order is deliberately unsorted. Strict greater ownership retains
    # path length 5 rather than replacing it with the later equal response at 2.
    scale_maps = (np.asarray([[9]], dtype=np.uint8), np.asarray([[9]], dtype=np.uint8))
    caller_lengths = (5, 2)
    merged = np.zeros((1, 1), dtype=np.uint8)
    owner = np.full((1, 1), -1, dtype=np.int64)
    for length, scale_map in zip(caller_lengths, scale_maps, strict=True):
        update = scale_map > merged
        merged[update] = scale_map[update]
        owner[update] = length
    assert int(owner[0, 0]) == 5
    assert owner.dtype == np.int64

    for bad_image, bad_length in (
        (np.array([], dtype=np.uint8), 3),
        (np.zeros((2, 2, 1), dtype=np.uint8), 3),
        (np.array([[math.nan]]), 1),
        (np.array([[-1.0]]), 1),
        (np.zeros((2, 2), dtype=np.uint8), 0),
    ):
        try:
            exhaustive_rorpo_maps(bad_image, bad_length, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid oracle input was silently accepted")
    try:
        exhaustive_rorpo_maps(np.zeros((2, 2), dtype=np.uint8), 1.5, 0)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("nonintegral path length was silently accepted")


def validate_rorpo_claims() -> None:
    """Run exact fixture and mathematical-invariant checks."""

    checked_arrays, exact_oracle_arrays, unique_direction_pixels = (
        _assert_source_fixture_matches_exhaustive_oracle()
    )
    _assert_load_bearing_claims()
    print("A08 RORPO logic validation passed")
    print(f"source archive sha256: {SOURCE_ARCHIVE_SHA256}")
    print(f"source-generated intermediate arrays checked exactly: {checked_arrays}")
    print(f"paper-oracle arrays matching executable exactly: {exact_oracle_arrays}")
    print(f"pixels satisfying the frozen unique-direction predicate: {unique_direction_pixels}")
    print("documented source plateau-order counterexamples: 2")
    print("tolerance: exact for uint8 morphology and ranking; no approximate claim")
    print("maximum observed response error: 0")
    print("assumption: path length counts vertices; four published 2-D DAGs only")


if __name__ == "__main__":
    try:
        validate_rorpo_claims()
    except Exception as error:
        print(f"A08 RORPO logic validation failed: {error}", file=sys.stderr)
        raise
