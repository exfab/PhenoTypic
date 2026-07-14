"""Clean-room behavioral tests for the uint8 bright-ridge RORPO core."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import phenotypic.sdk_.reconnect._rorpo as rorpo_module
from phenotypic.sdk_.reconnect._rorpo import (
    ORIENTATION_STEPS,
    _direction_from_ranked,
    _path_opening,
    _single_scale_rorpo,
    rorpo,
)


_FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "reconnect"
    / "rorpo"
    / "source_fixture.json"
)
if not _FIXTURE.exists():
    _FIXTURE = (
        Path(__file__).parents[4]
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-13-fungi-detection-method-ports"
        / "refs"
        / "rorpo"
        / "source_fixture.json"
    )


def _decode(encoded: dict[str, object]) -> np.ndarray:
    """Decode one source-free fixture array."""
    shape = tuple(int(value) for value in encoded["shape"])  # type: ignore[arg-type]
    return np.asarray(encoded["data"]).reshape(shape)


def _fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _case(name: str) -> dict[str, object]:
    cases = _fixture()["cases"]
    return next(case for case in cases if case["name"] == name)  # type: ignore[union-attr]


def _in_bounds(shape: tuple[int, int], row: int, column: int) -> bool:
    return 0 <= row < shape[0] and 0 <= column < shape[1]


def _enumerate_complete_paths(
    mask: np.ndarray,
    steps: tuple[tuple[int, int], ...],
    length: int,
) -> Iterable[tuple[tuple[int, int], ...]]:
    """Enumerate complete directed paths independently of production DP."""

    def extend(path: tuple[tuple[int, int], ...]):
        if len(path) == length:
            yield path
            return
        row, column = path[-1]
        for row_step, column_step in steps:
            neighbor = row + row_step, column + column_step
            if _in_bounds(mask.shape, *neighbor) and mask[neighbor]:
                yield from extend(path + (neighbor,))

    for row, column in np.argwhere(mask):
        yield from extend(((int(row), int(column)),))


def _paper_path_opening(
    image: np.ndarray,
    length: int,
    steps: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Reconstruct Equation 2 by exhaustive upper-level path enumeration."""
    output = np.zeros_like(image)
    for level in np.unique(image):
        if level == 0:
            continue
        mask = image >= level
        survivors = np.zeros(mask.shape, dtype=bool)
        for path in _enumerate_complete_paths(mask, steps, length):
            for point in path:
                survivors[point] = True
        output[survivors] = level
    return output


def _clipped_square_dilation(image: np.ndarray, robustness: int) -> np.ndarray:
    """Apply the contract's clipped square maximum independently."""
    radius = robustness // 2
    output = np.zeros_like(image)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            output[row, column] = np.max(
                image[
                    max(0, row - radius) : row + radius + 1,
                    max(0, column - radius) : column + radius + 1,
                ]
            )
    return output


@pytest.mark.parametrize(
    "name",
    [
        "constant",
        "horizontal_l_minus_1",
        "horizontal_l",
        "horizontal_l_plus_1",
        "vertical_l",
        "diagonal_o3_l",
        "diagonal_o4_l",
        "curved_admissible",
        "isotropic_blob",
        "multilevel",
        "gap_robustness_0",
        "gap_robustness_1",
        "gap_robustness_2",
        "border_horizontal",
        "rank_tie",
    ],
)
def test_sanitized_fixture_matches_intermediates_and_public_output(name: str) -> None:
    """Match all stable source observations and paper-authoritative plateau cases."""
    case = _case(name)
    arrays = case["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    length = int(case["path_length"])
    robustness = int(case["robustness"])
    actual = _single_scale_rorpo(image, length, robustness)
    dilated = _clipped_square_dilation(image, robustness)
    plateau_drift_cases = {"gap_robustness_2", "border_horizontal"}

    for orientation in range(4):
        raw_key = f"po_raw_o{orientation + 1}"
        robust_key = f"rpo_o{orientation + 1}"
        paper = _paper_path_opening(
            dilated, length, ORIENTATION_STEPS[orientation]
        )
        assert_array_equal(actual.path_openings[orientation], paper)
        assert_array_equal(actual.robust_openings[orientation], np.minimum(image, paper))
        if name not in plateau_drift_cases:
            assert_array_equal(actual.path_openings[orientation], _decode(arrays[raw_key]))  # type: ignore[index]
            assert_array_equal(actual.robust_openings[orientation], _decode(arrays[robust_key]))  # type: ignore[index]

    # Once the source's unstable plateau ordering changes a raw opening, every
    # source-derived downstream array may change too. The paper definition above
    # remains authoritative, so do not treat those derived fixture arrays as an
    # oracle for these two cases.
    if name in plateau_drift_cases:
        sorted_openings = np.sort(np.stack(actual.robust_openings, axis=-1), axis=-1)
        assert_array_equal(actual.rank_values, sorted_openings)
        assert_array_equal(actual.response, sorted_openings[..., 3] - sorted_openings[..., 0])
        assert np.all(actual.response <= image)
        return

    for rank in range(4):
        assert_array_equal(
            actual.rank_values[..., rank],
            _decode(arrays[f"rank_value_ascending_{rank + 1}"]),  # type: ignore[index]
        )
        assert_array_equal(
            actual.rank_orientations[..., rank],
            _decode(arrays[f"rank_orientation_ascending_{rank + 1}"]),  # type: ignore[index]
        )

    assert_array_equal(actual.response, _decode(arrays["intensity_reconstructed"]))  # type: ignore[index]
    assert_array_equal(
        actual.direction_threshold,
        _decode(arrays["direction_threshold_gt_1"]).astype(bool),  # type: ignore[index]
    )
    for split in range(3):
        assert_array_equal(
            actual.split_costs[..., split],
            _decode(arrays[f"direction_split_cost_high_{split + 1}"]),  # type: ignore[index]
        )
        assert_array_equal(
            actual.selected_orientations[..., split],
            _decode(arrays[f"direction_selected_orientation_{split + 1}"]),  # type: ignore[index]
        )
        assert_array_equal(
            actual.correction_signs[..., split],
            _decode(arrays[f"direction_correction_sign_{split + 1}"]),  # type: ignore[index]
        )
        assert_allclose(
            actual.raw_vectors[..., split, 0],
            _decode(arrays[f"direction_raw_x_{split + 1}"]),  # type: ignore[index]
            rtol=0.0,
            atol=0.0,
        )
        assert_allclose(
            actual.raw_vectors[..., split, 1],
            _decode(arrays[f"direction_raw_y_{split + 1}"]),  # type: ignore[index]
            rtol=0.0,
            atol=0.0,
        )
    assert_array_equal(actual.selected_count, _decode(arrays["direction_selected_count"]))  # type: ignore[index]
    assert np.all(actual.response <= image)


def test_atomic_plateaus_override_two_unstable_source_arrays() -> None:
    """The paper upper-level union, not source sort order, controls equal plateaus."""
    border = _case("border_horizontal")
    arrays = border["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    paper = _paper_path_opening(image, 4, ORIENTATION_STEPS[0])
    assert_array_equal(paper, np.full(image.shape, 7, dtype=np.uint8))
    assert not np.array_equal(paper, _decode(arrays["po_raw_o1"]))  # type: ignore[index]

    gap = _case("gap_robustness_2")
    gap_arrays = gap["arrays"]
    gap_image = _decode(gap_arrays["input"]).astype(np.uint8)  # type: ignore[index]
    dilated = _clipped_square_dilation(gap_image, 2)
    gap_paper = _paper_path_opening(dilated, 5, ORIENTATION_STEPS[0])
    assert not np.array_equal(gap_paper, _decode(gap_arrays["po_raw_o1"]))  # type: ignore[index]


def test_r_equals_one_is_identity_and_r_equals_two_is_anti_extensive() -> None:
    image = _decode(_case("gap_robustness_0")["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
    zero = _single_scale_rorpo(image, 5, 0)
    one = _single_scale_rorpo(image, 5, 1)
    two = _single_scale_rorpo(image, 5, 2)
    assert_array_equal(one.response, zero.response)
    for first, second in zip(one.robust_openings, zero.robust_openings, strict=True):
        assert_array_equal(first, second)
    assert np.all(two.response <= image)
    assert two.response[3, 3] == 0


def test_path_length_counts_vertices_and_curved_paths_are_admissible() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 2:5] = 200
    assert np.count_nonzero(_path_opening(image, 3, 1) == 200) == 3
    assert not np.any(_path_opening(image, 4, 1))

    curve = _decode(_case("curved_admissible")["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
    assert np.any(_path_opening(curve, 4, 2) > 7)


def test_public_directions_are_row_column_unit_axial_vectors() -> None:
    expected = {
        "horizontal_l": (0.0, 1.0),
        "vertical_l": (1.0, 0.0),
        "diagonal_o3_l": (2**-0.5, 2**-0.5),
        "diagonal_o4_l": (2**-0.5, -(2**-0.5)),
    }
    for name, vector in expected.items():
        case = _case(name)
        image = _decode(case["arrays"]["input"]).astype(np.uint8)  # type: ignore[index]
        result = rorpo(image, (int(case["path_length"]),), int(case["robustness"]))
        assert np.any(result.direction_valid)
        assert_allclose(
            result.direction_vector[result.direction_valid],
            np.broadcast_to(vector, result.direction_vector[result.direction_valid].shape),
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        )
        norms = np.linalg.norm(result.direction_vector[result.direction_valid], axis=1)
        assert_allclose(norms, 1.0, rtol=0.0, atol=8.0 * np.finfo(np.float64).eps)


def _one_pixel_direction(
    values: tuple[int, int, int, int],
    orientations: tuple[int, int, int, int],
    response: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the public direction predicate for one synthetic ranked pixel."""
    ranked_values = np.asarray(values, dtype=np.uint8).reshape(1, 1, 4)
    ranked_orientations = np.asarray(orientations, dtype=np.int64).reshape(1, 1, 4)
    intensity = np.asarray([[response]], dtype=np.uint8)
    outputs = _direction_from_ranked(
        ranked_values,
        ranked_orientations,
        intensity,
    )
    return outputs[0], outputs[8], outputs[9]


def test_direction_requires_response_strictly_greater_than_one() -> None:
    threshold, direction, valid = _one_pixel_direction(
        (0, 0, 0, 1),
        (0, 1, 2, 3),
        1,
    )
    assert not threshold[0, 0]
    assert not valid[0, 0]
    assert_array_equal(direction, 0.0)


def test_direction_requires_a_unique_truncated_split_cost() -> None:
    threshold, direction, valid = _one_pixel_direction(
        (0, 0, 0, 2),
        (0, 1, 2, 3),
        2,
    )
    assert threshold[0, 0]
    assert not valid[0, 0]
    assert_array_equal(direction, 0.0)


def test_direction_requires_a_strict_rank_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Select the two largest values uniquely while placing the split inside an
    # equal-valued plateau. The orientation pair is nonperpendicular so only the
    # rank-boundary predicate can invalidate this construction.
    monkeypatch.setattr(
        rorpo_module,
        "_truncated_split_cost",
        lambda _values, high_count: {1: 2, 2: 0, 3: 1}[high_count],
    )
    threshold, direction, valid = _one_pixel_direction(
        (0, 5, 5, 10),
        (1, 3, 2, 0),
        10,
    )
    assert threshold[0, 0]
    assert not valid[0, 0]
    assert_array_equal(direction, 0.0)


def test_direction_requires_a_unique_sign_correction() -> None:
    # The two selected private axes are perpendicular. Both correction signs
    # therefore have the same integer-truncated angle objective.
    _, direction, valid = _one_pixel_direction(
        (0, 0, 10, 10),
        (2, 3, 1, 0),
        10,
    )
    assert not valid[0, 0]
    assert_array_equal(direction, 0.0)


def test_direction_requires_a_nonzero_corrected_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the otherwise defensive zero-sum branch to be the unique winning
    # correction: o3 - o2 - o1 = (0, 0) in the private convention.
    monkeypatch.setattr(
        rorpo_module,
        "_pairwise_angle_objective",
        lambda vectors: 0 if np.linalg.norm(np.sum(vectors, axis=0)) == 0.0 else 1,
    )
    _, direction, valid = _one_pixel_direction(
        (0, 10, 10, 10),
        (3, 0, 1, 2),
        10,
    )
    assert not valid[0, 0]
    assert_array_equal(direction, 0.0)


def test_multiscale_strict_ties_retain_first_caller_length() -> None:
    fixture = _fixture()["multiscale"]
    arrays = fixture["arrays"]
    image = _decode(arrays["input"]).astype(np.uint8)  # type: ignore[index]
    result = rorpo(image, (2, 3, 5), robustness=0)
    assert_array_equal(result.response, _decode(arrays["intensity_source"]))  # type: ignore[index]
    assert_array_equal(result.winning_scale, _decode(arrays["winning_scale"]))  # type: ignore[index]

    tied = rorpo(image, (3, 2, 3), robustness=0)
    scale_three = _single_scale_rorpo(image, 3, 0).response
    scale_two = _single_scale_rorpo(image, 2, 0).response
    positive_ties = (scale_three == scale_two) & (scale_three > 0)
    assert np.any(positive_ties)
    assert_array_equal(tied.winning_scale[positive_ties], 3)


def test_outputs_have_frozen_dtypes_shapes_and_invalid_sentinels() -> None:
    image = np.zeros((5, 7), dtype=np.uint8)
    result = rorpo(image, (1,), robustness=0)
    assert result.response.shape == image.shape
    assert result.response.dtype == np.uint8
    assert result.direction_vector.shape == image.shape + (2,)
    assert result.direction_vector.dtype == np.float64
    assert result.direction_valid.dtype == np.bool_
    assert result.winning_scale.dtype == np.int64
    assert_array_equal(result.direction_vector, 0.0)
    assert_array_equal(result.direction_valid, False)
    assert_array_equal(result.winning_scale, -1)


def test_inputs_are_not_mutated() -> None:
    image = np.arange(49, dtype=np.uint8).reshape(7, 7)
    before = image.copy()
    rorpo(image, (2, 3), robustness=2)
    assert_array_equal(image, before)


@pytest.mark.parametrize(
    ("image", "lengths", "robustness"),
    [
        (np.array(1, dtype=np.uint8), (3,), 0),
        (np.zeros((0, 2), dtype=np.uint8), (3,), 0),
        (np.zeros((2, 2), dtype=np.float32), (3,), 0),
        (np.zeros((2, 2), dtype=np.int64), (3,), 0),
        (np.zeros((2, 2), dtype=np.uint8), [], 0),
        (np.zeros((2, 2), dtype=np.uint8), (), 0),
        (np.zeros((2, 2), dtype=np.uint8), (True,), 0),
        (np.zeros((2, 2), dtype=np.uint8), (1.5,), 0),
        (np.zeros((2, 2), dtype=np.uint8), (0,), 0),
        (np.zeros((2, 2), dtype=np.uint8), (1,), True),
        (np.zeros((2, 2), dtype=np.uint8), (1,), -1),
        (np.zeros((2, 2), dtype=np.uint8), (1,), 1.5),
    ],
)
def test_invalid_inputs_raise_value_error(
    image: np.ndarray,
    lengths: object,
    robustness: object,
) -> None:
    with pytest.raises(ValueError):
        rorpo(image, lengths, robustness)  # type: ignore[arg-type]
