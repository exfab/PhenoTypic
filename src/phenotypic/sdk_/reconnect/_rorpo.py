"""Clean-room uint8 implementation of two-dimensional RORPO."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage


UInt8Array: TypeAlias = npt.NDArray[np.uint8]
Int64Array: TypeAlias = npt.NDArray[np.int64]
Float64Array: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

StepSet: TypeAlias = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

ORIENTATION_STEPS: tuple[StepSet, StepSet, StepSet, StepSet] = (
    ((1, -1), (1, 0), (1, 1)),
    ((-1, 1), (0, 1), (1, 1)),
    ((-1, 0), (-1, 1), (0, 1)),
    ((-1, -1), (-1, 0), (0, -1)),
)

# Each axis has a strictly positive dot product with every forward step in the
# matching orientation. Sorting by this score therefore gives a topological order.
_TOPOLOGICAL_AXES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, -1),
)

# Private source convention is (column, row). Public output swaps this once.
_PRIVATE_DIRECTION_VECTORS: Float64Array = np.array(
    ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class RorpoResult:
    """Public multiscale RORPO result.

    Attributes:
        response: Unnormalized uint8 RORPO intensity.
        direction_vector: Unit axial direction in ``(row, column)`` order.
        direction_valid: Five-part unique-direction validity predicate.
        winning_scale: First caller-supplied winning path length, or ``-1``.
    """

    response: UInt8Array
    direction_vector: Float64Array
    direction_valid: BoolArray
    winning_scale: Int64Array


@dataclass(frozen=True)
class _RorpoScaleResult:
    """Complete source-visible state for one path length."""

    path_openings: tuple[UInt8Array, UInt8Array, UInt8Array, UInt8Array]
    robust_openings: tuple[UInt8Array, UInt8Array, UInt8Array, UInt8Array]
    rank_values: UInt8Array
    rank_orientations: Int64Array
    response: UInt8Array
    direction_threshold: BoolArray
    split_costs: Int64Array
    selected_count: Int64Array
    selected_orientations: Int64Array
    correction_signs: Int64Array
    raw_vectors: Float64Array
    corrected_vectors: Float64Array
    corrected_sum: Float64Array
    direction_vector: Float64Array
    direction_valid: BoolArray


def _validate_image(image: np.ndarray) -> UInt8Array:
    """Validate the deliberately narrow clean-room arithmetic boundary."""
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.dtype(np.uint8):
        raise ValueError("image must have dtype exactly uint8")
    return image


def _validate_path_lengths(path_lengths: tuple[int, ...]) -> tuple[int, ...]:
    """Validate path lengths while preserving caller order and duplicates."""
    if not isinstance(path_lengths, tuple) or not path_lengths:
        raise ValueError("path_lengths must be a nonempty tuple")
    validated: list[int] = []
    for length in path_lengths:
        if isinstance(length, (bool, np.bool_)) or not isinstance(
            length, (int, np.integer)
        ):
            raise ValueError("path lengths must be positive nonboolean integers")
        integer = int(length)
        if integer < 1:
            raise ValueError("path lengths must be positive nonboolean integers")
        validated.append(integer)
    return tuple(validated)


def _validate_robustness(robustness: int) -> int:
    """Validate the nonnegative integer robustness parameter."""
    if isinstance(robustness, (bool, np.bool_)) or not isinstance(
        robustness, (int, np.integer)
    ):
        raise ValueError("robustness must be a nonnegative nonboolean integer")
    integer = int(robustness)
    if integer < 0:
        raise ValueError("robustness must be a nonnegative nonboolean integer")
    return integer


def _topological_order(
    shape: tuple[int, int], orientation: int
) -> npt.NDArray[np.intp]:
    """Return flat pixel indices in one orientation's DAG order."""
    rows, columns = np.indices(shape, dtype=np.int64)
    row_weight, column_weight = _TOPOLOGICAL_AXES[orientation]
    scores = row_weight * rows + column_weight * columns
    return np.argsort(scores, axis=None, kind="stable")


def _binary_path_survivors(
    mask: BoolArray,
    length: int,
    orientation: int,
) -> BoolArray:
    """Return pixels belonging to a complete directed path of ``length`` vertices."""
    if length == 1:
        return mask.copy()
    if length > mask.size:
        return np.zeros(mask.shape, dtype=np.bool_)

    steps = ORIENTATION_STEPS[orientation]
    order = _topological_order(mask.shape, orientation)
    forward = np.zeros(mask.shape, dtype=np.int64)
    backward = np.zeros(mask.shape, dtype=np.int64)
    height, width = mask.shape

    for flat_index in order:
        row, column = divmod(int(flat_index), width)
        if not mask[row, column]:
            continue
        best = 0
        for row_step, column_step in steps:
            previous_row = row - row_step
            previous_column = column - column_step
            if (
                0 <= previous_row < height
                and 0 <= previous_column < width
                and mask[previous_row, previous_column]
            ):
                best = max(best, int(forward[previous_row, previous_column]))
        forward[row, column] = min(length, best + 1)

    for flat_index in order[::-1]:
        row, column = divmod(int(flat_index), width)
        if not mask[row, column]:
            continue
        best = 0
        for row_step, column_step in steps:
            next_row = row + row_step
            next_column = column + column_step
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and mask[next_row, next_column]
            ):
                best = max(best, int(backward[next_row, next_column]))
        backward[row, column] = min(length, best + 1)

    return mask & (forward + backward - 1 >= length)


def _path_opening(image: UInt8Array, length: int, orientation: int) -> UInt8Array:
    """Compute the atomic upper-level grayscale path opening from Equation 2."""
    if orientation not in range(4):
        raise ValueError("orientation must be one of 0, 1, 2, or 3")
    if length == 1:
        return image.copy()
    output = np.zeros(image.shape, dtype=np.uint8)
    for level in np.unique(image):
        if level == 0:
            continue
        upper_level = image >= level
        survivors = _binary_path_survivors(upper_level, length, orientation)
        output[survivors] = level
    return output


def _clipped_square_dilation(image: UInt8Array, robustness: int) -> UInt8Array:
    """Apply the clipped square maximum with radius ``floor(R / 2)``."""
    radius = robustness // 2
    if radius == 0:
        return image.copy()
    size = 2 * radius + 1
    return cast(
        UInt8Array,
        ndimage.maximum_filter(image, size=size, mode="constant", cval=0),
    )


def _population_standard_deviation(values: UInt8Array) -> np.float32:
    """Compute the frozen float32 population standard deviation."""
    float_values = values.astype(np.float32, copy=False)
    return np.std(float_values, dtype=np.float32)


def _truncated_split_cost(values: UInt8Array, high_count: int) -> int:
    """Return one uint8-truncated low/high intraclass standard-deviation cost."""
    boundary = 4 - high_count
    low_cost = _population_standard_deviation(values[:boundary])
    high_cost = _population_standard_deviation(values[boundary:])
    return int(np.uint8(np.float32(low_cost + high_cost)))


def _pairwise_angle_objective(vectors: Float64Array) -> int:
    """Return the integer-truncated sum of pairwise angles in degrees."""
    total = 0.0
    for first in range(vectors.shape[0]):
        for second in range(first + 1, vectors.shape[0]):
            denominator = np.linalg.norm(vectors[first]) * np.linalg.norm(
                vectors[second]
            )
            cosine = float(np.dot(vectors[first], vectors[second]) / denominator)
            total += math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return int(total)


def _direction_from_ranked(
    rank_values: UInt8Array,
    rank_orientations: Int64Array,
    response: UInt8Array,
) -> tuple[
    BoolArray,
    Int64Array,
    Int64Array,
    Int64Array,
    Int64Array,
    Float64Array,
    Float64Array,
    Float64Array,
    Float64Array,
    BoolArray,
]:
    """Compute every direction intermediate and the five-part public predicate."""
    height, width, _ = rank_values.shape
    threshold = response > 1
    split_costs = np.full((height, width, 3), -1, dtype=np.int64)
    selected_count = np.zeros((height, width), dtype=np.int64)
    selected_orientations = np.full((height, width, 3), -1, dtype=np.int64)
    correction_signs = np.zeros((height, width, 3), dtype=np.int64)
    raw_vectors = np.zeros((height, width, 3, 2), dtype=np.float64)
    corrected_vectors = np.zeros_like(raw_vectors)
    corrected_sum = np.zeros((height, width, 2), dtype=np.float64)
    direction = np.zeros((height, width, 2), dtype=np.float64)
    valid = np.zeros((height, width), dtype=np.bool_)

    for row, column in np.argwhere(threshold):
        values = rank_values[row, column]
        costs = np.array(
            [_truncated_split_cost(values, count) for count in (1, 2, 3)],
            dtype=np.int64,
        )
        split_costs[row, column] = costs
        minimum_cost = int(np.min(costs))
        cost_is_unique = np.count_nonzero(costs == minimum_cost) == 1
        high_count = int(np.flatnonzero(costs == minimum_cost)[0]) + 1
        selected_count[row, column] = high_count

        orientation_indices = rank_orientations[row, column, ::-1][:high_count]
        selected_orientations[row, column, :high_count] = orientation_indices
        vectors = _PRIVATE_DIRECTION_VECTORS[orientation_indices]
        raw_vectors[row, column, :high_count] = vectors

        assignments: list[tuple[int, ...]] = []
        objectives: list[int] = []
        for trailing_signs in itertools.product((1, -1), repeat=high_count - 1):
            sign_assignment = (1, *trailing_signs)
            assignments.append(sign_assignment)
            objectives.append(
                _pairwise_angle_objective(
                    vectors
                    * np.asarray(sign_assignment, dtype=np.float64)[:, None]
                )
            )
        minimum_objective = min(objectives)
        assignment_is_unique = objectives.count(minimum_objective) == 1
        chosen = objectives.index(minimum_objective)
        chosen_signs = np.asarray(assignments[chosen], dtype=np.int64)
        corrected = vectors * chosen_signs[:, None]
        correction_signs[row, column, :high_count] = chosen_signs
        corrected_vectors[row, column, :high_count] = corrected
        vector_sum = cast(
            Float64Array,
            np.sum(corrected, axis=0, dtype=np.float64),
        )
        corrected_sum[row, column] = vector_sum
        norm = float(np.linalg.norm(vector_sum))

        boundary = 4 - high_count
        boundary_is_strict = bool(values[boundary - 1] < values[boundary])
        if not (
            cost_is_unique
            and boundary_is_strict
            and assignment_is_unique
            and norm > 0.0
        ):
            continue

        private_direction = vector_sum / norm
        public_direction = np.array(
            (private_direction[1], private_direction[0]), dtype=np.float64
        )
        if public_direction[0] < 0.0 or (
            public_direction[0] == 0.0 and public_direction[1] < 0.0
        ):
            public_direction *= -1.0
        direction[row, column] = public_direction
        valid[row, column] = True

    return (
        threshold,
        split_costs,
        selected_count,
        selected_orientations,
        correction_signs,
        raw_vectors,
        corrected_vectors,
        corrected_sum,
        direction,
        valid,
    )


def _single_scale_rorpo(
    image: UInt8Array,
    length: int,
    robustness: int,
) -> _RorpoScaleResult:
    """Compute all RORPO products for one already-validated scale."""
    dilated = _clipped_square_dilation(image, robustness)
    path_openings = cast(
        tuple[UInt8Array, UInt8Array, UInt8Array, UInt8Array],
        tuple(_path_opening(dilated, length, orientation) for orientation in range(4)),
    )
    robust_openings = cast(
        tuple[UInt8Array, UInt8Array, UInt8Array, UInt8Array],
        tuple(np.minimum(image, opened) for opened in path_openings),
    )
    orientation_stack = np.stack(robust_openings, axis=2)
    rank_orientations = np.argsort(
        orientation_stack, axis=2, kind="stable"
    ).astype(np.int64)
    rank_values = np.take_along_axis(
        orientation_stack, rank_orientations, axis=2
    )
    response = rank_values[..., 3] - rank_values[..., 0]
    (
        direction_threshold,
        split_costs,
        selected_count,
        selected_orientations,
        correction_signs,
        raw_vectors,
        corrected_vectors,
        corrected_sum,
        direction_vector,
        direction_valid,
    ) = _direction_from_ranked(rank_values, rank_orientations, response)
    return _RorpoScaleResult(
        path_openings=path_openings,
        robust_openings=robust_openings,
        rank_values=rank_values,
        rank_orientations=rank_orientations,
        response=response,
        direction_threshold=direction_threshold,
        split_costs=split_costs,
        selected_count=selected_count,
        selected_orientations=selected_orientations,
        correction_signs=correction_signs,
        raw_vectors=raw_vectors,
        corrected_vectors=corrected_vectors,
        corrected_sum=corrected_sum,
        direction_vector=direction_vector,
        direction_valid=direction_valid,
    )


def rorpo(
    image: np.ndarray,
    path_lengths: tuple[int, ...],
    robustness: int = 0,
) -> RorpoResult:
    """Compute clean-room uint8 RORPO for bright ridges.

    Args:
        image: Nonempty two-dimensional ``numpy.uint8`` bright-ridge image.
        path_lengths: Nonempty tuple of positive, nonboolean vertex counts. Caller
            order and duplicates are preserved for strict first-winner scale ties.
        robustness: Nonnegative, nonboolean integer. The square dilation radius is
            ``floor(robustness / 2)``; therefore 0 and 1 are identical.

    Returns:
        RORPO intensity, canonical row-column axial direction, validity, and scale.

    Raises:
        ValueError: Any input lies outside the frozen uint8 bright-ridge contract.
    """
    values = _validate_image(image)
    lengths = _validate_path_lengths(path_lengths)
    robust_radius_code = _validate_robustness(robustness)

    response = np.zeros(values.shape, dtype=np.uint8)
    direction_vector = np.zeros(values.shape + (2,), dtype=np.float64)
    direction_valid = np.zeros(values.shape, dtype=np.bool_)
    winning_scale = np.full(values.shape, -1, dtype=np.int64)
    for length in lengths:
        scale = _single_scale_rorpo(values, length, robust_radius_code)
        update = scale.response > response
        response[update] = scale.response[update]
        direction_vector[update] = scale.direction_vector[update]
        direction_valid[update] = scale.direction_valid[update]
        winning_scale[update] = length

    return RorpoResult(
        response=response,
        direction_vector=direction_vector,
        direction_valid=direction_valid,
        winning_scale=winning_scale,
    )


__all__ = ["RorpoResult", "rorpo"]
