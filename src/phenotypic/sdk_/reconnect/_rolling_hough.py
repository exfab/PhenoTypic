"""Clark source-faithful Rolling Hough numerical core."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from scipy import ndimage


Float64Array: TypeAlias = npt.NDArray[np.float64]
Int64Array: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ClarkRollingHoughResult:
    """Complete in-memory output of Clark's rho-zero Rolling Hough core.

    Attributes:
        theta: Hough-normal angles in radians on ``[0, pi)`` with shape ``(T,)``.
        support_counts: Angle-dependent rho-zero support with shape ``(T,)``.
        raw_counts: Integer rho-zero counts with shape ``(H, W, T)``. Values are
            adapter zeros outside ``eligible``.
        threshold_residual: Source threshold residuals with shape ``(H, W, T)``.
            Rejected bins are signed zero after the source's multiplicative mask.
        response: Unnormalized residual sum with shape ``(H, W)``.
        orientation: Axial Hough-normal angle in ``(0, pi]`` with shape ``(H, W)``.
            Invalid pixels are NaN.
        eligible: Source rolling-window mask with shape ``(H, W)``.
        valid: Boolean positive-residual mask with shape ``(H, W)``.
    """

    theta: Float64Array
    support_counts: Int64Array
    raw_counts: Int64Array
    threshold_residual: Float64Array
    response: Float64Array
    orientation: Float64Array
    eligible: BoolArray
    valid: BoolArray


def _positive_integer(value: object, *, name: str, odd: bool = False) -> int:
    """Validate one source integer parameter without accepting Boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        qualifier = "positive odd integer" if odd else "positive integer"
        raise TypeError(f"{name} must be a {qualifier}")
    integer = int(value)
    if integer <= 0 or (odd and integer % 2 == 0):
        qualifier = "positive odd integer" if odd else "positive integer"
        raise ValueError(f"{name} must be a {qualifier}")
    return integer


def _validated_threshold_fraction(value: object) -> float:
    """Validate the finite closed-unit-interval source threshold."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("threshold_fraction must be a real number")
    fraction = float(value)
    if not math.isfinite(fraction):
        raise ValueError("threshold_fraction must be finite")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("threshold_fraction must be in [0, 1]")
    return fraction


def _validated_image(image: np.ndarray) -> Float64Array:
    """Validate the deliberately narrow source arithmetic boundary."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("image must be two-dimensional")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.dtype != np.dtype(np.float64):
        raise TypeError("image must have dtype exactly float64")
    return image


def _circular_kernel(diameter: int) -> Int64Array:
    """Return the source inclusive integer-radius disk."""
    radius = diameter // 2
    row_coordinates, column_coordinates = (
        np.indices((diameter, diameter)) - radius
    )
    return np.less_equal(
        np.hypot(row_coordinates, column_coordinates), radius
    ).astype(np.int64)


def _all_within_diameter_are_good(
    shape: tuple[int, int], bad: BoolArray, diameter: int
) -> BoolArray:
    """Reproduce Clark's edge restriction and clipped bad-pixel disk clearing."""
    height, width = shape
    radius = diameter // 2
    mask: BoolArray = np.zeros(shape, dtype=np.bool_)
    if height > 2 * radius and width > 2 * radius:
        mask[radius : height - radius, radius : width - radius] = True

    offsets = np.argwhere(_circular_kernel(diameter)) - radius
    for bad_row, bad_column in np.argwhere(bad):
        rows = np.clip(bad_row + offsets[:, 0], 0, height - 1)
        columns = np.clip(bad_column + offsets[:, 1], 0, width - 1)
        mask[rows, columns] = False
    return mask


def _source_masks(
    image: Float64Array, window_diameter: int, smoothing_radius: int
) -> tuple[BoolArray, BoolArray]:
    """Build the source smoothing and second-halo rolling masks."""
    smoothing_mask = _all_within_diameter_are_good(
        image.shape,
        np.logical_not(np.isfinite(image)),
        2 * smoothing_radius + 1,
    )
    eligible = _all_within_diameter_are_good(
        image.shape,
        np.logical_not(smoothing_mask),
        window_diameter,
    )
    return smoothing_mask, eligible


def _clark_preprocessing(
    image: Float64Array, window_diameter: int, smoothing_radius: int
) -> tuple[
    Int64Array,
    BoolArray,
    BoolArray,
    Float64Array,
    Float64Array,
    Float64Array,
    BoolArray,
]:
    """Return every source-visible preprocessing intermediate."""
    smoothing_mask, eligible = _source_masks(
        image, window_diameter, smoothing_radius
    )
    smoothing_kernel = _circular_kernel(2 * smoothing_radius + 1)
    correlated: Float64Array = ndimage.correlate(image, smoothing_kernel)
    smoothed = correlated / np.sum(smoothing_kernel)
    with np.errstate(invalid="ignore"):
        unsharp = image - smoothed
    bitmask = np.logical_and(smoothing_mask, np.greater(unsharp, 0.0))
    return (
        smoothing_kernel,
        smoothing_mask,
        eligible,
        correlated,
        smoothed,
        unsharp,
        bitmask,
    )


def _center_line_geometry(
    window_diameter: int, theta: Float64Array
) -> tuple[Int64Array, Int64Array]:
    """Rasterize the source's round-to-even rho-zero Hough center lines."""
    radius = window_diameter // 2
    circular_window = _circular_kernel(window_diameter)
    row_coordinates, column_coordinates = (
        np.indices((window_diameter, window_diameter)) - radius
    )
    distances = column_coordinates[:, :, None] * np.cos(
        theta
    ) + row_coordinates[:, :, None] * np.sin(theta)
    center_lines = np.logical_and(
        circular_window[:, :, None], np.round(distances) == 0.0
    ).astype(np.int64)
    return circular_window, center_lines


def _threshold_counts(
    counts: Int64Array, support_counts: Int64Array, threshold_fraction: float
) -> Float64Array:
    """Apply Clark's subtract-then-``>=`` multiplicative threshold exactly."""
    residual = np.true_divide(counts, support_counts) - threshold_fraction
    residual *= np.greater_equal(residual, 0.0)
    return residual


def _axial_orientation(weights: Float64Array, theta: Float64Array) -> float:
    """Collapse one residual spectrum to Clark's axial Hough-normal angle."""
    y_component: np.float64 = np.sum(weights * np.sin(2.0 * theta))
    x_component: np.float64 = np.sum(weights * np.cos(2.0 * theta))
    rough_angle = 0.5 * np.arctan2(y_component, x_component)
    return float(np.pi - math.fmod(float(rough_angle + np.pi), float(np.pi)))


def clark_rolling_hough(
    image: np.ndarray,
    window_diameter: int,
    smoothing_radius: int,
    threshold_fraction: float,
) -> ClarkRollingHoughResult:
    """Compute Clark's original rho-zero Rolling Hough Transform.

    The input is smoothed with the source inclusive disk and SciPy reflect boundary,
    then converted to a strict-positive unsharp bitmask. Only centers surviving both
    the smoothing and rolling circular halos are evaluated. Theta parameterizes the
    Hough line normal, not the filament tangent. Counts are raw integer support;
    response is the unnormalized positive threshold-residual sum.

    Args:
        image: Nonempty two-dimensional array with dtype exactly ``float64``.
            Nonfinite pixels are retained as source bad pixels and invalidate nearby
            centers. Finite zero and negative values are allowed.
        window_diameter: Positive odd diameter of the circular rolling domain.
        smoothing_radius: Positive integer radius of the smoothing disk.
        threshold_fraction: Finite threshold fraction in ``[0, 1]``.

    Returns:
        Frozen result containing theta, supports, counts, threshold residuals, raw
        response, axial Hough-normal orientation, eligibility, and validity.

    Raises:
        TypeError: An input has an unsupported container, dtype, or scalar type.
        ValueError: An input has an invalid shape, range, or parity.
    """
    values = _validated_image(image)
    diameter = _positive_integer(
        window_diameter, name="window_diameter", odd=True
    )
    radius = _positive_integer(smoothing_radius, name="smoothing_radius")
    fraction = _validated_threshold_fraction(threshold_fraction)

    _, _, eligible, _, _, _, bitmask = _clark_preprocessing(
        values, diameter, radius
    )
    theta_count = int(
        math.ceil(np.pi * (diameter - 1) / np.sqrt(np.float64(2.0)))
    )
    theta = np.linspace(
        0.0, np.pi, theta_count, endpoint=False, dtype=np.float64
    )
    circular_window, center_lines = _center_line_geometry(diameter, theta)
    support_counts = np.einsum(
        "ijt,ij->t", center_lines, circular_window, dtype=np.int64
    )

    height, width = values.shape
    raw_counts = np.zeros((height, width, theta_count), dtype=np.int64)
    threshold_residual = np.zeros(
        (height, width, theta_count), dtype=np.float64
    )
    window_radius = diameter // 2
    for row, column in np.argwhere(eligible):
        local_window = bitmask[
            row - window_radius : row + window_radius + 1,
            column - window_radius : column + window_radius + 1,
        ]
        counts = np.einsum(
            "ijt,ij->t", center_lines, local_window, dtype=np.int64
        )
        raw_counts[row, column] = counts
        threshold_residual[row, column] = _threshold_counts(
            counts, support_counts, fraction
        )

    valid = np.any(threshold_residual > 0.0, axis=2)
    response: Float64Array = np.sum(
        threshold_residual, axis=2, dtype=np.float64
    )
    orientation: Float64Array = np.full(values.shape, np.nan, dtype=np.float64)
    for row, column in np.argwhere(valid):
        orientation[row, column] = _axial_orientation(
            threshold_residual[row, column], theta
        )

    return ClarkRollingHoughResult(
        theta=theta,
        support_counts=support_counts,
        raw_counts=raw_counts,
        threshold_residual=threshold_residual,
        response=response,
        orientation=orientation,
        eligible=eligible,
        valid=valid,
    )


__all__ = ["ClarkRollingHoughResult", "clark_rolling_hough"]
