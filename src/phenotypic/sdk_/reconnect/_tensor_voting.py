"""Oriented 2-D stick tensor voting.

This module ports only Trevor Linton's ``calc_vote_stick`` stage from MATLAB
Central File Exchange 21051, version 1.0.0.0. The source reconciliation and
license notices are stored under the matching design specification's
``refs/tensor_voting`` directory.
"""

from __future__ import annotations

import math

import numba
import numpy as np


def _source_window_radius(sigma: float) -> int:
    """Return half the exact odd support window used by the MATLAB source."""
    window_size = (
        math.floor(
            math.ceil(math.sqrt(-math.log(0.01) * sigma**2) * 2.0) / 2.0
        )
        * 2
        + 1
    )
    return (window_size - 1) // 2


def _validate_tensor_vote_inputs(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and copy tensor-voting inputs into contiguous float64 arrays."""
    response_array = np.asarray(response, dtype=np.float64)
    theta_array = np.asarray(theta, dtype=np.float64)

    if response_array.ndim != 2:
        raise ValueError("response must be a 2-D array")
    if theta_array.ndim != 2:
        raise ValueError("theta must be a 2-D array")
    if response_array.shape != theta_array.shape:
        raise ValueError("response and theta must have the same shape")
    if not np.all(np.isfinite(response_array)):
        raise ValueError("response must contain only finite values")
    if np.any(response_array < 0.0):
        raise ValueError("response must be nonnegative")
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values")

    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or sigma_value <= 0.0:
        raise ValueError("sigma must be finite and greater than zero")

    return (
        np.ascontiguousarray(response_array),
        np.ascontiguousarray(theta_array),
        sigma_value,
    )


@numba.njit(cache=True)
def _tensor_vote_components_kernel(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the three symmetric tensor components in source order."""
    height, width = response.shape
    component_a = np.empty((height, width), dtype=np.float64)
    component_b = np.empty((height, width), dtype=np.float64)
    component_d = np.empty((height, width), dtype=np.float64)

    # Contract adaptation: encode the tangent as Linton's rank-1 normal tensor.
    for row in range(height):
        for column in range(width):
            normal_x = -math.sin(theta[row, column])
            normal_y = math.cos(theta[row, column])
            amplitude = response[row, column]
            component_a[row, column] = amplitude * normal_x * normal_x
            component_b[row, column] = amplitude * normal_x * normal_y
            component_d[row, column] = amplitude * normal_y * normal_y

    curvature_weight = (-16.0 * math.log2(0.1) * (sigma - 1.0)) / (math.pi * math.pi)
    sigma_squared = sigma * sigma

    # MATLAB find() traverses columns first. Preserve that accumulation order.
    for voter_column in range(width):
        for voter_row in range(height):
            # The source decomposes the complete input field once before any vote
            # is accumulated. Rebuild that immutable input tensor here rather than
            # allowing received votes to cast again later in the traversal.
            input_normal_x = -math.sin(theta[voter_row, voter_column])
            input_normal_y = math.cos(theta[voter_row, voter_column])
            input_amplitude = response[voter_row, voter_column]
            input_a = input_amplitude * input_normal_x * input_normal_x
            input_b = input_amplitude * input_normal_x * input_normal_y
            input_d = input_amplitude * input_normal_y * input_normal_y
            half_trace = (input_a + input_d) / 2.0
            centered_a = input_a - half_trace
            half_delta = math.sqrt(centered_a * centered_a + input_b * input_b)
            lambda1 = half_trace + half_delta
            lambda2 = half_trace - half_delta
            vote_amplitude = lambda1 - lambda2
            if vote_amplitude <= 0.0:
                continue

            eigen_angle = math.atan2(half_delta - centered_a, input_b)
            principal_x = math.cos(eigen_angle)
            principal_y = math.sin(eigen_angle)
            tangent_x = -principal_y
            tangent_y = principal_x
            base_angle = math.atan2(tangent_y, tangent_x)

            first_row = max(0, voter_row - radius)
            last_row = min(height - 1, voter_row + radius)
            first_column = max(0, voter_column - radius)
            last_column = min(width - 1, voter_column + radius)

            for target_row in range(first_row, last_row + 1):
                offset_y = float(voter_row - target_row)
                for target_column in range(first_column, last_column + 1):
                    offset_x = float(target_column - voter_column)

                    local_x = tangent_x * offset_x + tangent_y * offset_y
                    local_y = -tangent_y * offset_x + tangent_x * offset_y
                    local_angle = math.atan2(local_y, local_x)

                    # Vote tensor direction uses the signed, unscaled local angle.
                    vote_angle = 2.0 * local_angle + base_angle
                    vote_normal_x = -math.sin(vote_angle)
                    vote_normal_y = math.cos(vote_angle)

                    attenuation_angle = abs(local_angle)
                    if attenuation_angle > math.pi / 2.0:
                        attenuation_angle = math.pi - attenuation_angle
                    # The selected archive explicitly includes this fourfold fork.
                    attenuation_angle *= 4.0
                    if attenuation_angle > math.pi / 2.0:
                        continue

                    distance = math.sqrt(local_x * local_x + local_y * local_y)
                    if distance == 0.0 or attenuation_angle == 0.0:
                        arc_length = distance
                    else:
                        arc_length = attenuation_angle * distance / math.sin(attenuation_angle)
                    if distance == 0.0:
                        curvature = 0.0
                    else:
                        curvature = 2.0 * math.sin(attenuation_angle) / distance

                    decay = math.exp(
                        -(
                            arc_length * arc_length
                            + curvature_weight * curvature * curvature
                        )
                        / sigma_squared
                    )
                    weighted_vote = vote_amplitude * decay
                    component_a[target_row, target_column] += (
                        weighted_vote * vote_normal_x * vote_normal_x
                    )
                    component_b[target_row, target_column] += (
                        weighted_vote * vote_normal_x * vote_normal_y
                    )
                    component_d[target_row, target_column] += (
                        weighted_vote * vote_normal_y * vote_normal_y
                    )

    return component_a, component_b, component_d


def _tensor_vote_components(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return accumulated Cartesian ``(xx, xy, yy)`` tensor components."""
    response_array, theta_array, sigma_value = _validate_tensor_vote_inputs(
        response, theta, sigma
    )
    radius = _source_window_radius(sigma_value)
    return _tensor_vote_components_kernel(response_array, theta_array, sigma_value, radius)


def tensor_vote(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast oriented stick votes and return raw curve and junction saliencies.

    ``theta`` is an axial tangent angle in Cartesian radians, with x increasing
    along image columns and y increasing upward. Input pixels with positive
    ``response`` cast votes. The source's finite support is clipped at image
    boundaries, and its initial rank-1 tensor is retained in the accumulator.

    Runtime is ``O(H * W + A * (2 * r + 1) ** 2)`` for image shape ``H x W``,
    ``A`` positive response pixels, and source support radius ``r``. Callers
    should therefore provide a sparse token field, not an everywhere-positive
    dense image.

    Args:
        response: Finite, nonnegative 2-D token-amplitude array.
        theta: Same-shaped finite axial tangent field in radians.
        sigma: Positive source scale. It determines both vote decay and support.

    Returns:
        A tuple ``(stick_saliency, ball_saliency)`` of unnormalized float64
        arrays. Stick saliency is ``lambda1 - lambda2`` and ball saliency is
        ``lambda2`` after accumulated-tensor decomposition.

    Raises:
        ValueError: If arrays are not finite same-shaped 2-D inputs, response is
            negative, or sigma is not finite and positive.
    """
    component_a, component_b, component_d = _tensor_vote_components(
        response, theta, sigma
    )
    delta = np.hypot(component_a - component_d, 2.0 * component_b)
    stick_saliency = delta
    ball_saliency = (component_a + component_d - delta) / 2.0
    return stick_saliency, ball_saliency
