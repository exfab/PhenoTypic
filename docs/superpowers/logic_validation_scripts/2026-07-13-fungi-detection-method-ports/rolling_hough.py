"""Validate A09 Clark Rolling Hough numerical claims from source fixtures.

This script imports neither ``phenotypic`` nor either pinned implementation. It
re-derives the load-bearing transform equations directly from fixture inputs and
captured geometric masks using only the standard library and NumPy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import numpy.typing as npt


ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIRECTORY = ROOT / "tests/fixtures/reconnect/rolling_hough"
MANIFEST_PATH = FIXTURE_DIRECTORY / "manifest.json"


def _rederive_orientation(
    residual: npt.NDArray[np.float64], theta: npt.NDArray[np.float64]
) -> float:
    """Apply the published doubled-angle axial collapse in float64."""

    y_component = np.sum(residual * np.sin(2.0 * theta))
    x_component = np.sum(residual * np.cos(2.0 * theta))
    rough_angle = 0.5 * np.arctan2(y_component, x_component)
    return float(np.pi - math.fmod(float(rough_angle + np.pi), float(np.pi)))


def _assert_case_geometry(
    fixture: dict[str, npt.NDArray[np.generic]], prefix: str
) -> int:
    """Re-derive one full case from captured masks and image-local windows."""

    image = fixture[prefix + "image"]
    if image.dtype != np.float64:
        raise AssertionError(f"{prefix}: frozen input dtype is not float64")
    diameter = int(fixture[prefix + "window_diameter"])
    radius = diameter // 2
    theta = fixture[prefix + "theta"].astype(np.float64)
    expected_theta_count = math.ceil(math.pi * (diameter - 1) / math.sqrt(2.0))
    if theta.shape != (expected_theta_count,):
        raise AssertionError(f"{prefix}: theta-count equation drifted")
    np.testing.assert_array_equal(
        theta,
        np.linspace(0.0, np.pi, expected_theta_count, endpoint=False),
    )

    center_lines = fixture[prefix + "center_lines"].astype(np.int64)
    circular_window = fixture[prefix + "circular_window"].astype(np.int64)
    support = np.sum(center_lines * circular_window[:, :, None], axis=(0, 1))
    np.testing.assert_array_equal(support, fixture[prefix + "support_counts"])

    bitmask = fixture[prefix + "bitmask"].astype(np.int64)
    eligible = fixture[prefix + "window_mask"].astype(bool)
    raw_counts = fixture[prefix + "raw_counts"]
    expected_counts = np.zeros_like(raw_counts)
    for row, column in zip(*np.nonzero(eligible), strict=True):
        window = bitmask[
            row - radius : row + radius + 1,
            column - radius : column + radius + 1,
        ]
        expected_counts[row, column] = np.sum(
            center_lines * window[:, :, None], axis=(0, 1)
        )
    np.testing.assert_array_equal(expected_counts, raw_counts)

    fraction = float(fixture[prefix + "threshold_fraction"])
    expected_residual = np.zeros(raw_counts.shape, dtype=np.float64)
    candidate = raw_counts[eligible] / support - fraction
    candidate[candidate < 0.0] = 0.0
    expected_residual[eligible] = candidate
    residual = fixture[prefix + "threshold_residual"]
    np.testing.assert_array_equal(expected_residual, residual)
    valid = np.any(expected_residual > 0.0, axis=2)
    if fixture[prefix + "valid"].dtype != np.bool_:
        raise AssertionError(f"{prefix}: dense validity is not Boolean")
    np.testing.assert_array_equal(valid, fixture[prefix + "valid"])
    np.testing.assert_array_equal(
        np.sum(expected_residual, axis=2), fixture[prefix + "raw_response"]
    )

    orientation = fixture[prefix + "derived_orientation"]
    for row, column in zip(*np.nonzero(valid), strict=True):
        expected_angle = _rederive_orientation(expected_residual[row, column], theta)
        expected_bits = int(np.float64(expected_angle).view(np.uint64))
        observed_bits = int(np.float64(orientation[row, column]).view(np.uint64))
        if observed_bits != expected_bits:
            raise AssertionError(
                f"{prefix}: orientation drifted from the zero-ULP pinned equation"
            )
    if not np.all(np.isnan(orientation[~valid])):
        raise AssertionError(f"{prefix}: invalid orientation sentinel drifted")
    return int(np.count_nonzero(eligible))


def validate_rolling_hough_claims() -> None:
    """Run exact geometry, threshold, validity, response, and angle checks."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2:
        raise AssertionError("A09 fixture schema drifted")
    with np.load(FIXTURE_DIRECTORY / manifest["fixture"], allow_pickle=False) as archive:
        fixture = {key: np.asarray(archive[key]) for key in archive.files}
    evaluated_centers = sum(
        _assert_case_geometry(fixture, f"c{case_index:02d}_")
        for case_index in range(1, 6)
    )

    # Applying the transform to its own full circular support creates exact
    # threshold equality at fraction one, which must remain numeric zero.
    np.testing.assert_array_equal(
        fixture["local_circle_fraction_one_residual"], 0.0
    )

    print("A09 Rolling Hough logic validation passed")
    print(f"source-fixture centers independently re-counted: {evaluated_centers}")
    print("integer geometry/count tolerance: exact")
    print("float residual/response tolerance: exact pinned-operation order")
    print("orientation tolerance: zero ULP on the pinned source fixture")
    print("coherence: deferred and absent")


if __name__ == "__main__":
    try:
        validate_rolling_hough_claims()
    except Exception as error:
        print(f"A09 Rolling Hough logic validation failed: {error}", file=sys.stderr)
        raise
