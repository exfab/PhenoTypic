"""Source-fidelity and behavioral tests for the Clark Rolling Hough core."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phenotypic.sdk_.reconnect._rolling_hough import (
    _axial_orientation,
    _center_line_geometry,
    _clark_preprocessing,
    _threshold_counts,
    clark_rolling_hough,
)


_FIXTURE = (
    Path(__file__).parents[3]
    / "fixtures"
    / "reconnect"
    / "rolling_hough"
    / "clark_rht_source.npz"
)


@pytest.mark.parametrize("case_index", range(1, 6))
def test_source_fixture_matches_every_core_output_and_intermediate(
    case_index: int,
) -> None:
    """All captured preprocessing and result fields match the pinned source."""
    prefix = f"c{case_index:02d}_"
    with np.load(_FIXTURE, allow_pickle=False) as fixture:
        image = fixture[prefix + "image"]
        diameter = int(fixture[prefix + "window_diameter"])
        smoothing_radius = int(fixture[prefix + "smoothing_radius"])
        threshold_fraction = float(fixture[prefix + "threshold_fraction"])
        (
            smoothing_kernel,
            smoothing_mask,
            eligible,
            correlated,
            smoothed,
            unsharp,
            bitmask,
        ) = _clark_preprocessing(image, diameter, smoothing_radius)
        result = clark_rolling_hough(
            image,
            window_diameter=diameter,
            smoothing_radius=smoothing_radius,
            threshold_fraction=threshold_fraction,
        )
        circular_window, center_lines = _center_line_geometry(
            diameter, result.theta
        )

        for actual, key in (
            (smoothing_kernel, "smoothing_kernel"),
            (smoothing_mask, "smoothing_mask"),
            (eligible, "window_mask"),
            (correlated, "correlated"),
            (smoothed, "smoothed"),
            (unsharp, "unsharp"),
            (bitmask, "bitmask"),
            (circular_window, "circular_window"),
            (center_lines, "center_lines"),
            (result.theta, "theta"),
            (result.support_counts, "support_counts"),
            (result.raw_counts, "raw_counts"),
            (result.threshold_residual, "threshold_residual"),
            (result.response, "raw_response"),
            (result.orientation, "derived_orientation"),
            (result.eligible, "window_mask"),
            (result.valid, "valid"),
        ):
            assert_array_equal(actual, fixture[prefix + key], err_msg=key)

        # Negative rejected bins retain the source multiplication's signed zero.
        assert_array_equal(
            np.signbit(result.threshold_residual),
            np.signbit(fixture[prefix + "threshold_residual"]),
        )


def test_diameter_eleven_geometry_counts_and_angles_match_source() -> None:
    """Round-to-even rho-zero lines preserve every local source template."""
    with np.load(_FIXTURE, allow_pickle=False) as fixture:
        theta = fixture["local_theta"]
        circular_window, center_lines = _center_line_geometry(11, theta)
        assert_array_equal(circular_window, fixture["local_circle_window"])
        support = np.einsum("ijt,ij->t", center_lines, circular_window)
        assert_array_equal(support, fixture["local_support_counts"])

        for name in (
            "horizontal",
            "vertical",
            "diagonal",
            "crossing",
            "gap",
            "circle",
        ):
            window = fixture[f"local_{name}_window"]
            counts = np.einsum("ijt,ij->t", center_lines, window)
            assert_array_equal(counts, fixture[f"local_{name}_counts"])
            angle = _axial_orientation(counts.astype(np.float64), theta)
            assert np.float64(angle).view(np.uint64) == fixture[
                f"local_{name}_source_angle"
            ].view(np.uint64)


def test_geometry_contains_round_to_nearest_even_half_ties() -> None:
    """NumPy half ties are rounded to even before selecting rho zero."""
    theta = np.array([0.0], dtype=np.float64)
    _, center_lines = _center_line_geometry(3, theta)

    # x = +/-0.5 cannot occur on the integer grid, so calibrate the exact
    # primitive used by the geometry with synthetic rho coordinates.
    rounded = np.round(np.array([-1.5, -0.5, 0.5, 1.5]))
    assert_array_equal(rounded, [-2.0, -0.0, 0.0, 2.0])
    assert center_lines[1, 1, 0] == 1
    assert center_lines[1, 0, 0] == 0


def test_threshold_equality_is_zero_and_rejected_values_keep_negative_zero() -> (
    None
):
    """The source >= mask accepts equality numerically but sparse validity does not."""
    counts = np.array([[2, 1, 0]], dtype=np.int64)
    support = np.array([2, 2, 2], dtype=np.int64)
    residual = _threshold_counts(counts, support, 1.0)

    assert_array_equal(residual, [[0.0, 0.0, 0.0]])
    assert_array_equal(np.signbit(residual), [[False, True, True]])
    assert not np.any(residual > 0.0)


def test_raw_response_is_not_globally_normalized() -> None:
    image = np.zeros((17, 19), dtype=np.float64)
    image[8, 2:17] = 5.0
    result = clark_rolling_hough(
        image,
        window_diameter=5,
        smoothing_radius=1,
        threshold_fraction=0.2,
    )
    assert result.response.max() > 1.0
    assert_array_equal(
        result.response, np.sum(result.threshold_residual, axis=2)
    )


def test_constant_image_returns_defined_empty_result() -> None:
    result = clark_rolling_hough(
        np.full((15, 17), 7.25, dtype=np.float64),
        window_diameter=5,
        smoothing_radius=1,
        threshold_fraction=0.7,
    )
    assert not np.any(result.raw_counts)
    assert not np.any(result.threshold_residual)
    assert not np.any(result.response)
    assert not np.any(result.valid)
    assert np.all(np.isnan(result.orientation))


def test_nonfinite_pixels_invalidate_both_source_halos() -> None:
    image = np.zeros((17, 17), dtype=np.float64)
    image[8, :] = 1.0
    image[8, 8] = np.nan
    (
        _,
        smoothing_mask,
        eligible,
        _,
        _,
        _,
        _,
    ) = _clark_preprocessing(image, 5, 1)

    assert not smoothing_mask[8, 8]
    assert not smoothing_mask[8, 7]
    assert not eligible[8, 5:12].any()
    assert eligible.dtype == np.bool_


def test_orientation_is_axial_hough_normal_with_source_mapping() -> None:
    theta = np.linspace(0.0, np.pi, 8, endpoint=False)
    horizontal_normal = np.zeros(8)
    horizontal_normal[0] = 1.0
    vertical_normal = np.zeros(8)
    vertical_normal[4] = 1.0

    assert _axial_orientation(horizontal_normal, theta) == pytest.approx(np.pi)
    assert _axial_orientation(vertical_normal, theta) == pytest.approx(
        np.pi / 2.0
    )
    assert _axial_orientation(
        horizontal_normal, theta + np.pi
    ) == pytest.approx(np.pi)


def test_outputs_have_frozen_shapes_and_dtypes() -> None:
    image = np.zeros((9, 11), dtype=np.float64)
    image[4, 2:9] = 1.0
    result = clark_rolling_hough(
        image,
        window_diameter=3,
        smoothing_radius=1,
        threshold_fraction=0.5,
    )
    theta_count = math.ceil(math.pi * 2.0 / math.sqrt(2.0))

    assert result.theta.shape == (theta_count,)
    assert result.support_counts.shape == (theta_count,)
    assert result.raw_counts.shape == image.shape + (theta_count,)
    assert result.threshold_residual.shape == image.shape + (theta_count,)
    assert result.response.shape == image.shape
    assert result.orientation.shape == image.shape
    assert result.eligible.shape == image.shape
    assert result.valid.shape == image.shape
    assert result.theta.dtype == np.float64
    assert result.support_counts.dtype == np.int64
    assert result.raw_counts.dtype == np.int64
    assert result.threshold_residual.dtype == np.float64
    assert result.response.dtype == np.float64
    assert result.orientation.dtype == np.float64
    assert result.eligible.dtype == np.bool_
    assert result.valid.dtype == np.bool_


def test_input_is_not_mutated() -> None:
    image = np.arange(225, dtype=np.float64).reshape(15, 15)
    image[5, 5] = np.nan
    before = image.copy()
    clark_rolling_hough(
        image,
        window_diameter=5,
        smoothing_radius=1,
        threshold_fraction=0.5,
    )
    assert_array_equal(image, before)


@pytest.mark.parametrize(
    ("image", "exception", "message"),
    [
        ([[1.0]], TypeError, "image must be a numpy.ndarray"),
        (np.array(1.0), ValueError, "image must be two-dimensional"),
        (
            np.zeros((0, 2), dtype=np.float64),
            ValueError,
            "image must not be empty",
        ),
        (
            np.zeros((1, 1), dtype=np.float32),
            TypeError,
            "dtype exactly float64",
        ),
        (np.zeros((1, 1), dtype=np.int64), TypeError, "dtype exactly float64"),
        (np.zeros((1, 1), dtype=np.bool_), TypeError, "dtype exactly float64"),
        (
            np.zeros((1, 1), dtype=np.complex128),
            TypeError,
            "dtype exactly float64",
        ),
    ],
)
def test_invalid_images_raise(
    image: object, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        clark_rolling_hough(
            image,  # type: ignore[arg-type]
            window_diameter=5,
            smoothing_radius=1,
            threshold_fraction=0.5,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"window_diameter": True},
            "window_diameter must be a positive odd integer",
        ),
        (
            {"window_diameter": 0},
            "window_diameter must be a positive odd integer",
        ),
        (
            {"window_diameter": 4},
            "window_diameter must be a positive odd integer",
        ),
        (
            {"window_diameter": 5.0},
            "window_diameter must be a positive odd integer",
        ),
        (
            {"smoothing_radius": False},
            "smoothing_radius must be a positive integer",
        ),
        (
            {"smoothing_radius": 0},
            "smoothing_radius must be a positive integer",
        ),
        (
            {"smoothing_radius": 1.0},
            "smoothing_radius must be a positive integer",
        ),
        ({"threshold_fraction": np.nan}, "threshold_fraction must be finite"),
        ({"threshold_fraction": np.inf}, "threshold_fraction must be finite"),
        ({"threshold_fraction": -0.1}, "threshold_fraction must be in"),
        ({"threshold_fraction": 1.1}, "threshold_fraction must be in"),
        (
            {"threshold_fraction": True},
            "threshold_fraction must be a real number",
        ),
    ],
)
def test_invalid_parameters_raise(
    kwargs: dict[str, object], message: str
) -> None:
    parameters: dict[str, object] = {
        "window_diameter": 5,
        "smoothing_radius": 1,
        "threshold_fraction": 0.5,
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        clark_rolling_hough(
            np.ones((9, 9), dtype=np.float64),
            **parameters,  # type: ignore[arg-type]
        )


def test_small_image_has_no_eligible_centers_without_border_fabrication() -> (
    None
):
    result = clark_rolling_hough(
        np.ones((3, 3), dtype=np.float64),
        window_diameter=7,
        smoothing_radius=2,
        threshold_fraction=0.5,
    )
    assert not np.any(result.eligible)
    assert not np.any(result.raw_counts)
    assert not np.any(result.valid)


def test_orientation_roundoff_bound_is_ulp_derived() -> None:
    theta = np.linspace(0.0, np.pi, 23, endpoint=False)
    weights = np.arange(1.0, 24.0)
    actual = _axial_orientation(weights, theta)
    doubled = 2.0 * theta
    expected = np.pi - math.fmod(
        0.5
        * math.atan2(
            math.fsum((weights * np.sin(doubled)).tolist()),
            math.fsum((weights * np.cos(doubled)).tolist()),
        )
        + np.pi,
        np.pi,
    )
    # The production/source reduction and fsum may reassociate 23 additions.
    bound = 32.0 * np.spacing(max(abs(actual), abs(expected), 1.0))
    assert_allclose(actual, expected, rtol=0.0, atol=bound)
