"""Source-fidelity and behavioral tests for oriented tensor voting."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from phenotypic.sdk_.reconnect._tensor_voting import (
    _source_window_radius,
    _tensor_vote_components,
    tensor_vote,
)


_FIXTURE_DIRECTORY = Path("tests/fixtures/reconnect/tensor_voting")


def _canonical_fixture_hash(
    fixture: dict[str, np.ndarray], required_keys: list[str]
) -> str:
    """Hash MAT content while excluding its nondeterministic container header."""
    digest = hashlib.sha256()
    for key in sorted(required_keys):
        value = np.asarray(fixture[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(struct.pack("<Q", value.ndim))
        for extent in value.shape:
            digest.update(struct.pack("<Q", extent))
        if value.dtype.kind in "US":
            digest.update(b"utf8\0")
            for item in value.reshape(-1, order="C"):
                encoded = str(item).encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
        else:
            canonical_dtype = value.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
            digest.update(canonical_dtype.str.encode("ascii") + b"\0")
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _load_source_fixture() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    manifest = json.loads(
        (_FIXTURE_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    fixture_path = _FIXTURE_DIRECTORY / str(manifest["fixture"])
    fixture = loadmat(fixture_path)
    present_keys = {key for key in fixture if not key.startswith("__")}
    assert present_keys == set(manifest["required_keys"])
    observed_hash = _canonical_fixture_hash(fixture, manifest["required_keys"])
    assert observed_hash == manifest["canonical_content_sha256"]
    return manifest, fixture


def _source_comparison_bound(expected: np.ndarray, term_count: int) -> np.ndarray:
    """Bound direct sums plus component/eigenvalue reassociation roundoff."""
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma = (term_count * unit_roundoff) / (1.0 - term_count * unit_roundoff)
    # Each vote component includes two trig products, one decay product, and a
    # source-vs-libm transcendental evaluation. Sixteen ulps per term bounds the
    # observed cross-runtime component error; Weyl adds a factor of two to saliency.
    return 32.0 * gamma * np.maximum(np.abs(expected), 1.0)


def test_source_generated_fixture_matches_all_outputs() -> None:
    """The port matches every MATLAB-visible numeric output in the fixture."""
    _, fixture = _load_source_fixture()
    response = fixture["response"]
    theta = fixture["theta"]
    sigma = float(fixture["sigma"][0, 0])

    component_a, component_b, component_d = _tensor_vote_components(
        response, theta, sigma
    )
    stick, ball = tensor_vote(response, theta, sigma)

    # Five voters plus the retained input tensor can contribute at a pixel.
    for actual, key in (
        (component_a, "accumulated_a"),
        (component_b, "accumulated_b"),
        (component_d, "accumulated_d"),
        (stick, "stick"),
        (ball, "ball"),
    ):
        expected = fixture[key]
        bound = _source_comparison_bound(expected, term_count=6)
        assert np.all(np.abs(actual - expected) <= bound), key

    assert_array_equal(
        fixture["window_size"], np.array([[_source_window_radius(sigma) * 2 + 1]])
    )


@pytest.mark.parametrize(
    ("response", "theta", "sigma", "message"),
    [
        (np.zeros(3), np.zeros(3), 2.0, "response must be a 2-D array"),
        (np.zeros((2, 2)), np.zeros(4), 2.0, "theta must be a 2-D array"),
        (
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            2.0,
            "response and theta must have the same shape",
        ),
        (
            np.array([[np.nan]]),
            np.zeros((1, 1)),
            2.0,
            "response must contain only finite values",
        ),
        (
            np.array([[-1.0]]),
            np.zeros((1, 1)),
            2.0,
            "response must be nonnegative",
        ),
        (
            np.ones((1, 1)),
            np.array([[np.inf]]),
            2.0,
            "theta must contain only finite values",
        ),
        (np.ones((1, 1)), np.zeros((1, 1)), 0.0, "sigma must be finite"),
        (np.ones((1, 1)), np.zeros((1, 1)), np.inf, "sigma must be finite"),
    ],
)
def test_invalid_inputs_raise(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        tensor_vote(response, theta, sigma)


def test_zero_and_empty_inputs_preserve_shape_and_dtype() -> None:
    for shape in ((0, 0), (0, 3), (4, 5)):
        stick, ball = tensor_vote(np.zeros(shape), np.zeros(shape), 2.25)
        assert stick.shape == shape
        assert ball.shape == shape
        assert stick.dtype == np.float64
        assert ball.dtype == np.float64
        assert_array_equal(stick, np.zeros(shape))
        assert_array_equal(ball, np.zeros(shape))


def test_isolated_token_has_retained_and_cast_self_vote() -> None:
    response = np.zeros((7, 7), dtype=np.float64)
    theta = np.zeros_like(response)
    response[3, 3] = 1.75

    component_a, component_b, component_d = _tensor_vote_components(
        response, theta, 2.25
    )
    stick, ball = tensor_vote(response, theta, 2.25)

    assert component_a[3, 3] == pytest.approx(0.0, abs=1e-15)
    assert component_b[3, 3] == pytest.approx(0.0, abs=1e-15)
    assert component_d[3, 3] == pytest.approx(3.5, abs=1e-14)
    assert stick[3, 3] == pytest.approx(3.5, abs=1e-14)
    assert ball[3, 3] == pytest.approx(0.0, abs=1e-14)


def test_closed_form_saliency_decomposition() -> None:
    response = np.zeros((9, 9), dtype=np.float64)
    theta = np.zeros_like(response)
    response[4, 2] = 1.0
    response[4, 6] = 1.0
    theta[4, 6] = np.pi / 2.0

    component_a, component_b, component_d = _tensor_vote_components(
        response, theta, 2.25
    )
    expected_eigenvalues = np.linalg.eigvalsh(
        np.stack(
            (
                np.stack((component_a, component_b), axis=-1),
                np.stack((component_b, component_d), axis=-1),
            ),
            axis=-2,
        )
    )
    stick, ball = tensor_vote(response, theta, 2.25)
    assert_allclose(stick, expected_eigenvalues[..., 1] - expected_eigenvalues[..., 0])
    eigensolver_bound = (
        8.0
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(component_a))), float(np.max(np.abs(component_d))), 1.0)
    )
    assert_allclose(ball, expected_eigenvalues[..., 0], rtol=0.0, atol=eigensolver_bound)


def test_positive_linearity_for_fixed_active_mask() -> None:
    response = np.zeros((9, 11), dtype=np.float64)
    theta = np.zeros_like(response)
    response[4, 2] = 0.4
    response[4, 8] = 1.2
    theta[4, 8] = np.pi / 6.0

    stick, ball = tensor_vote(response, theta, 2.25)
    scaled_stick, scaled_ball = tensor_vote(3.5 * response, theta, 2.25)
    assert_allclose(scaled_stick, 3.5 * stick, rtol=3e-15, atol=3e-15)
    assert_allclose(scaled_ball, 3.5 * ball, rtol=3e-15, atol=3e-15)


def test_axial_pi_periodicity() -> None:
    response = np.zeros((9, 9), dtype=np.float64)
    response[2, 4] = 0.75
    response[6, 4] = 1.25
    theta = np.linspace(-0.7, 0.8, response.size).reshape(response.shape)
    actual = tensor_vote(response, theta, 2.25)
    shifted = tensor_vote(response, theta + np.pi, 2.25)
    for field, shifted_field in zip(actual, shifted, strict=True):
        assert_allclose(field, shifted_field, rtol=3e-14, atol=3e-14)


def test_rotation_and_transpose_controls() -> None:
    response = np.zeros((11, 13), dtype=np.float64)
    theta = np.zeros_like(response)
    response[5, 2] = 1.0
    response[7, 8] = 0.7
    theta[7, 8] = 0.3
    stick, ball = tensor_vote(response, theta, 2.25)

    rotated = tensor_vote(
        np.rot90(response), np.rot90(theta) + np.pi / 2.0, 2.25
    )
    transposed = tensor_vote(response.T, np.pi / 2.0 - theta.T, 2.25)
    covariance_bound = 32.0 * np.finfo(np.float64).eps
    assert_allclose(rotated[0], np.rot90(stick), rtol=0.0, atol=covariance_bound)
    assert_allclose(rotated[1], np.rot90(ball), rtol=0.0, atol=covariance_bound)
    assert_allclose(transposed[0], stick.T, rtol=0.0, atol=covariance_bound)
    assert_allclose(transposed[1], ball.T, rtol=0.0, atol=covariance_bound)


def test_boundary_vote_is_cropped_not_wrapped() -> None:
    response = np.zeros((15, 15), dtype=np.float64)
    theta = np.zeros_like(response)
    response[0, 0] = 1.0
    stick, ball = tensor_vote(response, theta, 2.25)

    assert stick[0, 0] == pytest.approx(2.0, abs=1e-14)
    assert stick[0, -1] == 0.0
    assert ball[0, -1] == 0.0
    assert stick[-1, -1] == 0.0
    assert ball[-1, -1] == 0.0


def test_threshold_crossing_is_discontinuous_only_at_zero_activity() -> None:
    theta = np.zeros((5, 5), dtype=np.float64)
    zero = np.zeros_like(theta)
    tiny = zero.copy()
    tiny[2, 2] = 1e-100

    zero_stick, _ = tensor_vote(zero, theta, 1.0)
    tiny_stick, _ = tensor_vote(tiny, theta, 1.0)
    assert_array_equal(zero_stick, zero)
    assert tiny_stick[2, 2] == pytest.approx(2e-100, rel=1e-14)


def test_inputs_are_not_mutated() -> None:
    response = np.zeros((7, 7), dtype=np.float32)
    theta = np.full((7, 7), 0.3, dtype=np.float32)
    response[3, 3] = 1.0
    response_before = response.copy()
    theta_before = theta.copy()

    tensor_vote(response, theta, 2.25)

    assert_array_equal(response, response_before)
    assert_array_equal(theta, theta_before)


def test_support_radius_matches_source_rounding_not_plain_ceiling() -> None:
    sigma = 0.52
    exact_radius = math_floor_ceil_source_radius(sigma)
    assert _source_window_radius(sigma) == exact_radius
    assert exact_radius != int(np.ceil(sigma * np.sqrt(-np.log(0.01))))


def math_floor_ceil_source_radius(sigma: float) -> int:
    """Independent spelling of the MATLAB odd-window conversion."""
    doubled_extent = np.sqrt(-np.log(0.01) * sigma**2) * 2.0
    return int(np.floor(np.ceil(doubled_extent) / 2.0))
