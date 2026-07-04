"""Analytic-phantom tests for the structure-tensor orientation field."""
from __future__ import annotations

import numpy as np

from phenotypic.util._orientation_field import orientation_field


def _parallel_stripes(n=64, period=8.0):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.sin(2 * np.pi * yy / period).astype(np.float64)


def test_parallel_bundle_is_coherent_and_non_turning():
    phi, coh, grad = orientation_field(_parallel_stripes(), sigma_d=1.5, sigma_i=4.0)
    interior = (slice(12, 52), slice(12, 52))
    assert coh[interior].mean() > 0.8          # highly coherent
    assert grad[interior].mean() < 1e-2        # orientation ~constant -> no turning


def test_isotropic_noise_is_incoherent():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((64, 64))
    _, coh, _ = orientation_field(field, sigma_d=1.5, sigma_i=6.0)
    assert coh[16:48, 16:48].mean() < 0.35     # no dominant orientation


def test_output_shapes_and_ranges():
    phi, coh, grad = orientation_field(_parallel_stripes())
    for a in (phi, coh, grad):
        assert a.shape == (64, 64)
    assert np.all(coh >= -1e-9) and np.all(coh <= 1 + 1e-9)
    assert np.all(grad >= 0)
    assert np.all(phi <= np.pi / 2 + 1e-9) and np.all(phi > -np.pi / 2 - 1e-9)
