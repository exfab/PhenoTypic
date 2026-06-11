import numpy as np
import pytest
from phenotypic.util._robust_color_stats import robust_color_center


def test_robust_center_symmetric_cloud():
    pts = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    assert np.allclose(robust_color_center(pts), [0.0, 0.0, 0.0], atol=1e-3)


def test_robust_center_resists_single_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (99, 1))
    pts = np.vstack([cluster, [50_000.0, 50_000.0, 50_000.0]])
    assert np.allclose(robust_color_center(pts), [50.0, 10.0, 20.0], atol=1.0)


def test_robust_center_single_point_returns_it():
    assert np.allclose(robust_color_center(np.array([[3.0, 4.0, 5.0]])), [3.0, 4.0, 5.0])


def test_robust_center_identical_points():
    assert np.allclose(robust_color_center(np.tile([7.0, 7.0, 7.0], (5, 1))), [7.0, 7.0, 7.0])


def test_robust_center_empty_returns_nan():
    out = robust_color_center(np.empty((0, 3)))
    assert out.shape == (3,) and np.isnan(out).all()
from phenotypic.util._robust_color_stats import (
    medoid_ciede2000,
    delta_e2000_spread,
)


def test_medoid_is_an_actual_input_pixel():
    rng = np.random.default_rng(0)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(50, 3))
    center, deltas = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert any(np.allclose(center, p) for p in lab)  # center IS a real pixel
    assert deltas.shape == (50,)
    assert np.all(deltas >= 0)


def test_medoid_central_for_one_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (40, 1))
    lab = np.vstack([cluster, [10.0, -50.0, 60.0]])
    center, _ = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert np.allclose(center, [50.0, 10.0, 20.0])


def test_medoid_subsample_is_reproducible():
    rng = np.random.default_rng(1)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(5000, 3))
    c1, d1 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    c2, d2 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    assert np.allclose(c1, c2)
    assert d1.shape == (5000,)  # spread uses ALL pixels, not the subsample


def test_medoid_single_and_empty():
    c, d = medoid_ciede2000(np.array([[40.0, 5.0, -5.0]]))
    assert np.allclose(c, [40.0, 5.0, -5.0]) and np.allclose(d, [0.0])
    c0, d0 = medoid_ciede2000(np.empty((0, 3)))
    assert np.isnan(c0).all() and d0.size == 0


def test_delta_e2000_spread_values():
    deltas = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    med, mean, p95 = delta_e2000_spread(deltas)
    assert med == 2.0
    assert mean == pytest.approx(21.2)
    assert p95 == pytest.approx(np.percentile(deltas, 95))


def test_delta_e2000_spread_empty_is_nan():
    med, mean, p95 = delta_e2000_spread(np.array([]))
    assert np.isnan(med) and np.isnan(mean) and np.isnan(p95)
