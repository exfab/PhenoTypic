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
