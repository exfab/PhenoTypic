import numpy as np
import pytest

from phenotypic.abc_ import GridFinder
from phenotypic.grid import CenteredAutoGridFinder, CenteredAutoGridFinderFallbackWarning


def test_is_gridfinder_and_constructs_keyword_only():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    assert isinstance(f, GridFinder)
    assert (f.nrows, f.ncols) == (8, 12)
    with pytest.raises(Exception):
        CenteredAutoGridFinder(8, 12)  # positional construction is rejected


def test_warning_class_is_userwarning():
    assert issubclass(CenteredAutoGridFinderFallbackWarning, UserWarning)


def test_compute_bounds_centers_fit_ceiling_and_percentile_floor():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    # 12 columns at pitch 404, 8 rows; occupied x spans cols 0..11, y spans rows 1..6
    H, W = 3152, 5066
    x = np.array([311 + 404 * j for j in range(12)], dtype=float)          # full column span
    y = np.array([162 + 404 * i for i in (1, 2, 3, 5, 6)], dtype=float)     # rows 1..6 occupied
    p_min, p_max = f._compute_bounds(x, y, H, W)
    # ceiling = min(H/(R-1), W/(C-1)) = min(450.3, 460.5) = 450.3
    assert p_max == pytest.approx(min(H / 7, W / 11), rel=1e-6)
    # floor uses percentile span; must be <= true pitch 404 and < p_max (valid window)
    assert p_min < p_max
    assert p_min <= 404.0 + 1e-6


def _lattice_points(p, cx, cy, R, C, occupied):
    """occupied: iterable of (i,j) cells. Returns x (cols/CC), y (rows/RR) arrays."""
    xs, ys = [], []
    for (i, j) in occupied:
        xs.append(cx + (j - (C - 1) / 2) * p)
        ys.append(cy + (i - (R - 1) / 2) * p)
    return np.array(xs, float), np.array(ys, float)


def test_estimate_pitch_recovers_fundamental_with_empty_rows():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, W / 2, H / 2
    # occupy rows {1,2,3,5,6} (empty edge + interior rows), scattered columns
    occ = [(1, 0), (1, 4), (1, 8), (2, 3), (2, 7), (2, 11),
           (3, 1), (3, 5), (3, 9), (5, 0), (5, 4), (5, 8), (6, 0), (6, 6), (6, 11)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    p0, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok
    assert p0 == pytest.approx(p_true, abs=3.0)   # not the p/2=202 octave


def test_estimate_pitch_flags_degenerate_on_random_points():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    rng = np.random.default_rng(0)
    H, W = 3152, 5066
    x = rng.uniform(0, W, 40); y = rng.uniform(0, H, 40)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    _, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok is False  # no real periodicity -> caller will fall back
