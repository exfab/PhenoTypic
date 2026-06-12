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
