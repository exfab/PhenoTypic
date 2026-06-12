import numpy as np
import pytest

from phenotypic.abc_ import GridFinder
from phenotypic.grid import CenteredAutoGridFinder, CenteredAutoGridFinderFallbackWarning
from phenotypic.schema import GRID


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
    x = rng.uniform(0, W, 40)
    y = rng.uniform(0, H, 40)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    _, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok is False  # no real periodicity -> caller will fall back


def test_center_candidates_include_true_center():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0   # offset from image center (2533,1576)
    occ = [(1, 0), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (1, 8), (5, 4)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    cx_c = f._center_candidates(x, p_true, f.ncols, W)
    cy_c = f._center_candidates(y, p_true, f.nrows, H)
    assert min(abs(np.array(cx_c) - cx)) < 0.5 * p_true
    assert min(abs(np.array(cy_c) - cy)) < 0.5 * p_true


def test_icp_recovers_params_from_good_seed():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (5, 4), (2, 7), (6, 0)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    out = f._icp_refine(x, y, cx + 30, cy - 25, p_true + 6)  # seed within ~0.1 cell
    assert out is not None
    rcx, rcy, rp, res = out
    assert rp == pytest.approx(p_true, abs=1.0)
    assert rcx == pytest.approx(cx, abs=2.0) and rcy == pytest.approx(cy, abs=2.0)
    assert res < 0.05 * p_true


def test_multistart_rejects_one_cell_shift():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (2, 3), (3, 5), (5, 8), (6, 11), (3, 1), (5, 4), (2, 7), (6, 0)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    cx_c = f._center_candidates(x, p_true, f.ncols, W)
    cy_c = f._center_candidates(y, p_true, f.nrows, H)
    best = f._multi_start_refine(x, y, p_true, cx_c, cy_c)
    assert best is not None
    bcx, bcy, bp, res = best
    assert bcx == pytest.approx(cx, abs=2.0) and bcy == pytest.approx(cy, abs=2.0)


def test_icp_singular_returns_none_all_one_cell():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    # all points in a tiny cluster -> every assignment rounds to one cell -> det ~ 0
    x = np.array([2500.0, 2503.0, 2498.0])
    y = np.array([1570.0, 1572.0, 1569.0])
    out = f._icp_refine(x, y, 2533.0, 1576.0, 404.0)
    assert out is None


def test_axis_edges_length_sorted_clipped():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    edges = f._axis_edges(center=1575.0, p=404.0, n_cells=8, image_dim=3152)
    assert len(edges) == 9                  # n+1
    assert np.all(np.diff(edges) > 0)       # sorted ascending
    assert edges[0] >= 0 and edges[-1] <= 3152


def test_fit_grid_returns_edges_and_lands_on_lattice():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, 2545.0, 1575.0
    occ = [(1, 0), (1, 4), (1, 8), (2, 3), (2, 7), (2, 11), (3, 1), (3, 5),
           (3, 9), (5, 0), (5, 4), (5, 8), (6, 0), (6, 6), (6, 11)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)
    assert len(row_edges) == 9 and len(col_edges) == 13
    # every colony falls strictly inside the bin of its true cell
    for (i, j), xi, yi in zip(occ, x, y):
        assert row_edges[i] <= yi <= row_edges[i + 1]
        assert col_edges[j] <= xi <= col_edges[j + 1]


def _edges_ok(row_edges, col_edges, R, C, H, W):
    return (len(row_edges) == R + 1 and len(col_edges) == C + 1
            and np.all(np.diff(row_edges) > 0) and np.all(np.diff(col_edges) > 0)
            and row_edges[0] >= 0 and row_edges[-1] <= H
            and col_edges[0] >= 0 and col_edges[-1] <= W)


def test_zero_and_one_colony_uniform_no_crash():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    for x, y in [(np.array([]), np.array([])), (np.array([2500.0]), np.array([1570.0]))]:
        re, ce = f._fit_grid_from_centers(x, y, H, W)
        assert _edges_ok(re, ce, 8, 12, H, W)


def test_degenerate_response_falls_back_without_exception():
    f = CenteredAutoGridFinder(nrows=8, ncols=12, warn=True)
    rng = np.random.default_rng(1)
    H, W = 3152, 5066
    x = rng.uniform(0, W, 30)
    y = rng.uniform(0, H, 30)
    with pytest.warns(CenteredAutoGridFinderFallbackWarning):
        re, ce = f._fit_grid_from_centers(x, y, H, W)
    assert _edges_ok(re, ce, 8, 12, H, W)


def test_grid_image_default_finder_is_centered():
    from phenotypic import GridImage
    img = GridImage(arr=np.zeros((400, 600, 3), dtype=np.uint8), nrows=8, ncols=12)
    assert isinstance(img.grid_finder, CenteredAutoGridFinder)


def test_integration_decimated_synth_plate():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    image = OtsuDetector().apply(load_synth_yeast_plate())
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    df = f.measure(image)
    assert str(GRID.ROW_NUM) in df.columns and str(GRID.COL_NUM) in df.columns
    # rows/cols within range, no NaN section for detected objects
    assert df[str(GRID.ROW_NUM)].dropna().between(0, 7).all()
    assert df[str(GRID.COL_NUM)].dropna().between(0, 11).all()


def test_dense_plate_pitch_above_naive_ceiling_is_found():
    """Regression for the bound-inversion: true pitch exceeds min(H/R,W/C)."""
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 404.0, W / 2, H / 2          # 404 > min(H/8,W/12)=394
    occ = [(i, j) for i in range(8) for j in range(12)]   # fully dense
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    re, ce = f._fit_grid_from_centers(x, y, H, W)
    # recovered column pitch ~ 404 (not capped at 394)
    assert np.mean(np.diff(ce[1:-1])) == pytest.approx(p_true, abs=2.0)
