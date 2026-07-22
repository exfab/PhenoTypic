import numpy as np
import pandas as pd
import pytest

from phenotypic.abc_ import GridFinder
from phenotypic.grid import CenteredAutoGridFinder, CenteredAutoGridFinderFallbackWarning
from phenotypic.schema import BBOX, GRID


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
    # Fundamental recovered despite empty rows. (The p/2 octave is OUTSIDE this
    # window; genuine octave rejection is covered by the clustered test below.)
    assert p0 == pytest.approx(p_true, abs=3.0)


def test_estimate_pitch_picks_fundamental_not_octave_when_window_contains_it():
    """Clustered occupancy whose pitch floor admits p/2 and p/3 inside the search
    window: the 'largest-p above floor' rule must pick the fundamental (200) where
    a plain argmax of the comb response could land on an octave."""
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3152, 5066
    p_true, cx, cy = 200.0, W / 2, H / 2
    occ = [(i, j) for i in (3, 4, 5) for j in (4, 5, 6, 7)]  # tight central 3x4 cluster
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)
    p_min, p_max = f._compute_bounds(x, y, H, W)
    assert p_min < p_true / 2          # window genuinely contains the p/2 octave
    p0, ok = f._estimate_pitch(x, y, p_min, p_max)
    assert ok
    assert p0 == pytest.approx(p_true, abs=4.0)   # fundamental, not 100 or ~66.7


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
    out = f._icp_refine(
        x, y, cx + 30, cy - 25, p_true + 6, p_min=300.0, p_max=450.0
    )  # seed within ~0.1 cell
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
    # A one-cell-shifted seed alone gets TRAPPED on a high-residual lattice: edge
    # colonies (col 0, col 11) clip when their index shifts, so the fit can't match.
    shifted = f._icp_refine(
        x, y, cx + p_true, cy, p_true, p_min=300.0, p_max=450.0
    )
    assert shifted is not None
    assert shifted[3] > 0.15 * p_true                      # the trap: large residual
    # Multi-start over all candidates escapes the trap by selecting the lowest-
    # residual placement (the true center, ~0 residual). Remove multi-start and this
    # decisive gap (res << shifted) disappears.
    cx_c = f._center_candidates(x, p_true, f.ncols, W)
    cy_c = f._center_candidates(y, p_true, f.nrows, H)
    best, saw_refined = f._multi_start_refine(
        x,
        y,
        p_true,
        300.0,
        450.0,
        cx_c,
        cy_c,
        H,
        W,
    )
    assert best is not None
    assert saw_refined
    bcx, bcy, bp, res = best
    assert res < 0.05 * p_true                             # decisively beats the trap
    assert res < shifted[3] / 3
    assert bcx == pytest.approx(cx, abs=2.0) and bcy == pytest.approx(cy, abs=2.0)


def test_icp_singular_returns_none_all_one_cell():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    # all points in a tiny cluster -> every assignment rounds to one cell -> det ~ 0
    x = np.array([2500.0, 2503.0, 2498.0])
    y = np.array([1570.0, 1572.0, 1569.0])
    out = f._icp_refine(
        x, y, 2533.0, 1576.0, 404.0, p_min=300.0, p_max=450.0
    )
    assert out is None


def test_axis_edges_length_sorted_clipped():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    edges = f._axis_edges(center=1575.0, p=404.0, n_cells=8, image_dim=3152)
    assert len(edges) == 9                  # n+1
    assert np.all(np.diff(edges) > 0)       # sorted ascending
    assert edges[0] >= 0 and edges[-1] <= 3152


def test_axis_edge_validation_rejects_reported_and_mirrored_collapse():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)

    reported_rows = f._axis_edges(
        center=1983.0, p=403.2, n_cells=8, image_dim=3002
    )
    assert reported_rows[-2] == reported_rows[-1] == 3002.0
    assert not f._axis_edges_are_valid(reported_rows, 8, 3002)

    mirrored_cols = f._axis_edges(
        center=2800.0, p=350.0, n_cells=12, image_dim=4500
    )
    assert mirrored_cols[-2] == mirrored_cols[-1] == 4500.0
    assert not f._axis_edges_are_valid(mirrored_cols, 12, 4500)

    one_pitch_up = f._axis_edges(
        center=1579.8, p=403.2, n_cells=8, image_dim=3002
    )
    assert f._axis_edges_are_valid(one_pitch_up, 8, 3002)


def test_multistart_uses_downstream_effective_column_bound(monkeypatch):
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W, p = 3002, 1099, 99.0
    feasible_cx = (W - 1) / 2.0
    subpixel_last_cell_cx = 603.5

    pre_assignment_edges = f._axis_edges(
        subpixel_last_cell_cx, p, f.ncols, W
    )
    assert f._axis_edges_are_valid(pre_assignment_edges, f.ncols, W)
    downstream_edges = np.clip(pre_assignment_edges, 0, W - 1)
    assert not f._axis_edges_are_valid(downstream_edges, f.ncols, W - 1)

    def fake_icp_refine(self, x, y, cx, cy, p0, p_min, p_max):
        del self, x, y, p0, p_min, p_max
        residual = 0.2 if cx == feasible_cx else 0.1
        return float(cx), float(cy), p, residual

    monkeypatch.setattr(
        CenteredAutoGridFinder, "_icp_refine", fake_icp_refine
    )
    best, saw_refined = f._multi_start_refine(
        np.array([feasible_cx]),
        np.array([H / 2.0]),
        p,
        90.0,
        110.0,
        [feasible_cx, subpixel_last_cell_cx],
        [H / 2.0],
        H,
        W,
    )

    assert saw_refined
    assert best is not None
    assert best[0] == feasible_cx
    col_edges = f._axis_edges(best[0], best[2], f.ncols, W - 1)
    table = pd.DataFrame({str(BBOX.CENTER_CC): [54.0, 153.0]})
    assigned = f._add_col_number_info(table, col_edges, (H, W))
    assert assigned[str(GRID.COL_NUM)].notna().all()


def test_multistart_rejects_collapsed_fit_and_maps_rows_b_through_g(monkeypatch):
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W, p = 3002, 5066, 403.2
    feasible_cy = 1579.8
    collapsed_cy = 1983.0

    def fake_icp_refine(
        self, x, y, cx, cy, p0, p_min, p_max
    ):  # pragma: no cover - signature documents the private seam
        del self, x, y, p0, p_min, p_max
        residual = 0.2 if cy == feasible_cy else 0.1
        return float(cx), float(cy), p, residual

    monkeypatch.setattr(
        CenteredAutoGridFinder, "_icp_refine", fake_icp_refine
    )
    best, saw_refined = f._multi_start_refine(
        np.array([W / 2.0]),
        np.array([571.8]),
        p,
        350.0,
        430.0,
        [W / 2.0],
        [feasible_cy, collapsed_cy],
        H,
        W,
    )

    assert saw_refined
    assert best is not None
    assert best[1] == feasible_cy
    row_edges = f._axis_edges(best[1], best[2], f.nrows, H)
    assert f._axis_edges_are_valid(row_edges, f.nrows, H)

    row_centers = 571.8 + p * np.arange(6)
    table = pd.DataFrame({str(BBOX.CENTER_RR): row_centers})
    assigned = f._add_row_number_info(table, row_edges, (H, W))
    assert assigned[str(GRID.ROW_NUM)].astype(int).tolist() == list(range(1, 7))


def test_reported_sparse_rows_fit_end_to_end_without_duplicate_bins():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W, p = 3002, 5066, 403.2
    cx, cy = W / 2.0, 1579.8
    occupied = [(i, j) for i in range(1, 7) for j in range(12)]
    x, y = _lattice_points(p, cx, cy, 8, 12, occupied)

    row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)

    assert _edges_ok(row_edges, col_edges, 8, 12, H, W)
    row_centers = np.unique(y)
    table = pd.DataFrame({str(BBOX.CENTER_RR): row_centers})
    assigned = f._add_row_number_info(table, row_edges, (H, W))
    assert assigned[str(GRID.ROW_NUM)].astype(int).tolist() == list(range(1, 7))


def test_multistart_tied_residual_prefers_nearest_image_center(monkeypatch):
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W, p = 3002, 5066, 300.0

    def fake_icp_refine(self, x, y, cx, cy, p0, p_min, p_max):
        del self, x, y, p0, p_min, p_max
        return float(cx), float(cy), p, 1.0

    monkeypatch.setattr(
        CenteredAutoGridFinder, "_icp_refine", fake_icp_refine
    )
    best, _ = f._multi_start_refine(
        np.array([W / 2.0]),
        np.array([H / 2.0]),
        p,
        250.0,
        350.0,
        [W / 2.0],
        [1400.0, 1600.0],
        H,
        W,
    )

    assert best is not None
    assert best[1] == 1600.0


def test_bounded_solution_reoptimizes_center_for_clamped_pitch():
    solution = np.array([400.0, 300.0, 500.0])
    x = np.array([0.0, 1000.0])
    y = np.array([100.0, 700.0])
    a = np.array([-1.0, 1.0])
    b = np.array([-1.0, 1.0])

    bounded = CenteredAutoGridFinder._bounded_solution(
        solution, x, y, a, b, p_min=200.0, p_max=400.0
    )

    assert bounded == pytest.approx((500.0, 400.0, 400.0))


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


def test_zero_span_degenerate_response_uses_strict_centered_fallback():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3002, 5066
    x = np.full(8, W / 2.0)
    y = np.full(8, H / 2.0)

    row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)

    assert _edges_ok(row_edges, col_edges, 8, 12, H, W)


def test_all_refined_candidates_infeasible_warns_and_falls_back(
    monkeypatch,
):
    f = CenteredAutoGridFinder(nrows=8, ncols=12, warn=True)
    H, W, p = 3002, 5066, 403.2
    occ = [(i, j) for i in range(1, 7) for j in range(1, 11)]
    x, y = _lattice_points(p, W / 2.0, H / 2.0, 8, 12, occ)

    def fake_icp_refine(self, x, y, cx, cy, p0, p_min, p_max):
        del self, x, y, cx, cy, p0, p_min, p_max
        return W / 2.0, 1983.0, p, 0.1

    monkeypatch.setattr(
        CenteredAutoGridFinder, "_icp_refine", fake_icp_refine
    )
    with pytest.warns(
        CenteredAutoGridFinderFallbackWarning,
        match="invalid-geometry",
    ):
        row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)

    assert _edges_ok(row_edges, col_edges, 8, 12, H, W)
    assert row_edges[1:-1] + row_edges[-2:0:-1] == pytest.approx(H)
    assert np.mean(np.diff(row_edges[1:-1])) == pytest.approx(p, abs=1.0)


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


def test_valid_off_center_fit_beyond_one_pitch_is_preserved():
    f = CenteredAutoGridFinder(nrows=8, ncols=12)
    H, W = 3002, 4200
    p_true = 300.0
    cx, cy = W / 2.0, H / 2.0 + p_true + 1.0
    occ = [(i, j) for i in range(8) for j in range(12)]
    x, y = _lattice_points(p_true, cx, cy, 8, 12, occ)

    row_edges, col_edges = f._fit_grid_from_centers(x, y, H, W)

    assert _edges_ok(row_edges, col_edges, 8, 12, H, W)
    assert row_edges[0] == pytest.approx(602.0, abs=2.0)
    assert row_edges[-1] == pytest.approx(H, abs=2.0)
    assert np.mean(np.diff(col_edges)) == pytest.approx(p_true, abs=2.0)
