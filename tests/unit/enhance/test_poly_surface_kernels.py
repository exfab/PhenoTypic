"""Kernel tests for SubtractPolySurface (design.md §4; drift-register.md)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import (
    MAX_FIT_POINTS,
    RobustFit,
    evaluate_surface,
    fit_surface_coefficients,
    flatten_surface,
    legendre_design,
    level_lines,
    normalized_axis,
    robust_least_squares,
    solve_least_squares,
    term_powers,
)

from ._poly_surface_synth import NOISE, grid, line_plate, rmse, surface_plate


class TestNormalizedAxis:
    def test_endpoints_and_spacing(self):
        t = normalized_axis(5)
        np.testing.assert_array_equal(t, [-1.0, -0.5, 0.0, 0.5, 1.0])
        assert t.dtype == np.float64

    def test_divides_by_n_minus_one(self):
        """Gwyddion's normalization reaches +1 exactly at the last sample (n-1, not n)."""
        assert normalized_axis(4)[-1] == 1.0
        assert normalized_axis(7)[3] == 0.0

    def test_rejects_fewer_than_two_samples(self):
        with pytest.raises(ValueError, match="at least 2"):
            normalized_axis(1)


class TestTermPowers:
    def test_independent_is_the_tensor_product(self):
        assert term_powers(2, independent=True) == (
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))

    def test_total_degree_keeps_p_plus_q_at_most_degree(self):
        assert term_powers(2, independent=False) == ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 11])
    def test_term_counts(self, degree):
        assert len(term_powers(degree, independent=True)) == (degree + 1) ** 2
        assert len(term_powers(degree, independent=False)) == (degree + 1) * (degree + 2) // 2


class TestLegendreDesign:
    def test_column_is_legendre_of_u_times_legendre_of_v(self):
        u, v = np.array([0.3]), np.array([-0.4])
        a = legendre_design(u, v, 2, ((1, 0), (0, 1), (2, 1)))
        p2_of_u = 0.5 * (3 * 0.3**2 - 1)
        np.testing.assert_allclose(a[0], [0.3, -0.4, p2_of_u * -0.4], atol=1e-15)

    @pytest.mark.parametrize("independent", [True, False])
    def test_spans_the_monomial_term_set(self, independent):
        """Drift D1: same span as Gwyddion's monomials, so any monomial surface is fitted exactly."""
        rng = np.random.default_rng(0)
        _, _, u, v = grid(13, 17)
        terms = term_powers(3, independent)
        z = sum(c * u**p * v**q for c, (p, q) in zip(rng.normal(size=len(terms)), terms))
        a = legendre_design(u.ravel(), v.ravel(), 3, terms)
        np.testing.assert_allclose(a @ solve_least_squares(a, z.ravel()), z.ravel(), atol=1e-12)


class TestSolveLeastSquares:
    def test_recovers_exact_coefficients(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(40, 4))
        x = np.array([0.5, -1.0, 2.0, 0.25])
        np.testing.assert_allclose(solve_least_squares(a, a @ x), x, atol=1e-12)

    def test_rank_deficient_design_raises(self):
        """Drift D2: Gwyddion silently zeroes the coefficients; we raise."""
        a = np.ones((10, 2))
        with pytest.raises(ValueError, match="rank"):
            solve_least_squares(a, np.arange(10.0))


class TestEvaluateSurface:
    @pytest.mark.parametrize("independent", [True, False])
    def test_matches_dense_design_evaluation(self, independent):
        rng = np.random.default_rng(2)
        height, width, degree = 11, 19, 3
        terms = term_powers(degree, independent)
        coef = rng.normal(size=len(terms))
        _, _, u, v = grid(height, width)
        dense = (legendre_design(u.ravel(), v.ravel(), degree, terms) @ coef).reshape(height, width)
        surface = evaluate_surface(coef, terms, degree, height, width)
        assert surface.shape == (height, width)
        np.testing.assert_allclose(surface, dense, atol=1e-13)


class TestRobustLeastSquares:
    def test_exact_data_returns_exact_coefficients(self):
        _, _, u, v = grid(20, 30)
        a = legendre_design(u.ravel(), v.ravel(), 1, term_powers(1, independent=False))
        x = np.array([0.4, 0.02, -0.03])
        fit = robust_least_squares(a, a @ x, clip_sigma=3.0, max_iter=10)
        assert isinstance(fit, RobustFit)
        np.testing.assert_allclose(fit.coef, x, atol=1e-12)

    def test_zero_scale_stops_without_dividing(self):
        """A constant residual set has MAD 0; the loop must stop, not divide (Review Focus 1)."""
        fit = robust_least_squares(np.ones((50, 1)), np.full(50, 0.37), clip_sigma=3.0, max_iter=10)
        assert fit.rounds == 0
        np.testing.assert_allclose(fit.coef, [0.37])
        assert fit.kept.all()

    def test_recovers_background_under_colonies(self):
        """R1/R2 at 25% cover: robust within 0.5 sigma, plain lstsq beyond 1 sigma, >= 10x apart."""
        z, background = surface_plate(cover=0.25, seed=2025)
        _, _, u, v = grid(*z.shape)
        terms = term_powers(3, independent=True)
        a = legendre_design(u.ravel(), v.ravel(), 3, terms)
        robust = (a @ robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=10).coef).reshape(z.shape)
        plain = (a @ solve_least_squares(a, z.ravel())).reshape(z.shape)
        assert rmse(robust, background) < 0.5 * NOISE
        assert rmse(plain, background) > 1.0 * NOISE
        assert rmse(plain, background) > 10 * rmse(robust, background)

    def test_clipped_points_never_return(self):
        """The mask is carried forward between rounds, so the kept set only shrinks (spec §4.3)."""
        z, _ = surface_plate(cover=0.40, seed=2030)
        _, _, u, v = grid(*z.shape)
        a = legendre_design(u.ravel(), v.ravel(), 3, term_powers(3, independent=True))
        previous = None
        for rounds in range(1, 6):
            kept = robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=rounds).kept
            if previous is not None:
                assert not (kept & ~previous).any(), f"a clipped point returned at max_iter={rounds}"
            previous = kept

    def test_max_iter_caps_the_rounds_and_convergence_stops_early(self):
        z, _ = surface_plate(cover=0.10, seed=2031)
        _, _, u, v = grid(*z.shape)
        a = legendre_design(u.ravel(), v.ravel(), 3, term_powers(3, independent=True))
        assert robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=1).rounds == 1
        converged = robust_least_squares(a, z.ravel(), clip_sigma=3.0, max_iter=50)
        assert 1 <= converged.rounds < 50

    def test_never_fits_from_fewer_points_than_terms(self):
        """Drift D3: a round that would leave < n_terms inliers is rejected, so no rank error."""
        rng = np.random.default_rng(3)
        for _ in range(300):
            n = int(rng.integers(4, 9))
            a = np.stack([np.ones(n), np.linspace(-1, 1, n), np.linspace(-1, 1, n) ** 2], axis=1)
            z = rng.standard_cauchy(n)
            fit = robust_least_squares(a, z, clip_sigma=1.0, max_iter=10)
            assert fit.kept.sum() >= 3
            assert np.isfinite(fit.coef).all()


class TestFitSurfaceCoefficients:
    def test_below_the_cap_every_pixel_is_used(self):
        z, _ = surface_plate(height=60, width=90, cover=0.0, seed=2032)
        _, _, u, v = grid(*z.shape)
        terms = term_powers(3, independent=True)
        direct = solve_least_squares(legendre_design(u.ravel(), v.ravel(), 3, terms), z.ravel())
        coef = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(coef, direct, atol=1e-13)

    def test_subsampled_fit_is_within_the_standard_error_bound(self):
        """R4: RMS distance from the full fit <= 3*sigma*sqrt(p/n_sub)."""
        z, _ = surface_plate(height=200, width=300, cover=0.0, seed=2033)
        terms = term_powers(3, independent=True)
        full = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10, max_fit_points=10**9)
        sub = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                       clip_sigma=3.0, max_iter=10, max_fit_points=6000)
        n_sub = len(range(0, 200, 4)) * len(range(0, 300, 4))  # stride ceil(sqrt(60000/6000)) = 4
        bound = 3 * NOISE * np.sqrt(len(terms) / n_sub)
        diff = rmse(evaluate_surface(sub, terms, 3, 200, 300), evaluate_surface(full, terms, 3, 200, 300))
        assert 0 < diff <= bound

    def test_per_axis_stride_keeps_enough_rows_on_a_thin_image(self):
        """Spec §4.8: a 4 x 5000 strip at degree 3 must keep >= 4 distinct rows (no rank error)."""
        rng = np.random.default_rng(4)
        _, _, u, v = grid(4, 5000)
        z = 0.3 + 0.1 * u * v**3 + rng.normal(0, NOISE, (4, 5000))
        coef = fit_surface_coefficients(z, degree=3, independent=True, fit="lstsq",
                                        clip_sigma=3.0, max_iter=10, max_fit_points=100)
        assert np.isfinite(coef).all()

    def test_default_cap_is_the_spec_constant(self):
        assert MAX_FIT_POINTS == 262_144


class TestLevelLines:
    def test_removes_per_row_offsets_and_slopes_exactly(self):
        rng = np.random.default_rng(5)
        _, _, u, _ = grid(40, 60)
        z = 0.4 + rng.normal(0, 0.05, (40, 1)) + rng.normal(0, 0.03, (40, 1)) * u
        out = level_lines(z, line_order=1, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out, np.full_like(z, z.mean()), atol=1e-12)

    @pytest.mark.parametrize("line_order", [0, 1, 3])
    def test_every_row_is_leveled_to_the_input_mean(self, line_order):
        """Spec §4.5 / L4: avg is the input's global mean, taken BEFORE leveling."""
        z, _, _ = line_plate(height=50, width=80, cover=0.2, seed=2034)
        out = level_lines(z, line_order=line_order, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out.mean(axis=1), np.full(50, z.mean()), atol=1e-12)

    def test_degree_zero_equals_the_row_shift_form(self):
        """L5: z - (rowmean - mean(rowmeans)) on an unmasked field."""
        z, _, _ = line_plate(height=30, width=45, cover=0.1, seed=2035)
        rowmeans = z.mean(axis=1, keepdims=True)
        out = level_lines(z, line_order=0, fit="lstsq", clip_sigma=3.0, max_iter=10)
        np.testing.assert_allclose(out, z - (rowmeans - rowmeans.mean()), atol=1e-12)

    def test_robust_recovers_rows_that_are_mostly_background(self):
        """R2 per line: rows < 20% own colony fraction within 0.5 sigma; lstsq worse than 1 sigma."""
        z, background, colonies = line_plate(cover=0.25, seed=2026)
        avg = z.mean()
        robust_fit = z - level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) + avg
        plain_fit = z - level_lines(z, line_order=1, fit="lstsq", clip_sigma=3.0, max_iter=10) + avg
        sparse = colonies.mean(axis=1) < 0.2
        def err(f):
            return np.sqrt(np.mean((f[sparse] - background[sparse]) ** 2, axis=1))

        assert sparse.sum() > 50
        assert err(robust_fit).max() < 0.5 * NOISE
        assert err(plain_fit).max() > 1.0 * NOISE

    def test_rows_are_independent_across_block_boundaries(self):
        """Robust per-row state must not leak between rows or blocks (515 rows > 2 blocks of 256)."""
        z, _, _ = line_plate(height=515, width=64, cover=0.2, seed=2036)
        whole = level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) - z.mean()
        for r in (0, 255, 256, 511, 514):
            row = z[r:r + 1]
            alone = level_lines(row, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10) - row.mean()
            np.testing.assert_allclose(whole[r], alone[0], atol=1e-12)

    def test_short_heavy_tailed_rows_never_fail(self):
        """Drift D3 per line: a round leaving < line_order+1 inliers is rejected for that row."""
        rng = np.random.default_rng(6)
        z = rng.standard_cauchy((200, 6))
        out = level_lines(z, line_order=3, fit="robust", clip_sigma=1.0, max_iter=10)
        assert np.isfinite(out).all()

    def test_a_row_that_is_mostly_colony_is_not_shifted_by_the_mean(self):
        """Drift D3: a degenerate row keeps its last fit rather than gaining the global mean."""
        rng = np.random.default_rng(7)
        z = 0.3 + rng.normal(0, NOISE, (20, 100))
        z[7, 5:95] += 0.4                                    # row 7 is 90% colony
        out = level_lines(z, line_order=1, fit="robust", clip_sigma=3.0, max_iter=10)
        assert np.isfinite(out[7]).all()
        assert not np.allclose(out[7], z[7] + z.mean())


DEFAULTS = dict(order=3, independent=True, line_order=1, line_axis="row",
                fit="lstsq", clip_sigma=3.0, max_iter=10)


def flatten(z, **overrides):
    return flatten_surface(z, **{**DEFAULTS, **overrides})


class TestOffset:
    def test_lstsq_subtracts_the_mean(self):
        z, _ = surface_plate(height=40, width=50, cover=0.1, seed=2040)
        out = flatten(z, method="offset")
        np.testing.assert_allclose(out, z - z.mean(), atol=1e-15)

    def test_robust_finds_the_agar_level_under_colonies(self):
        rng = np.random.default_rng(2028)
        from ._poly_surface_synth import colony_domes
        z = 0.4 + colony_domes(300, 450, 0.25, rng) + rng.normal(0, NOISE, (300, 450))
        level = z - flatten(z, method="offset", fit="robust")
        assert abs(float(level.flat[0]) - 0.4) < 0.5 * NOISE
        assert abs(z.mean() - 0.4) > 1.0 * NOISE


class TestPlane:
    def test_removes_tilt_about_gwyddions_pivot(self):
        """Drift D7 / L2: out = a + bx*W/2 + by*H/2 everywhere; NOT the true-centre value."""
        height, width, a, bx, by = 21, 34, 0.35, 0.002, 0.001
        i, j, _, _ = grid(height, width)
        z = a + bx * j + by * i
        out = flatten(z, method="plane")
        np.testing.assert_allclose(out, np.full_like(z, a + bx * width / 2 + by * height / 2), atol=1e-12)
        assert abs(out.mean() - (z.mean() + 0.5 * (bx + by))) < 1e-12

    def test_robust_plane_flattens_the_background_under_colonies(self):
        rng = np.random.default_rng(2041)
        from ._poly_surface_synth import colony_domes
        i, j, _, _ = grid(300, 450)
        background = 0.3 + 0.0004 * j - 0.0003 * i
        colonies = colony_domes(300, 450, 0.25, rng)
        z = background + colonies + rng.normal(0, NOISE, (300, 450))
        out = flatten(z, method="plane", fit="robust")
        flattened_background = (out - colonies)[colonies == 0]
        assert np.std(flattened_background) < 1.5 * NOISE


class TestPolynomial:
    def test_removes_its_own_form_exactly_and_lands_at_zero(self):
        _, _, u, v = grid(30, 41)
        z = 0.3 + 0.05 * u - 0.02 * v + 0.04 * u**2 * v + 0.01 * u**3 * v**3
        out = flatten(z, method="polynomial", order=3, independent=True)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)

    def test_independent_keeps_the_u3v3_term_and_total_degree_does_not(self):
        _, _, u, v = grid(30, 41)
        z = 0.3 + 0.05 * u**3 * v**3
        assert np.max(np.abs(flatten(z, method="polynomial", order=3, independent=True))) < 1e-10
        assert rmse(flatten(z, method="polynomial", order=3, independent=False), 0.0) > 1e-3

    def test_robust_polynomial_recovers_the_background(self):
        z, background = surface_plate(cover=0.25, seed=2025)
        out = flatten(z, method="polynomial", order=3, independent=True, fit="robust")
        assert rmse(z - out, background) < 0.5 * NOISE


class TestLineDispatch:
    def test_column_axis_is_row_axis_on_the_transpose(self):
        z, _, _ = line_plate(height=40, width=55, cover=0.1, seed=2042)
        col = flatten(z, method="line", line_axis="column", line_order=2)
        row_of_t = flatten(z.T.copy(), method="line", line_axis="row", line_order=2)
        np.testing.assert_allclose(col, row_of_t.T, atol=1e-12)

    def test_line_levels_to_the_input_mean(self):
        z, _, _ = line_plate(height=40, width=55, cover=0.1, seed=2043)
        out = flatten(z, method="line", line_order=1)
        np.testing.assert_allclose(out.mean(axis=1), np.full(40, z.mean()), atol=1e-12)


class TestContract:
    @pytest.mark.parametrize("method", ["offset", "plane", "polynomial", "line"])
    def test_returns_new_float64_and_never_mutates_input(self, method):
        z, _ = surface_plate(height=30, width=40, cover=0.1, seed=2044)
        before = z.copy()
        out = flatten(z, method=method)
        assert out.dtype == np.float64 and out is not z
        np.testing.assert_array_equal(z, before)


class TestValidation:
    @pytest.mark.parametrize("shape", [(1, 10), (10, 1)])
    def test_single_row_or_column_is_rejected(self, shape):
        with pytest.raises(ValueError, match="at least 2"):
            flatten(np.zeros(shape), method="plane")

    def test_polynomial_needs_order_plus_one_samples_per_axis(self):
        flatten(np.random.default_rng(8).random((4, 4)), method="polynomial", order=3)
        with pytest.raises(ValueError, match="order"):
            flatten(np.random.default_rng(8).random((3, 4)), method="polynomial", order=3)

    def test_plane_works_on_two_by_two(self):
        assert np.isfinite(flatten(np.array([[0.1, 0.2], [0.3, 0.5]]), method="plane")).all()

    def test_column_lines_need_line_order_plus_one_rows(self):
        with pytest.raises(ValueError, match="line_order"):
            flatten(np.random.default_rng(9).random((2, 50)), method="line",
                    line_axis="column", line_order=2)

    @pytest.mark.parametrize("bad", [dict(method="huber"), dict(fit="ransac"), dict(line_axis="diag")])
    def test_unknown_strings_are_rejected(self, bad):
        with pytest.raises(ValueError):
            flatten(np.ones((5, 5)), **{"method": "plane", **bad})


class TestReviewFocus:
    @pytest.mark.parametrize("fit", ["lstsq", "robust"])
    @pytest.mark.parametrize("method, level", [("offset", 0.0), ("plane", 0.42),
                                               ("polynomial", 0.0), ("line", 0.42)])
    def test_constant_image_lands_at_its_documented_level(self, method, level, fit):
        """Review Focus 1: flat detect_mat; robust must stop on zero MAD (spec §4.6 levels)."""
        out = flatten(np.full((25, 35), 0.42), method=method, fit=fit)
        np.testing.assert_allclose(out, level, atol=1e-12)

    @pytest.mark.parametrize("method", ["offset", "plane", "polynomial", "line"])
    def test_majority_saturated_image_is_finite(self, method):
        """Review Focus 2: >= 50% pixels exactly 1.0 -- MAD over the kept set can be 0."""
        rng = np.random.default_rng(10)
        z = 0.3 + rng.normal(0, NOISE, (60, 80))
        z[:, :48] = 1.0
        assert np.isfinite(flatten(z, method=method, fit="robust")).all()

    def test_dark_colonies_are_clipped_too(self):
        """Review Focus 3 / drift D15: an un-inverted plate (dark colonies on bright agar)."""
        rng = np.random.default_rng(2027)
        from ._poly_surface_synth import colony_domes
        _, _, u, v = grid(300, 450)
        background = 0.70 + 0.05 * u - 0.04 * v + 0.03 * u * v
        z = background - colony_domes(300, 450, 0.25, rng) + rng.normal(0, NOISE, (300, 450))
        out = flatten(z, method="polynomial", order=2, independent=True, fit="robust")
        assert rmse(z - out, background) < 0.5 * NOISE
