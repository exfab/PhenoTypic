"""Kernel tests for SubtractPolySurface (design.md §4; drift-register.md)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import (
    MAX_FIT_POINTS,
    RobustFit,
    evaluate_surface,
    fit_surface_coefficients,
    legendre_design,
    normalized_axis,
    robust_least_squares,
    solve_least_squares,
    term_powers,
)

from ._poly_surface_synth import NOISE, grid, rmse, surface_plate


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
