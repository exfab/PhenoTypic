"""Kernel tests for SubtractPolySurface (design.md §4; drift-register.md)."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.enhance._poly_surface_kernels import (
    evaluate_surface,
    legendre_design,
    normalized_axis,
    solve_least_squares,
    term_powers,
)

from ._poly_surface_synth import grid


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
