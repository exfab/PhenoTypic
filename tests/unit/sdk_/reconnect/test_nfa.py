"""Tests for the clean-room binomial NFA statistic."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from phenotypic.sdk_.reconnect import _nfa as nfa_module
from phenotypic.sdk_.reconnect._nfa import NFAResult, binomial_nfa


FIXTURE_ROOT = Path(__file__).parents[3] / "fixtures" / "reconnect" / "nfa"
# Across all eight exact golden log tails on the locked SciPy 1.16.3 runtime,
# the observed maximum is five float64 ULP at ``large_n_rare``.
_GOLDEN_LOG10_TAIL_MAX_ULP = 5
_FLOAT64_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2
# Calibrated against the independent 120-digit Decimal oracle on the supported
# CI platforms. macOS reaches 3,297 ULP; Ubuntu reaches 3,819 ULP because the
# fallback's ``math.lgamma`` path is supplied by the platform libm.
_MILLION_FALLBACK_MAX_ULP = 3_819


def _exact_tail(n: int, k: int, p: Fraction) -> Fraction:
    return sum(
        Fraction(math.comb(n, successes))
        * p**successes
        * (1 - p) ** (n - successes)
        for successes in range(k, n + 1)
    )


def _positive_float_ulp_distance(first: float, second: float) -> int:
    first_bits = int(np.float64(first).view(np.uint64))
    second_bits = int(np.float64(second).view(np.uint64))
    return abs(first_bits - second_bits)


def _golden_log10_tail_tolerance(exact_log10_tail: float) -> float:
    """Return the locked-runtime, five-ULP golden-tail envelope."""

    ulp = abs(float(np.spacing(np.float64(abs(exact_log10_tail)))))
    return _GOLDEN_LOG10_TAIL_MAX_ULP * ulp


def _golden_score_tolerance(log10_n_tests: float, exact_log10_tail: float) -> float:
    """Bound calibrated tail error plus one rounded float64 addition."""

    gamma_1 = _FLOAT64_UNIT_ROUNDOFF / (1 - _FLOAT64_UNIT_ROUNDOFF)
    addition_bound = gamma_1 * (abs(log10_n_tests) + abs(exact_log10_tail))
    return _golden_log10_tail_tolerance(exact_log10_tail) + addition_bound


def _decimal_rare_tail_log10(n: int, k: int, p: Decimal) -> float:
    with localcontext() as context:
        context.prec = 120
        one = Decimal(1)
        normalized_term = one
        normalized_sum = one
        for successes in range(k + 1, n + 1):
            normalized_term *= (
                Decimal(n - successes + 1)
                / Decimal(successes)
                * p
                / (one - p)
            )
            previous = normalized_sum
            normalized_sum += normalized_term
            if normalized_sum == previous:
                break
        log_tail = (
            Decimal(math.comb(n, k)).ln()
            + Decimal(k) * p.ln()
            + Decimal(n - k) * (one - p).ln()
            + normalized_sum.ln()
        )
        return float(log_tail / Decimal(10).ln())


def test_source_fixture_exact_fields_and_score_convention():
    fixture = json.loads((FIXTURE_ROOT / "lsd_source.json").read_text())
    for record in fixture["records"]:
        n_tests = 10.0 ** record["log10_n_tests"]
        actual = binomial_nfa(
            np.array(record["n"], dtype=np.int64),
            np.array(record["k"], dtype=np.int64),
            record["p"],
            n_tests=n_tests,
        )
        exact_log10_tail = float(record["exact_log10_tail"])
        exact_log10_nfa = float(record["log10_n_tests"]) + exact_log10_tail
        exact_score = float(record["exact_score"])
        tail_tolerance = _golden_log10_tail_tolerance(exact_log10_tail)
        score_tolerance = _golden_score_tolerance(
            float(record["log10_n_tests"]), exact_log10_tail
        )
        assert actual.log10_n_tests == math.log10(n_tests)
        np.testing.assert_allclose(
            actual.log10_binomial_tail,
            exact_log10_tail,
            rtol=0,
            atol=tail_tolerance,
        )
        np.testing.assert_allclose(
            actual.log10_nfa,
            exact_log10_nfa,
            rtol=0,
            atol=score_tolerance,
        )
        np.testing.assert_allclose(
            actual.score,
            exact_score,
            rtol=0,
            atol=score_tolerance,
        )
        assert actual.score == -actual.log10_nfa
        assert bool(actual.meaningful) == (exact_score >= 0.0)

        source_drift = float(record["source_minus_exact_score"])
        if source_drift != 0.0:
            source_score = float(record["source_score"])
            assert source_score == exact_score + source_drift
            np.testing.assert_allclose(
                actual.score - source_score,
                -source_drift,
                rtol=0,
                atol=score_tolerance,
            )


def test_golden_fixture_rejects_sf_k_instead_of_sf_k_minus_one():
    fixture = json.loads((FIXTURE_ROOT / "lsd_source.json").read_text())
    record = next(
        item for item in fixture["records"] if item["name"] == "ordinary_upper_tail"
    )
    exact_log10_tail = float(record["exact_log10_tail"])
    tolerance = _golden_log10_tail_tolerance(exact_log10_tail)
    actual = binomial_nfa(
        np.array(record["n"], dtype=np.int64),
        np.array(record["k"], dtype=np.int64),
        record["p"],
        n_tests=10.0 ** record["log10_n_tests"],
    )
    np.testing.assert_allclose(
        actual.log10_binomial_tail,
        exact_log10_tail,
        rtol=0,
        atol=tolerance,
    )


def test_exact_rational_small_tail_grid_stays_within_calibrated_256_ulp():
    probabilities = (
        Fraction(1, 100),
        Fraction(1, 8),
        Fraction(1, 2),
        Fraction(9, 10),
        Fraction(999, 1000),
    )
    n_values: list[int] = []
    k_values: list[int] = []
    p_values: list[float] = []
    expected: list[float] = []
    for n in range(31):
        for k in range(n + 1):
            for probability in probabilities:
                n_values.append(n)
                k_values.append(k)
                p_values.append(float(probability))
                expected.append(float(_exact_tail(n, k, probability)))

    actual = binomial_nfa(
        np.asarray(n_values, dtype=np.int64),
        np.asarray(k_values, dtype=np.int64),
        np.asarray(p_values, dtype=np.float64),
        n_tests=1.0,
    )
    reconstructed = np.power(10.0, actual.log10_binomial_tail)
    maximum_ulp = max(
        _positive_float_ulp_distance(observed, reference)
        for observed, reference in zip(reconstructed, expected, strict=True)
    )
    assert maximum_ulp <= 256


def test_million_trial_underflow_uses_finite_bounded_fallback():
    n = 1_000_000
    k = 1_000
    p = 0.0001
    assert binom.sf(k - 1, n, p) == 0.0

    actual = binomial_nfa(
        np.array(n, dtype=np.int64),
        np.array(k, dtype=np.int64),
        p,
        n_tests=1.0,
    )
    expected = _decimal_rare_tail_log10(n, k, Decimal(1) / Decimal(10_000))

    assert np.isfinite(actual.log10_binomial_tail)
    assert _positive_float_ulp_distance(
        float(-actual.log10_binomial_tail),
        -expected,
    ) <= _MILLION_FALLBACK_MAX_ULP


def test_minimum_positive_probability_uses_additive_log_ratio():
    fixture = json.loads((FIXTURE_ROOT / "contract_edges.json").read_text())
    record = next(
        case for case in fixture["cases"] if case["name"] == "minimum_positive_float64"
    )
    actual = binomial_nfa(
        np.array(record["n"], dtype=np.int64),
        np.array(record["k"], dtype=np.int64),
        record["p"],
        n_tests=1.0,
    )
    observed = float(actual.log10_binomial_tail)
    expected = float(record["expected_log10_tail"])
    assert np.isfinite(observed)
    assert _positive_float_ulp_distance(-observed, -expected) <= 1


def test_exact_edges_and_inclusive_equality_skip_backend(monkeypatch: pytest.MonkeyPatch):
    def fail_backend(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("exact branches must not call SciPy")

    monkeypatch.setattr(nfa_module._scipy_binom, "logsf", fail_backend)
    actual = binomial_nfa(
        np.array([0, 20, 20, 20, 1], dtype=np.int64),
        np.array([0, 0, 1, 20, 1], dtype=np.int64),
        np.array([0.0, 0.125, 0.0, 0.125, 1.0]),
        n_tests=1.0,
        epsilon=1.0,
    )
    np.testing.assert_array_equal(actual.log10_binomial_tail[[0, 1, 4]], 0.0)
    assert np.isneginf(actual.log10_binomial_tail[2])
    expected_all_successes = 20 * np.log10(0.125)
    assert actual.log10_binomial_tail[3] == expected_all_successes
    assert np.isposinf(actual.score[2])
    assert bool(actual.meaningful[0])


@pytest.mark.parametrize("backend_value", [math.nan, 0.1, math.inf])
def test_invalid_backend_log_survival_raises(
    monkeypatch: pytest.MonkeyPatch,
    backend_value: float,
):
    monkeypatch.setattr(
        nfa_module._scipy_binom,
        "logsf",
        lambda *args, **kwargs: np.array([backend_value]),
    )
    with pytest.raises(FloatingPointError, match="SciPy returned an invalid"):
        binomial_nfa(
            np.array([10], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([0.5]),
            n_tests=1.0,
        )


def test_backend_underflow_outside_rare_upper_tail_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        nfa_module._scipy_binom,
        "logsf",
        lambda *args, **kwargs: np.array([-np.inf]),
    )
    with pytest.raises(FloatingPointError, match="outside the reviewed rare"):
        binomial_nfa(
            np.array([10], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([0.5]),
            n_tests=1.0,
        )


def test_broadcast_dtypes_ownership_contiguity_and_immutability():
    n = np.array([[10], [20]], dtype=np.uint32)
    k = np.array([1, 2, 3], dtype=np.int16)
    p = np.array(0.25, dtype=np.float32)
    actual = binomial_nfa(n, k, p, n_tests=12.0)

    assert actual.n.shape == (2, 3)
    assert actual.n.dtype == np.int64
    assert actual.k.dtype == np.int64
    assert actual.p.dtype == np.float64
    assert actual.log10_binomial_tail.dtype == np.float64
    assert actual.log10_nfa.dtype == np.float64
    assert actual.score.dtype == np.float64
    assert actual.meaningful.dtype == np.bool_
    for field in fields(NFAResult):
        value = getattr(actual, field.name)
        if isinstance(value, np.ndarray):
            assert value.flags.c_contiguous
            assert value.flags.owndata
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flat[0] = 0
    n[:] = 0
    k[:] = 0
    assert np.all(actual.n[[0, 1], 0] == [10, 20])
    assert np.all(actual.k[0] == [1, 2, 3])
    assert "nfa" not in {field.name for field in fields(NFAResult)}


def test_empty_broadcast_result_is_permitted_but_test_count_remains_conservative():
    actual = binomial_nfa(
        np.empty((0, 1), dtype=np.int64),
        np.empty((0, 1), dtype=np.int64),
        np.full((1, 3), 0.5, dtype=np.float64),
        n_tests=1.0,
    )
    assert actual.n.shape == (0, 3)
    assert actual.log10_binomial_tail.shape == (0, 3)
    assert actual.meaningful.shape == (0, 3)


@pytest.mark.parametrize(
    ("n", "k", "p", "n_tests", "epsilon", "error", "match"),
    [
        ([1.0], [1], [0.5], 1.0, 1.0, TypeError, "n must have an integer"),
        ([1], [True], [0.5], 1.0, 1.0, TypeError, "k must have an integer"),
        ([1], [1], [True], 1.0, 1.0, TypeError, "p must have a real"),
        ([-1], [0], [0.5], 1.0, 1.0, ValueError, "0 <= n"),
        ([1_000_001], [1], [0.5], 1.0, 1.0, ValueError, "reviewed"),
        ([10], [-1], [0.5], 1.0, 1.0, ValueError, "0 <= k <= n"),
        ([10], [11], [0.5], 1.0, 1.0, ValueError, "0 <= k <= n"),
        ([10], [1], [math.nan], 1.0, 1.0, ValueError, "finite values"),
        ([10], [1], [-0.1], 1.0, 1.0, ValueError, "finite values"),
        ([10], [1], [1.1], 1.0, 1.0, ValueError, "finite values"),
        ([10], [1], [0.5], 0.5, 1.0, ValueError, "n_tests"),
        ([10], [1], [0.5], True, 1.0, TypeError, "n_tests"),
        ([10], [1], [0.5], 1.0, 0.0, ValueError, "epsilon"),
        ([10], [1], [0.5], 1.0, True, TypeError, "epsilon"),
    ],
)
def test_invalid_inputs_raise(
    n: object,
    k: object,
    p: object,
    n_tests: object,
    epsilon: object,
    error: type[Exception],
    match: str,
):
    with pytest.raises(error, match=match):
        binomial_nfa(
            np.asarray(n),
            np.asarray(k),
            np.asarray(p),
            n_tests=n_tests,
            epsilon=epsilon,
        )


def test_nonbroadcastable_inputs_raise():
    with pytest.raises(ValueError, match="broadcastable"):
        binomial_nfa(
            np.ones((2,), dtype=np.int64),
            np.ones((3,), dtype=np.int64),
            0.5,
            n_tests=1.0,
        )
