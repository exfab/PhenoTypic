"""Validate the numerical contract for the A07 binomial NFA statistic.

This script is source-independent and never imports :mod:`phenotypic`. It
re-derives finite binomial tails, false-alarm control, and logarithmic edge
behavior from their definitions.
"""

from __future__ import annotations

import itertools
import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
from scipy.stats import binom


UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0
LOG10_E = 1.0 / math.log(10.0)
MAX_TRIALS = 1_000_000
MAX_FALLBACK_TERMS = 1_000_000
# Exhaustive calibration below on SciPy 1.16.3 reaches exactly 256 ULP at
# n=26, k=24, p=1/100 after the public log10-to-probability round trip.
SMALL_TAIL_ULP_LIMIT = 256.0
# Observed against the 120-digit Decimal oracle below on the locked runtime.
MILLION_FALLBACK_ULP_LIMIT = 3_297.0
MINIMUM_POSITIVE_ULP_LIMIT = 1.0


def exact_binomial_tail(n: int, k: int, p: Fraction) -> Fraction:
    """Compute a binomial survival probability with exact rational arithmetic."""
    if k == 0:
        return Fraction(1)
    return sum(
        (
            Fraction(math.comb(n, j)) * p**j * (1 - p) ** (n - j)
            for j in range(k, n + 1)
        ),
        start=Fraction(0),
    )


def decimal_rare_tail_log10(
    n: int, k: int, p: Fraction, *, precision: int = 120
) -> Decimal:
    """Compute a rare tail with exact combinatorics and high-precision Decimal."""
    with localcontext() as context:
        context.prec = precision
        decimal_p = Decimal(p.numerator) / Decimal(p.denominator)
        decimal_q = Decimal(1) - decimal_p
        log_first = (
            Decimal(math.comb(n, k)).ln()
            + Decimal(k) * decimal_p.ln()
            + Decimal(n - k) * decimal_q.ln()
        )
        relative_term = Decimal(1)
        relative_sum = Decimal(1)
        relative_tolerance = Decimal(10) ** Decimal(-(precision - 20))
        for j in range(k, n):
            ratio = (
                Decimal(n - j)
                / Decimal(j + 1)
                * decimal_p
                / decimal_q
            )
            relative_term *= ratio
            relative_sum += relative_term
            if j + 1 == n:
                break
            next_ratio = (
                Decimal(n - j - 1)
                / Decimal(j + 2)
                * decimal_p
                / decimal_q
            )
            remainder_bound = relative_term * next_ratio / (Decimal(1) - next_ratio)
            if remainder_bound <= relative_tolerance * relative_sum:
                break
        return +(log_first + relative_sum.ln()) / Decimal(10).ln()


def assert_calibrated_ulps(
    actual: float, expected: float, *, limit: float, context: str
) -> None:
    """Compare a libm/SciPy path against an independently derived ULP envelope."""
    error_ulps = abs(actual - expected) / math.ulp(expected)
    if error_ulps > limit:
        raise AssertionError(
            f"{context} exceeded calibrated envelope: {error_ulps} ULP > {limit} ULP"
        )


def stable_log10_tail(n: int, k: int, p: float) -> float:
    """Evaluate the intended stable log10 survival contract."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be an integer")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")
    if n < 0 or n > MAX_TRIALS:
        raise ValueError(f"n must be between 0 and {MAX_TRIALS}")
    if k < 0 or k > n:
        raise ValueError("k must satisfy 0 <= k <= n")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be finite and between 0 and 1")
    if k == 0:
        return 0.0
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return 0.0
    if k == n:
        return n * math.log10(p)
    scipy_log_tail = float(binom.logsf(k - 1, n, p))
    validate_scipy_log_tail(scipy_log_tail)
    if math.isfinite(scipy_log_tail):
        return scipy_log_tail * LOG10_E
    if k <= n * p:
        raise ArithmeticError("SciPy returned -inf outside the rare upper tail")
    term_count = n - k + 1
    if term_count > MAX_FALLBACK_TERMS:
        raise ArithmeticError(
            "rare-tail fallback would exceed its explicit work bound: "
            f"{term_count} > {MAX_FALLBACK_TERMS}"
        )
    fallback = rare_tail_log(n, k, p)
    if not math.isfinite(fallback) or fallback > 0.0:
        raise ArithmeticError("rare-tail fallback returned an invalid log probability")
    return fallback * LOG10_E


def validate_scipy_log_tail(log_tail: float) -> None:
    """Reject backend values outside the logarithm-of-probability range."""
    if math.isnan(log_tail) or log_tail > 0.0:
        raise ArithmeticError(
            "binomial log survival must be finite or -inf and no greater than zero"
        )


def validate_n_tests(n_tests: float) -> None:
    """Validate the conservative nonempty candidate-family count."""
    if isinstance(n_tests, bool) or not math.isfinite(n_tests) or n_tests < 1.0:
        raise ValueError("n_tests must be finite and at least one")


def logaddexp(left: float, right: float) -> float:
    """Add two positive quantities represented by natural logarithms."""
    maximum = max(left, right)
    return maximum + math.log1p(math.exp(min(left, right) - maximum))


def rare_tail_log(n: int, k: int, p: float) -> float:
    """Sum a rare binomial tail with a proved monotone remainder bound."""
    log_term = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )
    log_tail = log_term
    for j in range(k + 1, n + 1):
        log_ratio = (
            math.log(n - j + 1)
            - math.log(j)
            + math.log(p)
            - math.log1p(-p)
        )
        log_remainder_bound = geometric_remainder_log_bound(log_term, log_ratio)
        if remainder_cannot_change_log(log_tail, log_remainder_bound):
            break
        log_term += log_ratio
        log_tail = logaddexp(log_tail, log_term)
    return log_tail


def geometric_remainder_log_bound(log_term: float, log_ratio: float) -> float:
    """Bound all terms after the current one using decreasing tail ratios."""
    ratio = math.exp(log_ratio)
    if ratio >= 1.0:
        return math.inf
    return log_term + log_ratio - math.log1p(-ratio)


def remainder_cannot_change_log(log_tail: float, log_bound: float) -> bool:
    """Return whether an omitted tail changes log-tail by at most half an ULP."""
    relative_log_bound = log_bound - log_tail
    if relative_log_bound > math.log(sys.float_info.max):
        return False
    log_error_bound = math.log1p(math.exp(relative_log_bound))
    return log_error_bound <= 0.5 * math.ulp(log_tail)


def assert_basic_operations_close(
    actual: float, expected: float, *, operations: int
) -> None:
    """Compare basic arithmetic using unit roundoff and a counted gamma bound."""
    gamma = operations * UNIT_ROUNDOFF / (1.0 - operations * UNIT_ROUNDOFF)
    bound = gamma * max(1.0, abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=bound):
        raise AssertionError(
            f"{actual!r} != {expected!r}; derived absolute bound {bound!r}"
        )


def validate_exact_small_tails() -> None:
    """Compare stable survival logs with exact rational small-tail values."""
    probabilities = (
        Fraction(1, 100),
        Fraction(1, 8),
        Fraction(1, 2),
        Fraction(9, 10),
        Fraction(999, 1000),
    )
    worst_ulps = 0.0
    worst_case: tuple[int, int, Fraction] | None = None
    for n in range(0, 31):
        for k in range(0, n + 1):
            for p in probabilities:
                exact = exact_binomial_tail(n, k, p)
                expected = float(exact)
                stable = 10.0 ** stable_log10_tail(n, k, float(p))
                error_ulps = abs(stable - expected) / math.ulp(expected)
                if error_ulps > worst_ulps:
                    worst_ulps = error_ulps
                    worst_case = (n, k, p)
    if worst_ulps > SMALL_TAIL_ULP_LIMIT:
        raise AssertionError(
            "small-tail SciPy regression exceeded the calibrated ULP envelope: "
            f"{worst_ulps} ULP at {worst_case}; limit={SMALL_TAIL_ULP_LIMIT}"
        )


def enumerate_bernoulli_tail(n: int, k: int, p: Fraction) -> Fraction:
    """Enumerate Bernoulli strings with exact rational arithmetic."""
    probability = Fraction(0)
    for outcome in itertools.product((0, 1), repeat=n):
        successes = sum(outcome)
        if successes >= k:
            probability += p**successes * (1 - p) ** (n - successes)
    return probability


def validate_exhaustive_false_alarm_control() -> None:
    """Prove expected detections for small predetermined families are bounded."""
    for n in range(1, 9):
        for p in (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)):
            probabilities = [
                enumerate_bernoulli_tail(n, k, p) for k in range(n + 1)
            ]
            for n_tests in (1, 3, 11):
                for epsilon in (Fraction(1, 20), Fraction(1, 2), Fraction(1)):
                    detection_probability = sum(
                        Fraction(math.comb(n, k))
                        * p**k
                        * (1 - p) ** (n - k)
                        for k in range(n + 1)
                        if n_tests * probabilities[k] <= epsilon
                    )
                    expected_detections = n_tests * detection_probability
                    if expected_detections > epsilon:
                        raise AssertionError(
                            "false-alarm expectation exceeded epsilon: "
                            f"{expected_detections} > {epsilon}"
                        )


def validate_basic_log_nfa_arithmetic() -> None:
    """Isolate basic add/negate rounding from calibrated libm operations."""
    log_tail = -math.log10(3.0)
    log_n_tests = math.log10(11.0)
    exact_sum_of_float_inputs = float(
        Fraction.from_float(log_tail) + Fraction.from_float(log_n_tests)
    )
    log_nfa = log_tail + log_n_tests
    assert_basic_operations_close(log_nfa, exact_sum_of_float_inputs, operations=1)
    if -log_nfa != 0.0 - log_nfa:
        raise AssertionError("score negation changed the represented value")


def validate_monotonicity() -> None:
    """Check tail and NFA monotonicity in every load-bearing scalar."""
    n = 100
    tails_by_k = [stable_log10_tail(n, k, 0.2) for k in range(n + 1)]
    if any(left < right for left, right in itertools.pairwise(tails_by_k)):
        raise AssertionError("log tail must be nonincreasing as k increases")

    for k in (1, 10, 50, 100):
        tails_by_p = [stable_log10_tail(n, k, p) for p in (0.01, 0.1, 0.5)]
        if any(left > right for left, right in itertools.pairwise(tails_by_p)):
            raise AssertionError("log tail must be nondecreasing as p increases")

    tail = stable_log10_tail(50, 20, 0.2)
    log_nfas = [math.log10(n_tests) + tail for n_tests in (1.0, 10.0, 1e6)]
    if any(left >= right for left, right in itertools.pairwise(log_nfas)):
        raise AssertionError("log NFA must strictly increase with n_tests")


def validate_threshold_and_edges() -> None:
    """Pin inclusive threshold equality and exact probability limits."""
    n_tests = 10.0
    epsilon = 10.0
    log10_nfa = math.log10(n_tests) + stable_log10_tail(20, 0, 0.125)
    if not log10_nfa <= math.log10(epsilon):
        raise AssertionError("exact NFA threshold equality must pass")

    edge_expectations = (
        ((0, 0, 0.0), 0.0),
        ((5, 0, 0.0), 0.0),
        ((5, 1, 0.0), -math.inf),
        ((0, 0, 1.0), 0.0),
        ((5, 5, 1.0), 0.0),
        ((5, 5, 0.25), 5.0 * math.log10(0.25)),
    )
    for arguments, expected in edge_expectations:
        actual = stable_log10_tail(*arguments)
        if math.isinf(expected):
            if actual != expected:
                raise AssertionError(f"edge {arguments} returned {actual}")
        else:
            if actual != expected:
                raise AssertionError(f"exact edge {arguments} returned {actual}")

    log_tail = stable_log10_tail(1_000_000, 1_000, 0.0001)
    if not math.isfinite(log_tail) or log_tail >= -100.0:
        raise AssertionError(f"large-n rare tail lost log stability: {log_tail}")
    if 10.0**log_tail != 0.0:
        raise AssertionError("large-n control should expose raw-tail underflow")


def validate_domain_and_backend_guards() -> None:
    """Pin the tested work domain and the reviewed SciPy failure guard."""
    boundary = stable_log10_tail(MAX_TRIALS, 1_000, 0.0001)
    if not math.isfinite(boundary) or boundary >= 0.0:
        raise AssertionError("maximum supported n lost a valid rare-tail result")

    # SciPy 1.16.3 returned a positive logsf (about 4.8315) for this input.
    # The public domain must reject it before delegating to the backend.
    invalid_n = 2**63 - 1
    invalid_k = invalid_n // 2
    try:
        stable_log10_tail(invalid_n, invalid_k, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("signed-int64 endpoint escaped the n-domain guard")

    for invalid_log_tail in (math.nan, UNIT_ROUNDOFF, 4.831532860197917):
        try:
            validate_scipy_log_tail(invalid_log_tail)
        except ArithmeticError:
            pass
        else:
            raise AssertionError(
                f"invalid backend log survival was accepted: {invalid_log_tail}"
            )

    validate_scipy_log_tail(-math.inf)
    validate_scipy_log_tail(0.0)


def validate_probability_endpoints() -> None:
    """Pin exact zero, one, minimum-positive, and maximum-below-one behavior."""
    minimum_positive = float(np.nextafter(0.0, 1.0))
    maximum_below_one = float(np.nextafter(1.0, 0.0))

    expected = float(
        decimal_rare_tail_log10(
            1000, 999, Fraction.from_float(minimum_positive)
        )
    )
    minimum_tail = stable_log10_tail(1000, 999, minimum_positive)
    assert_calibrated_ulps(
        minimum_tail,
        expected,
        limit=MINIMUM_POSITIVE_ULP_LIMIT,
        context="minimum-positive fallback",
    )
    if not math.isfinite(minimum_tail) or minimum_tail >= 0.0:
        raise AssertionError("minimum-positive p lost its finite log-tail")

    assert stable_log10_tail(1000, 1, 0.0) == -math.inf
    assert stable_log10_tail(1000, 1000, 1.0) == 0.0
    assert stable_log10_tail(1000, 999, maximum_below_one) <= 0.0


def validate_candidate_count_domain() -> None:
    """Reject zero or fractional-below-one candidate-family upper bounds."""
    for valid in (1, 1.0, 11.5, 1e12):
        validate_n_tests(valid)
    for invalid in (True, 0.9999999999999999, 0.5, 0.0, -1.0, math.inf, math.nan):
        try:
            validate_n_tests(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid n_tests was accepted: {invalid}")


def validate_geometric_remainder_bound() -> None:
    """Prove the decreasing-ratio bound against exact rational tail remainders."""
    n = 20
    p = Fraction(1, 8)
    k = 5
    term = Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k)
    previous_ratio = Fraction(1)
    for j in range(k, n):
        ratio = Fraction(n - j, j + 1) * p / (1 - p)
        if not 0 <= ratio < previous_ratio <= 1:
            raise AssertionError("rare-tail ratios must be in [0,1) and decrease")
        remaining = exact_binomial_tail(n, j + 1, p)
        exact_bound = term * ratio / (1 - ratio)
        if remaining > exact_bound:
            raise AssertionError("geometric remainder bound fell below exact remainder")

        term *= ratio
        previous_ratio = ratio


def validate_decimal_fallback_oracle() -> None:
    """Compare the million-trial fallback to independent 120-digit arithmetic."""
    expected = float(
        decimal_rare_tail_log10(1_000_000, 1_000, Fraction(1, 10_000))
    )
    actual = stable_log10_tail(1_000_000, 1_000, 0.0001)
    assert_calibrated_ulps(
        actual,
        expected,
        limit=MILLION_FALLBACK_ULP_LIMIT,
        context="million-trial rare-tail fallback",
    )


def validate_nfa_contract() -> None:
    """Run all independent numerical derivations."""
    validate_exact_small_tails()
    validate_exhaustive_false_alarm_control()
    validate_basic_log_nfa_arithmetic()
    validate_monotonicity()
    validate_threshold_and_edges()
    validate_domain_and_backend_guards()
    validate_probability_endpoints()
    validate_candidate_count_domain()
    validate_geometric_remainder_bound()
    validate_decimal_fallback_oracle()


if __name__ == "__main__":
    try:
        validate_nfa_contract()
    except Exception as error:  # pragma: no cover - command-line failure report
        print(f"A07 NFA logic validation FAILED: {error}", file=sys.stderr)
        raise
    print("A07 NFA logic validation PASSED")
