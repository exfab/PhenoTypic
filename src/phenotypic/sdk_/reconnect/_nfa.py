"""Source-independent binomial number-of-false-alarms statistic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom as _scipy_binom


_MAX_TRIALS = 1_000_000
_LOG_10 = math.log(10.0)


@dataclass(frozen=True, slots=True)
class NFAResult:
    """Numerically stable binomial NFA fields.

    Attributes:
        n: Broadcast trial counts as a read-only int64 array.
        k: Broadcast aligned counts as a read-only int64 array.
        p: Broadcast null probabilities as a read-only float64 array.
        log10_n_tests: Base-10 logarithm of the candidate-family bound.
        log10_binomial_tail: Base-10 logarithm of ``P[Binomial(n, p) >= k]``.
        log10_nfa: Base-10 logarithm of the number of false alarms.
        score: Negative ``log10_nfa``.
        meaningful: Inclusive ``NFA <= epsilon`` decision.
    """

    n: np.ndarray
    k: np.ndarray
    p: np.ndarray
    log10_n_tests: float
    log10_binomial_tail: np.ndarray
    log10_nfa: np.ndarray
    score: np.ndarray
    meaningful: np.ndarray


def _validated_integer_array(value: object, *, name: str) -> np.ndarray:
    """Return an integer array without changing its values."""

    array = np.asarray(value)
    if array.dtype == np.bool_ or array.dtype.kind not in "iu":
        raise TypeError(f"{name} must have an integer NumPy dtype other than boolean")
    return array


def _validated_probability_array(value: object) -> np.ndarray:
    """Convert a real probability input to float64."""

    raw = np.asarray(value)
    if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
        raise TypeError("p must have a real numeric dtype other than boolean")
    probabilities = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("p must contain only finite values in [0, 1]")
    return probabilities


def _validated_scalar(value: object, *, name: str, minimum: float) -> float:
    """Validate a finite real scalar with an inclusive lower bound."""

    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a finite real scalar other than boolean")
    scalar = float(raw)
    if not math.isfinite(scalar) or scalar < minimum:
        relation = ">= 1" if minimum == 1.0 else "> 0"
        raise ValueError(f"{name} must be finite and {relation}")
    return scalar


def _logaddexp_scalar(first: float, second: float) -> float:
    """Add two finite log-domain values without NumPy scalar dispatch."""

    if second > first:
        first, second = second, first
    return first + math.log1p(math.exp(second - first))


def _rare_upper_tail_log_probability(n: int, k: int, p: float) -> float:
    """Evaluate a nonzero rare upper tail using bounded log recurrence.

    The caller establishes ``0 < p < 1``, ``0 < k < n``, ``k > n*p``, and
    ``n-k+1 <= 1_000_000``. The ratio between consecutive PMF terms is formed
    additively in log space. A monotone geometric bound permits termination only
    when the entire omitted remainder cannot change the natural-log sum by half
    an ulp.
    """

    term_count = n - k + 1
    if term_count > _MAX_TRIALS:
        raise ValueError("rare-tail fallback exceeds the reviewed work bound")

    log_p = math.log(p)
    log_one_minus_p = math.log1p(-p)
    log_term = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * log_p
        + (n - k) * log_one_minus_p
    )
    if not math.isfinite(log_term):
        raise FloatingPointError("rare-tail initial log-PMF is not finite")

    log_sum = log_term
    for j in range(k + 1, n + 1):
        log_ratio = (
            math.log(n - j + 1)
            - math.log(j)
            + log_p
            - log_one_minus_p
        )
        log_term += log_ratio
        log_sum = _logaddexp_scalar(log_sum, log_term)

        if j == n:
            break
        log_next_ratio = (
            math.log(n - j)
            - math.log(j + 1)
            + log_p
            - log_one_minus_p
        )
        next_ratio = math.exp(log_next_ratio)
        if not 0.0 <= next_ratio < 1.0:
            raise FloatingPointError("rare-tail recurrence lost its monotone ratio")
        if next_ratio == 0.0:
            break
        log_remainder_bound = (
            log_term
            + log_next_ratio
            - math.log1p(-next_ratio)
        )
        log_relative_bound = log_remainder_bound - log_sum
        maximum_log_change = math.log1p(math.exp(log_relative_bound))
        if maximum_log_change <= 0.5 * math.ulp(log_sum):
            break

    if not math.isfinite(log_sum) or log_sum > 0.0:
        raise FloatingPointError("rare-tail fallback produced an invalid log probability")
    return log_sum


def _readonly_array(value: np.ndarray, *, dtype: np.dtype[np.generic]) -> np.ndarray:
    """Return an owned, C-contiguous, read-only array."""

    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def binomial_nfa(
    n: np.ndarray,
    k: np.ndarray,
    p: np.ndarray | float,
    *,
    n_tests: float,
    epsilon: float = 1.0,
) -> NFAResult:
    r"""Compute a binomial number-of-false-alarms statistic.

    For each broadcast element, this function evaluates
    :math:`P[X \ge k]` for :math:`X \sim \operatorname{Binomial}(n,p)`, then
    returns :math:`\log_{10}(N_{tests} P[X \ge k])`. Candidate geometry and
    the complete-family bound are caller responsibilities; this helper never
    infers them from image dimensions or input length.

    Args:
        n: Broadcastable integer trial counts in ``[0, 1_000_000]``.
        k: Broadcastable integer success counts satisfying ``0 <= k <= n``.
        p: Broadcastable finite null probabilities in ``[0, 1]``.
        n_tests: Finite scalar upper bound on the complete candidate family,
            at least one.
        epsilon: Finite positive meaningfulness threshold. Equality is
            meaningful.

    Returns:
        An immutable :class:`NFAResult` preserving the common broadcast shape.
        Intentional impossible-event logs are ``-inf`` and their scores are
        ``+inf``. Raw NFA values are not exposed because they can underflow.

    Raises:
        TypeError: An input has an unsupported dtype or is not scalar where
            required.
        ValueError: Inputs cannot broadcast or violate the reviewed domain.
        FloatingPointError: SciPy or the bounded fallback produces an invalid
            log probability.
    """

    n_input = _validated_integer_array(n, name="n")
    k_input = _validated_integer_array(k, name="k")
    p_input = _validated_probability_array(p)
    n_tests_value = _validated_scalar(n_tests, name="n_tests", minimum=1.0)
    epsilon_value = _validated_scalar(
        epsilon,
        name="epsilon",
        minimum=np.nextafter(0.0, 1.0),
    )

    try:
        n_broadcast, k_broadcast, p_broadcast = np.broadcast_arrays(
            n_input,
            k_input,
            p_input,
        )
    except ValueError as error:
        raise ValueError("n, k, and p must be broadcastable to one shape") from error

    if np.any(n_broadcast < 0):
        raise ValueError("n must satisfy 0 <= n <= 1000000")
    if np.any(n_broadcast > _MAX_TRIALS):
        raise ValueError("n exceeds the reviewed 1000000-trial domain")
    if np.any(k_broadcast > n_broadcast) or np.any(k_broadcast < 0):
        raise ValueError("k must satisfy 0 <= k <= n")

    n_values = np.asarray(n_broadcast, dtype=np.int64)
    k_values = np.asarray(k_broadcast, dtype=np.int64)
    p_values = np.asarray(p_broadcast, dtype=np.float64)
    log_tail = np.full(n_values.shape, np.nan, dtype=np.float64)

    k_zero = k_values == 0
    p_one = p_values == 1.0
    p_zero_impossible = (p_values == 0.0) & ~k_zero
    exact_all_successes = (
        (k_values == n_values)
        & ~k_zero
        & (p_values > 0.0)
        & (p_values < 1.0)
    )
    log_tail[k_zero | p_one] = 0.0
    log_tail[p_zero_impossible] = -np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        log_tail[exact_all_successes] = (
            n_values[exact_all_successes]
            * np.log(p_values[exact_all_successes])
        )

    general = np.isnan(log_tail)
    if np.any(general):
        backend_log_tail = np.asarray(
            _scipy_binom.logsf(
                k_values[general] - 1,
                n_values[general],
                p_values[general],
            ),
            dtype=np.float64,
        )
        backend_log_tail = np.broadcast_to(
            backend_log_tail,
            n_values[general].shape,
        )
        if np.any(np.isnan(backend_log_tail)) or np.any(backend_log_tail > 0.0):
            raise FloatingPointError("SciPy returned an invalid binomial log survival")
        log_tail[general] = backend_log_tail

    fallback = np.isneginf(log_tail) & (p_values > 0.0) & (p_values < 1.0)
    for flat_index in np.flatnonzero(fallback):
        current_n = int(n_values.flat[flat_index])
        current_k = int(k_values.flat[flat_index])
        current_p = float(p_values.flat[flat_index])
        if current_k <= current_n * current_p:
            raise FloatingPointError(
                "SciPy underflowed outside the reviewed rare upper-tail fallback"
            )
        log_tail.flat[flat_index] = _rare_upper_tail_log_probability(
            current_n,
            current_k,
            current_p,
        )

    if np.any(np.isnan(log_tail)) or np.any(log_tail > 0.0):
        raise FloatingPointError("binomial log survival is outside [-inf, 0]")

    log10_tail = np.array(log_tail / _LOG_10, dtype=np.float64, copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log10_tail[exact_all_successes] = (
            n_values[exact_all_successes]
            * np.log10(p_values[exact_all_successes])
        )
    log10_n_tests = math.log10(n_tests_value)
    log10_nfa = log10_n_tests + log10_tail
    score = -log10_nfa
    meaningful = log10_nfa <= math.log10(epsilon_value)

    return NFAResult(
        n=_readonly_array(n_values, dtype=np.dtype(np.int64)),
        k=_readonly_array(k_values, dtype=np.dtype(np.int64)),
        p=_readonly_array(p_values, dtype=np.dtype(np.float64)),
        log10_n_tests=log10_n_tests,
        log10_binomial_tail=_readonly_array(
            log10_tail,
            dtype=np.dtype(np.float64),
        ),
        log10_nfa=_readonly_array(log10_nfa, dtype=np.dtype(np.float64)),
        score=_readonly_array(score, dtype=np.dtype(np.float64)),
        meaningful=_readonly_array(meaningful, dtype=np.dtype(np.bool_)),
    )


__all__ = ["NFAResult", "binomial_nfa"]
