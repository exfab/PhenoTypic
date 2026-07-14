# A07 NFA frozen reference contract

## Authority and scope

The mathematical authority is the peer-reviewed IPOL paper. Its pages 37 to 39 define the
uniform-orientation null, the binomial survival event, the complete candidate-family factor,
and the expected-false-alarm theorem. The executable oracle is the official peer-reviewed LSD
1.6 `nfa` function at `source/lsd.c:1033-1158`. The paper is more than five years old, but the
load-bearing claims here are definitions and a theorem rather than a current empirical result.

The paper has an internal notation fork: its meaningful-event definition, proof, and algorithmic
test use `NFA <= epsilon`, while one theorem display prints `NFA < epsilon`. The executable detector
also accepts strictly (`source/lsd.c:2137-2139`). The public contract follows the paper's explicit
definition and proof, so equality is meaningful; drift N04 records both strict variants.

The production scope is only the pure statistic `binomial_nfa`. Geometry, orientation counting,
label filtering, and `NFAValidation` are deferred. LSD does not source the proposed PhenoTypic
label adapter, and a label map alone cannot determine `n`, `k`, `p`, or `N_tests`.

## Public numerical contract

Inputs `n`, `k`, and `p` are broadcast to a common shape. `n` and `k` must have integer NumPy
dtypes other than boolean and satisfy `0 <= k <= n <= 1_000_000`. `p` is converted to float64 and
must be finite in `[0, 1]`. `n_tests` is scalar, finite, and at least one; `epsilon` is scalar,
finite, and strictly positive.
Empty broadcast results are permitted. The one-million-trial ceiling is the tested numerical and
computational domain, not a mathematical property of NFA. It caps the source-free fallback at
most one million recurrence updates after the exact `k=0` shortcut and includes the required
one-million-pixel rare-tail control. [Based on general reasoning -- no specific citation
available.]

For every broadcast element, define

\[
B(n,k,p)=\Pr[X\ge k],\qquad X\sim\operatorname{Binomial}(n,p),
\]

\[
\log_{10}\operatorname{NFA}=\log_{10}(N_{tests})+\log_{10}B(n,k,p),
\]

\[
\operatorname{score}=-\log_{10}\operatorname{NFA}.
\]

Production first uses SciPy's log survival path, mathematically equivalent to
`binom.logsf(k - 1, n, p) / log(10)`. SciPy 1.16 can still return `-inf` after its
underlying survival probability underflows even when the mathematical tail is nonzero. In that
case production must use a source-free stable rare-tail fallback based on log-PMF terms and
log-addition. Each recurrence ratio is evaluated additively as
`log(n-j+1) - log(j) + log(p) - log1p(-p)`; multiplying the ratio in value space underflows for
valid subnormal probabilities. It does not reproduce LSD's 10-percent truncation. The fallback
is permitted only for a rare upper tail (`k > n*p`) containing at most 1,000,000 terms. The
public `n` ceiling makes that work bound unconditional. The fallback must retain finite log tails
for both `n=1_000_000, k=1_000, p=0.0001`, where raw probability underflows, and
`n=1000, k=999, p=np.nextafter(0, 1)`, where a value-space recurrence ratio is zero.

For `k > n*p`, successive term ratios decrease. After including term `t_j`, let
`q_j = t_(j+1)/t_j = ((n-j)/(j+1))*p/(1-p)`. The uncomputed remainder is bounded by
`t_j*q_j/(1-q_j)`. Early termination is permitted only when
`log1p(remainder_bound/current_sum) <= 0.5*ulp(log(current_sum))`; otherwise summation continues,
up to the explicit term-count ceiling. The standalone suite proves the ratio monotonicity and
bound against exact rational remainders. [Based on direct geometric-series reasoning -- no
specific citation available.]

Every SciPy result must be in `[-inf, 0]`. A NaN or positive log survival is an invalid backend
result and must raise rather than be returned or normalized. This guard is load-bearing: SciPy
1.16.3 returned approximately `4.831532860197917` for
`n=2^63-1, k=(2^63-1)//2, p=0.5`. That input is now rejected by the public trial ceiling before
SciPy is called. This is evidence that accepting the whole signed-int64 range was unsafe, not a
claim that every SciPy version retains the defect.
Exact edge semantics are:

- `k=0`, including `n=k=0`: tail 1 and log-tail 0.
- `k=n`, with `0 < p < 1`: log-tail `n*log10(p)`.
- `p=0, k>0`: tail 0, log-tail `-inf`, score `+inf`.
- `p=1`: tail 1 for every valid `k <= n`.
- NaN is never a valid result. Intentional `-inf` log tails and `+inf` scores are valid.

The small-tail accuracy gate enumerates every `n=0..30`, every valid `k`, and the exact rational
probabilities `1/100`, `1/8`, `1/2`, `9/10`, and `999/1000`. Expected tails are computed with
`Fraction` arithmetic before one final float64 rounding. On the locked SciPy 1.16.3 path, the
public log10-to-probability round trip has an exhaustively observed maximum error of 256 ULP at
`n=26, k=24, p=1/100`; 256 ULP is therefore the pinned regression envelope, not an unexplained
absolute tolerance. The rare fallback has two independent 120-digit Decimal controls built from
exact `math.comb` coefficients and a normalized value-space recurrence, structurally different
from the production-style `lgamma`/log-addition path. On the locked runtime, the million-trial
case is within exactly 3,297 ULP of that oracle and the minimum-positive case within one ULP.
Those three finite grids/controls are the calibrated libm/SciPy accuracy promise; the contract
does not extrapolate an untested global ULP bound.

False-alarm enumeration uses `Fraction` throughout and therefore needs no floating tolerance.
The separate log-NFA add/negate check treats its libm results as fixed inputs and uses the standard
`gamma_m = m*u/(1-m*u)` basic-operation bound with unit roundoff
`u = np.finfo(float64).eps/2`; it makes no accuracy claim about libm itself. [Based on standard
floating-point error reasoning -- no specific citation available.]

`meaningful` is exactly

```text
log10_nfa <= log10(epsilon)
```

so equality passes. Returned `n` and `k` are signed int64 arrays; `p`, log fields, and score are
float64 arrays; `meaningful` is boolean. The result preserves broadcast shape and contains no raw
NFA field because exponentiating can underflow.

## Candidate-family obligation

`n_tests >= 1` is explicit. It is a conservative upper bound on a nonempty candidate family and
therefore cannot be fractional below one or zero. An empty broadcast result remains permitted but
still uses `n_tests >= 1`; there is no separate empty-family scalar semantic. The function does
not infer the count from labels, image size, or the number of rows in its input. A caller must
provide a predetermined upper bound covering its complete
candidate-selection procedure, including any estimated axes, tolerances, widths, refinements, or
other data-selected alternatives.

For faithful LSD rectangles only, the source freezes

\[
N_{tests}=11(XY)^{5/2},
\]

where `X` and `Y` are the scaled angle image dimensions. The factor 11 covers the initial
precision plus two groups of five precision refinements (`source/lsd.c:2081-2095`). This formula
must not be reused for a different candidate family without a new derivation.

## Licensing barrier

The official executable is AGPL-3.0-or-later while PhenoTypic is Apache-2.0. The oracle agent may
compile and run the reference, but production must be written by a clean implementer from this
source-free behavioral contract, the published paper, and generated fixtures. No production body
may be copied or structurally transcribed from `source/lsd.c`.
