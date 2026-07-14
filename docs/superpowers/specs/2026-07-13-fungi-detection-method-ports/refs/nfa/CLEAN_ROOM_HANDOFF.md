# A07 clean-room implementation handoff

## Allowed production scope

Implement only a pure, vectorized `binomial_nfa` statistic. Do not implement orientation
counting, label filtering, a detector adapter, or an `ObjectRefiner`. Those seams require a
separate candidate-family derivation and review.

The function accepts broadcastable integer arrays `n` and `k`, and a broadcastable float array or
scalar `p`, plus scalar keyword-only `n_tests` and `epsilon`. It returns immutable arrays for the
broadcast `n`, `k`, and `p`; scalar `log10_n_tests`; and arrays `log10_binomial_tail`, `log10_nfa`,
`score`, and `meaningful`.

Compute

\[
B(n,k,p)=\sum_{j=k}^{n}{n\choose j}p^j(1-p)^{n-j},
\]

\[
\log_{10}\mathrm{NFA}=\log_{10}(N_{tests})+\log_{10}B(n,k,p),
\qquad
\mathrm{score}=-\log_{10}\mathrm{NFA}.
\]

The inclusive decision is `log10_nfa <= log10(epsilon)`. This formula and its false-alarm
interpretation are established in the supplied peer-reviewed IPOL paper, especially pages 37 to
39 and 45. The paper is from 2012, more than five years old, but these claims are mathematical
definitions and a proved bound, not a current empirical performance claim.

## Validation and dtype rules

- Require integer NumPy dtypes for `n` and `k`; reject boolean, float, object, and values outside
  `0 <= k <= n <= 1_000_000`.
- Require `0 <= k <= n` after broadcasting.
- Convert `p` to float64 and require finite values in `[0, 1]`.
- Require finite scalar `n_tests >= 1` and `epsilon > 0`; reject booleans. Empty broadcast output
  still uses the conservative minimum `n_tests=1`; there is no fractional/zero empty-family mode.
- Permit an empty broadcast result.
- Return `n` and `k` as int64, numerical probability/log fields as float64, and `meaningful` as
  boolean, preserving the broadcast shape.
- `k=0`, including `n=k=0`, has tail one and log-tail zero.
- `p=0, k>0` has tail zero, log-tail `-inf`, and score `+inf`.
- `p=1` has tail one for every valid `k <= n`.
- Intentional infinities above are valid. NaN is always an implementation error.
- Do not expose raw NFA because exponentiation can underflow.

Use SciPy's binomial log survival for ordinary values. SciPy 1.16 returns `-inf` for some nonzero
extreme tails after raw probability underflow. Supply an independently designed log-domain rare
tail fallback only when `k > n*p` and `n-k+1 <= 1_000_000`. The required control is
`n=1_000_000`, `k=1_000`, `p=0.0001`: its log tail must remain finite while its raw float64 tail is
zero. Reject every SciPy log survival that is NaN or greater than zero. In particular, reject
`n=2^63-1, k=(2^63-1)//2, p=0.5` at input validation; SciPy 1.16.3 produced an impossible positive
log survival for that input. The one-million-trial ceiling is an explicit work bound: after the
`k=0` shortcut, the direct fallback can execute no more than one million recurrence updates. It is
a product-domain decision based on bounded work, not a theorem from the paper. Do not accept
unexplained fixed tolerances.

Compute every fallback ratio additively as
`log(n-j+1) - log(j) + log(p) - log1p(-p)`. A value-space ratio is forbidden because it
underflows at the valid control `n=1000, k=999, p=np.nextafter(0, 1)`. Special-case `k=0`, `k=n`,
and `p=0/1`; include exact controls for zero, one, minimum-positive float64, and the maximum
float64 below one.

For small-tail accuracy, use exact rational arithmetic for the five-probability exhaustive grid
specified in `REFERENCE_CONTRACT.md`. The locked SciPy 1.16.3 public round trip reaches 256 ULP at
`n=26, k=24, p=1/100`; that exhaustively measured value is the regression envelope. Independently
derive the fallback controls with 120-digit `Decimal`, exact `math.comb` coefficients, and a
normalized value-space recurrence. On the locked runtime, the million-trial fallback differs from
that oracle by exactly 3,297 ULP and the minimum-positive control by one ULP. These finite cases are
the complete calibrated libm/SciPy accuracy promise; do not infer a global ULP bound. Prove the
small false-alarm families with exact `Fraction` arithmetic and no tolerance. Only the isolated
basic add/negate check may use `gamma_m = m*u/(1-m*u)`, with the actual operation count and unit
roundoff `u=np.finfo(float64).eps/2`; that bound treats libm outputs as fixed inputs and makes no
claim about libm accuracy. Do not introduce unexplained epsilon multipliers.

`n_tests` is always explicit. It must cover the caller's complete predetermined candidate family,
including every data-selected alternative. The pure function must never infer it from image size,
labels, or input row count.

## Permitted evidence

A clean implementer may read only:

- this file;
- `REFERENCE_CONTRACT.md`;
- `paper/2012_IPOL_LSD.pdf` and the rendered paper pages;
- `CLEAN_ROOM_ALLOWLIST.txt`, `CLEAN_ROOM_ATTESTATION.md`, and `ATTRIBUTION.md`;
- `tests/fixtures/reconnect/nfa/lsd_source.json`;
- `tests/fixtures/reconnect/nfa/contract_edges.json`;
- the algorithm plan and ordinary PhenoTypic architecture/style guides.

The JSON fixture's source executable exposes only one numerical output, `-log10(NFA)`. Its exact
tail fields were independently re-derived from the published finite sum and intentionally differ
slightly from the executable's documented approximation for some records. Production targets the
exact fields, with source-score comparison used only to detect sign, tail-index, and test-count
mistakes.

The clean implementer must not open the vendored source directory, archive, source harness,
reconciliation, drift register, oracle-generation script, or standalone oracle logic script. The
reviewer may inspect both sides and runs the oracle logic script after production is complete.

The orchestrator must first run `prepare_clean_room.py` from a clean exact commit. The exporter
copies tracked files into a new directory with no `.git` directory or Git objects and excludes all
A07-specific evidence except the exact paths in `CLEAN_ROOM_ALLOWLIST.txt`. The implementer works
only inside that export, completes `CLEAN_ROOM_ATTESTATION.md`, and returns changed files plus a
source-free patch. The orchestrator, not the implementer, applies that patch to the real worktree.
Any access by the implementer to the oracle checkout invalidates the barrier.
