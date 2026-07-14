# A07: clean-room binomial NFA statistic

**Implementer:** one clean-room 5.6-sol/high-effort algorithm turn
**Oracle agent:** source-exposed agent that never authors production
**Reviewer:** independent 5.6-sol/high-effort turn that may inspect both sides
**Shape:** Keystone statistical core only; label/orientation and detector seams deferred
**Blocked by:** corrected C8, sanitized evidence gate, and S00

## Corrected contract

The IPOL LSD 1.6 executable is AGPL-3.0-or-later and may be used only by the oracle side of this
clean-room port. Production is independently derived from the published binomial-tail equation,
the sanitized behavioral contract, and source-generated numeric fixtures.

```python
@dataclass(frozen=True)
class NFAResult:
    n: np.ndarray
    k: np.ndarray
    p: np.ndarray
    log10_n_tests: float
    log10_binomial_tail: np.ndarray
    log10_nfa: np.ndarray
    score: np.ndarray
    meaningful: np.ndarray

def binomial_nfa(
    n: np.ndarray,
    k: np.ndarray,
    p: np.ndarray | float,
    *,
    n_tests: float,
    epsilon: float = 1.0,
) -> NFAResult: ...
```

Define `score = -log10_nfa` and
`meaningful = log10_nfa <= log10(epsilon)`. Intentional `-inf` log tails and `+inf` scores are
valid for zero-probability events; NaN is invalid. Validate broadcastable integer `n/k`,
`0 <= k <= n <= 1_000_000`, broadcastable finite `p` in `[0,1]`, finite `n_tests >= 1`, and
finite `epsilon > 0`. Arrays remain int64 containers, but values outside the reviewed numerical
domain raise before SciPy is called. Outputs are float64 or bool, broadcast to one common shape,
C-contiguous, and read-only.

Callers explicitly supply `n`, `k`, `p`, and `n_tests`. The helper does not infer orientation
probabilities, candidate families, image-size test counts, axes, labels, or masks. All orientation
counting, label mapping, wrapper/refiner behavior, and detector integration are deferred.

Use SciPy's binomial log survival only when its result is finite and nonpositive. Reject NaN,
positive, or otherwise impossible results. For a mathematically nonzero rare upper tail reported
as `-inf`, use a source-independent log-PMF/log-ratio sum bounded by `n <= 1_000_000`. Compute
ratios additively with `log(n-j+1) - log(j) + log(p) - log1p(-p)` and use a derived monotone
geometric remainder bound. Special-case `k=0`, `k=n`, and `p=0/1` before general evaluation.

Inclusive equality is intentional. The paper defines meaningful events with `NFA <= epsilon` and
uses that relation in its proof, although one displayed theorem prints `<`; the executable detector
uses strict acceptance. Reconciliation must record this three-way notation fork and the reason the
paper's definition and proof control the public contract.

## Clean-room and licensing boundary

The oracle checkout contains the complete pinned LSD source and license. The production implementer
receives a sanitized checkout containing only this plan, the published mathematical contract as
permitted for use, synthetic numeric fixtures, and clean-room handoff documents. It must contain no
restricted source files or restricted Git objects. The root `NOTICE`, path-scoped third-party
notice, and package manifests must state which reference artifacts are distributed and under which
terms. Human licensing review remains required before release.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_nfa.py
tests/unit/sdk_/reconnect/test_nfa.py
tests/fixtures/reconnect/nfa/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/nfa.py
refs/nfa corpus, clean-room handoff, provenance, reconciliation, and drift
```

1. Oracle agent pins the IPOL paper and LSD executable, licenses, revisions, and complete corpus.
2. Oracle agent compiles the unmodified source, captures all source-visible scalar outputs, and
   emits sanitized project-authored fixtures.
3. Freeze the numerical domain, SciPy accuracy/error policy, invalid-result checks, fallback
   termination bound, equality, `n_tests`, dtype, broadcasting, and invalid inputs.
4. A clean implementer writes red controls and production using only the sanitized allowlist.
5. The independent logic script uses `math.comb` and `math.fsum` for exact small tails, exhaustive
   small Bernoulli families, monotonicity, equality, edge cases, and derived tolerance checks.
6. Reviewer reruns both source and independent oracles, audits the information barrier, kills every
   mutant, and checks that no label/detector seam exists.

## Required controls and fixture keys

Fixture `n/k/p`, `n_tests`, epsilon, raw/log tail, log NFA, score, and meaningful flags. Include
`k=0`, `k=n`, `p=0`, `p=1`, minimum-positive `p`, maximum float below one, equality, maximum
reviewed `n`, invalid-domain values, SciPy `-inf` fallback, and injected NaN/positive backend
results. Exact small cases use exact arithmetic; large cases use a derived log-domain error bound.

## Required mutants

- use `sf(k)` instead of `sf(k-1)` or CDF instead of survival;
- omit `n_tests`, use natural log as log10, or reverse score sign;
- make the meaningful threshold strict;
- accept `n > 1_000_000`, `n_tests < 1`, or a positive/NaN backend log survival;
- underflow multiplicative ratios at minimum-positive `p`;
- omit exact `k=0`, `k=n`, or `p=0/1` branches;
- trust backend `-inf` for a mathematically nonzero tail;
- use an unexplained fixed tolerance or unbounded fallback loop.

There are no A07 detector or wrapper seam mutants in this release.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/nfa.py
uv run pytest tests/unit/sdk_/reconnect/test_nfa.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_nfa.py
uv run ruff check src/phenotypic/sdk_/reconnect/_nfa.py tests/unit/sdk_/reconnect/test_nfa.py
```

After S02, the reviewer returns only for the pure-core public export, notice, packaging, and
import-boundary seams.
