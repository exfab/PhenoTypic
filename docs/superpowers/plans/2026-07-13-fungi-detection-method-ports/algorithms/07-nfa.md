# A07: A-contrario / NFA validation

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone statistical core followed by detector or wrapper Seam
**Blocked by:** C8 and S00

## Corrected contract

A label map alone cannot compute the stated NFA. Separate the numerical binomial tail from
geometry/orientation counting:

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

@dataclass(frozen=True)
class SegmentAlignmentCounts:
    label_ids: np.ndarray
    n: np.ndarray
    k: np.ndarray
    p: np.ndarray
    valid: np.ndarray

def binomial_nfa(
    n: np.ndarray,
    k: np.ndarray,
    p: np.ndarray | float,
    *,
    n_tests: float,
    epsilon: float = 1.0,
) -> NFAResult: ...

def segment_alignment_counts(
    labels: np.ndarray,
    orientations: np.ndarray,
    label_ids: np.ndarray,
    segment_axes: np.ndarray,  # row i is the independently supplied axis for label_ids[i]
    orientation_valid: np.ndarray,
    *,
    angle_tolerance: float,
) -> SegmentAlignmentCounts: ...
```

Define `score = -log10_nfa` and
`meaningful = log10_nfa <= log10(epsilon)`. Intentional `-inf` log tails and `+inf` scores are valid
for zero-probability events, while NaN is always invalid. Validate integer broadcastable `n/k`,
`0 <= k <= n`, broadcastable `p` in `[0,1]`, finite `n_tests > 0`, finite `epsilon > 0`, unique
positive `label_ids`, same-shape labels/orientations/valid mask, one finite axis per label, and
explicit missing-axis failure. Label 0 is never tested. The set of positive values in `labels` must
equal `label_ids`; an extra/unmatched positive label or a requested absent label raises rather than
being silently retained or rejected.

For directed level-line angles with a source-defined window, LSD uses its own probability model
and image-size test count ([LSD derivation](https://dev.ipol.im/~morel/Dossier_MVA_2010_Cours_Transparents_Documents/Cours_1_texte_Line_Segment_Detector.pdf)). Directed mode is deferred from this release rather than exposing an undefined model. PhenoTypic's phase orientation is axial,
as shown by doubled-angle coherence at
`src/phenotypic/sdk_/branch_pathfinding/_cost_surface.py:30-54`; a symmetric `±tau` window then has
probability `2*tau/pi`, not the directed LSD probability. Freeze the candidate family, `n`, `k`,
axis source, tolerance, orientation period, `N_tests`, equality threshold, and selection-bias
assumptions.
Constrain axial tolerance to `[0, pi/2]`. Candidate axes must be supplied independently of the
orientation samples being tested, unless the axis-estimation search is explicitly included in the
candidate-family/test-count derivation.

Invalid orientation pixels are excluded from both `n` and `k`. A label with at least one valid
orientation has `valid=True` and is tested using only those samples. A label with zero valid samples
has `n=k=0`, `valid=False`, is not passed to `binomial_nfa`, and is rejected by the detector adapter.
Fixture and test all-valid, partially valid, zero-valid, absent-label, and extra-label cases.

Prefer integration inside `FilamentousFungiDetector` while phase orientation is live. Defer a
generic `ObjectRefiner` until it has a valid independent orientation source. Keep calculations in
log10 space to avoid underflow; preserve retained label IDs rather than relabeling.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_nfa.py
tests/unit/sdk_/reconnect/test_nfa.py
tests/fixtures/reconnect/nfa/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/nfa.py
refs/nfa corpus and reconciliation
```

1. Pin IPOL/TPAMI LSD, MIT binomial NFA source, general NFA theorem, licenses, and revisions.
2. Decide exact SciPy tail versus source's approximate/truncated computation, then record drift.
3. Specify candidate geometry and whether `N_tests` is explicit, LSD rectangles, or a provably
   predetermined candidate count. Derive and document an upper bound for the detector's complete
   candidate-selection family, including any axis/refinement attempts. Never infer it silently from
   data-selected fragments. The exact false-alarm enumeration must exercise that complete selection
   procedure, not only independent Bernoulli tails.
4. Write exact small-n enumeration and source goldens before production code.
5. Implement stable `binomial_nfa`, then alignment counting as a separate layer.
6. Integrator applies the gate where phase orientation and candidate geometry coexist, maps result
   rows back through `label_ids`, zeros rejected labels, preserves retained IDs, and returns the
   filtered label map plus binary mask. Add dedicated detector integration tests. If a wrapper is
   later justified, add protected-layer, forwarding, doctest, serialization, and tune tests.
7. Reviewer audits null-model assumptions, candidate-selection independence, exact tails, mutants,
   and label preservation.

## Logic-validation script

Use `math.comb` and `math.fsum` to compute

\[
B(n,k,p)=\sum_{j=k}^n {n\choose j}p^j(1-p)^{n-j}.
\]

Compare the intended log-survival computation, exhaustively enumerate small Bernoulli test families
to verify expected false alarms no greater than epsilon, derive directed/axial alignment
probabilities, test axial wrap, monotonicity in `k`/`N_tests`/tolerance, exact threshold equality,
large-n log stability, and edge cases `n=0`, `k=0`, `k=n`, `p=0/1`. Monte Carlo is secondary only,
with fixed seed and an explicit confidence bound.

Fixture all inputs and outputs: candidates/axes/orientations, `n/k/p`, test count, raw/log tail,
log NFA, score, flags, kept IDs, filtered label map, and binary mask.

## Required core mutants

- `sf(k)` rather than `sf(k-1)` or CDF rather than survival;
- wrong axial probability or degrees/radians;
- omit circular wrap;
- data-selected rather than declared support count;
- omit test count, factor, or exponent;
- natural log labeled log10;
- strict rather than inclusive threshold;
- score-sign reversal or raw-probability underflow;
- include background or estimate axis circularly from tested angles;
- count invalid orientations, test a zero-valid label, or accept unmatched label IDs.

## Required post-S01 seam mutants

- preserve a zero-valid label, relabel survivors, or map rows to the wrong label IDs;
- recompute/use a different orientation field or change retained-label raster values;

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/nfa.py
uv run pytest tests/unit/sdk_/reconnect/test_nfa.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_nfa.py
uv run ruff check src/phenotypic/sdk_/reconnect/_nfa.py tests/unit/sdk_/reconnect/test_nfa.py
```

After S01, the algorithm reviewer returns for:

```bash
uv run pytest tests/unit/detect/test_filamentous_fungi_nfa.py -q
uv run mypy src/phenotypic/detect/_filamentous_fungi_detector.py
uv run ruff check src/phenotypic/detect/_filamentous_fungi_detector.py tests/unit/detect/test_filamentous_fungi_nfa.py
```
