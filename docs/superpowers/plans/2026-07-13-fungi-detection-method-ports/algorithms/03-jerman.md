# A03: Jerman vesselness

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Deferred after incomplete G0 source-fidelity review
**Blocked by:** unavailable authenticated full paper needed to freeze C5

## G0 decision

A03 is deferred from this release. The author executable source was located and exercised, but
the full authoritative paper could not be retrieved after the university-authenticated browser
lost network access. The paper is required to decide whether the selected executable implements
the complete published Hessian chain and response contract, and to review that decision
independently. An implementation drafted from the executable alone cannot pass G0 under
`01-reference-and-validation.md`, so no A03 production code, wrapper, export, or fixture enters
this release.

Resume A03 by restoring authenticated paper access, pinning and hashing the paper locally,
reconciling it line by line with the executable source, then obtaining an independent G0 review.
The contract proposal below remains planning material, not an approved implementation contract.

## Contract decision

Choose one claim before coding:

1. faithful author-MATLAB Hessian chain plus Jerman response; or
2. Jerman response law over scikit-image Hessians, explicitly labeled an adapted port.

The author distribution states that its 2-D vesselness implementation accompanies the 2016 work
([author implementation](https://jp.mathworks.com/matlabcentral/fileexchange/63171-jerman-enhancement-filter)). Reusing scikit-image changes derivative kernels, padding, eigenvalue
ordering, scale normalization, and potentially threshold behavior. Every difference needs a drift
row.

Recommended interface after the choice:

```python
def jerman_vesselness(
    image: np.ndarray,
    sigmas: Iterable[float],
    tau: float,
    black_ridges: bool,
    *,
    mode: BoundaryMode,
    cval: float = 0.0,
) -> np.ndarray: ...
```

Validate finite 2-D input, nonempty positive sigmas, source-supported tau, boundary mode, safe
constant input, and no mutation. Freeze helper dtype/range only after selecting faithful versus
adapted scope; any cast or normalization absent from the authority receives a drift row. The
wrapper must satisfy the repository's finite float32 `[0,1]` `detect_mat` boundary. It mirrors
the tuple-coercion and per-scale memory discipline in
`src/phenotypic/enhance/_focus_edge_sato.py:85-128`.
Select the wrapper's no-argument defaults, including boundary mode, only after the source fork is
resolved; add border fixtures that distinguish MATLAB-replicate from scikit-image alternatives.

## Owned files

```text
src/phenotypic/sdk_/reconnect/_jerman.py
src/phenotypic/enhance/_focus_edge_jerman.py
tests/unit/sdk_/reconnect/test_jerman.py
tests/unit/enhance/test_focus_edge_jerman.py
tests/fixtures/reconnect/jerman/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/jerman.py
refs/jerman corpus and reconciliation
```

## Tasks

1. Pin the TMI paper, complete author code, inherited Kroon Hessian source, licenses, runtime, and
   scikit-image source if adapting.
2. Reconcile Hessian, polarity, eigenvalue sorting, tau clamp, cutoffs, scale maximum,
   normalization, dtype, and padding.
3. Capture non-square/off-axis, crossing, multi-width, polarity, boundary, weak-ridge, and constant
   fixtures with every intermediate/public output. Include unsorted and duplicate sigmas and pin
   whether output is order/duplicate invariant; separately test list, tuple, and generator inputs.
4. Write the algebraic standalone oracle and red helper/wrapper tests.
5. Implement helper, streaming scale maximum, safe-zero branch, and no hidden wrapper behavior.
6. Add wrapper forwarding spy, tuple coercion, doctest, serialization, taxonomy, and tune fields.
7. Reviewer reruns the selected source and mutation matrix.

## Logic-validation script

For the polarity-adjusted response ratio, re-derive the complete piecewise law:

\[
V(r)=
\begin{cases}
0, & r\le 0,\\
\dfrac{27r^2(1-r)}{(1+r)^3}, & 0<r<\tfrac12,\\
1, & r\ge\tfrac12,
\end{cases}
\qquad
V'(r)=\frac{54r(1-2r)}{(1+r)^4}\ \text{on }(0,\tfrac12).
\]

Check piecewise branches and continuity at \(r=1/2\), lambda-rho regularization, positive scale
intensity invariance, exact saturation at 1 without redundant post-normalization, safe all-zero
behavior, offset/polarity behavior,
and analytic Hessians for a Gaussian ridge and equal-eigenvalue crossing. Compare Frangi
analytically: its crossing/line shape ratio is parameter-dependent and below one, not identically
zero. If the faithful Hessian is ported, independently reproduce smoothing/finite differences with
SciPy and compare against closed-form ridge curvature.

## Required mutants

- drop tau regularization;
- use signed rather than polarity-adjusted maximum;
- change denominator/factor/cube;
- reverse saturation inequality or zero mask;
- select wrong eigenvalue or polarity;
- use last/min scale rather than maximum;
- remove safe-zero handling;
- change source cutoffs or scale normalization;
- hardcode wrapper argument;
- remove tuple coercion or dtype contract.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/jerman.py
uv run pytest tests/unit/sdk_/reconnect/test_jerman.py tests/unit/enhance/test_focus_edge_jerman.py -q
uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_jerman.py -q
uv run mypy src/phenotypic/sdk_/reconnect src/phenotypic/enhance/_focus_edge_jerman.py
uv run ruff check src/phenotypic/sdk_/reconnect src/phenotypic/enhance/_focus_edge_jerman.py
```

After S02, the algorithm reviewer returns for:

```bash
uv run pytest tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py -q
```
