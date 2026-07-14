# A04: multiscale bowler-hat transform

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Deferred after incomplete G0 executable-oracle review
**Blocked by:** unavailable licensed MATLAB runtime and unresolved source license/formula fork

## G0 decision

A04 is deferred from this release. The available executable oracle could not run because the
configured MATLAB license server returned error `-15,570`. The signed-versus-absolute residual,
footprint rasterization, boundary behavior, and final normalization fork therefore could not be
regenerated or reviewed. Source licensing also remains unresolved. Under
`01-reference-and-validation.md`, a paper-only implementation without the selected executable
fixture and license decision cannot pass G0, so no A04 production code, wrapper, export, or
fixture enters this release.

Resume A04 only after the licensed runtime is available, the oracle outputs are regenerated, and
an independent reviewer freezes the formula fork and compatible implementation boundary. The
contract proposal below remains planning material, not an approved implementation contract.

## Contract decision

The design and preliminary remote inspection suggest a possible fork between the paper's absolute
residual and an author MATLAB signed residual followed by global min-max normalization. This is a
hypothesis, not a frozen source claim, until D1 stores both complete artifacts locally and cites the
relevant `refs/...:line` ranges. Pin both and record the selected authority. The primary paper is
older than five years but defines this historical method
([Sazak et al.](https://arxiv.org/abs/1709.05495)). Resolve source licensing before copying code;
absent a compatible license, implement from the paper equation and use source outputs only as an
oracle.

```python
def bowler_hat(
    image: np.ndarray,
    scales: Sequence[int],
    n_orientations: int,
) -> np.ndarray: ...
```

Freeze whether scales are sampled lengths or `1..dmax`, orientation grid
`k*180/n_orientations`, one-pixel line rasterization/anchor, disk radius rounding, boundary mode,
signed/absolute residual, final normalization, dtype, and safe zero-range behavior. Use
`class FocusEdgeBowlerHat(FocusEdge)`. `FocusEdge` already inherits footprint behavior through
`ImageEnhancer` (`src/phenotypic/abc_/CLAUDE.md:90-94`), and all numerical footprints belong in
the pure helper.

Validate a finite, nonempty 2-D numeric array without mutation; a nonempty sequence of positive
integer scales; source-decided duplicate/order behavior; and a positive orientation count with a
precisely defined half-open angular grid. Reject Boolean/complex or nonfinite inputs unless the
source contract explicitly supports them.
Exact 90-degree covariance is required only when `n_orientations` is even, because only then does
the sampled half-open angle grid close under a 90-degree permutation. For odd counts, compare the
rotated output to the independently rerasterized oracle instead of asserting exact covariance.
Before the wrapper is dispatched, D0 must select source/domain-justified default scales and
orientation count so `FocusEdgeBowlerHat()` is valid with no arguments. Store scales as a tuple,
coerce JSON lists back to tuples, and pin default construction plus round-trip behavior.

## Owned files

```text
src/phenotypic/sdk_/reconnect/_bowler_hat.py
src/phenotypic/enhance/_focus_edge_bowler_hat.py
tests/unit/sdk_/reconnect/test_bowler_hat.py
tests/unit/enhance/test_focus_edge_bowler_hat.py
tests/fixtures/reconnect/bowler_hat/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/bowler_hat.py
refs/bowler_hat corpus and reconciliation
```

## Tasks

1. Pin paper, author code/test image, commit, MATLAB morphology semantics, license, and hashes.
2. Resolve absolute/signed, scale/disk, line rasterization, padding, normalization, and dtype forks.
3. Generate fixtures containing every line/disk footprint, per-angle opening, per-scale maxima and
   differences, raw maximum, and normalized output.
4. Write an independent direct-enumeration morphology oracle and red tests.
5. Implement streaming orientation and scale maxima to avoid the reference's full
   `H x W x scale x orientation` stack while preserving arithmetic order.
6. Add wrapper, forwarding spy, doctest, serialization, taxonomy, tune, and invariant tests.
7. Reviewer audits license, source fork, fixture intermediates, mutations, and peak memory.

## Logic-validation script

Build discrete line/disk masks independently, then implement grayscale erosion/dilation by direct
neighborhood min/max enumeration. Check aligned/off-angle bars, X-junction, blob-plus-line,
border line, constant input, scale order/duplicates, and the selected source fork's intensity law.
For a raw residual, test offset invariance and positive intensity-scale equivariance. For the
globally min-max-normalized fork, test positive affine invariance plus defined constant-range
behavior. Also test exact 90-degree/reflection covariance. For other sampled angles, compare only against the
pinned discrete rasterization oracle because arbitrary rotations do not preserve a square pixel
lattice exactly. Use exact equality for binary morphology and an ulp-derived bound only for final
normalization.

## Required mutants

- add/remove `abs` or reverse subtraction;
- min instead of max over orientation;
- difference global rather than paired per-scale maxima;
- wrong reduction order;
- disk radius `d` or floor instead of source rounding;
- include/omit an angular endpoint;
- interpolate a rotated rectangle instead of exact line rasterization;
- per-scale instead of final normalization;
- unsigned integer subtraction wrap;
- NaN on constant input;
- changed boundary mode.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/bowler_hat.py
uv run pytest tests/unit/sdk_/reconnect/test_bowler_hat.py tests/unit/enhance/test_focus_edge_bowler_hat.py -q
uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_bowler_hat.py -q
uv run mypy src/phenotypic/sdk_/reconnect src/phenotypic/enhance/_focus_edge_bowler_hat.py
uv run ruff check src/phenotypic/sdk_/reconnect src/phenotypic/enhance/_focus_edge_bowler_hat.py
```

After S02, the algorithm reviewer returns for:

```bash
uv run pytest tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py tests/unit/enhance/test_detect_mat_invariant.py -q
```
