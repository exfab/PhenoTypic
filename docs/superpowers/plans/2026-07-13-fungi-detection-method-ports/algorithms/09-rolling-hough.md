# A09: Rolling Hough Transform

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone transform core and Leaf diagnostic wrapper
**Blocked by:** C10 and S00

## Corrected contract

The helper must return enough information for both promised wrapper outputs. Pin the Clark paper,
official/source-author implementation, exact FilFinder RHT implementation/version, licenses, and
defaults. The RHT was introduced as a local measure of coherent linearity
([Clark et al. 2014](https://arxiv.org/abs/1312.1338)); the exact discretization must come from the
pinned code, not the abstract.

```python
@dataclass(frozen=True)
class RollingHoughResult:
    angular_accumulator: np.ndarray
    response: np.ndarray
    orientation: np.ndarray
    coherence: np.ndarray
    valid: np.ndarray

def rolling_hough(
    image: np.ndarray,
    window_diameter: int,
    smoothing_diameter: int,
    coherence_fraction: float,
) -> RollingHoughResult: ...
```

`angular_accumulator` has shape `image.shape + (n_theta,)` and stores raw integer center-line
counts from the pinned source. If the source also normalizes the distribution, expose that only as
a separately named derived field with a written rounding bound. `response`, `coherence`, and axial
`orientation` are explicitly derived capability fields. `valid` is false wherever the source gate
accepts no angle; invalid helper orientations use `NaN`, while the wrapper applies the finite
mapping frozen below. Do not
invent a response silently. Freeze odd-diameter/radius rules, normalized smoothing kernel, convolution
padding, bitmask comparator, angular grid, center-line rasterization, threshold rounding and
inclusive comparison, border support, axial collapse, crossing behavior, undefined-angle sentinel,
and dtype. Keep raw radians in the helper and map to `[0,1]` only in the wrapper.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_rolling_hough.py
src/phenotypic/enhance/_focus_edge_rolling_hough.py
tests/unit/sdk_/reconnect/test_rolling_hough.py
tests/unit/enhance/test_focus_edge_rolling_hough.py
tests/fixtures/reconnect/rolling_hough/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
refs/rolling_hough corpus and reconciliation
```

1. Resolve port-versus-wrap wording: production is dependency-free. Use a pinned Clark/source-
   author executable as the full-pipeline oracle. FilFinder is only a compatible binary-skeleton
   accumulator cross-check because its modified RHT reports filament/branch angular distributions,
   not this plan's image preprocessing or pixelwise response/coherence contract
   ([FilFinder tutorial](https://fil-finder.readthedocs.io/en/latest/tutorial.html)).
2. Freeze preprocessing, diameter, angle, line, gate, border, collapse, sentinel, and output
   equations.
3. Capture smoothing, unsharp, bitmask, theta grid, accumulator, validity, accepted bins, and every public
   output for asymmetric, boundary, gap, crossing, border, constant, and non-default cases.
4. Write a direct small-array oracle and red helper tests.
5. Implement the pure helper and deterministic kernel.
6. Add `RollingHoughOutput = Literal["response", "orientation"]`, wrapper normalization, spy,
   diagnostic warning, doctest, serialization, taxonomy, and tune fields. The orientation wrapper
   maps valid axial radians to `[0, 1)` by division by `pi` and writes `0.0` at invalid pixels so
   `detect_mat` stays finite. Document that wrapper output alone cannot distinguish an invalid
   orientation from a valid zero angle; inferential callers must use the pure helper's `valid` mask.
7. Add FilFinder only to the dev oracle environment for binary-skeleton accumulator comparisons;
   committed tests consume fixtures without it. Record preprocessing and derived products as
   PhenoTypic adaptations unless the pinned source-author executable supplies them.
8. Reviewer reruns FilFinder/source oracle and mutations.

## Logic-validation script

Independently build the normalized smoothing kernel, direct convolution, unsharp bitmask, angular
grid, center lines, integer counts, exact threshold gate, validity, and axial collapse. Check constant input,
analytic template count, exact gate and one-pixel-below cases, positive-scale invariance, additive-
offset invariance only in the padding-independent valid interior, separately sourced border behavior,
horizontal/vertical/diagonal recovery, 90-degree covariance, line reversal modulo \(\pi\), crossing
multi-orientation behavior, allowed/disallowed gap, border rule, and all fixture outputs. Counts are
exact; angle error is bounded by half the angular bin width plus floating rounding.

## Required mutants

- wrong smoothing kernel/normalization;
- `>=0` versus `>0` bitmask;
- swap diameters or radius off by one;
- square rather than circular window;
- full Hough rather than rho-zero center line;
- row/column or degree/radian error;
- wrong gate rounding or strictness;
- wrong angular endpoint or 2-pi periodicity;
- arithmetic rather than axial weighted mean;
- arbitrary angle on zero response;
- pass `NaN` through the orientation wrapper or map an invalid pixel to a nonzero finite angle;
- wrong border mode;
- wrapper maps orientation as response or hardcodes a field.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
uv run pytest tests/unit/sdk_/reconnect/test_rolling_hough.py tests/unit/enhance/test_focus_edge_rolling_hough.py -q
uv run pytest tests/unit/abc_/test_enhancer_taxonomy.py tests/unit/tune/test_enhance_annotations.py tests/unit/tune/test_annotation_coverage.py tests/unit/sdk_/test_typing_aliases.py -q
uv run pytest --doctest-modules src/phenotypic/enhance/_focus_edge_rolling_hough.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_rolling_hough.py src/phenotypic/enhance/_focus_edge_rolling_hough.py
uv run ruff check src/phenotypic/sdk_/reconnect/_rolling_hough.py src/phenotypic/enhance/_focus_edge_rolling_hough.py
```
