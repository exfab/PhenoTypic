# A09: Rolling Hough Transform

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone Clark transform core only; coherence and wrapper deferred
**Blocked by:** C10 and S00

## Corrected contract

The helper must return enough information for both promised wrapper outputs. Pin the Clark paper,
official/source-author implementation, exact FilFinder RHT implementation/version, licenses, and
defaults. The RHT was introduced as a local measure of coherent linearity
([Clark et al. 2014](https://arxiv.org/abs/1312.1338)); the exact discretization must come from the
pinned code, not the abstract.

```python
@dataclass(frozen=True)
class ClarkRollingHoughResult:
    theta: np.ndarray
    support_counts: np.ndarray
    raw_counts: np.ndarray
    threshold_residual: np.ndarray
    response: np.ndarray
    orientation: np.ndarray
    eligible: np.ndarray
    valid: np.ndarray

def clark_rolling_hough(
    image: np.ndarray,
    window_diameter: int,
    smoothing_radius: int,
    threshold_fraction: float,
) -> ClarkRollingHoughResult: ...
```

The core accepts only a nonempty 2-D float64 array. Integer, Boolean, float16, float32,
extended-float, and complex inputs are rejected. `window_diameter` is positive and odd;
`smoothing_radius` is a positive source radius, not a translated diameter; and
`threshold_fraction` is finite in `[0, 1]`.

`raw_counts` has shape `image.shape + (n_theta,)` and stores exact int64 rho-zero center-line
counts. The core also exposes exact angle-dependent `support_counts`, source residuals,
unnormalized residual-sum response, Hough-normal axial orientation, the source rolling-window
eligibility mask, and a dense Boolean validity conversion. Coherence is not sourced and is
deferred. Invalid orientations are NaN. No normalization, wrapper mapping, or `detect_mat`
conversion belongs in this release.

Freeze the inclusive disk smoothing footprint, SciPy reflect correlation, strict `> 0` bitmask,
bad-pixel halos, angular count/grid, round-to-nearest-even rho-zero rasterization, angle-dependent
supports, threshold equality producing zero residual, doubled-angle collapse, crossings,
empty-result behavior, axes, and dtypes. Dense `valid` is exactly
`np.any(threshold_residual > 0, axis=2)` and is recorded as representation drift D09.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_rolling_hough.py
tests/unit/sdk_/reconnect/test_rolling_hough.py
tests/fixtures/reconnect/rolling_hough/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
refs/rolling_hough corpus and reconciliation
```

1. Resolve port-versus-wrap wording: production is dependency-free. Use a pinned Clark/source-
   author executable as the full-pipeline oracle. FilFinder is only a compatible binary-skeleton
   accumulator cross-check because its modified RHT reports filament/branch angular distributions,
   not this plan's image preprocessing or pixelwise response/coherence contract
   ([FilFinder tutorial](https://fil-finder.readthedocs.io/en/latest/tutorial.html)).
2. Freeze float64-only input, preprocessing, source radius, diameter, angle, line, gate, border,
   collapse, sentinel, dense-valid drift, and output equations.
3. Capture smoothing, unsharp, bitmask, theta grid, accumulator, validity, accepted bins, and every public
   output for asymmetric, boundary, gap, crossing, border, constant, and non-default cases.
4. Write a direct small-array oracle and red helper tests.
5. Implement the pure helper and deterministic kernel.
6. Defer coherence, response normalization, smoothing-diameter translation, wrapper output,
   doctest, serialization, taxonomy, tune fields, and any `detect_mat` conversion.
7. Keep FilFinder only in the oracle environment for simple binary-skeleton angle comparisons;
   committed tests consume fixtures without it. Record preprocessing and derived products as
   PhenoTypic adaptations unless the pinned source-author executable supplies them.
8. Reviewer reruns FilFinder/source oracle and mutations.

## Logic-validation script

Independently build the disk smoothing footprint, direct reflect correlation, unsharp bitmask,
eligibility halos, angular grid, rho-zero center lines, support counts, integer raw counts, exact
threshold residual, dense validity, response, and axial collapse. Recount every eligible center;
check all odd diameters 1 through 31, exact gate equality, zero-weight and constant cases,
nonfinite-pixel halos, horizontal/vertical/diagonal recovery, 90-degree covariance, line reversal
modulo \(\pi\), crossings, borders, and all fixture outputs. Counts/residuals/response are exact;
the reviewed source-orientation controls require zero ULP.

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
- wrong border mode;
- derive validity from raw counts instead of positive residual;
- accept or silently convert a non-float64 image;
- add coherence, normalization, or a wrapper-derived field.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
uv run pytest tests/unit/sdk_/reconnect/test_rolling_hough.py tests/unit/sdk_/reconnect/test_import_rules.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_rolling_hough.py
uv run ruff check src/phenotypic/sdk_/reconnect/_rolling_hough.py tests/unit/sdk_/reconnect/test_rolling_hough.py
```
