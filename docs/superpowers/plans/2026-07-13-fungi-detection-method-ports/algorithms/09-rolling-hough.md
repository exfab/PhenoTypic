# A09: Clark Rolling Hough core

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn

**Reviewer:** independent 5.6-sol/high-effort turn

**Shape:** source-faithful numerical core only

**Blocked by:** independent G0 PASS and shared scaffold S00

## Corrected contract

The executable authority is the source-author `seclark/RHT` repository at immutable MIT commit
`4d06f9fa4cafe9022011a0bec0315390d7e23c39`. The approved candidate is narrower than the original
plan: it reproduces Clark preprocessing and the rho-zero rolling transform but does not invent a
pixelwise coherence statistic or enhancement wrapper.

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

Input is exactly a nonempty 2-D `float64` NumPy array. There is no implicit integer or float32
conversion. `window_diameter` is a positive odd integer, `smoothing_radius` is a positive integer,
and `threshold_fraction` is finite in `[0, 1]`.

The result exposes both source-near integer counts and float64 threshold residuals. `orientation`
is the Hough-normal axial angle, not the filament tangent. Invalid orientation is NaN. Dense
Boolean `valid` is an explicit adapter conversion from sparse positive-residual emission. The
complete formulas, axes, boundaries, sentinels, dtype rules, and deviations are frozen in
`refs/rolling_hough/SOURCE_CONTRACT.md` and `DRIFT.md`.

Coherence remains deferred. A mean-resultant length, peak ratio, or similarly named field would be
a new derived capability requiring its own equation, tie behavior, tests, drift row, and review.
The fixture parameter previously misnamed `coherence_fraction` is `threshold_fraction`.

## Owned production files

```text
src/phenotypic/sdk_/reconnect/_rolling_hough.py
tests/unit/sdk_/reconnect/test_rolling_hough.py
tests/fixtures/reconnect/rolling_hough/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/
```

The algorithm implementer must not edit wrappers, public exports, registries, dependency files, or
the deferred coherence design.

## Execution steps

1. Reviewer approves the pinned paper/source corpus, narrow source contract, drift register,
   fixture, verifier pins, and paper redistribution disposition.
2. Write red tests for every public field, all captured preprocessing intermediates, exact counts,
   float64 residual bounds, invalid inputs, empty output, borders, nonfinite masking, and dtypes.
3. Implement the four stages separately: preprocessing and eligibility masks, theta/center-line
   rasterization, raw count and residual accumulation, then response/orientation collapse.
4. Compare every source-visible output and intermediate with the golden fixture. Use exact integer
   comparison and documented ULP-derived floating bounds.
5. Run required mutants individually and map each to a named test.
6. Independent reviewer performs line-by-line source reconciliation and signs the exact commit.

## Required mutants

- interpret smoothing radius as a diameter;
- use a noninclusive disk or different border mode;
- change strict-positive unsharp masking;
- change theta count, include the endpoint, swap row/column, or use tangent angle;
- replace round-to-nearest-even during center-line rasterization;
- use constant rather than angle-dependent support;
- change threshold equality to accepted positive output;
- compute validity from raw counts or return integer validity;
- normalize the local response globally;
- return pi instead of NaN for invalid orientation;
- accept or silently convert integer/float32 inputs;
- reproduce the source empty-output exception.

## Focused gate

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/generate_fixture.py
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/verify_fixture.py
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/rolling_hough.py
uv run pytest tests/unit/sdk_/reconnect/test_rolling_hough.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_rolling_hough.py
uv run ruff check
```

No wrapper, coherence, biological-performance, or globally normalized output claim is approved by
this core gate.
