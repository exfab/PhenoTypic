# A09 Clark Rolling Hough source contract

## Approved candidate scope

The selected executable authority is the source-author `seclark/RHT` repository at immutable MIT
commit `4d06f9fa4cafe9022011a0bec0315390d7e23c39`. It can authorize a dependency-free numerical core
that reproduces Clark preprocessing and the original, rho-zero Rolling Hough Transform. It cannot
authorize the broader planned `RollingHoughResult.coherence` field or a PhenoTypic enhancement
wrapper.

The candidate clean API is intentionally narrower than the current algorithm plan:

```python
@dataclass(frozen=True)
class ClarkRollingHoughResult:
    theta: np.ndarray                # float64, (T,), Hough-normal radians [0, pi)
    support_counts: np.ndarray       # int64, (T,)
    raw_counts: np.ndarray           # int64, (H, W, T)
    threshold_residual: np.ndarray   # float64, (H, W, T)
    response: np.ndarray             # float64, (H, W), unnormalized residual sum
    orientation: np.ndarray          # float64, (H, W), Hough normal in (0, pi]
    eligible: np.ndarray             # bool, (H, W), source rolling-window mask
    valid: np.ndarray                # bool, (H, W), any positive residual

def clark_rolling_hough(
    image: np.ndarray,
    window_diameter: int,
    smoothing_radius: int,
    threshold_fraction: float,
) -> ClarkRollingHoughResult: ...
```

No production is authorized until an independent G0 reviewer approves this exact contract.

## Frozen behavior

1. `image` is a nonempty, two-dimensional, real numeric array. The source converts it to floating
   arithmetic for masking. Nonfinite pixels are source-supported bad pixels and invalidate nearby
   centers. Boolean and complex inputs are rejected at the Python boundary as D01.
2. `window_diameter` is a positive odd integer. `smoothing_radius` is a positive integer, matching
   the executable parameter rather than renaming it as a diameter. `threshold_fraction` is finite
   and in `[0, 1]`.
3. The smoothing footprint is the inclusive integer-radius disk of diameter
   `2 * smoothing_radius + 1`. Correlation uses SciPy's default reflect boundary, is divided by the
   footprint's pixel count, and is subtracted from the original image. The bitmask comparator is
   strict `> 0` (`source_clark/rht/rht.py:530-556`).
4. Source default bad-pixel flags are retained: nonfinite values are bad, while finite zero and
   negative values are not intrinsically bad (`source_clark/rht/rht.py:90-94,403-448`). The
   smoothing mask excludes the footprint radius at edges and near bad pixels. The rolling-window
   mask then excludes a second circular halo of radius `window_diameter // 2`
   (`source_clark/rht/rht.py:450-488,518-528`).
5. The theta count is
   `ceil(pi * (window_diameter - 1) / sqrt(2))`; the grid is float64 on `[0, pi)` with no endpoint
   (`source_clark/rht/rht.py:259-264,780-792`). Theta parameterizes the Hough line normal, not the
   filament tangent (`clark_2014_fibers_cpp.tex:90-104`).
6. The rolling domain is the inclusive integer-radius disk. Each angle's rho-zero center line is
   generated from the Hough normal equation with NumPy round-to-nearest-even. The exact discrete
   support is angle dependent (`source_clark/rht/rht.py:530-537,592-664`).
7. At every eligible center, `raw_counts` is the int64 rho-zero count for each theta. Values outside
   `eligible` are adapter zeros and must not be interpreted as evaluated counts.
8. `support_counts` applies the same transform to the full circular window. The source residual is
   `raw_counts / support_counts - threshold_fraction`, multiplied by its `>= 0` mask. Exact
   equality is numerically zero, so a bin is accepted only when its residual is positive
   (`source_clark/rht/rht.py:799-846`). Values outside `eligible` are adapter zeros.
9. `valid` is true only where at least one residual is positive, matching sparse source emission.
   `response` is the unnormalized sum of residuals, matching the source value before persistence
   (`source_clark/rht/rht.py:842-846`). No global normalization belongs in this core.
10. At valid pixels, `orientation` uses the source doubled-angle weighted sum and its `(0, pi]`
    mapping. It is the axial Hough-normal angle. Invalid orientation is adapter `NaN`, not the
    source helper's zero-weight value of pi (`source_clark/rht/rht.py:667-692`). Crossings with an
    exactly cancelling doubled-angle vector retain the executable's floating `atan2` behavior;
    no new tie-break is invented.
11. Counts and supports are frozen to int64. Theta, residual, response, and orientation are float64.
    Masks are Boolean. The source's later float32 persistence conversion is not part of this
    in-memory core (`source_clark/rht/rht.py:83-85,842-856`).
12. A constant or otherwise empty-result image returns zero residual/response, false `valid`, and
    NaN orientation. This avoids the executable persistence path's empty-array `IndexError` while
    preserving all defined numerical products; it is registered as D06.

## Explicit exclusions

- No `coherence` field. A mean-resultant-length or peak ratio would be a new derived capability.
- No globally normalized response or `detect_mat` wrapper.
- No smoothing-diameter translation.
- No FilFinder runtime dependency. FilFinder v1.8 is only a test oracle for axial normal
  orientation on simple binary skeletons.
- No claim of improved fungal detection. That requires a separate ground-truth benchmark.
