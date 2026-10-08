# SubtractPolySurface — polynomial surface flattening (design)

**Date:** 2026-10-08
**Status:** design approved section by section in conversation (2026-10-08); this written spec
awaits user review.
**Companions:** [`references.md`](./references.md) (every source, cited `path:line`),
[`drift-register.md`](./drift-register.md) (every deviation, one row each),
`docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/` (the numeric
claims, re-derived and executable).

---

## 1. Objective

Add `SubtractPolySurface`, an `ImageEnhancer` that flattens slow background variation in
`detect_mat` by fitting and subtracting a low-order polynomial. One `method` parameter switches
between four methods:

| `method` | Fits | Gwyddion tool it ports |
|---|---|---|
| `"offset"` | a constant (order 0) | *Level → Zero Mean Value* |
| `"plane"` | a plane (order 1) | *Level → Plane Level* |
| `"polynomial"` | a 2-D polynomial surface, order ≥ 2 | *Level → Polynomial Background* |
| `"line"` | a low-order polynomial per scan line | *Correct Data → Align Rows*, **Polynomial** method |

Conventions and defaults follow **Gwyddion 2.71** (Nečas & Klapetek 2012), the reference for
SPM-style leveling. Fitting reuses numpy and scipy (`numpy.polynomial.legendre`,
`numpy.linalg.lstsq`, `scipy.stats.median_abs_deviation`); no fitting math is hand-written beyond
the outlier-clipping loop.

### Non-goals

- **Masks.** Gwyddion can include or exclude a mask in every tool; enhancers run before detection
  and have none. `fit="robust"` is the substitute (drift D9).
- Align Rows' non-polynomial methods (Median — Align Rows' own default — Modus, Median difference,
  Matching, Facet tilt, trimmed means), Level Rotate, Fix Zero, distinct per-axis degrees,
  background extraction (drift D10).
- Any change to existing enhancers.

## 2. Decisions taken in conversation (2026-10-08)

| # | Decision | Alternatives rejected |
|---|---|---|
| Q1 | The colony-bias treatment is **selectable**: `fit = "lstsq" \| "robust"` | robust-only; plain-only; exclude an existing `objmap` |
| Q2 | Approach **A**: numpy.polynomial basis + `lstsq`; robust = iterative sigma-clipping | sklearn `PolynomialFeatures` + `RANSACRegressor`/`HuberRegressor`; `scipy.optimize.least_squares` with a robust loss. Brainstorming probe on a 600×900 synthetic plate (σ = 0.01): at 40% colony cover `HuberRegressor` erred 0.1035 — no better than plain least squares (0.1037) — `least_squares(loss="cauchy")` 0.0016, sigma-clipping 0.0002 and fastest. |
| Q3 | **Split order fields**: `order` (polynomial only) and `line_order` (line only) | one shared `order` whose legal range would depend on `method`, making an Optuna trial `method="polynomial", order=1` fail validation |
| Q4 | **Defaults match Gwyddion's**, citing Gwyddion and the literature | the probe-optimal defaults (`polynomial`/2/`robust`) proposed first |
| Q5 | **Level convention per method, as each Gwyddion tool does** (§4.6) | always → 0; always keep the mean; a `keep_mean` flag |
| Q6 | **Replicate Plane Level's half-pixel pivot** (drift D7) | pivot at the true centre |
| Q7 | **`max_iter = 10`**, not astropy's `niter = 3` (drift D12) | 3; 3 rounds × 5 inner clip passes |
| Q8 | **Clean-room port**: Gwyddion's GPL source and GFDL guide stay outside the repository and away from the implementer (references.md §1) | vendoring the C files under `refs/` |

## 3. Public API

```python
class SubtractPolySurface(NormalizedOutputMixin, BackgroundSubtraction):
    method:      SurfaceMethod = "plane"
    order:       Annotated[int, TuneSpec(2, 5)] = Field(3, ge=2, le=11)
    independent: bool = True
    line_order:  Annotated[int, TuneSpec(0, 3)] = Field(1, ge=0, le=5)
    line_axis:   LineAxis = "row"
    fit:         SurfaceFit = "lstsq"
    clip_sigma:  Annotated[float, TuneSpec(2.0, 4.0)] = Field(3.0, gt=0.0)
    max_iter:    Annotated[int, TuneSpec(tunable=False)] = Field(10, ge=1)
    # norm: NormOut = "clip"  -- appended last by NormalizedOutputMixin
```

`SurfaceMethod`, `SurfaceFit`, `LineAxis` are `Literal` `TypeAlias`es added to
`phenotypic.sdk_.typing_` (type-only closed sets with no documentation surface of their own — the
`adding-an-operation` convention, like `BoundaryMode`):

```python
SurfaceMethod: TypeAlias = Literal["offset", "plane", "polynomial", "line"]
SurfaceFit:    TypeAlias = Literal["lstsq", "robust"]
LineAxis:      TypeAlias = Literal["row", "column"]
```

| Field | Default | Bounds | Used by | Default's source |
|---|---|---|---|---|
| `method` | `"plane"` | closed set | all | Gwyddion has no cross-tool default; Plane Level is its first leveling step and the simplest (user-approved) |
| `order` | `3` | `2 ≤ order ≤ 11` | `polynomial` | `col_degree = row_degree = max_degree = 3` (`polylevel.c:141-146`); upper bound is Gwyddion's `MAX_DEGREE = 11` (`polylevel.c:33`); lower bound is the request ("order 2 and up") |
| `independent` | `True` | bool | `polynomial` | `independent = TRUE` (`polylevel.c:149`). `True`: per-axis degree, term set `{u^p v^q : p, q ≤ order}` — `(order+1)²` terms. `False`: total degree, `{u^p v^q : p + q ≤ order}` — `(order+1)(order+2)/2` terms. Both axes share `order` (Gwyddion's `same_degree = TRUE`, `polylevel.c:148`). |
| `line_order` | `1` | `0 ≤ line_order ≤ 5` | `line` | Align Rows polynomial `max_degree` default 1, range 0–`MAX_DEGREE` = 5 (`linematch.c:44, 173`) |
| `line_axis` | `"row"` | closed set | `line` | Align Rows direction default *horizontal* (`linematch.c:171-172`) |
| `fit` | `"lstsq"` | closed set | all | Gwyddion has no robust estimator; plain least squares is its only fit |
| `clip_sigma` | `3.0` | `> 0` | `fit="robust"` | astropy `SigmaClip(sigma=3.0)` (`astropy_sigma_clipping.py:177`) |
| `max_iter` | `10` | `≥ 1` | `fit="robust"` | drift D12 (astropy's `niter = 3` is not converged at 40% cover) |
| `norm` | `"clip"` | `NormOut` | all | `NormalizedOutputMixin` convention |

**Fields a method does not use are ignored, not rejected**, and each field's docstring entry says
which methods read it. Rejecting them would make a tune trial fail whenever the sampler varies an
unused field.

**`TuneSpec` windows.** `order` 2–5 and `line_order` 0–3 are search windows inside the validity
bounds (a 6th-order or higher surface on a plate image is a curiosity, not a search target);
`clip_sigma` 2–4 brackets astropy's 3.0. None is literature-verified — mark each
`# TODO: review bound (unverified vs literature)`, the repository's convention for exactly this.
`max_iter` is a convergence cap, not a knob: `tunable=False`. `method`, `fit`, `line_axis` and
`independent` are non-numeric and outside the annotation-coverage gate.

## 4. Behavioural contract

This section is the **source-free contract** the implementer works from (clean-room, §8). Every
statement is a fact about behaviour; `references.md` §4 carries the citations.

Notation: `z` is `detect_mat` as float64, shape `(H, W)`; `i` the row index `0..H−1`; `j` the
column index `0..W−1`.

### 4.1 Coordinates

- Surfaces: `u = 2j/(W−1) − 1`, `v = 2i/(H−1) − 1`, both in `[−1, 1]`.
- Lines (after the `line_axis` transpose, §4.5): each line of length `N` uses
  `t = 2k/(N−1) − 1`, `k = 0..N−1`.

### 4.2 Basis and solver (drift D1)

- Build the design matrix from **Legendre** polynomials of `u` and `v` (or of `t`):
  `numpy.polynomial.legendre.legvander` on each axis, multiplied column-wise for the chosen term
  set. The term *set* (which `(p, q)` pairs) is exactly Gwyddion's (§3, `independent`); only the
  basis polynomials differ, and they span the same space.
- Solve with `numpy.linalg.lstsq`. If the returned rank is below the number of terms, **raise
  `ValueError`** (drift D2).
- Evaluate a fitted surface on the full grid with `numpy.polynomial.legendre.leggrid2d(v_axis,
  u_axis, C)`, where `C[q, p]` holds the coefficient of `L_p(u)·L_q(v)` (zero for terms outside the
  set). Never build an `H·W × n_terms` matrix for evaluation (B3).

### 4.3 Robust fit (`fit="robust"`; drift D11–D15)

The structure of astropy's `FittingWithOutlierRemoval`, with a single-pass clip:

```
fit on all points                                   # astropy_fitting.py:1006
repeat up to max_iter times:                        # :1017
    r      = data − model                            # residuals at ALL points
    center = median(r[kept]);  scale = mad_std(r[kept])
    if scale == 0: stop                              # nothing left to clip
    new    = kept AND |r − center| ≤ clip_sigma·scale   # clipped points never return (:1027)
    if count(new) < n_terms: stop, keep previous fit # drift D3
    if count(new) == count(kept): stop               # converged (:1105-1108)
    kept = new;  refit on kept                       # :1085
```

`mad_std` is `scipy.stats.median_abs_deviation(r, scale="normal")`. For `method="line"` the loop
runs **independently per line**: each line has its own `kept`, `center`, `scale`, and stopping
state, and a line that stops keeps its last fit while the others continue.

### 4.4 Per method

**`offset`.** Subtract a constant: the mean of `z` (`lstsq`), or the robust loop's constant
(design matrix `[1]`). → output level 0.

**`plane`.** Fit the terms `{1, u, v}` (per §4.2/§4.3) to get `b_u`, `b_v`. Convert to
per-pixel-index slopes: `bx = 2·b_u/(W−1)`, `by = 2·b_v/(H−1)`. Subtract **only the tilt**,
pivoted at `(W/2, H/2)`:

```
out = z − bx·(j − W/2) − by·(i − H/2)
```

The fitted constant is discarded. Consequence: `mean(out) = mean(z) + ½(bx + by)`. The pivot is
**not** the true centre `((W−1)/2, (H−1)/2)` — that is Gwyddion's behaviour, replicated on purpose
(drift D7; L1, L2). Do not "correct" it.

**`polynomial`.** Fit the term set of §3 (`order`, `independent`) and subtract the **whole**
fitted surface, constant included. → output level 0 (to rounding, under `lstsq`; L6).

**`line`.** See §4.5.

### 4.5 `line` method

1. If `line_axis == "column"`, transpose `z`; level its rows; transpose back. (`"row"` levels rows
   directly.) Gwyddion's vertical direction is implemented exactly this way.
2. Compute `avg = mean(z)` over the **whole input**, once, before leveling — the plain mean, even
   under `fit="robust"`.
3. Fit every line with the degree-`line_order` basis on `t` (§4.1). Under `lstsq`, fit all lines
   in **one** `lstsq` call with the lines as right-hand sides. Under `robust`, run §4.3 per line.
4. `out = z − fit + avg`: every line is leveled to the input's global mean (L4). For
   `line_order = 0` this equals Gwyddion's row-mean-shift form (L5).

### 4.6 Output level, per method (decision Q5)

| `method` | Where the background ends up | Gwyddion behaviour followed |
|---|---|---|
| `offset` | 0 | Zero Mean Value |
| `plane` | `mean(z) + ½(bx + by)` (the mean, up to the pivot shift) | Plane Level |
| `polynomial` | 0 | Polynomial Background |
| `line` | `mean(z)`, on every line | Align Rows (Polynomial) |

Switching `method` therefore also changes the output level. The class docstring carries this
table. Under `norm="clip"` (the default), methods that land at 0 lose the negative half of the
background noise; the docstring points to `norm="rescale"` or `norm=None` for when that matters
(drift D5).

### 4.7 Validation and degenerate input

- `order < 2` for `method="polynomial"` cannot occur (field bound); `order`, `line_order` are
  validated by their `Field` bounds regardless of `method`.
- At `apply` time, raise `ValueError` with an actionable message when:
  - `H < 2` or `W < 2` (the coordinate normalization divides by `H−1`, `W−1`);
  - `method="polynomial"`: `H < order + 1` or `W < order + 1` (too few distinct coordinates per
    axis; covers both term sets, since total-degree mode also reaches `u^order` and `v^order`);
  - `method="line"`: the line length `N < line_order + 1`;
  - the solver reports a rank below the number of terms (drift D2).
- `ImageOperation` wraps an exception raised in `_operate`, destroying its type (see
  `2026-07-08-alt-phase-detection/drift-register.md` M10, `_image_operation.py:422, 469`). Tests
  assert on the `__cause__` chain, as `TestSigmaOnfOneIsRejectedAtApplyTime` does.

### 4.8 Large images (drift D4)

- Whole-image surfaces (`offset` under `robust`, `plane`, `polynomial`) are fitted on a strided
  subsample when `H·W > MAX_FIT_POINTS = 262_144` (a private module constant, not a field).
  Let `d` be the surface's per-axis degree — 0 for `offset`, 1 for `plane`, `order` for
  `polynomial`. Stride `s = ceil(sqrt(H·W / MAX_FIT_POINTS))`, then per axis
  `s_a = min(s, max(1, (n_a − 1) // max(d, 1)))`, so that each axis keeps at least `d + 1`
  distinct coordinates (samples at `0, s_a, …, d·s_a ≤ n_a − 1`). Subsampled points carry their **full-grid** coordinates (§4.1). Below the cap,
  `s = 1`: every pixel is used, exactly as Gwyddion does.
- The surface is then evaluated on the full grid (§4.2). Measured cost of subsampling: RMS 7.4e-5
  from the full fit against a 2.9e-4 bound (R4).
- `offset` under `lstsq` is `numpy.mean` over every pixel; no subsampling.
- `line` is never subsampled. Under `robust`, process lines in blocks (e.g. 256 lines) so the
  per-iteration temporaries are bounded by the block, not the image — the project's
  memory-awareness rule. Batched per-block solves may use weighted normal equations
  (`numpy.linalg.solve` on stacked `(k+1)×(k+1)` systems): with the Legendre basis on `[−1, 1]`
  and `line_order ≤ 5` these are well conditioned.

### 4.9 Output

Compute in float64; cast the result back to `detect_mat`'s dtype; apply `self._apply_norm()`;
write `image.detect_mat[:]`. `rgb` and `gray` are untouched.

## 5. Implementation guidance

- **Files** (the `_monogenic_kernels.py` / `_focus_edge_monogenic_phase.py` split):
  - `src/phenotypic/enhance/_poly_surface_kernels.py` — private, pure float64 numpy functions
    implementing §4 on a bare array, with no `Image` and no `norm`. This is where the numeric
    contract lives, and what the golden fixture and the tight-tolerance tests exercise.
  - `src/phenotypic/enhance/_subtract_poly_surface.py` — the operation: fields, validation,
    `_operate` = read `detect_mat` as float64 → kernel → cast → `_apply_norm`. Bases
    `(NormalizedOutputMixin, BackgroundSubtraction)`, the order `ContrastLog` uses
    (`enhance/_contrast_log.py:16`), so `norm` is appended last.
- **`detect_mat` is float32** (measured 2026-10-08: `Image(arr=<float64>)` yields a float32
  `detect_mat`). Operation-level assertions therefore carry float32-derived tolerances
  (`≤ 1e-6` on `[0, 1]` data, ~8·eps32); float64-tight assertions target the kernel.
- **Helpers, not one long `_operate`:** coordinate/term-set construction, the lstsq solve with its
  rank check, the robust loop, surface evaluation, and per-method assembly are separate private
  functions with explicit names (project rule: no generic `run()`/`process()`).
- **Imports:** numpy and scipy are already runtime dependencies (`pyproject.toml`). Import
  `scipy.stats` inside the function that uses it (lazy-import rule, root `CLAUDE.md` *Gotchas*).
- **Docstring** (Google style, `abc_/CLAUDE.md` layout, microbiology context):
  - what each method fits; the §4.6 level table;
  - that defaults reproduce Gwyddion 2.71, with the citation (Nečas & Klapetek 2012,
    doi:10.2478/s11534-011-0096-2);
  - that `fit="lstsq"` (the Gwyddion default) is biased upward by colonies — measured 2.3–9.5 σ of
    background noise at 10–40% plate cover — and that `fit="robust"` is the recommended setting for
    plates;
  - the robust limits (drift D16): reliable to about 40% foreground for surfaces, and **per line**
    for `method="line"` — a scan line through a row of colony centres on an arrayed plate is mostly
    colony and cannot be leveled;
  - that `method="line"` is Align Rows' *Polynomial* method, not its default *Median*;
  - `Consider Also:` `SubtractGaussian`, `SubtractRollingBall`, `FlattenIllumination`.
  - A runnable doctest on `load_synth_yeast_plate()`.
- **Do not** read anything under the Gwyddion reference tree (§8).

## 6. Registration touchpoints

| File | Change |
|---|---|
| `src/phenotypic/sdk_/typing_.py` | the three `Literal` aliases (§3) |
| `src/phenotypic/enhance/__init__.py` | import + `__all__` (this also lists it in the GUI builder's enhancer dropdown — no GUI chrome change, no `FEATURES.md` row) |
| `tests/unit/abc_/test_enhancer_taxonomy.py` | add to the `BackgroundSubtraction` roster |
| `tests/unit/tune/test_enhance_annotations.py` | rows for `order`, `line_order`, `clip_sigma` windows and the `Field` bounds (keeps the annotation-coverage gate green) |
| `docs/source/explanation/what_enhancement_does.md` | one bullet beside `SubtractGaussian` / `SubtractRollingBall` |

## 7. Verification

### 7.1 Logic-validation scripts (written and passing with this spec)

`docs/superpowers/logic_validation_scripts/2026-10-08-subtract-poly-surface/` — stdlib + numpy +
scipy, never imports `phenotypic`, non-zero exit on failure. Run with any interpreter that has
numpy/scipy, e.g. `uv run python <script>`.

| Script | Claims | Result (2026-10-08) |
|---|---|---|
| `basis_equivalence.py` | B1 Legendre ≡ monomial LS surface, both term sets, orders 2–11; B2 monomial Gram cond 2.27e15 vs Legendre 4.7e2 at order 11 (tensor); B3 `leggrid2d` ≡ dense evaluation | 0 failures; worst B1 gap 4.4e-12 |
| `level_conventions.py` | L1 slope conversion; L2 plane mean shift `+½(bx+by)` (and a true-centre control); L3 pixel-monomial ≡ Legendre line residuals; L4 every line → input mean; L5 degree-0 shift form; L6 offset/polynomial → 0 | 0 failures |
| `robust_and_subsample.py` | R1 lstsq bias; R2 robust recovery at defaults (surfaces ≤ 40% cover; lines < 20% own fraction); R3 `max_iter=10` converged; R4 subsample bound; R5 `niter=3` not converged at 40%; R6/R7 the 50% breakdown, surface and per line | 0 failures |

Measured, `robust_and_subsample.py` (300×450, σ = 0.01, soft-edged colony domes, tensor order 3;
RMSE in units of σ):

| plate cover | seed | surface lstsq | robust ×3 | robust ×10 | worst sparse line, lstsq | worst sparse line, robust ×10 |
|---|---|---|---|---|---|---|
| 0.10 | 1 / 2 / 3 | 2.59 / 2.27 / 2.65 | 0.036 / 0.020 / 0.053 | 0.036 / 0.020 / 0.053 | 5.51 / 5.48 / 7.09 | 0.19 / 0.18 / 0.19 |
| 0.25 | 1 / 2 / 3 | 5.93 / 6.02 / 5.97 | 0.023 / 0.041 / 0.045 | 0.021 / 0.039 / 0.042 | 5.82 / 5.08 / 5.20 | 0.18 / 0.16 / 0.17 |
| 0.40 | 1 / 2 / 3 | 8.55 / 9.36 / 9.54 | **0.147 / 0.660 / 1.049** | 0.046 / 0.020 / 0.055 | 6.63 / 4.63 / 4.48 | 0.10 / 0.11 / 0.12 |
| 0.50 | 1 / 2 / 3 | 11.4 / 11.4 / 11.2 | 11.4 / 11.4 / 9.75 | **11.4 / 11.4** / 0.039 | — | — |

"Sparse line" = a line whose own colony fraction is < 20%. Lines ≥ 50% colony fail at up to 20.7 σ.

The `max_iter` decision (Q7) came from a sweep of `(outer rounds, inner clip passes)`; at 40% cover
over three seeds, the maximum error was 1.05 σ for `(3, 1)`, 0.12 σ for `(3, 5)`, 0.055 σ for
`(10, 1)` (converged by round 7), and the same as `(10, 1)` for `(50, 1)`.

### 7.2 Behavioural tests — `tests/unit/enhance/test_subtract_poly_surface.py`

1. **Exact recovery:** for every method under `lstsq`, a noise-free synthetic background of the
   method's own form is removed to `≤ 1e-10` by the float64 **kernel**, and to `≤ 1e-6` through
   the operation (float32 `detect_mat`, §5), and the output level matches §4.6.
2. **Level convention:** per method, the output mean equals §4.6's formula (plane: including the
   `+½(bx+by)` shift; line: every line's mean equals the input mean).
3. **`independent`:** a `u³v³` term is removed by `independent=True, order=3` and left (measurably)
   by `independent=False, order=3`.
4. **Line mode:** injected per-row offsets and slopes are removed; `line_axis="column"` on `z`
   equals `line_axis="row"` on `z.T`, transposed.
5. **Robust vs lstsq:** on a synthetic plate at 25% cover, `robust` beats `lstsq` by ≥ 10× and is
   within 0.5 σ of the true background (bounds from §7.1, not guessed).
6. **Robust limits:** a line ≥ 90% colony keeps a valid fit (drift D3 — no `+avg` shift, no
   exception).
7. **Subsampling:** with `MAX_FIT_POINTS` monkeypatched low, the subsampled surface is within the
   R4 bound of the full fit; below the cap the result is bit-identical to an unsubsampled fit.
8. **Contract:** `rgb`/`gray` unchanged; dtype preserved; `norm` = `clip`/`rescale`/`None` each
   behave; `to_json()`/`from_json()` round-trip; `model_json_schema()` field order ends in `norm`.
9. **Validation:** every §4.7 condition raises, asserted through the `__cause__` chain.
10. **Doctest:** the class docstring example runs.

### 7.3 Golden fixture from the Gwyddion oracle

- **Oracle task (separate agent, clean-room side):** build Gwyddion 2.71's `libgwyddion` +
  `libprocess` as a **Slurm job** (they need glib, not GTK), and a small C harness that calls, on
  fixed seeded arrays (including one non-square, one odd-sized, one with `W ≠ H` parity):
  `gwy_data_field_fit_plane` + the Plane Level constant + `gwy_data_field_plane_level`;
  `gwy_data_field_fit_poly` + `gwy_data_field_subtract_poly` for both term sets at orders 2, 3, 5;
  `gwy_data_field_row_level_poly` at degrees 0, 1, 3, both directions (via transpose);
  Zero Mean Value. It writes **numbers only** — inputs, every output array, the fitted coefficients
  — to `tests/fixtures/enhance/subtract_poly_surface_gwyddion.npz` with a provenance JSON beside it
  (Gwyddion version, tarball sha256, harness sha256, build flags). Numeric outputs of a program are
  not covered by its copyright; the harness source stays outside the repository with the other
  references.
- **Implementer side:** a test loads the fixture and requires the float64 **kernel**
  (`_poly_surface_kernels.flatten_surface`, `fit="lstsq"`) to reproduce every output to a
  tolerance derived from D1 (B1's bound for the order in question), not a guessed `rtol`. The
  operation is not compared against the fixture: its float32 `detect_mat` would swamp D1.
- **Fallback** if the build is infeasible on the cluster: the oracle agent writes the fixture by
  executing the §4 contract in an independent numpy re-derivation. That fixture pins the contract,
  not Gwyddion; the plan must say so wherever it is used.

### 7.4 Load-bearing proof and mutation matrix

- **Proof:** move the plane pivot to the true centre; the golden fixture (and test 2) must fail.
  Restore it from saved bytes, never `git checkout --`.
- **Mutants**, each a single change, each killed by at least one test:
  drop the line-mode `+avg`; compute `avg` after leveling instead of before; swap the
  independent/total term sets; normalize with `W` instead of `W−1`; skip the column transpose;
  one-sided clipping; let clipped points return; stop at `max_iter` without the convergence check;
  pivot at the true centre; subtract the plane's constant; omit the rank check.

## 8. Clean-room protocol

Mirrors `2026-07-13-fungi-detection-method-ports/refs/nfa/ATTRIBUTION.md`.

- **Oracle side** (this spec's author, the oracle agent, reviewers): may read the out-of-repo
  Gwyddion tree.
- **Implementer side:** receives `design.md`, `drift-register.md`, the logic-validation scripts,
  and the oracle's numeric fixture. **Must not open** `gwyddion-2.71/` or `gwyddion-user-guide/`
  in the reference tree, and the brief must say so. The astropy files (BSD-3-Clause) are
  permitted.
- An `ATTRIBUTION.md` beside the fixture records: the Gwyddion citation, that no Gwyddion source
  was copied or transcribed, and what the implementer was given.

## 9. Open items for the plan

- Confirm on the cluster that `libprocess` builds headless (glib only) inside a Slurm job; if not,
  take the §7.3 fallback and label the fixture accordingly.
- The Nečas & Klapetek PDF is not on disk (publisher bot challenge). Add it to the reference tree
  if a copy can be obtained; nothing in the contract depends on it.
