# CenteredAutoGridFinder — Design

**Date:** 2026-06-11
**Status:** Draft v2 — adversarial review incorporated (2026-06-11); pre-implementation
**Worktree/branch:** `feature-new-grid-finder` / `worktree-feature-new-grid-finder`
**Author:** brainstorming session (Alexander Nguyen + Claude)

---

## 1. Motivation

The existing `AutoGridFinder` fits a regular grid by estimating pitch from the
**span** of detected object centers: `pitch = (c_max - c_min) / (n_expected - 1)`.
This is accurate when the outermost cells are occupied (dense and moderately
sparse plates) but degrades on **truly sparse plates**, because:

1. **Span-based pitch needs the edge cells occupied.** If an entire outer row or
   column is colony-free, `c_max - c_min` underestimates the true grid extent and
   the pitch is wrong.
2. **Image-derived *minimum* pitch is unreliable.** Prior iterations that fell
   back to `image_dim / n_expected` failed whenever the plate did not fill the
   frame edge-to-edge (margins, letterboxing, asymmetric crop).

We want a finder whose pitch comes from the **periodicity** of the detected
objects (which survives missing colonies) rather than their **extent** (which does
not), anchored by the physical fact that the plate sits roughly centered in the
frame.

### Goals

- Robust grid placement on sparse plates, down to a floor of **2 colonies**.
- Object-driven pitch; image dimension may bound the **maximum** pitch but never
  the minimum.
- Deterministic, closed-form refinement — no 2-D/3-D brute-force search.
- Drop-in `GridFinder` that satisfies the existing axis-aligned edge contract.

### Non-goals

- Rotation handling inside the finder (delegated upstream to `GridAligner`).
- Non-square (anisotropic) pitch.
- Collision resolution (delegated to existing `refine/` operations).
- GUI builder registration and a bespoke diagnostic dashboard (later, if adopted).

---

## 2. Constraints & assumptions

| Assumption | Source | Consequence |
|---|---|---|
| Image center ≈ grid center | user constraint | center seeded at image center; search **widened to the full in-frame offset range** (§4) so off-center plates are recoverable |
| Square cells (equal sides) | user constraint | a single isotropic pitch `p` (not `p_x`, `p_y`) |
| Known `nrows = R`, `ncols = C`, even spacing | plate format | grid fully determined by `(p, cx, cy)` |
| Plate de-rotated before finder | `GridAligner` upstream | finder is axis-aligned and **separable** in x/y |
| Detections reasonably clean | upstream detect/merge | finder tolerates a few outliers via robust trim, not pervasive noise |
| `nrows`/`ncols` match the physical plate | caller responsibility | **not guarded internally** — wrong `R`/`C` silently produces a wrong grid (see §9.6). Per the minimal-surface decision, this is documented, not detected. |
| Plate center reachable within the search box | widened center search (§4) | the search box now spans the full in-frame offset, so off-center plates are handled; a plate whose true center is outside the frame is out of scope. |

**Explicit risk (recorded, not solved here):** because rotation is handled
upstream, `GridAligner` must itself be reliable on sparse plates — it also fits
from colony centroids and could mis-rotate a sparse plate, which this finder would
then inherit as an axis misalignment. See §9.

---

## 3. Positioning

- New class **`CenteredAutoGridFinder`** in
  `src/phenotypic/grid/_centered_auto_grid_finder.py`, subclassing `GridFinder`.
- Exported from `src/phenotypic/grid/__init__.py` (public API) so the GUI/`from_json`
  can discover it.
- **New default GridFinder for grid images** (user decision, 2026-06-11). Two wiring
  sites flip from `AutoGridFinder` to `CenteredAutoGridFinder`:
  - `_core/_image_parts/_grid_image_handler.py` (~L97) — the `grid_finder is None`
    default in `GridImageHandler.__init__`.
  - `_core/_pipeline_parts/_image_pipeline_core.py` (~L1142) — the finder injected
    into a pipeline when a `GridImage` needs one.
- `AutoGridFinder` is **retained** (not deleted) and stays importable/serializable —
  it is simply no longer the default. Existing saved images/pipelines that name
  `AutoGridFinder` continue to deserialize unchanged.
- **Blast radius (must be handled in the plan):** flipping the default changes
  behavior for every `GridImage` constructed without an explicit finder. Expect
  fallout in (a) tests asserting the default finder *type*, (b) any grid-output
  golden/snapshot that used `AutoGridFinder`'s placement, (c) serialization
  round-trip tests that assumed the default class name. The dense-plate regression
  test (§10) is now a **gate**, not a nicety: as the default, `CenteredAutoGridFinder`
  must not regress the dense plates `AutoGridFinder` already handles.
- Name rationale: `CenteredAutoGridFinder` advertises the defining assumption — the
  grid is anchored at the image center — and mirrors the existing `AutoGridFinder`
  naming so its role as the automatic default is obvious.

---

## 4. Parameter model

The grid is a 3-parameter model after de-rotation:

- `p` — single isotropic pitch (pixels between adjacent cell centers)
- `(cx, cy)` — grid center in pixel coordinates

Cell-center positions:

```
col j center:  cx + (j - (C-1)/2) * p     for j = 0 .. C-1
row i center:  cy + (i - (R-1)/2) * p     for i = 0 .. R-1
```

### Search bounds

```
x_span = pct(x, 95) - pct(x, 5)                       # robust span (percentile, not min/max)
y_span = pct(y, 95) - pct(y, 5)
p_min  = max( x_span / (C-1),  y_span / (R-1) )       # object-derived FLOOR
p_max  = min( H / (R-1),  W / (C-1) )                 # image-derived CEILING (outermost cell CENTERS fit frame)
grid_extent_x = (C-1) * p ;  grid_extent_y = (R-1) * p
center ∈ image_center ± ( (image_extent - grid_extent)/2 + p )   # full in-frame offset, per axis
```

- The floor is sound because occupied cells span `≤ (C-1)·p`, so `p ≥ x_span/(C-1)`
  (tight on dense, loose-but-valid on sparse) — the old span formula reused as a
  **lower bound**, never a point estimate. Using a **5th–95th percentile** span
  (not raw min/max) stops a single spurious edge detection from inflating the span
  and inverting the bounds (challenger #4).
- The ceiling is the pitch at which the **outermost cell centers** reach the frame
  edge (`(R-1)·p = H`), i.e. `min(H/(R-1), W/(C-1))` — *not* the half-cell-padded
  `min(H/R, W/C)`. The latter understates the true ceiling: a frame-filling plate
  has true pitch up to `H/(R-1) > min(H/R, W/C)`, so the naive ceiling silently caps
  the pitch search **below** the truth (challenger #1, Critical). **Confirmed on
  real data** (§13): `SaltTolerantSparsePlate` has true pitch 404 px while
  `min(H/R, W/C) = 394 px < 404`; the centers-fit ceiling `min(H/(R-1), W/(C-1)) =
  450 px` contains it with headroom and needs no arbitrary margin.
- **Center search width** spans the full range the plate could be offset while
  remaining in frame: `± ((image_extent − grid_extent)/2 + p)` per axis. This
  covers hand-loaded / asymmetrically-cropped / sub-frame plates whose true center
  is more than one pitch from the image center (challenger #5). The image center is
  used only to *order* candidates, never to clamp them away.
- `min_gap` (smallest inter-colony spacing) is deliberately **not** used as a
  bound — a single over-segmented or doublet colony would collapse it.
- If `p_min ≥ p_max` even *after* the margin, treat it as a genuine contradiction
  (bad detection outliers or wrong `R/C`) → fallback ladder §6.

---

## 5. Algorithm (Approach A: comb-response → ICP)

### Stage 0 — Extract centers

- Pull per-object centers from the objects table. **Fit** on
  `BBOX.DIST_WEIGHTED_CENTER_RR/CC` — matching `AutoGridFinder`'s choice.
- **Center-column upgrade (in scope, folded in 2026-06-11).** That column is
  currently computed as `ndi.maximum_position(dt)` — the single deepest DT pixel
  (argmax), which makes its name a misnomer. On a **budding/dumbbell yeast colony**
  the DT has two peaks and argmax snaps the center onto *one lobe*. Fix: compute the
  true **DT-weighted centroid** `ndi.center_of_mass(dt, labels=objmap, index=labels)`
  instead. One statistic, robust to *both* failure modes: thin filament hyphae carry
  tiny DT weight so the center stays on the body (the original reason for using DT),
  and a doublet's two lobes balance to the neck ≈ the true pin position. No detection
  is needed — and note the tempting "DT-far-from-centroid ⇒ fall back" detector is
  *unusable* because filaments also have DT≠centroid, so distance cannot tell
  "trust-DT (filament)" from "trust-centroid (doublet)" apart; the weighted centroid
  sidesteps the discrimination entirely. Blast radius is narrow — only the grid
  finders + `grid/_grid_fit_report.py` read this column and no migration goldens
  reference it — so the change is **in-place** in `measure/_measure_bounds.py` (no
  new column, no schema surface). `AutoGridFinder` inherits the improvement.
  Implemented as plan **Task 0**.
- **Assign** (final output) via the standard `GridFinder` helper, which bins the
  geometric `BBOX.CENTER_RR/CC` through `pd.cut`. (Fit-center vs assign-center
  divergence is intentional and matches existing behavior.)

### Stage 1 — Bounds

Compute `p_min`, `p_max` per §4. If `p_min ≥ p_max` (contradiction — frame too
cropped for the object span), see fallback ladder §6.

### Stage 2 — Pitch via comb-response (circular order parameter)

For a candidate period `p`, form the pooled response over both axes:

```
z_x(p) = Σ_k exp( i·2π·x_k / p )
z_y(p) = Σ_k exp( i·2π·y_k / p )
R(p)   = |z_x(p)| / N  +  |z_y(p)| / N        ∈ [0, 2]
```

- `|z(p)|/N ∈ [0,1]` is the mean resultant length (Kuramoto order parameter): 1 ⟺
  colonies perfectly periodic at `p`; ~0 ⟺ residuals `mod p` uniform.
- Robust to missing colonies: removing a colony drops one term; it never shifts the
  phase or destroys the peak. (Contrast span, which needs edge cells.)

**Fundamental selection (octave safety).** `R(p)` also peaks at subharmonics
`p*/2, p*/3, …`. Scan `p` over `[p_min, p_max]` (linear sampling, `n_pitch_samples`
points); identify **strict local maxima** (a sample exceeding both neighbours),
keep those with `R ≥ response_floor · R_peak`, and choose the **largest-`p`**
survivor (the coarsest strong period = the fundamental). Golden-section refine
around it for sub-sample precision. The challenger confirmed this rule survives
every-other-column, every-other-row, and half-plate layouts *as long as the true
pitch is inside `[p_min, p_max]`* — which the §4 bound fixes now guarantee. If no
strict interior maximum clears the floor (flat/degenerate response), see §6.

### Stage 3 — Center candidates from phase + prior

For the chosen `p*`, `φ_x = arg(z_x(p*))` locates the column lattice **modulo
`p*`**, and `φ_y` the row lattice. The remaining global placement is an **integer**
ambiguity: `cx = base_x + m_x·p*` for integer `m_x` (parity of `C` shifts `base` by
`p/2`; handled in implementation). Enumerate every `(m_x, m_y)` whose resulting
center lies inside the widened center box (§4) — typically 1–3 candidates per axis.
**Do not commit to one here.** All surviving candidates pass to Stage 4, which
selects by residual. The image-center prior only *orders* candidates (nearest
first), it never discards.

### Stage 4 — Multi-start ICP refine (closed-form, robust)

Run the refinement below from **each** Stage-3 candidate, then keep the candidate
with the lowest final mean residual. A wrong integer placement yields a
self-consistent but high-residual fit (~450× higher in the challenger's
experiment), so the correct registration wins cleanly. This multi-start is what
defeats the one-cell-shift trap: there is no reliable in-place "escape" once ICP
locks onto a shifted lattice, so we instead *start* from every plausible placement
and let the residual choose. Before residual selection, discard any refined
placement whose clipped row or column edges are non-finite or not strictly
increasing. This feasibility check removes whole-pitch registrations that would
collapse an outer cell at the frame boundary. Among residual ties within the
sub-pixel tolerance, explicitly prefer the fitted center nearest the image center.

Each refinement iterates (`max_iter`, default ~5):

1. **Assign** each object to its nearest cell index:
   ```
   j_k = clip( round( (x_k - cx)/p + (C-1)/2 ), 0, C-1 )
   i_k = clip( round( (y_k - cy)/p + (R-1)/2 ), 0, R-1 )
   ```
2. **Singularity guard:** if the 3×3 design matrix is near-singular
   (`det(A) < ε` — e.g. all objects rounded into one cell so every `a_k` is
   equal), abandon this candidate (it cannot constrain `p`) → §6.
3. **Solve** the 3×3 normal equations for `(cx, cy, p)` with indices fixed
   (`a_k = j_k-(C-1)/2`, `b_k = i_k-(R-1)/2`):
   ```
   [ N      0      Σa_k          ] [cx]   [ Σx_k                 ]
   [ 0      N      Σb_k          ] [cy] = [ Σy_k                 ]
   [ Σa_k   Σb_k   Σ(a_k²+b_k²)  ] [p ]   [ Σ(a_k·x_k + b_k·y_k) ]
   ```
   The shared `p` (third row) is where the square-cell constraint couples x and y.
4. **Trim & re-solve (one pass):** compute residuals from the all-object solve,
   mark inliers (`|residual| ≤ residual_fraction · p`), and re-solve on inliers.
   The next iteration re-assigns *all* objects (trimmed ones get another chance).
   Same `residual_fraction` knob/semantics as `AutoGridFinder`.
5. Clamp `p` to `[p_min, p_max]` each iteration.

The loop **terminates in at most `max_iter` iterations**; it is **not** guaranteed
to reach the global optimum from a poor start (the three parameters are coupled, so
the Lloyd/k-means convergence proof does *not* apply) — which is precisely why we
multi-start and select by residual. If the best candidate's mean residual still
exceeds `residual_fraction · p`, escalate to the fallback ladder §6 (warn).

### Stage 5 — Emit edges

Cell centers → edges as the **midlines between adjacent centers**, with outer
edges extrapolated by `± p/2`, producing `R+1` row edges and `C+1` col edges,
clipped to image bounds. One outer edge may clip while leaving a valid partial
edge cell. Two or more edges clipping to the same boundary is invalid because it
creates a zero-width cell. The finder enforces finite, bounded, strictly increasing
edges before `_operate` calls `_get_grid_info(...)` for faithful, many-to-one
assignment (no collision handling). Column feasibility uses the assignment layer's
effective upper bound `W - 1`, so its final clipping step cannot collapse a
sub-pixel partial cell that looked valid against `W`.

---

## 6. Fallback ladder (the sparse tail)

Evaluated top-down; first matching rule applies. Every fallback emits a
`CenteredAutoGridFinderFallbackWarning` when `warn=True`.

| Condition | Behavior |
|---|---|
| `N == 0` | Uniform centered grid at `p = p_max` (largest fitting); no objects to assign. |
| `N == 1` | Uniform centered grid at `p = p_max`; the single object assigns to whatever cell it falls in. No pitch is inferable from one point. |
| `2 ≤ N ≤ min_fit_objects` (default 6) | Run comb-response + multi-start ICP, but the pitch is **bounded-ambiguous**: with few inter-colony vectors, `R(p)` peaks at `gap/k` for several integers `k`. Pick the fundamental within `[p_min, p_max]`; the Stage-4 multi-start then selects the placement with lowest residual (image-center prior only orders candidates). Document that this is a best-effort bounded choice, not a confident fit. |
| `p_min ≥ p_max` | Contradiction (object span exceeds what the frame can hold at the expected count → likely a detection outlier or wrong `R/C`). Clamp: drop to image-fit pitch `p = p_max`, center at image center, warn loudly. |
| degenerate comb-response (`R_peak < absolute_floor`) | No periodicity detectable; fall back to uniform centered grid at the span-derived `p_min` (object floor), warn. |
| ICP failed (best multi-start residual `> residual_fraction·p`, **or** every candidate tripped the singularity guard) | Use the comb-response `p` (if it cleared the floor) with center = image center as a uniform grid; else uniform centered grid at `p_min`. Warn. |
| Every refined registration has invalid edge geometry | Reject the registrations and use the bounded comb-response `p` with center = image center. Warn with the `invalid-geometry` reason. |

The floor is **2 colonies** for any *inferred* pitch; `N ∈ {0,1}` only guarantees a
non-crashing centered default.

---

## 7. Output / contract

- Implements `get_row_edges(image) -> np.ndarray` (len `R+1`) and
  `get_col_edges(image) -> np.ndarray` (len `C+1`): midlines between fitted cell
  centers, clipped to image bounds via `_clip_row_edges`/`_clip_col_edges`.
- `_operate(image) -> pd.DataFrame` computes edges then delegates to
  `_get_grid_info(image, row_edges, col_edges)`.
- **Collisions:** faithful many-to-one, **no flag** — identical to the current
  contract. Multiple objects in one section all receive the same `ROW_NUM`,
  `COL_NUM`, `ROW_MAJOR_IDX`. Resolution is the caller's choice of `refine/` op
  (`KeepNearestCenter`, `KeepSectionLargest`, `MergeWithinSection`, …). The
  recommended companion refiner is documented in the class docstring; nothing is
  enforced.

---

## 8. Pydantic fields & TuneSpec annotations

Mirrors `AutoGridFinder` conventions (keyword-only pydantic fields, `Annotated[...,
TuneSpec(...)]`). Every numeric field on a `grid/` op is pulled into
`tests/unit/tune/test_annotation_coverage.py` and must be covered.

| Field | Type / annotation | Intent |
|---|---|---|
| `nrows` | `Annotated[int, TuneSpec(tunable=False)] = 8` | structural |
| `ncols` | `Annotated[int, TuneSpec(tunable=False)] = 12` | structural |
| `residual_fraction` | `Annotated[float, TuneSpec(0.1, 0.5)] = 0.25` | robust-trim threshold (fraction of pitch) |
| `n_pitch_samples` | `Annotated[int, TuneSpec(tunable=False)] = 512` | comb-scan resolution; algorithmic, not a plate knob |
| `response_floor` | `Annotated[float, TuneSpec(0.5, 0.95)] = 0.8` | relative peak threshold for fundamental selection |
| `max_iter` | `Annotated[int, TuneSpec(tunable=False)] = 5` | ICP iteration cap |
| `min_fit_objects` | `Annotated[int, TuneSpec(tunable=False)] = 6` | floor below which we treat the fit as bounded-ambiguous |
| `warn` | `bool = False` | emit `CenteredAutoGridFinderFallbackWarning` |

**`ClassVar` constants** (not tunable fields — internal algorithm thresholds, à la
`AutoGridFinder._SPAN_TOLERANCE`):

- `SPAN_PCT_LOW = 5`, `SPAN_PCT_HIGH = 95` — robust-span percentiles (§4).
- `ABSOLUTE_FLOOR` — minimum comb-response peak to treat the response as non-degenerate (§6).
- `DET_EPS` — singularity threshold for the ICP design matrix (§5 Stage 4).

(Exact *field* set may shrink after review — fewer knobs is better. Anything that is
an internal threshold rather than a plate-dependent knob should be a `ClassVar`, not
a `TuneSpec` field.)

---

## 9. Risks & open questions

Updated after the adversarial review (2026-06-11). Items 1–3 are **resolved in this
spec**; 4–6 are **accepted/documented**; 7–8 remain genuine residual risks.

1. **Dense-plate bound inversion (was Critical) — RESOLVED + FIELD-VALIDATED.**
   `p_max` is now the centers-fit ceiling `min(H/(R-1), W/(C-1))` and `p_min` uses a
   percentile span (§4), so the true pitch is never capped out of the search range.
   Confirmed on `SaltTolerantSparsePlate`, whose true pitch (404 px) exceeds the old
   naive ceiling (394 px) — see §13. Regression-tested by a dense-plate fixture (§10).
2. **ICP one-cell-shift trap (was High) — RESOLVED.** Replaced the (false)
   convergence guarantee with **multi-start over the integer center candidates +
   residual selection** (§5 Stage 4). The shifted local minima have ~450× the
   residual and lose. A post-fit residual check escalates to §6 if even the best
   loses.
3. **Off-center plate (was Medium) — RESOLVED.** Center search widened to the full
   in-frame offset (§4); the prior no longer clamps the truth away.
4. **2-colony ambiguity — ACCEPTED/DOCUMENTED.** Irreducible; the `k`-selection is a
   bounded guess. Documented as best-effort, never presented as confident.
5. **Wrong `nrows`/`ncols` — ACCEPTED/DOCUMENTED (per decision).** No internal
   guard; the class docstring states `R/C` must match the physical plate. A
   mismatch produces a wrong grid silently — caller's responsibility.
6. **Fit-center vs assign-center divergence — ACCEPTED.** Fitting on DT-weighted
   centers but binning on geometric centroids could assign a very elongated colony
   to an adjacent cell. Low risk on well-separated colonies; covered by a test
   fixture, no special handling.
7. **`GridAligner` on sparse plates — RESIDUAL RISK.** The rotation-upstream
   decision is only as good as `GridAligner`'s sparse robustness; it also fits from
   centroids and may degrade at 2–10 colonies. Validate; possible follow-up to
   harden it or add a residual-tilt sanity check. Out of scope for this finder.
8. **Clustered-sparse octave — RESIDUAL RISK (low).** When colonies occupy a small
   central region, `p*/2` can sit above `p_min`; fundamental-selection mitigates but
   does not fully eliminate. Covered by a dedicated octave test fixture (§10).

---

## 10. Testing plan

**Unit (synthetic lattice, no image needed):** generate a known `R×C` grid at
chosen `(p, cx, cy)`, sample a subset of cells, add Gaussian jitter, assert
recovered `p, cx, cy` within tolerance across:

- occupancy sweep: 100% → 5% of cells occupied;
- **dense frame-filling plate** (≥96 % fill, true pitch ≈ `H/(R-1)`): assert no
  bound inversion and pitch within tolerance (regression for challenger #1);
- empty outer rows/columns;
- clustered-corner occupancy (octave stress);
- **off-center plate** (true center offset > one pitch from image center):
  multi-start recovers the correct registration (regression for challenger #5);
- **boundary-collapse registration**: a lower-residual whole-pitch placement that
  clips two edges to one boundary is rejected in favor of the feasible placement;
- **single clipped outer edge**: remains valid and does not trigger recentering;
- **one-cell-shifted seed**: assert multi-start selects the correct placement, not
  the shifted local minimum (regression for challenger #2);
- 2-colony floor (assert bounded, non-crashing, within `[p_min,p_max]`);
- 0/1 colony (assert centered default, no exception);
- **all objects in one cell**: singularity guard fires → fallback grid, **no
  `LinAlgError`** (regression for challenger #8);
- octave test: assert the recovered pitch is the fundamental, not `p/2`;
- single-outlier span: one far spurious detection does **not** invert the bounds
  (regression for challenger #4);
- collision: two objects in one cell → both assigned same section, `(p,cx,cy)`
  unchanged vs the no-collision baseline.

**Integration:** decimate `load_synth_yeast_plate()` to N colonies; run
`OtsuDetector → CenteredAutoGridFinder`; assert sane edge count/spacing and that known
colonies map to expected cells.

**Determinism:** identical input → identical edges (fixed, no RNG).

**Doctest:** one runnable example on `load_synth_yeast_plate()` per the project
docstring convention.

---

## 11. Implementation sketch (file-level)

- `src/phenotypic/grid/_centered_auto_grid_finder.py` — `CenteredAutoGridFinder`,
  `CenteredAutoGridFinderFallbackWarning`, private helpers
  (`_compute_bounds`, `_comb_response`, `_estimate_pitch`,
  `_seed_center_candidates`, `_multi_start_refine` (wraps the closed-form
  `_icp_refine` + residual selection + singularity guard), `_centers_to_edges`).
- `src/phenotypic/grid/__init__.py` — export both new symbols.
- `src/phenotypic/_core/_image_parts/_grid_image_handler.py` — flip the
  `grid_finder is None` default to `CenteredAutoGridFinder`.
- `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py` — flip the injected
  default finder to `CenteredAutoGridFinder`.
- `tests/unit/grid/test_centered_auto_grid_finder.py` — the suite in §10.
- **Test fallout sweep:** update any test asserting the default finder type, any
  grid golden/snapshot, and serialization round-trips that assumed `AutoGridFinder`
  as default (find via `grep -rn "AutoGridFinder" tests` + a full `pytest` run).
- Docstring follows the `ImageOperation`-subclass pattern (abc_/CLAUDE.md):
  summary → Args → Returns → Raises → detail → two doctests.

---

## 12. Out of scope (explicit)

Rotation in-finder; non-square pitch; collision resolution; GUI builder
registration; bespoke dashboard (the existing `AutoGridFinder.dashboard()` is not
reused — a viz pass is a separate follow-up if `CenteredAutoGridFinder` is adopted).

---

## 13. Empirical validation (real plate, 2026-06-11)

The core algorithm was prototyped and run on
`src/phenotypic/data/snp-imager-samples/SaltTolerantSparsePlate.png` (6016×4012),
detected with the user's pipeline (crop → blur → SubtractGaussian → Otsu →
RemoveLowCircularity → RemoveBorderObjects; the `RemoveByFeature` eccentricity
filter was corrected to `max_value=0.75`, i.e. drop elongated debris — the supplied
`min_value=0.75` inverted the test and deleted 17 of 18 round colonies). Cropped
frame `H×W = 3152×5066`; **18 colonies** on an `8×12` grid occupying rows
`{1,2,3,5,6}` (empty top, bottom, and one interior row) — a genuine empty-edge/
interior-row sparse case.

**Result (3-parameter fit, R=8, C=12):**

| Quantity | Value |
|---|---|
| Comb-response pitch | **403.8 px**, picked as the fundamental over the 202/135/101 px subharmonics, in **both** axes (square confirmed) |
| Fitted center | (2545, 1575) px ≈ image center (2533, 1576) — center-at-image-center assumption holds |
| Mean residual | **12.1 px = 3.0 % of pitch**; max 34 px |
| Assignment | **18 colonies → 18 distinct cells, no collisions**; cells match the visual layout exactly |

**Design claims confirmed:** (a) the comb-response recovers the true pitch from 18
sparse colonies with whole empty rows, where span-based pitch would fail; (b) the
"largest-`p` above floor" rule selects the fundamental, not an octave; (c) the
phase-seeded multi-start ICP locks the center at the image center and converges to
3 % residual; (d) **challenger #1 is real** — true pitch 404 px > naive ceiling
`min(H/R,W/C)=394 px`, which would have capped the search below the truth; the
centers-fit ceiling `min(H/(R-1),W/(C-1))=450 px` (now adopted, §4) contains it.

Prototype scripts archived under `/tmp/grid_exp/` (not committed): `fit_probe.py`
(comb scan), `full_fit.py` (bounds → pitch → phase → multi-start ICP → overlay).
