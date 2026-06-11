# Robust colorimetric measures for `MeasureColor` — Design Spec

**Date:** 2026-06-10
**Branch:** `worktree-robust-lab-color-measures`
**Status:** Design approved (pending written-spec review)

## 1. Motivation

`MeasureColor` currently emits a "kitchen-sink" suite: for **every** channel of
**four** color spaces (CIE XYZ, xy chromaticity, CIE L\*a\*b\*, HSV) it computes
eight statistics (Min, Q1, Mean, Median, Q3, Max, StdDev, CoeffVar), plus two
`ChromaEstimated` columns — ~90 columns, most of which are redundant,
device-dependent, or not meaningful for colorimetric comparison.

Two problems motivate a redesign:

1. **No robust, outlier-resistant color summary.** Per-channel arithmetic means
   are pulled by specular highlights, agar bleed-through at colony edges, and
   debris. There is no single robust "center color" and no perceptually
   meaningful within-colony consistency measure.
2. **Output overload.** The 8-stats-×-many-channels matrix buries the few
   columns that matter for colorimetric phenotyping and inflates every
   downstream artifact (per-image parquet, master, analysis, dashboard).

This redesign replaces the per-channel suites for the **two colorimetrically
useful spaces** (L\*a\*b\* and HSV) with compact, robust, multivariate summaries,
and demotes the non-colorimetric spaces (XYZ, xy) to opt-in/hidden.

## 2. Goals / Non-goals

**Goals**
- A robust, multivariate **center color** in CIE L\*a\*b\* (perceptually uniform).
- A perceptually meaningful **within-colony consistency** scalar set (ΔE2000).
- A robust HSV summary that **correctly handles hue circularity**.
- A **hex swatch** column for plot visualization.
- **Trim** the output to the colorimetrically useful set; hide XYZ/xy.
- Auto-update the generated measurement **reference table** doc.

**Non-goals**
- Changing other measurers (`MeasureIntensity`, `MeasureTexture`, …).
- ΔE2000-based *location* optimization (medoid covers the ΔE2000 center cleanly;
  see §4.2).
- Backward-compatible column aliases — this is a hard cutover (goldens
  regenerated).

## 3. Background — math decisions (and why)

All per-object math operates on the object's pixel vectors extracted via
`array[image.objmap[:] == label]`.

### 3.1 Robust center in L\*a\*b\*

CIE L\*a\*b\* is designed so Euclidean distance ≈ perceived difference (ΔE76).
For `N` pixels `pᵢ = (L*ᵢ, a*ᵢ, b*ᵢ)`:

- **ΔE76 geometric median** `c_gm = argmin_{c∈ℝ³} Σᵢ‖pᵢ − c‖₂`. The Fermat–Weber
  point; the multivariate generalization of the median. Convex objective →
  unique minimum → solved by **Weiszfeld iteration**
  (`c⁽ᵗ⁺¹⁾ = Σ pᵢ/‖pᵢ−c⁽ᵗ⁾‖ ÷ Σ 1/‖pᵢ−c⁽ᵗ⁾‖`), an inverse-distance reweighting
  that crushes outlier weight. Breakdown point 0.5. Continuous/smooth, but can
  land at a color no pixel has.

- **ΔE2000 medoid** `c_med = argmin_{c∈{pᵢ}} Σᵢ ΔE₀₀(pᵢ, c)`. The argmin is over
  **actual pixels** and the metric is the canonical **ΔE2000**. A medoid works
  with any dissimilarity (no metric axioms needed) — essential because **ΔE2000
  is not a metric** (it violates the triangle inequality; Sharma et al., 2005),
  so there is no clean "ΔE2000 geometric median." The medoid is always a real
  colony color and O(N²) (subsampled to bound cost).

Both are reported (the user wants `GeoMedian` and `Medoid` columns), with the ΔE
form named in each description.

### 3.2 Within-colony consistency (anchored to the medoid)

Hard-wired to **ΔE2000**, measured from the **medoid** center to **all** pixels:

- `median ΔE2000 from medoid` — robust "perceptual MAD."
- `mean ΔE2000 from medoid` — equals the medoid's own minimized objective ÷ N
  (residual of the optimization that defined the medoid); the color-science
  uniformity standard.
- `P95 ΔE2000 from medoid` — near-worst-case deviation; flags
  sectoring/contamination (P95 chosen over raw max, which is a single-pixel
  artifact).

Plus one classical, Euclidean spread scalar:

- `LabTotalVariance = var(L*) + var(a*) + var(b*)` = trace of the 3×3 covariance
  about the arithmetic mean (= mean-squared ΔE76 spread). Inherently Euclidean;
  ΔE2000 does not apply to second moments.

### 3.3 Robust HSV — cone-Cartesian embedding

HSV **cannot** use the Lab approach directly: hue is **circular** (350° and 10°
are 20° apart) and HSV is **not perceptually uniform** (no ΔE). Naive
geometric-median/Euclidean-variance on raw H,S,V is mathematically wrong.

Embed each HSV pixel (`H∈[0,1]`, `S,V∈[0,1]`) into Cartesian **cone**
coordinates before estimation:

```
θ = 2π·H
x = S·V·cos(θ)
y = S·V·sin(θ)
z = V
```

Then:
- **Robust HSV center** = geometric median in `(x,y,z)`, converted **back** to
  `(H,S,V)` (`H = atan2(y,x)/2π mod 1`, chroma `r=√(x²+y²)`, `V=z`,
  `S = r/V` with `S=0` when `V=0`).
- **`HSVConeVariance`** = trace of the covariance of `(x,y,z)` (about the
  arithmetic mean in cone space) — a single 3D variance scalar, parallel to
  `LabTotalVariance`.

This handles circularity, and unreliable hue at low saturation/value collapses
toward the achromatic axis automatically (chroma `S·V → 0`).

### 3.4 Hex swatch (plot-only)

From the **ΔE2000 Lab medoid** (a real, perceptually-central colony color):
`Lab → XYZ → sRGB` (D65, via `colour`), clip to `[0,1]`, format `#RRGGBB`.
A real colony color is the most faithful swatch. **String column, strictly for
visualization** — must not enter numeric analysis (see §7 risk).

## 4. Output schema (final)

### 4.1 L\*a\*b\* (always on) — replaces the 8-stat L\*/a\*/b\* suite + ChromaEstimated

| Column label | Type | Meaning |
|---|---|---|
| `L*GeoMedian`, `a*GeoMedian`, `b*GeoMedian` | float | Coordinates of the **ΔE76 (Euclidean) geometric-median** center color. |
| `L*Medoid`, `a*Medoid`, `b*Medoid` | float | Coordinates of the **ΔE2000 medoid** center (real pixel minimizing total ΔE2000). |
| `DeltaE2000MedianFromMedoid` | float | Median ΔE2000 of pixels from the medoid (robust perceptual MAD). |
| `DeltaE2000MeanFromMedoid` | float | Mean ΔE2000 from the medoid (uniformity standard). |
| `DeltaE2000P95FromMedoid` | float | 95th-percentile ΔE2000 from the medoid (worst-case / sectoring flag). |
| `LabTotalVariance` | float | Trace of L\*a\*b\* covariance (sum of channel variances). |
| `MedoidColorHex` | str | sRGB hex of the ΔE2000 medoid color, **for plot visualization only**. |

### 4.2 HSV (always on) — replaces the 8-stat Hue/Saturation/Brightness suite

| Column label | Type | Meaning |
|---|---|---|
| `HueRobustMean`, `SaturationRobustMean`, `ValueRobustMean` | float | Cone-embedded geometric-median center, converted back to H,S,V. |
| `HSVConeVariance` | float | Trace of cone-Cartesian covariance (3D HSV spread). |

### 4.3 CIE XYZ & xy chromaticity — opt-in, hidden from the reference doc

- Existing 8-stat suites retained **unchanged** behind default-off flags
  (`include_XYZ: bool = False` (existing), `include_xy: bool = False` (new —
  `xy` is currently always-on)).
- **Removed from the generated reference table** regardless of flags (they are
  not part of the default colorimetric output).
- Rationale: preserve a power-user/legacy escape hatch at near-zero cost without
  redesigning de-emphasized spaces.

## 5. Parameters (`MeasureColor` fields)

| Field | Default | Purpose |
|---|---|---|
| `include_XYZ: bool` | `False` | (existing) opt-in legacy XYZ 8-stat suite. |
| `include_xy: bool` | `False` | (new) opt-in legacy xy 8-stat suite. |
| `geomedian_max_iter: int` | `50` | Weiszfeld iteration cap. |
| `geomedian_tol: float` | `1e-4` | Weiszfeld convergence tolerance (center movement). |
| `medoid_max_pixels: int` | `1000` | Subsample cap for the O(N²) medoid selection (spread still uses all pixels). |
| `random_seed: int` | `0` | Fixed seed for reproducible medoid subsampling. |

ΔE2000 is hard-wired (no metric toggle), via `colour.difference.delta_E_CIE2000`
(matches existing repo usage at
`correction/_color_correction/_helpers.py:585`).

## 6. Implementation outline

- **New pure-function helper module** `util/_robust_color_stats.py`
  (testable in isolation): `robust_color_center(pixels, max_iter, tol)`,
  `medoid_ciede2000(pixels, max_pixels, seed) -> (center, all_pixel_deltas)`,
  `delta_e2000_spread(deltas) -> (median, mean, p95)`, `hsv_to_cone(hsv)`,
  `cone_to_hsv(xyz)`, `lab_to_srgb_hex(lab)`. Exported from `phenotypic.util`.
  - `robust_color_center`: **reuses the existing, verified
    `phenotypic.util.geometric_median`** (Weiszfeld path) — pinned to
    `method='weiszfeld'` because the default `method='cohen'` is unimplemented
    (raises) — and adds empty/`n==1` guards. No new Weiszfeld code.
  - `medoid_ciede2000`: subsample to `max_pixels` (seeded) for the pairwise
    selection; compute final spread distances from the chosen medoid to **all**
    pixels (O(N)).
- **`MeasureColor._operate`**: single pass over `np.unique(objmap)` labels,
  extracting `Lab[mask]` and `hsv[mask]` per object; assemble the robust block
  into the results frame; keep the (now opt-in) XYZ/xy 8-stat paths behind their
  flags. Remove the L\*/a\*/b\* and Hue/Sat/Val 8-stat paths and the
  `ChromaEstimated` post-computation.
- **Schema (`phenotypic.schema`)**: add the new `ColorLab`/`ColorHSV` members and
  their header helpers; remove the per-channel-suite members from `ColorLab`
  and `ColorHSV` that are no longer emitted (keep `ColorXYZ`/`Colorxy` intact).
  New members auto-flow into `get_headers()`/`get_labels()`.

## 7. Risks & mitigations

- **String `MedoidColorHex` in a numeric measurement frame.** Aggregation (mean
  across replicates), master merge, post pipeline, analysis chain, and dashboards
  may assume numeric columns. **Mitigation:** audit those paths; ensure numeric
  reductions use `numeric_only=True` / `select_dtypes`, and that the hex column
  passes through aggregation/serialization untouched (it is viz-only, never
  analyzed). Add explicit tests.
- **Migration goldens are bit-exact** and the column set changes fundamentally.
  **Mitigation:** regenerate `tests/migration/_goldens/measure.MeasureColor*.parquet`
  via `scripts/capture_migration_goldens.py`; update the `with_xyz` scenario
  (and add an `with_xy` case if needed) in `tests/migration/_scenarios.py`.
  Note: fresh-worktree float drift can fail unrelated goldens — verify scope.
- **Medoid subsample cap is a silent accuracy bound** for colonies > 1000 px
  (selection only; spread uses all pixels). Acceptable speed/accuracy trade;
  documented in the field description.
- **Downstream references to removed columns.** Known touch points to update:
  `docs/source/_extensions/measurements_ref.py` (drop XYZ/xy from the
  MeasureColor doc entry), `src/phenotypic/_cli/_cli_readme_generator.py`
  (color schema mapping, lines ~196/213), `src/phenotypic/data/meas/all_meas.csv`
  (sample fixture used by ICC/serialization tests — regenerate),
  `docs/source/explanation/measurement_metrics_biological_meaning.md` (prose),
  `tests/unit/cli/test_cli_output_manager.py`,
  `tests/unit/util/test_measurement_outputs.py`.

## 8. Testing strategy

- **Unit (pure helpers):** geometric median vs hand-computed cases & known
  symmetric configs; outlier resistance; Weiszfeld zero-distance guard; medoid
  correctness on small sets; ΔE2000 spread values; cone round-trip
  `hsv → cone → hsv`; hex formatting; reproducibility under fixed seed;
  degenerate objects (single pixel, all-identical → spread/variance 0); empty
  object → NaN.
- **Schema:** new members appear in `get_headers()`; removed members absent;
  doctest on `load_synth_yeast_plate()`.
- **Integration:** pipeline serialization round-trip (`from_json` — all params
  defaulted); CLI output manager / measurement outputs with the new columns;
  hex column survives aggregation without breaking numeric reductions.
- **Migration:** regenerated goldens pass equivalence.
- **Quality gates:** `uv run mypy src/phenotypic`, `uv run ruff check --fix`.

## 9. Open items for written-spec review

1. **Column naming** — `HueRobustMean` vs `HueGeoMedian`; `MedoidColorHex` vs
   `PlotColorHex`; `HSVConeVariance` vs `HSVTotalVariance`. (Easy to adjust.)
2. **XYZ/xy when opted-in** stay as the classic 8-stat suite (not robustified) —
   confirm that asymmetry is acceptable for legacy spaces.
3. **Hex provenance** — convert the Lab medoid value via `colour`, or read the
   medoid pixel's original RGB directly (truer, needs pixel-coordinate tracking).
