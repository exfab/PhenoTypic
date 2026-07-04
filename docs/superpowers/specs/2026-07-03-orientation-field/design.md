# Design: `MeasureOrientationZones`

- **Date:** 2026-07-03
- **Status:** Design — approved for planning
- **Branch / worktree:** `orientation-field`
- **Related:** the orientation-field mathematics (gradients → structure tensor →
  `{coherence, |∇φ|, entropy}` → `{R, turning}`) are derived in the companion reference
  `docs/superpowers/explain/2026-07-03-gradient-to-orientation-field-metrics.md` (on the
  brainstorm branch). This spec assumes that background and does not re-derive it.

---

## 1. Motivation & summary

A new Regime-B `MeasureFeatures` operator, **`MeasureOrientationZones`**, that quantifies
hyphal **concentration** (alignment `R`) and **turning** (coherence-weighted `⟨|∇φ|⟩`)
from the structure-tensor orientation field, reported **overall** and per **dense** /
**sparse** growth zone.

It reuses the zone segmentation already produced by `MeasureSymmetricZones` (core / dense
/ sparse concentric radii from the colony-ness profile) but computes an *orientation* read
of each zone rather than a radial-symmetry read. Crucially, because the detected mask is
imperfect (it misses sections and is only trustworthy for the **radial extent**), the
orientation field is computed over a **mask-free tile** and aggregated over **radially
defined** regions bounded by the symmetric radius — not over raw mask pixels. A raw-mask
variant is emitted alongside the radial one purely so the mask's distortion can be *seen*.

---

## 2. Output schema

New header enum **`RADIAL_ORIENT_ZONES`** in
`src/phenotypic/schema/_radial_orient_zones.py`, category prefix
**`RadialOrientZones`**, mirroring `SYMMETRIC_ZONES`. Per object:

**Regions** `{Overall, Dense, Sparse}` × **variants** `{Radial, Mask}` × **metrics**
`{Concentration, Turning, Coherence}`.

| Metric | Meaning | Units / range |
|---|---|---|
| `Concentration` | coherence-weighted resultant length `R` of the doubled-angle field | dimensionless [0,1] |
| `Turning` | coherence-weighted mean `⟨|∇φ|⟩` | rad/px (rad/µm with scale) |
| `Coherence` | mean coherence `⟨C⟩` over the selector — confidence/QC readout | dimensionless [0,1] |

Column headers (18 + label), pattern `RadialOrientZones_<value>-<ZoneName>` where
`<value> = <Metric>_<Variant>` and `<ZoneName> ∈ {Overall, Dense, Sparse}` (the zone comes
last, hyphen-separated):

```
Object_Label
RadialOrientZones_Concentration_Radial-Overall   RadialOrientZones_Concentration_Mask-Overall
RadialOrientZones_Concentration_Radial-Dense      RadialOrientZones_Concentration_Mask-Dense
RadialOrientZones_Concentration_Radial-Sparse     RadialOrientZones_Concentration_Mask-Sparse
RadialOrientZones_Turning_Radial-Overall          RadialOrientZones_Turning_Mask-Overall
RadialOrientZones_Turning_Radial-Dense            RadialOrientZones_Turning_Mask-Dense
RadialOrientZones_Turning_Radial-Sparse           RadialOrientZones_Turning_Mask-Sparse
RadialOrientZones_Coherence_Radial-Overall        RadialOrientZones_Coherence_Mask-Overall
RadialOrientZones_Coherence_Radial-Dense          RadialOrientZones_Coherence_Mask-Dense
RadialOrientZones_Coherence_Radial-Sparse         RadialOrientZones_Coherence_Mask-Sparse
```

**Regions are radial** (centred on the inoculum centre, using the zone radii from §3):
`Overall = disk(0 .. symmetric_radius)`, `Dense = ring(core_end .. dense_end)`,
`Sparse = ring(dense_end .. sparse_end)`. The *core* zone is intentionally omitted for now
(trivial to add later — see §11).

Schema authoring follows the project rule: author `label`/`desc` only; **leave
`bio_desc=""`/`image=None`** for human domain authoring. `tier = 2` (descriptive trait).

---

## 3. Architecture

### 3.1 Shared zone-segmentation helper (extraction + regression guard)

Extract the colony-ness → zone-radii pipeline and per-object geometry out of
`MeasureSymmetricZones._compute_intermediates` into a reusable module:

- **New:** `src/phenotypic/measure/_zone_segmentation.py`
  - `@dataclass ZoneSegmentation` carrying the common core: `label`, `bbox_slice`
    (expanded), `centroid_rc` (local), `centroid_global`, `dist_map` (expanded),
    `intensity_crop`, `obj_mask` (expanded), `core_radius`, `symmetric_radius`,
    `core_end_radius`, `dense_end_radius`, `sparse_end_radius`, `zones_computed`.
  - `compute_zone_segmentation(image, prop, *, params) -> ZoneSegmentation` runs the
    existing pipeline (PELT core → Sholl angular → symmetric radius → colony-ness →
    threshold-crossing radii → expanded crop) and returns the dataclass.
- **Refactor:** `MeasureSymmetricZones._compute_intermediates` becomes a thin wrapper that
  calls `compute_zone_segmentation` and then adds its diagnostic-only fields
  (density_profile, sholl_counts, angular_R_profile, per-angle envelope, zone areas). Its
  public columns and `inspect()` behaviour stay **byte-identical**.
- **Regression guard (mandatory):** a test asserts `MeasureSymmetricZones().measure(img)`
  produces an identical DataFrame before/after the extraction on
  `load_synth_yeast_plate()` (golden-value or self-consistency check).

**Constraint:** the extraction is a pure refactor — no behaviour change to the existing
operator. Do it first, land it green, then build the new op on top.

### 3.2 New operator

`class MeasureOrientationZones(MeasureFeatures, FigureProvider)` in
`src/phenotypic/measure/_measure_orientation_zones.py`, exported from
`measure/__init__.py`. Per object it: (1) obtains a `ZoneSegmentation`, (2) picks the
compute tile (§4.1), (3) computes the orientation field on that tile (§4.2), (4) builds
the radial/mask selectors and aggregates the coherence-weighted metrics (§4.3), (5)
caches per-object intermediates for `inspect()`/`dashboard()`.

---

## 4. Orientation-field computation

### 4.1 Compute region (mask-free)

The structure tensor is computed over a **mask-free tile**:

- **Preferred:** the **grid-section tile** — when `image` is a `GridImage`, resolve the
  grid section containing the object centroid (via `image.grid`) and use its slice.
- **Fallback:** the **expanded crop** already built by `ZoneSegmentation`
  (`r_max·(1+extent_margin)` around the centre), used for non-grid images.

Both are supersets of the `symmetric_radius` disk (the aggregation is bounded there), so
they agree within the relevant area; the grid section is preferred per the requirement to
"read the field over the grid section." The op computes its own `dist_map` and centre in
the chosen tile's frame so radial selectors are consistent.

### 4.2 Structure tensor → `φ`, `C`, `|∇φ|`

On the chosen intensity tile (`intensity_source`, default **`detect_mat`**):

```
Ix = gaussian_filter(I, sigma_d, order=(0,1)); Iy = gaussian_filter(I, sigma_d, order=(1,0))
Jxx,Jyy,Jxy = gaussian_filter({Ix², Iy², Ix·Iy}, sigma_i)
φ = ½·atan2(2Jxy, Jxx−Jyy)
C = √((Jyy−Jxx)² + 4Jxy²) / (Jxx+Jyy + eps)          # coherence ∈ [0,1]
|∇φ| = ½·√(|∇cos2φ|² + |∇sin2φ|²)                     # doubled-angle, π-safe
```

Reuse existing structure-tensor helpers where available (`util/image_metrics.py`,
`sdk_/branch_pathfinding/_cost_surface.py`) rather than reimplementing; factor a small
shared `orientation_field(I, sigma_d, sigma_i) -> (phi, coherence, grad_phi)` if none fits
cleanly. Params: `sigma_d` (default ~1.5, ≈ hypha width), `sigma_i` (default ~4.0).

### 4.3 Zone selectors & aggregation

For each region build a boolean selector on the tile from `dist_map`:

- **Radial** variant: `r_lo ≤ dist_map < r_hi` (the ring / disk), all tile pixels.
- **Mask** variant: that **∩ `obj_mask`** (the imperfect detected mask).

Then aggregate, coherence-weighted, over each selector `S`:

```
Concentration R = | Σ_{S} C·(cos2φ, sin2φ) | / Σ_{S} C
Turning         =   Σ_{S} C·|∇φ|            / Σ_{S} C
Coherence       =   mean_{S} C
```

---

## 5. Edge cases & NaN semantics

- Object area < 10 px → all metrics `NaN`.
- Zone with zero width (e.g. collapsed `symmetric_radius`, `dense_end == core_end`) or
  `Σ_S C < eps` → that region's metrics `NaN` (can't define orientation), distinct from a
  legitimate low value.
- Non-grid image → expanded-crop fallback (no error).
- `zones_computed == False` (symmetric envelope collapsed) → Dense/Sparse `NaN`; Overall
  may still be computable if a symmetric disk exists, else `NaN`.

---

## 6. Visualization

Two figure surfaces, following the codebase conventions (`@figure` decorator; `inspect()`
= the single *saveable* primary figure and CLI `--save-inspect` surface; `dashboard()` =
a richer composed notebook figure that is **not** the save surface).

### 6.1 `inspect()` — primary, saveable (ships **A + C**)

`@figure(primary=True, controls={"base_layer": BASE_LAYER})`,
`inspect(self, image=None, base_layer="detect_mat", *, for_save=False) -> go.Figure`.
Plate-level plotly overview (like `MeasureSymmetricZones.inspect()`), with
legend-toggleable layers:

- **A — coherence-modulated quiver:** one short segment per ~8–16 px block; angle = block
  `φ`, **length + opacity ∝ block-mean `C`**; a *single* NaN-separated `Scattergl` trace
  so it stays fast plate-wide.
- **Zone rings:** symmetric-radius circle + core/dense/sparse boundary circles per object
  (reuse the existing circle/annulus polygon helpers).
- **C — per-zone summary glyph:** the resultant `R` arrow per zone (direction = mean
  orientation, length ∝ `R`) plus a small text badge of `R`/turning for both variants.
- Centroids. `for_save=True` force-shows layers for static raster export.

### 6.2 `dashboard()` — composed notebook diagnostic (adds **B**)

`dashboard(self, image=None, show=True) -> go.Figure` returning one vertically-composed
`make_subplots` figure (the `AutoGridFinder.dashboard()` / `GridFitReport.dash()` pattern),
built by a transient `_OrientationZonesReport` `FigureProvider` whose control-free
`@figure` panels compose into:

1. the `inspect()` overview (A + C + rings),
2. **B — coherence heatmap:** semi-transparent `C(x)` image layer showing *where*
   orientation is well-defined,
3. a per-zone concentration/turning summary (bar or table) for both Radial & Mask
   variants,
4. *(optional)* an orientation rose (angular histogram of `φ`, coherence-weighted) per
   zone.

---

## 7. Parameters

| Param | Default | Meaning |
|---|---|---|
| `intensity_source` | `"detect_mat"` | image array for the structure tensor (`"gray"` alt) |
| `sigma_d` | `1.5` | Gaussian-derivative (gradient) scale ≈ hypha width |
| `sigma_i` | `4.0` | structure-tensor integration scale |
| `quiver_block` | `12` | inspect quiver downsample block (px) |
| *(zone params)* | — | `n_annuli`, `pelt_penalty`, `symmetry_threshold`, `tau_core/dense/sparse`, `extent_margin`, `min_samples_per_ring`, `method` — passed through to `compute_zone_segmentation`, same defaults as `MeasureSymmetricZones` |

Keyword-only pydantic fields; normalization/guards in `field_validator`s per project
conventions (`adding-an-operation`).

---

## 8. Testing

- **Regression (blocking):** `MeasureSymmetricZones` output unchanged after the §3.1
  extraction, on `load_synth_yeast_plate()`.
- **Analytic phantoms** for the orientation field (synthetic intensity tiles with known
  answers):
  - parallel bundle → `R → 1`, turning `→ 0`;
  - smoothly fanning/rotating field → turning high, `R` moderate;
  - isotropic random → `R → 0`, coherence `→ 0`, metrics `NaN` if `ΣC ≈ 0`.
- **Zone-restriction correctness:** a phantom with different orientation in the dense vs
  sparse ring yields the expected per-zone values.
- **Invariances:** rotating the phantom leaves `R` magnitude & turning invariant while the
  resultant *direction* rotates; scale-covariance of turning.
- **Radial vs Mask:** agree when the ring is mask-filled; diverge (documented) when mask
  holes are injected.
- **Doctest** on `load_synth_yeast_plate()` per project rule; `inspect()`/`dashboard()`
  smoke tests (figure builds, no exceptions).

---

## 9. File inventory

**New**
- `src/phenotypic/schema/_radial_orient_zones.py` — `RADIAL_ORIENT_ZONES` enum.
- `src/phenotypic/measure/_zone_segmentation.py` — `ZoneSegmentation` +
  `compute_zone_segmentation`.
- `src/phenotypic/measure/_measure_orientation_zones.py` — the operator + `inspect()` +
  `dashboard()` + `_OrientationZonesReport`.
- `tests/unit/measure/test_measure_orientation_zones.py`,
  `tests/unit/measure/test_zone_segmentation_regression.py`.

**Changed**
- `src/phenotypic/measure/_measure_symmetric_zones.py` — refactor
  `_compute_intermediates` to consume `compute_zone_segmentation` (no behaviour change).
- `src/phenotypic/measure/__init__.py`, `src/phenotypic/schema/__init__.py` — exports.
- Possibly `util/image_metrics.py` — a shared `orientation_field()` helper if none fits.

---

## 10. Open questions / deferred

- **Core zone:** omitted per scope; adding a `*_Core_*` triple is mechanical once the
  dense/sparse pattern lands.
- **Entropy:** `Field_OrientEntropy` per zone is a natural third metric; deferred (keep the
  first cut to concentration + turning + coherence).
- **`R`/turning unification:** the single-scalar `{R, coherence-weighted ⟨|∇φ|⟩}` framing
  is already the output; no further reduction planned.
- **Per-object zoom in `inspect()`:** if the plate-wide quiver is heavy, add a per-object
  zoomed panel later.
- **Prune variants:** once the Radial-vs-Mask comparison has served its purpose, drop the
  Mask variant (mask is unreliable) to halve the column count.
