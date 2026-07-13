# `LightDetectFungi` — design

Status: proposal (not committed). A **lightweight** filamentous-fungi detector: composite the
branch enhancement and the center-fill enhancement into one response map, then segment it with
`TriangleDetector`. **No Dijkstra reconnection, no Voronoi labelling, no cost surface, no
tiling** — those stay in `FilamentousFungiDetector`. This is the fast path for "where is the
fungus" when per-colony reconnection is not needed.

Grounded in a live run on the production crop (`NeurosporaPipeV10.json`, `650/650/600/600`).

## 1 · Motivation

`FilamentousFungiDetector` does an enormous amount (inoculum detect → dual-mask branch detect →
grid Voronoi → Dijkstra fragment reconnection → final Voronoi), and takes ~minutes. Much of the
value — a filled-center fungal mask — comes from just two ideas developed this session:

- the **branch enhancement** recipe (`FlattenIllumination → ContrastStretching(70,99) →
  FocusEdgePhase`, i.e. the `FocusBranches` spec), and
- the **center-fill** (a `ManualGridPointDetector` at known wells ∩ a background subtraction),
  which fills the hollow inoculum core that edge-based phase congruency leaves — **without**
  `fill_holes` (which would also fill the legitimate gaps between hyphae).

`LightDetectFungi` composites those two and thresholds. It is a `ObjectDetector` (not a
`GridObjectDetector`) because it does no per-colony labelling.

## 2 · The scale problem (the crux — verified)

The two enhancements must be composited (pixel-wise `max`) into one map. **They are not on the
same energy scale out of the box.** Measured on the production crop:

| Map | min | max | p99 | mean |
|---|---|---|---|---|
| `FocusEdgePhase` branch (`pc_sum`) | 0.000 | 0.546 | **0.148** | 0.010 |
| `SubtractGaussian` center body | 0.000 | 0.397 | 0.142 | 0.013 |

**The PCT output *is* clipped to `[0, 1]` but does not fill it** — `pc_sum` is `Σ/n_orient`
then `np.clip(·, 0, 1)` (`_focus_edge_phase.py`), and on real plates it lives near the bottom of
the range (p99 ≈ 0.15, mean ≈ 0.01). `FocusEdgePhase` is **not** a `NormalizedOutputMixin`, so it
never rescales. That the two happen to share a scale here (both p99 ≈ 0.14) is **coincidental**
and must not be relied on: a center-fill that filled `[0, 1]` (e.g. a binary stamp, or any
`norm="rescale"` enhancer) would swamp the branches under `max`.

**Resolution — normalise each enhancement to `[0, 1]` before compositing.** Rescale each map by
its own robust range (`rescale_intensity(m, in_range=(0, p99.5(m)))`). Verified: after this,
branch p99 → 0.82, gated-center p99 → 1.0, composite p99 → 0.97, and both terms contribute to the
`max`. `CompositeEnhance` will **not** do this for you — its `norm` field is an *output* clip
policy, not per-child normalisation — so the normalisation is `LightDetectFungi`'s job (or a
range-filling step must be added to each child enhancer).

## 3 · Architecture (`_operate`)

```
branch_map = normalize( branch_enhancer(image).detect_mat )          # FocusBranches-style PCT, -> [0,1]
body_map   = normalize( background_subtractor(image).detect_mat )    # SubtractGaussian body, -> [0,1]
grid_mask  = center_detector(image).objmask                          # ManualGridPointDetector at wells
center_map = body_map * grid_mask                                    # center-fill, gated to known wells
composite  = maximum(branch_map, center_map)                         # balanced energies
image.detect_mat[:] = composite
TriangleDetector(...).apply(image, inplace=True)                     # segmentation backend
```

- **`normalize`** = `rescale_intensity(m, in_range=(0, percentile(m, 99.5)))` — the same-scale
  guarantee. A single private helper, reused for both maps.
- **Center gating.** `body_map` alone responds to cores *and* branches *and* any bright agar; the
  `grid_mask` (from the manual grid at exact coordinates) confines the *fill* to the known wells,
  which also drops the plate rim (the stamps never cover the wall). Verified: the gated center
  map is 60 clean discs on this plate. The grid detector is swappable (`InoculumDetector`, blob,
  etc.) for plates without known coordinates.
- **Composite** is `max` (union of evidence), matching how `FilamentousFungiDetector` combines its
  own two masks (`_combine_bg_removed_with_pct`).
- **Segmentation** is `TriangleDetector` on the composite, per request.

## 4 · What `TriangleDetector` gives you, and the trade-off

Triangle is a **permissive** single threshold (designed to catch faint objects). On the
normalised composite it fills the centers and captures branches, but it **also picks up agar
micro-texture** — the normalisation lifts the low-level background, and faint hyphae and faint
agar texture sit at the same level (the same tension the whole session has circled). Measured:
`fg ≈ 0.33`, **~46 000 connected components** (mostly tiny speckle + one fragment per branch tip,
since nothing reconnects them). This is the accepted cost of "light":

- `objmask` is a good filled-fungus mask; `objmap` is **fragmented** (no colony-level identity).
  If per-colony labels are needed, that is what `FilamentousFungiDetector` is for.
- Mitigations to document (not defaults): a **minimum-object-size filter** (a `refine` op) drops
  the speckle cheaply; a larger `min_wavelength` in the branch enhancer suppresses more agar; or
  swap `TriangleDetector` → `HysteresisDetector(low="triangle", high="otsu")` (what the heavy
  detector uses) for a cleaner double-threshold at the cost of the "single backend" simplicity.

## 5 · Fields / API

```python
class LightDetectFungi(ObjectDetector):
    branch_enhancer:        OperationField = FocusBranches(k=6)          # default: flatten→70-99→PCT(wl5, k6)
    center_detector:        OperationField = ManualGridPointDetector(...)  # or InoculumDetector
    background_subtractor:  OperationField = SubtractGaussian(sigma=300, n_iter=2)
    segmenter:              OperationField = TriangleDetector()
    normalize_percentile:   float = 99.5                                 # robust range top for rescale
    gate_center_to_grid:    bool = True
```

- All operations are `OperationField` (JSON round-trips the concrete class, GUI-editable) with
  live-default caveats handled in a `model_validator` (same pattern as
  `FilamentousFungiDetector.inoculum_detector`, which cannot hold a live pipeline at class-def
  time).
- If `FocusBranches` does not exist yet (companion spec), the default `branch_enhancer` is an
  inline `ImagePipeline([FlattenIllumination(300), ContrastStretching(70,99),
  FocusEdgePhase(n_orient=8, k=6, min_wavelength=5)])`.
- **`k=6`** (not `FocusBranches`'s own `k=2` completeness default) is chosen here because the
  light path exposes the segmenter directly to agar. The k-screen measured `k=6` giving the
  cleanest map (agar-strip `p99` 0.098 → 0.007 from `k=1` → `k=6`) at the cost of branch coverage
  (`active` 0.31 → 0.06); the light detector prioritises a clean mask over faint-tip recall. Lower
  it toward `2` when peripheral hyphae matter more than background cleanliness.

## 6 · Registration & tests

- `detect/_light_detect_fungi.py` → `LightDetectFungi`; re-export from `detect/__init__.py`
  (import + `__all__`).
- Tests (`tests/unit/detect/`):
  - **Scale invariance (the load-bearing one):** feed a branch map with p99 = 0.1 and a center
    map with p99 = 1.0; assert that after `_operate` the branch structure still appears in the
    final mask (i.e. the normalisation prevents the center from swamping it). Mutation: delete the
    `normalize` on the branch term and assert the branch pixels vanish from the mask — proves the
    normalisation is load-bearing.
  - **Center fill without hole-fill:** on a synthetic colony (solid core + radiating lines with
    gaps), assert the core is foreground **and** at least one inter-branch gap stays background
    (distinguishes this from `binary_fill_holes`).
  - **Composite energy:** both terms contribute — a pixel high only in branch and a pixel high
    only in center both survive to the composite.
  - **Contract:** `objmask`/`objmap` populated, `detect_mat` ends as the composite, `rgb`/`gray`
    untouched.
  - **Serialization round-trip** of the nested `OperationField`s.

## 7 · Excluded (explicitly, vs `FilamentousFungiDetector`)

Dijkstra fragment reconnection, the composite cost surface (anisotropy/coherence/MAD), tiling,
Euclidean Voronoi partition, connectivity correction, and the calibration/quality-filter cascade.
Consequently `LightDetectFungi` produces a **binary-quality fungal mask**, not individually
labelled colonies. Keep it that way — the moment per-colony reconnection is needed, reach for the
full detector.

## 8 · Open questions

1. **Cost-surface reuse is gone.** The heavy detector consumes the *full* `_PhaseCong3Result`
   (`M`, `m`, `orientation`) for its cost surface, which a black-box enhancer does not expose.
   `LightDetectFungi` sidesteps this (no cost surface), so no issue here — but it means the two
   detectors cannot share a single PCT pass if ever composed.
2. **Normalisation vs background lift.** `rescale(0, p99)` lifts agar texture and drives the
   triangle over-segmentation. Worth trying `in_range=(floor, p99)` with a small robust `floor`
   to push true background to 0 — but faint branches and faint agar are co-located, so this trades
   speckle for lost branch tips. Decide with a size-filter benchmark before committing a default.
3. **`k=6` default (set) vs branch completeness.** The k-screen picks `k=6` for a clean map
   (agar `p99` 0.007, `active` 0.06). This deliberately trades away faint peripheral hyphae; if a
   downstream needs those, `k≈2` recovers coverage (`active` 0.21) but leans harder on the
   segmenter/size-filter to hold agar back. `min_wavelength` is the other agar-suppression lever
   (larger = more suppression) and is worth a joint k × wl screen before hardening both.
4. **Ship the size filter as a default?** ~46k → a few hundred objects with a `min_object_area`
   drop; arguably it belongs inside `LightDetectFungi` rather than as a separate `refine` step,
   since a raw triangle mask is barely usable without it.
