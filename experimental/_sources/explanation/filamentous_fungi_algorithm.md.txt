# The Filamentous Fungi Detection Algorithm

Filamentous fungi grow as sprawling networks of thin, branching hyphae
that radiate from a dense inoculation point. Standard threshold
detectors fail on this morphology for three reasons:

- No single intensity cutoff captures both the dense colony centre *and*
  the thin peripheral hyphae.
- Hyphae frequently appear fragmented in imaging: gaps in signal break
  what is biologically one colony into many disconnected pieces.
- Neighbouring colonies interleave their hyphae and must still be
  individually labelled.

`FilamentousFungiDetector` handles all three in six phases. The class
exposes ~10 user-facing knobs grouped into three tiers:

1. **Scene parameters** — describe the sample's geometry.
2. **Tolerance knobs** — describe image quality and species morphology.
3. **Overrides** — escape hatches for non-standard imaging.

Plus four *hidden class-level tunables* (`beta`, `gamma`, `gauss_n_iter`,
`delta`, `pct_n_orient`) that are fixed to robust defaults. Override
only via subclassing if you really need to.

---

## Pipeline Overview

```
Phase 1: Inoculum Detection      ──▶ dense colony centres
Phase 2: Branch Detection        ──▶ binary mask of all hyphal pixels
Phase 3: Centre Filtering +      ──▶ initial colony labels
         Voronoi Partition          (connectivity not yet enforced)
Phase 4: Dijkstra Reconnection   ──▶ fragments routed back to centres
Phase 5: Final Voronoi           ──▶ final colony labels
```

Every hyphal pixel ends up labelled with its parent colony, respecting
both physical connectivity and spatial proximity.

---

## Phase 1: Inoculum Detection

Identify the dense colony centres first. These are the high-confidence
anchors around which the rest of the algorithm operates — they determine
both the Voronoi seeds and the Dijkstra source pixels. It's recommended if your
image is uniform enough to instead use `ManualGridPointDetector`, as it provides
guaranteed detection where you expect. InoculumDetector generally works, but can
fail depending on noise in your scene

**Parameters:**

| Param               | Role                                                                                                                                                                                                                        |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `inoculum_detector` | ObjectDetector or ImagePipeline for centres. Defaults to `InoculumDetector` + `GridSectionLargest` for grid plates. Replace with a species-specific centre detector if the default doesn't pick up your inoculation points. |

---

## Phase 2: Branch Detection

A deliberately *sensitive* detector captures every plausible hyphal
pixel, intentionally overestimating so that nothing real gets missed.
False positives are filtered in later phases.

Two signals are combined:

1. **Background-subtracted Gaussian response** — `SubtractGaussian` with
   σ sized to blur out the whole colony, leaving just the residual
   signal.
2. **Phase congruency** — an illumination-invariant edge response that
   highlights thin filaments even when their absolute intensity is low
   [[1]](#references). Hyphae are line-like; phase congruency is a
   line-sensitive feature detector.

The two signals are max-merged, then thresholded with a hysteresis
detector (triangle low, Otsu high) to produce a binary branch mask.

**Parameters:**

| Param                             | Role                                                                                                    | Default |
|-----------------------------------|---------------------------------------------------------------------------------------------------------|---------|
| `edge_noise_threshold`            | Phase congruency noise floor. ↑ = stricter (rejects more pixels as noise; preserves fewer thin hyphae). | 6.0     |
| `ignore_borders`                  | If True, drops branch components touching the image border.                                             | True    |
| `gauss_sigma` *(override)*        | SubtractGaussian σ. Auto-derived from `max_colony_radius_px × 1.2`.                                     | —       |
| `pct_min_wavelength` *(override)* | Log-Gabor minimum wavelength. Auto-derived from `min_branch_width_px × 2`.                              | —       |

**Hidden tunables:** `pct_n_orient = 8` (angular resolution for PCT),
`gauss_n_iter = 2` (background-subtraction iterations).

---

## Phase 3: Centre Filtering & Initial Voronoi Partition

The inoculum centroids seed a Euclidean Voronoi tessellation that
assigns every branch-mask pixel to its nearest centre.

First, branch components that don't overlap *any* inoculum centre are
dropped — debris and noise that happened to pass Phase 2 but don't
correspond to a real colony. Then centroids of the remaining overlapping
components become Voronoi seeds.

The result is an *initial* colony label map: every mask pixel has a
colony ID, but pixels within a given colony aren't necessarily
physically connected yet. That's what Phase 4 fixes.

**Parameters:** none directly — uses the Phase 1 centroids and the
Phase 2 mask.

---

## Phase 4: Dijkstra Reconnection

This is the algorithm's heavy lifting. "Pseudo-fragment" components
(assigned to a colony by Voronoi proximity but not physically touching
the main body) are bridged back via low-cost paths through the phase
congruency response.

### 4a. Cost surface construction

Each pixel's routing cost combines four features:

1. **Phase congruency energy** — paths prefer high-PCT-energy
   (signal-rich) pixels.
2. **Anisotropy** (weighted by `β`) — rewards directional (line-like)
   regions. Anisotropy appears in the denominator of the cost formula,
   so high anisotropy lowers cost; isotropic/non-directional pixels
   therefore appear expensive by comparison.
3. **Orientation coherence** — rewards contiguous oriented structure.
4. **Local MAD** (weighted by `γ`) — penalises noisy/textured regions.

Then two post-assembly penalties are added:

- **Gap penalty** (`gap_crossing_penalty`) — penalises paths that cross
  low-PCT-energy gaps proportional to the gap length.
- **Border penalty** (`border_margin_px`) — inflates cost within this
  many pixels of the image border, preventing shortcut paths that hug
  edges.

**Parameters:**

| Param                                  | Role                                                                       | Default |
|----------------------------------------|----------------------------------------------------------------------------|---------|
| `gap_crossing_penalty`                 | Strength of the distance-gap penalty. ↑ = harder to cross low-energy gaps. | 4.0     |
| `border_margin_px`                     | Width of the border penalty buffer.                                        | 50      |
| `mad_window` *(override)*              | Local MAD window. Auto-derived from `min_branch_width_px × 2 + 1`.         | —       |
| `coherence_window_radius` *(override)* | Orientation coherence window. Auto-derived from `min_branch_width_px × 5`. | —       |

**Hidden tunables:** `beta = 2.0` (anisotropy exponent), `gamma = 1.2`
(MAD penalty weight).

### 4b. Fragment pre-screening

Fragments stranded far from any routable territory are dropped *before*
Dijkstra runs — no point wasting compute on hopeless cases. A minimum
filter of radius `frag_reach_px` precomputes the lowest cost within
that 2D neighbourhood of every pixel. A fragment passes if at least
one of its boundary pixels sees a filtered value below `tau_screen` —
a threshold calibrated from known-good colony boundaries (specifically,
the 99th percentile of their min-cost envelope distribution), not an
absolute cutoff.

**Parameters:**

| Param           | Role                                                                  | Default |
|-----------------|-----------------------------------------------------------------------|---------|
| `frag_reach_px` | Max 2D distance from fragment boundary to the nearest routable pixel. | 10      |

### 4c. Tiled multi-source Dijkstra

For memory efficiency, the image is split into overlapping tiles. Each
tile runs a multi-source Dijkstra whose wavefront is seeded from the
*boundary* pixels of every colony body: all colony pixels (interior and
boundary) are pre-initialised to cost zero, but interior pixels are
marked visited immediately and never enter the heap, so propagation
starts cleanly from the boundary outward. The result is shortest-cost
paths from every pseudo-fragment back to its nearest main colony body.

Tile size must fit several colonies with routing headroom; tile overlap
must exceed the maximum reconnection distance so fragments near a tile
boundary can still see their parent.

**Parameters:**

| Param                       | Role                                                                            | Default |
|-----------------------------|---------------------------------------------------------------------------------|---------|
| `tile_size` *(override)*    | Tile side length. Auto-derived from `max_colony_radius_px × 4.8`.               | —       |
| `tile_overlap` *(override)* | Overlap between adjacent tiles. Auto-derived from `max_colony_radius_px × 2.4`. | —       |

**Hidden tunable:** `delta = 1.0` — radial retreat penalty that biases
Dijkstra against backtracking through already-visited territory.

### 4d. Path quality filtering

Not every Dijkstra path is accepted. A quality-filter cascade compares
each candidate against a distribution of *known-good* colony-skeleton
branches (extracted from the main colony bodies). Five metrics per path:

1. **Median raw cost** along the path (is it consistently cheap?).
2. **Max of windowed median costs** — the cost surface is sampled along
   the path; a sliding window of length `max_gap_length` computes the
   median cost within each window; the *maximum* of those per-window
   medians becomes the metric. A single-pixel cost spike is absorbed by
   the median, but a sustained bad stretch of length ≥ `max_gap_length`
   drives the windowed median up and flags the path.
3. **Band cost variance** — noisiness of the dilated band around the
   path.
4. **PCT energy band median** — average signal level along the path.
5. **Grayscale SNR** — local contrast vs. background.

Thresholds for each metric are set using `reconnection_tolerance` as an
IQR multiplier on the known-good distribution.

**Parameters:**

| Param                               | Role                                                                                         | Default |
|-------------------------------------|----------------------------------------------------------------------------------------------|---------|
| `max_gap_length`                    | Sliding window length for the "bad stretch" detection. Longer ⇒ tolerates longer gaps.       | 30      |
| `reconnection_tolerance`            | IQR multiplier for acceptance thresholds. ↑ = more permissive.                               | 2.5     |
| `path_dilation_radius` *(override)* | Path thickness for band sampling. Auto-derived from `max(1, min_branch_width_px // 2)`.      | —       |
| `snr_margin` *(override)*           | Background-ring offset for SNR filter. Auto-derived from `max(2, min_branch_width_px // 2)`. | —       |

Paths that survive the cascade get painted into the colony label map
with their assigned colony ID; the fragment pixels plus the dilated
reconnection path are labelled together.

---

## Phase 5: Final Voronoi Partition

With reconnected fragments now labelled, a final Voronoi partition
combines the Phase 3 filtered branch mask (components that overlapped
an inoculum centre) with all painted reconnection paths from Phase 4.
This ensures every foreground pixel gets a colony label that's
consistent with physical connectivity after reconnection, and that
connected components don't end up split across colony labels.

**Parameters:** none directly.

---

## Parameter Tuning Guide

The knobs fall into four tuning tiers. Work top-down: get scene
parameters right first, then adjust tolerances if results look wrong,
only touch overrides as a last resort.

### Tier 1: Always tune first (scene parameters)

These describe your sample. Wrong values here cascade into wrong
derived values for ~8 other parameters.

| Param                  | How to set it                                                                                                                                                                                       |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `max_colony_radius_px` | Measure the largest colony radius you expect in pixels. Too small ⇒ σ fails to blur out colonies, background subtraction breaks; too large ⇒ wasted memory on oversized tiles.                      |
| `min_branch_width_px`  | Measure the narrowest hyphal width you want to resolve. Too small ⇒ MAD/dilation windows shrink below Nyquist, missing real hyphae; too large ⇒ wavelength and windows merge neighbouring branches. |

### Tier 2: Tune based on image quality and species

These depend on your imaging conditions and organism. Start at defaults,
adjust if Phase 2 results look wrong (missing real hyphae / too much
noise) or Phase 4 reconnection behaves poorly.

| Param                    | When to raise                                                            | When to lower                                                   |
|--------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------|
| `edge_noise_threshold`   | Clean, high-contrast images with obvious hyphae                          | Dim or noisy images where real hyphae are being lost in Phase 2 |
| `reconnection_tolerance` | Legitimate fragments are not getting reconnected (permissiveness needed) | Obvious over-merging of nearby colonies (strictness needed)     |
| `max_gap_length`         | Species with long sparse hyphal segments                                 | Over-merging colonies across short gaps                         |

### Tier 3: Usually fine at defaults (spatial tolerances)

Sensible defaults cover most scenes. Touch only for specific failures.

| Param                  | Default | Typical reason to change                                                                                                      |
|------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------|
| `ignore_borders`       | `True`  | Set `False` if you need to keep colonies at image edges (mosaic/cropped views).                                               |
| `frag_reach_px`        | `10`    | Raise if legitimate but distant fragments are being dropped before Dijkstra.                                                  |
| `border_margin_px`     | `50`    | Lower if image edges carry useful signal; raise for noisy edges.                                                              |
| `gap_crossing_penalty` | `4.0`   | Lower (2.0) if gap-crossing paths are suppressed too aggressively; raise for very clean images where gaps truly mean absence. |

### Tier 4: Overrides (advanced; rarely needed)

If scene knobs capture your geometry well, auto-derivation works. Only
override when you have non-standard imaging (anisotropic pixels,
unusual magnification) or you're reproducing a pipeline with exact
fixed values.

| Override                  | Derived from           | Formula              |
|---------------------------|------------------------|----------------------|
| `gauss_sigma`             | `max_colony_radius_px` | `1.2 × R`            |
| `tile_size`               | `max_colony_radius_px` | `4.8 × R`            |
| `tile_overlap`            | `max_colony_radius_px` | `2.4 × R`            |
| `pct_min_wavelength`      | `min_branch_width_px`  | `2 × w`              |
| `mad_window`              | `min_branch_width_px`  | `2w + 1` (odd)       |
| `path_dilation_radius`    | `min_branch_width_px`  | `max(1, round(w/2))` |
| `snr_margin`              | `min_branch_width_px`  | `max(2, round(w/2))` |
| `coherence_window_radius` | `min_branch_width_px`  | `5 × w`              |

---

## Experimental tuning: what the defaults assume

The defaults are calibrated for:

- Plates imaged at a typical dissecting-microscope resolution.
- Colony radii roughly 50–300 px.
- Hyphal widths of 2–6 px.
- Moderate imaging noise (clean but not studio-quality).

If your setup differs substantially from these — imaging at very high
magnification, unusual organisms with atypical aspect ratios, or
phase-contrast optics with ringing artefacts — expect to iterate.
Specifically:

- **High-resolution imaging** (w > 6): raise `max_gap_length` and
  `frag_reach_px` proportionally.
- **Low-contrast / dim hyphae**: lower `edge_noise_threshold`, raise
  `reconnection_tolerance`.
- **Very dense plates with interleaved colonies**: lower
  `reconnection_tolerance`, raise `gap_crossing_penalty`. Accept a few
  extra fragments in exchange for cleaner separation.

---

## Preprocessing Requirements

The detector works best with upstream denoising and illumination
correction:

1. **`StableDenoise`** (BM3D) — removes Poisson-Gaussian noise without
   destroying thin filaments.
2. **`HomomorphicFilter`** — corrects uneven illumination so phase
   congruency isn't driven by intensity gradients.

The `FilamentousFungiPipeline` prefab chains these automatically.

---

## Quick Reference

```python
from phenotypic.detect import FilamentousFungiDetector

detector = FilamentousFungiDetector(
        # Scene: describe your sample
        max_colony_radius_px=250,  # tier 1 — always tune
        min_branch_width_px=3,  # tier 1 — always tune

        # Tolerances: describe your image quality
        edge_noise_threshold=6.0,  # tier 2 — tune if Phase 2 looks wrong
        reconnection_tolerance=2.5,  # tier 2 — tune if Phase 4 over/under-merges
        max_gap_length=30,  # tier 2 — species-dependent

        # Spatial tolerances (usually fine)
        ignore_borders=True,  # tier 3
        frag_reach_px=10,  # tier 3
        border_margin_px=50,  # tier 3
        gap_crossing_penalty=4.0,  # tier 3
)
```

---

## References

<a id="references"></a>

[1] P. Kovesi, "Image features from phase congruency," *Videre: J.
Computer Vision Research*, vol. 1, no. 3, pp. 1–26, 1999.

[2] A. F. Frangi et al., "Multiscale vessel enhancement filtering," in
*MICCAI*, 1998, pp. 130–137.
