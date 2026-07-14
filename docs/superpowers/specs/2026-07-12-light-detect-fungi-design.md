# `LightDetectFungi` — design

Status: proposal (not committed). A **lightweight** filamentous-fungi detector: build the branch
mask with **two-scale-`k` phase-congruency hysteresis**, union it with a grid-gated center-fill,
and (optionally) label per colony via a cheap grid-Voronoi assignment. **No Dijkstra cost-surface
reconnection, no tiling, no quality-filter cascade** — those stay in `FilamentousFungiDetector`.
The fast path to a clean, well-connected, filled fungal mask (and, with grid seeds, per-colony
labels) without the minutes-long machinery.

Grounded in live runs on the production crop (`NeurosporaPipeV10.json`, `650/650/600/600`).

## 1 · Motivation & flow

`FilamentousFungiDetector` (~minutes) does inoculum-detect → dual-mask branch-detect → grid
Voronoi → Dijkstra reconnection → final Voronoi. `LightDetectFungi` keeps the value from a few
cheap ideas developed this session:

- **branch mask via two-`k` hysteresis** — recovers most of the reconnection Dijkstra gives (§2);
- **center-fill** — `ManualGridPointDetector` at known wells ∩ a background subtraction, filling
  the hollow inoculum core that edge-based phase congruency leaves — **without** `fill_holes`;
- **grid-Voronoi labelling** (optional) — per-colony identity from known well coords, no Dijkstra.

```
strict = PCT(k=6)      ;  loose = PCT(k=4.5)          # two phase-congruency passes on the same enhanced base
branch_mask = two_k_hysteresis(strict, loose)          # §2 -> reconnected binary branch mask
center_mask = center_detector.objmask & (bg_subtract > otsu)   # §3 grid ∩ body, fills the core
colony_mask = branch_mask | center_mask                # union of binary evidence
objmap      = grid_voronoi_label(colony_mask, seeds)   # §4 optional per-colony id (else CC labels)
```

**No gamma pre-adjustment.** The pipeline runs on the raw cropped image — **no** `adjust_gamma`
front step (which the exploration notebooks originally carried). It is safe *and* marginally
better: the chain is contrast-adaptive/invariant end-to-end (homomorphic `FlattenIllumination`,
percentile `ContrastStretching`, contrast-invariant phase congruency, adaptive `otsu`/`triangle`),
so a global gamma is absorbed — and measured, gamma slightly *increased* fragmentation
(12.0 → 11.2 fragments/colony without it; the difference is faint-tip threshold flicker, systematically
in no-gamma's favour). Do not add a gamma step.

## 2 · Branch mask: two-scale-`k` hysteresis (the crux — verified)

`k` (the phase-congruency noise threshold) trades confidence for coverage: **high `k` = clean but
fragmented; low `k` = full but agar-flooded.** Neither single `k` is good. Hysteresis marries two:

- **strict `k=6`** → seeds: `seed = otsu(PCT k=6)` — clean, confident branch cores;
- **loose `k=4.5`** → candidates: `cand = triangle(PCT k=4.5)` — faint connectors **and** agar;
- **reconnect**: `keep = reconstruction(seed ∩ cand, mask=cand, "dilation")` — grow the seeds
  through the candidate mask; keep every candidate component that touches a seed.

**Binarize the two signals with *opposite* thresholds** — strict×strict for seeds, loose×loose
for candidates. Verified this is load-bearing: `otsu`-seed / `triangle`-cand is the only combo
that both reconnects and stays clean (using `otsu` on candidates drops the connectors → no
reconnection; using `triangle` on seeds admits agar into the seeds → flood).

Measured (grid-restricted per-colony fragments | agar false-positive, production crop):

| binarization | frags/colony | agar_fp |
|---|---|---|
| seed only (`otsu` k=6) | 22.0 | 0.001 |
| cand only (`triangle` k=4.5) | 26.0 | 0.045 (flood) |
| **two-`k`: `otsu` seed / `triangle` cand** | **12.0** | **0.012** |
| two-`k`: `otsu` / `otsu` | 22.3 | 0.002 |
| two-`k`: `triangle` / `triangle` | 26.0 | 0.041 |

**Nearly halves fragmentation (22 → 12) while agar stays clean** — the faint `k=4.5` hyphae that
bridge the `k=6` seed fragments are *recovered* (they touch a seed), and isolated `k=4.5` agar is
*rejected* (no seed). Cost: **two PCT passes (~45 s) + one `reconstruction`**. Decisively better
than coherence-enhancing diffusion (which managed only −7 to −13%).

**This is why the earlier continuous-composite + `TriangleDetector` plan is dropped for the branch
stage.** The two-`k` hysteresis *is* the branch segmentation (binarization built in), and it
sidesteps the cross-signal **energy-scale problem** entirely: each signal is thresholded at its own
scale, so there is no continuous `max()` of a low-energy PCT map (p99 ≈ 0.15) against a full-range
center map to normalise. (For the record, that scale mismatch — `FocusEdgePhase` clips to `[0,1]`
but fills only the bottom, and is not a `NormalizedOutputMixin` — was real and was the whole reason
a naive `max` composite failed; binarizing per-signal avoids it rather than patching it.)

## 3 · Center-fill (verified)

Phase congruency is an edge detector, so the inoculum core is a hole. Fill it with a *separate*
solid-body signal, gated to the known wells:

- `center_detector` (default `ManualGridPointDetector` at the plate's grid coords) stamps a disk at
  each of the 60 wells → `grid_mask`;
- `background_subtractor` (`SubtractGaussian(sigma=300)`) on the enhanced image keeps the bright
  solid cores (and, unavoidably, the plate rim);
- `center_mask = grid_mask & (body > otsu(body))` — the intersection snaps each stamp to real
  colony signal **and drops the plate rim** (the stamps never cover the wall). Verified: 60 clean
  discs. No `fill_holes`, so the legitimate inter-hypha gaps are preserved.

The grid detector is swappable (`InoculumDetector`, blob) for plates without known coordinates.

## 4 · Union & labelling

`colony_mask = branch_mask | center_mask` (both binary). Per-colony `objmap`: two-`k` still leaves
~12 components/colony, so connected-component labelling alone gives ~12 labels/colony. To get **one
label per colony**, assign every component to its nearest grid seed via **Euclidean Voronoi**
(`euclidean_voronoi_assign` + `connectivity_correct_labels` from `sdk_.branch_pathfinding` — the
*cheap* part of `FilamentousFungiDetector`, **not** the Dijkstra cost-surface part). With known
`ManualGridPointDetector` coords this is trivial and yields 60 labels. Optional
(`label_by_grid_voronoi`); off → `objmap` is the raw connected components.

## 5 · Fields / API

```python
class LightDetectFungi(GridObjectDetector):        # grid: uses grid seeds for center-fill + labelling
    # branch enhancement (shared pre-PCT base, then two PCT passes)
    branch_base:      OperationField = <ImagePipeline: FlattenIllumination(300), ContrastStretching(70,99)>
    n_orient:         int = 8
    min_wavelength:   float = 5.0
    k_strict:         float = 6.0                    # seed pass  (clean)
    k_loose:          float = 4.5                    # candidate pass (faint connectors)
    seed_thresh:      Literal["otsu","triangle"] = "otsu"      # strict
    cand_thresh:      Literal["otsu","triangle"] = "triangle"  # loose
    # center-fill
    center_detector:       OperationField = ManualGridPointDetector(...)   # or InoculumDetector
    background_subtractor: OperationField = SubtractGaussian(sigma=300, n_iter=2)
    # cleanup + labelling
    min_object_area:       int = 30                  # drop sub-branch speckle
    label_by_grid_voronoi: bool = True
```

- `OperationField`s JSON-round-trip the concrete class and are GUI-editable; live-default caveats
  handled in a `model_validator` (same pattern as `FilamentousFungiDetector.inoculum_detector`).
- **Why `k_strict=6` / `k_loose=4.5`:** the k-screen showed `k=6` is the cleanest single map
  (agar-strip `p99` 0.007) and `k=4.5` supplies the faint connectors; the hysteresis needs the gap
  between them. `k_loose` is the main tuning knob (§8).

## 6 · Registration & tests

- `detect/_light_detect_fungi.py` → `LightDetectFungi`; re-export from `detect/__init__.py`.
- Tests (`tests/unit/detect/`):
  - **Two-`k` reconnection (load-bearing):** a synthetic colony whose branches are broken at `k=6`
    but connected by faint pixels present at `k=4.5` → assert the reconnected mask has **fewer
    connected components** than the `k=6` seed mask, and that a pure-agar region stays background
    (agar_fp low). Mutation: swap to `otsu`/`otsu` binarization → assert the fragment count rises
    back to the seed-only level (proves the loose-candidate threshold is load-bearing).
  - **Binarization direction:** `triangle`-seed variant floods a pure-agar patch (agar_fp jumps) —
    pins that seeds must use the strict threshold.
  - **Center fill without hole-fill:** solid core + radiating lines with gaps → core is foreground
    **and** at least one inter-branch gap stays background (distinguishes from `binary_fill_holes`).
  - **Labelling:** with grid seeds and `label_by_grid_voronoi=True`, a plate of N wells → N labels;
    off → `objmap` == connected components.
  - **Contract & serialization round-trip.**

## 7 · Excluded (explicitly, vs `FilamentousFungiDetector`)

The Dijkstra **cost surface** (anisotropy/coherence/MAD), tiling, path quality-filter cascade, and
the two-pass grid→reconnect→final Voronoi. Two-`k` hysteresis recovers the *sub-threshold
connectors* Dijkstra would bridge (hyphae that exist in the faint `k=4.5` signal); only **truly
empty gaps** (no candidate pixels at all) remain unbridged — those still need Dijkstra. The **cheap
Euclidean-Voronoi labelling is retained** (§4); the expensive cost-surface reconnection is not.

## 8 · Open questions

1. **`k_loose` tuning.** `4.5` chosen from one plate; sweep `4.0 / 4.5 / 5.0` (lower = more
   connectors recovered but more agar admitted to candidates). The notebook
   `BranchReconnect_TwoK_Hysteresis.ipynb` is the vehicle.
2. **3-`k` ladder — tested, does not help (resolved).** Two formulations of `6 → 5 → 4.5` both
   failed to beat plain 2-`k`: a **nested-reconstruction** ladder is mathematically a *no-op*
   (morphological reconstruction floods whole components, so an intermediate mask nested between
   the extremes changes nothing — verified, 0.02% pixel diff vs 2-`k`); a **bounded-geodesic**
   ladder (limited growth per level) only trades a hair of agar (0.012→0.009) for a hair more
   fragments (12.0→13.0) and converges to 2-`k` as the reach grows. The residual ~12 fragments/
   colony are **true empty gaps** (no candidate pixels bridging even at `k=4.5`) — hysteresis can't
   span those; they are genuinely Dijkstra's job. **Keep the 2-level version.**
3. **Two PCT passes.** ~2× the branch cost. Still seconds (vs minutes for the full detector), so
   acceptable for "light" — but confirm on a full batch.
4. **Grid-Voronoi labelling default.** On for grid plates (known coords); needs a graceful CC-label
   fallback when no grid seeds are available. Resolve against how callers supply coordinates.
