# Voronoi Assignment Strategy for Filamentous Fungi

## Step 0: Copy this document

Copy this plan to `docs/design_outlines/voronoi_assignment_strategy.md` before making any code changes.

---

## Problem Statement

`FilamentousFungiDetector._voronoi_assign` assigns each pixel in the overall mask to the nearest colony marker. The current approach (flat-surface watershed, connectivity=2) produces **straight-line perpendicular-bisector boundaries** when the overall mask broadly connects two or more colonies into one connected component. On a flat surface, watershed BFS within a single connected component degenerates to chessboard-distance Voronoi — nearly identical to Euclidean Voronoi.

The straight-line boundaries cut through continuous fungal material, creating visually and biologically incorrect assignments.

### Biological Context: Anastomosis

The target organism (Neurospora) forms **anastomosis** — genuine hyphal fusion between colonies. The connections are real, not mask artifacts. The overall mask correctly shows one connected component because the colonies really are physically connected through thin hyphal bridges.

For analysis purposes, each colony must still be assigned distinct labels. The user's requirements:
- **At anastomosis bridge points** (where hyphae genuinely connect): distance-based separation is acceptable
- **Branches extending toward neighbors** (not in contact): must stay with their origin colony

**The problem only affects colonies that share a connected mask region.** Disconnected colonies (separate connected components) are assigned correctly regardless of approach.

---

## Two Independent Axes of Improvement

The assignment has two independent knobs:

| Axis | Current | Alternatives |
|------|---------|-------------|
| **Marker shape** | Single-pixel centroids | Full labeled center regions from inoculum detection |
| **Elevation surface** | Flat (zeros) | Inverted EDT, gradient, skeleton-based, random walker |

These combine freely. Each is analyzed below with honest failure-mode assessment.

---

## Axis 1: Marker Shape

### Current: Single-pixel centroids

`_create_markers_from_centers(overlap_objmap)` reduces each labeled center to one pixel at its center-of-mass. Watershed floods the entire mask from these tiny seeds. The boundary is the perpendicular bisector of two centroids.

### Alternative: Full labeled center regions

Skip `_create_markers_from_centers` and pass `overlap_objmap` directly as markers (masked to `overall_objmask`). Each colony is pre-labeled at its entire inoculum detection.

**Advantages:**
- Dramatically reduces the "disputed zone" that needs assignment
- Boundaries form between edges of center regions, not between centroids — more irregular, less straight
- Even with a flat surface, results improve proportionally to how large the center detections are
- Simplest change: one line in `_operate` (skip the centroid reduction step)

**Limitations:**
- If inoculum_detector produces very small centers (near-point), this approaches centroid behavior
- If two center regions touch within the overall mask, no boundary pixel is available between them
- Center regions must not overlap (they shouldn't, since they're independently labeled)

**Assessment: High value, low risk. Should be adopted regardless of which elevation surface is chosen.**

---

## Axis 2: Elevation Surface Options

### Option A: Flat Surface (current)

```python
elevation = np.zeros_like(mask, dtype=np.uint8)
```

Watershed = BFS = chessboard-distance Voronoi within connected components.

**Where it works:** Colonies in separate connected components; colonies connected only by narrow corridors (geodesic detour is longer than straight-line distance).

**Where it fails:** Colonies broadly connected through a wide overlap zone. BFS expands uniformly in all directions → straight-line boundary.

**Verdict: Insufficient for overlapping filamentous colonies.**

---

### Option B: Inverted EDT of Overall Mask

```python
edt = distance_transform_edt(mask)
elevation = -edt
```

EDT gives each pixel its distance to the nearest background pixel. Thick mask regions → high EDT (→ deep basins after negation). Thin connections → low EDT (→ ridges). Watershed boundaries form at ridges.

**Where it works:** Colonies connected by thin bridges (1–20px wide). EDT creates a clear ridge at the thin connection → clean split.

**Where it fails:** Colonies overlap in a wide zone (e.g., 200px broad junction). EDT is uniformly high throughout the overlap zone because all pixels are far from background. No ridge forms → boundary is determined by BFS arrival order (essentially arbitrary within the basin). This likely explains the "strange segments" seen previously.

**Additional concern:** EDT is dominated by the mask's overall outer boundary, not internal colony structure. Two colonies merging into one blob have one shared distance field that doesn't "know" about the two origins.

**Memory:** One extra array (float64 EDT). Negligible for typical plate images.

**Verdict: Helps for thin connections, unreliable for broad overlaps. Partial improvement over flat.**

---

### Option C: Compact Watershed (compactness parameter)

```python
segmentation.watershed(elevation, markers, mask=mask, compactness=C)
```

Adds penalty proportional to `C * euclidean_distance_from_seed²`. Biases toward compact (round) regions.

**Critical note for flat surface:** With flat elevation, the effective cost is purely `C * d²` — this is **Euclidean Voronoi regardless of compactness value**. Compactness on a flat surface changes nothing. It only adds value when combined with a non-flat elevation.

**With inverted EDT elevation:** Compactness provides a tiebreaker where the EDT landscape is ambiguous (flat overlap zones). The Euclidean proximity bias nudges boundaries toward equidistant points from seeds, which may be more regular than arbitrary BFS arrival order.

**Limitation for filamentous fungi:** Filamentous morphology is inherently non-compact (long branching structures). Compactness penalizes long branches extending far from the center, potentially mis-assigning them to a closer colony. This is the exact opposite of what we want — we want branches to stay with their origin colony regardless of proximity.

**Verdict: Counterproductive for filamentous morphology. Useful for round colonies (yeast), harmful for branching colonies (fungi).**

---

### Option D: Gradient-Based Elevation (Sobel on Original Image)

```python
gradient = filters.sobel(image_gray)
segmentation.watershed(gradient, markers, mask=mask, connectivity=2)
```

Boundaries follow intensity edges in the original image. Already proven in `WatershedDetector`.

**Where it works:** Colonies with distinct intensity profiles — the gradient field has high values at colony transitions, creating natural boundaries.

**Where it fails:** Overlapping hyphae from different colonies that look identical in intensity. Neurospora mycelium from different colonies often has similar intensity/density, so gradient signal at the junction may be weak or absent.

**Implementation cost:** Requires changing `_voronoi_assign` signature from `(markers, mask)` to `(markers, mask, image_data)`, plus adding `filters` import. Moderate refactor — breaks the static method's clean binary-only interface.

**Memory:** One extra array (float32/64 gradient).

**Verdict: Worth trying if colonies have intensity contrast at boundaries. Unreliable if overlapping hyphae are intensity-similar. Moderate implementation cost.**

---

### Option E: Random Walker

```python
from skimage.segmentation import random_walker
labels = random_walker(image_data, marker_labels, beta=130)
```

Each pixel is assigned to the marker it would most likely reach via random walk through the image, weighted by intensity gradients. Probabilistic and globally optimal.

**Where it works:** Complex shapes with even weak intensity gradients. Better than watershed at following subtle transitions. Handles irregular topologies well.

**Where it fails:** No intensity difference between overlapping colonies → random walker also becomes essentially distance-based. Very slow on large images (solves sparse linear system). Memory-intensive (sparse matrix).

**Implementation cost:** High. Changes algorithm entirely. Requires image data. Different API (`random_walker` returns labels directly, no elevation map concept). Needs labeled seed array (not just point markers — but full center masks provide this).

**Performance concern:** Neurospora plate images are large (4000x6000+). Random walker solves a sparse linear system of this size — could be minutes per image vs. milliseconds for watershed.

**Verdict: Theoretically best boundary quality. Practically too slow for batch phenotyping. Not recommended unless images are small or speed is irrelevant.**

---

### Option F: Skeleton-Based Tracing

```
1. Skeletonize(overall_mask) → 1px medial-axis network
2. Place markers at colony centers on skeleton
3. Flood-fill along skeleton from each marker → label each branch
4. Expand skeleton labels to fill original mask
```

Directly models the biological process: hyphae grow from colony centers along branching paths. Each skeleton branch traces back to exactly one origin colony.

**Where it works:** The skeleton preserves the branching topology. A hypha connected to colony A's center through the skeleton is correctly assigned to A, even if its tip is closer to colony B in Euclidean distance. This is the most biologically accurate approach.

**Where it fails:**
- **Noisy skeleton:** The overall mask has many small holes/speckles (visible in the screenshot). Skeletonizing a noisy mask produces many spurious branches that don't correspond to real hyphae.
- **Broad overlap zones:** Where two colonies' hyphae intermingle, the skeleton becomes a mesh rather than clean branches. Tracing through a mesh is ambiguous.
- **Re-expansion step:** After labeling the skeleton, expanding labels to fill the original mask reintroduces a Voronoi-like assignment (each mask pixel → nearest labeled skeleton pixel). This partially negates the skeleton advantage.

**Implementation cost:** High. Multi-step algorithm. Needs `skimage.morphology.skeletonize` (already in codebase as `Skeletonize` refiner). Skeleton flood-fill needs custom implementation or BFS. Expansion step needs `segmentation.expand_labels` or watershed from skeleton labels.

**Verdict: Most theoretically sound for filamentous morphology. Significant implementation complexity. Noisy masks severely degrade results. Potential future approach if mask quality improves.**

---

### Option G: Erode-Separate-Expand

```
1. Morphologically erode overall mask until colonies disconnect
2. Label disconnected components
3. Match each component to nearest marker
4. Expand labels back to original mask extent
```

**Where it works:** Colonies connected by thin bridges that erode away quickly.

**Where it fails:**
- **Different erosion levels:** Some colony pairs disconnect at 5px erosion, others at 50px. No single erosion radius works for all pairs.
- **Thin branches vanish:** Filamentous branches (1–5px wide) erode away entirely before thick connections break. They're then mis-assigned during expansion.
- **Small colonies vanish:** Small colonies may erode to nothing before large colony pairs disconnect.

**Verdict: Too lossy for filamentous morphology. Thin branches are the defining feature and this destroys them.**

---

## Summary Matrix

| Approach | Thin connections | Broad overlaps | Filament branches | Complexity | Image data needed |
|----------|:---:|:---:|:---:|:---:|:---:|
| **A. Flat (current)** | OK | Bad | Bad | None | No |
| **B. Inverted EDT** | Good | Unreliable | Neutral | Low | No |
| **C. Compactness** | — | Neutral | Harmful | Low | No |
| **D. Gradient (Sobel)** | Good | Good if contrast exists | Neutral | Medium | Yes |
| **E. Random walker** | Good | Good if contrast exists | Good | High | Yes |
| **F. Skeleton tracing** | Good | Moderate | Best | High | No |
| **G. Erode-expand** | Good | Bad | Destructive | Low | No |
| **Full center masks** | +Bonus | +Bonus | +Bonus | Trivial | No |

"Full center masks" is an independent improvement that stacks with any elevation option.

---

## Recommended Path (Incremental)

### Phase 1 — Use full center masks as markers
One-line change in `_operate`: pass `overlap_objmap` masked to `overall_objmask` instead of `_create_markers_from_centers(overlap_objmap)`. Evaluate visually.

### Phase 2 — Add inverted EDT elevation
Change `_voronoi_assign` to use `-distance_transform_edt(mask)`. Restore `distance_transform_edt` import. Evaluate visually. Combined with full masks, this may be sufficient.

### Phase 3 — If still insufficient, try gradient-based
Change `_voronoi_assign` signature to accept image data. Use `filters.sobel(image_data)` as elevation. Evaluate visually.

### Phase 4 — Explore skeleton tracing (if needed, separate feature)
Implement as an alternative to `_voronoi_assign`, not a replacement. More invasive design work needed.

---

## File

`src/phenotypic/detect/_filamentous_fungi_detector.py`

## Verification

1. `source .venv/bin/activate && python -m pytest tests/unit/detect/test_filamentous_fungi_detector.py -v`
2. Re-run Neurospora plate notebook overlay to compare boundaries visually at each phase
