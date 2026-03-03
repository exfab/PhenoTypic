# Filamentous Fungi Detection Pipeline

## The Core Problem

Filamentous fungi grow as a network of thin branching hyphae radiating from a
central inoculation point. Under a camera, these hyphae are:

1. **Faint** -- they are thin and semi-transparent, often barely above
   background noise.
2. **Fragmented** -- standard thresholding ("is this pixel bright enough to be
   fungus?") catches some segments but misses others, producing disconnected
   blobs instead of a continuous network.
3. **Entangled** -- neighboring colonies' hyphae can overlap or grow close
   together, making it unclear which branch belongs to which colony.

The pipeline addresses each of these in sequence.

---

## Pipeline Overview

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | GatBM3D | Denoise the raw image |
| 2 | HomomorphicFilter | Even out plate illumination |
| 3a | InoculumDetector | Find the central inoculation plugs |
| 3b | Dual-mask branch detection | Detect hyphal structures (fragmented) |
| 3c | Voronoi watershed | Assign detected pixels to colonies |
| 3d | Dijkstra reconnection | Trace orphaned fragments back to colonies |

---

## Step 1: Denoising (GatBM3D)

Camera sensors introduce two kinds of noise -- a baseline electronic hum
(Gaussian) and shot noise proportional to signal brightness (Poisson).
Together they create speckle that can look like, or obscure, fine hyphae.

**GatBM3D** stabilizes the noise (makes it uniform across bright and dim
areas), then uses a patch-matching denoiser: it finds similar small patches
across the image and averages them to suppress noise while preserving real
edges. This is done *before* any detection so that downstream steps do not
mistake noise for hyphae or miss real hyphae buried in noise.

## Step 2: Enhancement

### HomomorphicFilter (pipeline level)

Corrects uneven illumination -- if the left side of the plate is brighter, it
normalizes this so colony intensity is directly comparable everywhere.

### PhaseCongruencyEnhancer (inside the detector)

This is the key enhancement for filaments. Instead of looking at pixel
brightness, it detects *edges and lines* -- places where the image has
structure at multiple spatial scales. Hyphae show up as thin line-like
structures with strong phase congruency even when their absolute brightness is
low. This produces:

- **An energy map** -- how "structured" each pixel is.
- **An orientation map** -- the direction of local structure at each pixel.
- **Anisotropy** -- how elongated/directional the structure is (hyphae are
  highly anisotropic).

These maps become the basis for the cost surface used in reconnection.

## Step 3a: Inoculum Detection

The `InoculumDetector` finds the dense central plugs (inoculation points)
where fungi were spotted onto the plate. It uses size constraints
(`min_diameter` / `max_diameter`) to distinguish inocula from debris or merged
colonies. The pipeline is:

1. Gaussian background subtraction (remove large-scale gradients)
2. Median filter (suppress salt-and-pepper noise)
3. Multi-scale Laplacian-of-Gaussian blob enhancement (highlight round objects
   in the expected size range)
4. Contrast stretching and morphological smoothing
5. Round-peaks detection and optional GMM core extraction

The result is a set of labeled inoculum regions that serve as colony seeds for
all subsequent assignment.

## Step 3b: Fragmented Detection (Dual-Mask Approach)

Rather than relying on a single threshold, the detector builds two independent
masks and combines them:

### Mask A -- Gaussian branches

Subtracts the large-scale background (Gaussian blur), then uses Triangle
thresholding, followed by morphological opening/closing (erosion then dilation
to remove small debris, then dilation then erosion to close small gaps). This
catches the broader, more obvious structures but may include noise blobs.

### Mask B -- Phase congruency branches

Uses the phase congruency energy map with hysteresis thresholding (two
thresholds: a "definite" high threshold and a "maybe" low threshold -- pixels
above the low threshold are only kept if they connect to pixels above the high
threshold). This is more selective and catches thin filaments that Mask A may
miss.

### Overlap filter

Only keeps Mask A regions that have *any* overlap with Mask B. This eliminates
noise blobs from Mask A (they will not appear in both methods) while retaining
real structures detected by either method.

The result is still fragmented -- you get disconnected blobs of detected fungal
material with gaps between them.

## Step 3c: Voronoi Watershed Assignment

Each detected pixel needs to be assigned to a colony. The algorithm computes a
Euclidean distance transform from the inoculum regions -- each inoculum sits
at elevation zero (deepest basin) and the surface rises with distance. A
watershed flood then expands outward from each inoculum through the detected
mask. Boundaries form where two floods meet at equidistant points.

This keeps branches with their origin colony: the origin flood fills a branch
from its base (low elevation, near inoculum) before a neighbor's flood can
reach the branch tip.

After this step:

- **Central regions** -- blobs that overlap a known colony's inoculum are
  assigned to that colony.
- **Fragments** -- blobs that *don't* overlap any colony remain unassigned
  orphans.

## Step 3d: Why Reconnection Is Needed

The fragments from the previous step are real hyphal material (they passed both
detection masks) but the gap between them and the nearest colony was too faint
or thin to be detected. Without reconnection, colony extent would be
undercounted -- the measured colony area would only include the bright central
mass and miss distal hyphae.

## Step 3e: Dijkstra Branch Reconnection

The reconnection treats the image as a landscape where the "cost" of traveling
through each pixel is based on how un-hyphal that pixel looks.

### Cost surface construction

| Signal | High cost means... |
|--------|--------------------|
| Phase congruency energy | Featureless background (low energy) |
| Anisotropy | No directional structure (isotropic) |
| Orientation coherence | Nearby pixels point in different directions (not a filament) |
| Local intensity variability (MAD) | Noisy, not structured |
| Already-assigned colony pixels | Near-zero cost (known structure, easy to traverse) |

### Dijkstra shortest-path search

From every colony boundary pixel, the algorithm floods outward through the
cost surface (like water flowing downhill through channels). Each fragment gets
assigned to whichever colony can reach it via the cheapest path -- i.e., the
colony connected by the most hyphal-looking corridor.

The `delta` parameter penalizes paths that curve back toward the inoculum,
reflecting the biological prior that hyphae grow outward.

### Quality filtering

Not every traced path is real. The algorithm calibrates quality thresholds by
measuring the cost of *known-good* branches (skeleton paths within
already-assigned colonies), then rejects reconnection paths that are
significantly worse. Filters check:

- **Windowed cost** -- is the path consistently low-cost, or does it cross a
  high-cost gap?
- **SNR** -- is the path brighter than its immediate surroundings?
- **PCT energy** -- does the path follow actual image structure?

The `quality_k` parameter controls strictness: higher values accept more
borderline paths, lower values are more conservative.

### Tiling

Large plate images are processed in overlapping tiles for memory efficiency.
The `tile_size` and `tile_overlap` parameters control this partitioning.

### Result

The final output is a labeled map where each pixel of detected fungal
material -- including distal hyphae that were initially orphaned -- is assigned
to its parent colony.

---

## Measurements (post-detection)

| Measurement | What it tells you |
|-------------|-------------------|
| MeasureGridSpatial | Colony positions relative to the grid layout -- spacing, regularity, which well each colony occupies |
| MeasureShape | Per-colony morphology -- area, perimeter, circularity, eccentricity (how round vs elongated) |
| MeasureIntensity | Per-colony brightness statistics -- mean, median, std dev of pixel values within each colony |
| MeasureTexture | Surface texture features (Haralick) -- captures whether colony surfaces are smooth, rough, wrinkled, etc. |

---

## Key Tuning Parameters

| Group | Parameters | What they affect |
|-------|-----------|-----------------|
| Denoising | `bm3d_block_size`, `bm3d_stage_arg` | How aggressively to denoise (stronger = smoother but may lose fine hyphae) |
| Illumination | `homo_sigma`, `homo_gamma_low`, `homo_gamma_high` | Lighting correction strength |
| Inoculum sizing | `inoculum_min_diameter`, `inoculum_max_diameter` | Expected inoculum plug sizes in pixels |
| Reconnection strictness | `quality_k` | How strict when accepting traced paths as real (higher = more permissive) |
| Outward growth prior | `delta` | Penalizes paths that curve back toward the inoculum |
| Cost surface | `beta`, `gamma`, `r_coherence`, `mad_window` | Weights and radii for composite cost computation |
| Memory management | `tile_size`, `tile_overlap` | Splits large images into tiles for path tracing |