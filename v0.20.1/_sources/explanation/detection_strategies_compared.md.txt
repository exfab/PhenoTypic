# Detection Strategies Compared

PhenoTypic provides over 20 detectors, each implementing a different
strategy for separating colonies from background. This page compares the
major approaches and their failure modes.

## Threshold-Based Detectors

Threshold detectors compute a single intensity value that separates
foreground (colony) from background (agar). Pixels above or below the
threshold are classified as colony.

### Otsu's Method

Finds the threshold that minimizes the weighted sum of within-class
variances (equivalently, maximizes between-class variance). Assumes a
bimodal histogram — one peak for background, one for colonies [1].

**Strengths:** Fast, parameter-free, works well on clean plates.

**Failure mode:** When the histogram is not bimodal (e.g., few small
colonies on a large plate), the threshold drifts toward the dominant peak
and misses the minority class.

### Triangle Method

Draws a line from the histogram peak to the tail and finds the point of
maximum perpendicular distance. Effective when one class greatly
outnumbers the other [2].

**Strengths:** Robust to class imbalance (few colonies, large background).

**Failure mode:** Sensitive to histogram smoothness; noise can create
spurious peaks.

### Hysteresis Thresholding

Uses two thresholds: a *high* threshold for confident foreground and a
*low* threshold for uncertain regions. Uncertain pixels are included only
if connected to confident foreground. This captures soft colony edges
that a single threshold misses.

**Strengths:** Handles gradual intensity transitions at colony boundaries.

**Failure mode:** Requires good threshold selection; a poor low threshold
floods the mask.

## Region-Based Detectors

### Watershed

Treats the intensity image as a topographic surface and "floods" from
local minima. Watershed boundaries separate touching colonies that
threshold methods merge into one object.

**Strengths:** Separates touching and overlapping colonies.

**Failure mode:** Over-segmentation — small intensity variations create
too many regions. Requires careful seed selection or smoothing.

## Peak-Based Detectors

### RoundPeaksDetector

Designed for grid plates with round colonies. Finds intensity peaks in
each grid section, then expands outward to the colony boundary. Assumes
one dominant colony per well.

**Strengths:** Grid-aware, handles variable colony sizes, fast.

**Failure mode:** Assumes round morphology; fails on irregular or
filamentous colonies.

## Specialized Detectors

### FilamentousFungiDetector

Two-stage detector for branching morphology. First detects inoculation
points, then uses phase congruency and minimum-cost pathfinding to
reconnect fragmented hyphal branches.

**Strengths:** Captures thin, branching filaments that threshold
methods miss entirely.

**Failure mode:** Slow on large images; requires BM3D denoising
upstream for best results.

### CompositeDetector

Combines multiple detectors via union, intersection, or overlap.
Useful when no single detector handles all colony types in one plate.

## Choosing a Detector

```
Is the plate a standard 96/384-well grid with round colonies?
  ├── Yes → RoundPeaksDetector
  └── No
      Is the organism filamentous?
        ├── Yes → FilamentousFungiDetector
        └── No
            Are colonies touching?
              ├── Yes → WatershedDetector
              └── No
                  Is the histogram bimodal?
                    ├── Yes → OtsuDetector
                    └── No → TriangleDetector or HysteresisDetector
```

## References

[1] N. Otsu, "A threshold selection method from gray-level histograms,"
*IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1, pp. 62--66, Jan. 1979.

[2] G. W. Zack, W. E. Rogers, and S. A. Latt, "Automatic measurement of
sister chromatid exchange frequency," *J. Histochem. Cytochem.*, vol. 25,
no. 7, pp. 741--753, 1977.
