# The Filamentous Fungi Detection Algorithm

Filamentous fungi grow as networks of branching hyphae that are thin,
variable in intensity, and often fragmented in microscopy images.
Standard threshold detectors fail because a single intensity cutoff
cannot capture the full extent of the mycelium.

PhenoTypic's `FilamentousFungiDetector` uses a multi-stage approach
designed specifically for this morphology.

## Stage 1: Inoculum Detection

The dense central region of each fungal colony (the inoculation point)
is detected first using a conventional detector. This is the
high-confidence anchor for each colony.

The `inoculum_detector` parameter accepts any `ObjectDetector` or
`ImagePipeline`. By default, a simple threshold detector is used.

## Stage 2: Voronoi Partition

Detected inoculum centers are used to compute a Voronoi tessellation of
the image. This assigns every pixel to its nearest inoculum center by
Euclidean distance, effectively partitioning the plate into colony
territories.

## Stage 3: Hyphal Detection

An overall detector (or custom pipeline) identifies all foreground
pixels — both dense inoculum regions and thin hyphal branches. This
detection is intentionally sensitive, accepting false positives that
will be filtered later.

## Stage 4: Dijkstra Reconnection

When `enable_reconnection=True`, the detector uses phase congruency
and minimum-cost pathfinding to reconnect fragmented hyphal branches.

**Phase congruency** provides an illumination-invariant edge response
that highlights thin filaments even when their absolute intensity is
low. The phase congruency map serves as the cost surface for pathfinding.

**Dijkstra's algorithm** finds the lowest-cost path between disconnected
hyphal fragments, weighted by:

- **Anisotropy** (`beta`) — penalizes paths that deviate from the local
  filament orientation
- **MAD penalty** (`gamma`) — penalizes paths through high-variation
  (noisy) regions
- **Quality threshold** (`quality_k`) — IQR-based cutoff that rejects
  low-confidence reconnections

## Stage 5: Assignment

Reconnected hyphal pixels are assigned to their nearest inoculum center
via the Voronoi partition, producing the final labeled object map.

## Preprocessing Requirements

The detector works best with upstream denoising and illumination
correction:

1. **StableDenoise (BM3D)** — removes noise without destroying thin
   filaments
2. **HomomorphicFilter** — corrects uneven illumination across the plate

The `FilamentousFungiPipeline` prefab chains these automatically.

## Key Parameters

| Parameter | Effect | Default |
|-----------|--------|---------|
| `enable_reconnection` | Enable Dijkstra reconnection | True |
| `quality_k` | Reconnection permissiveness (higher = more) | 2.5 |
| `beta` | Anisotropy weight in cost function | 2.0 |
| `gamma` | MAD penalty weight | 1.2 |
| `tile_size` | Tile size for memory-efficient processing | 1200 |

## References

[1] P. Kovesi, "Image features from phase congruency," *Videre: J.
Computer Vision Research*, vol. 1, no. 3, pp. 1--26, 1999.

[2] A. F. Frangi et al., "Multiscale vessel enhancement filtering," in
*MICCAI*, 1998, pp. 130--137.
