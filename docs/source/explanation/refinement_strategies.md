# Refinement Strategies

After detection, the raw binary mask and labeled object map often contain
artifacts. Refiners clean up these artifacts without re-running detection.

## Common Artifacts

| Artifact                     | Cause                            | Refiner                                |
|------------------------------|----------------------------------|----------------------------------------|
| Border objects               | Partial colonies at image edges  | `RemoveBorderObjects`                  |
| Small fragments              | Noise, agar texture              | `SmallObjectRemover`                   |
| Holes in masks               | Uneven colony interior intensity | `MaskFill`                             |
| Jagged edges                 | Threshold quantization           | `MaskOpening`, `MaskClosing`           |
| Touching colonies merged     | Threshold too low                | `SeparateObjects`                      |
| One colony fragmented        | Threshold too high               | `NearestNeighborMerger`, `MaskClosing` |
| Multiple detections per well | Over-segmentation in grid plates | `ReduceSectionsByLine`                 |
| Oversized objects            | Merged colonies spanning wells   | `GridOversizedObjectRemover`           |

## Refinement Order

The recommended refinement sequence:

1. **RemoveBorderObjects** — remove partial edge colonies first
2. **SmallObjectRemover** — remove noise fragments
3. **MaskOpening** — smooth boundaries and break thin bridges
4. **MaskFill** — fill interior holes
5. **Grid-specific refiners** — align, reduce multiples, remove oversized

This order ensures that each refiner operates on progressively cleaner
masks.

## Morphological Refiners

Morphological operations use a structuring element (footprint) to reshape
mask boundaries.

- **MaskOpening** (erosion → dilation) — removes small protrusions,
  breaks thin bridges between touching colonies
- **MaskClosing** (dilation → erosion) — fills small gaps, bridges
  narrow breaks in fragmented colonies
- **MaskDilator** — expands mask boundaries outward
- **MaskErosion** — shrinks mask boundaries inward
- **MaskFill** — fills enclosed holes using binary flood fill

The `width` parameter controls the footprint size. Larger footprints
produce more aggressive changes.

## Grid-Specific Refiners

These refiners use grid position information and are only available
for `GridImage`:

- **GridAligner** — rotates the image to align the grid
- **GridOversizedObjectRemover** — removes objects larger than expected
  for a single well
- **ReduceSectionsByLine** — when multiple objects are in one grid
  section, keeps only the most prominent
- **MergeWithinSection** — merges all objects within a grid section into one

## Tuning Strategy

Start with the default refiner chain in a prefab pipeline. If the
results are not satisfactory:

1. Check the overlay — identify which artifact type persists
2. Add or adjust the corresponding refiner
3. Adjust `min_size`, `width`, or `border_size` parameters
4. Re-run and compare
