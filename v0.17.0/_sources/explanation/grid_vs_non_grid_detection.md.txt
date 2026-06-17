# Grid vs Non-Grid Detection

PhenoTypic supports both grid-arrayed and non-arrayed plate formats.
The choice between `Image` and `GridImage` affects which detectors,
refiners, and measurements are available.

## Non-Grid Detection (Image)

Standard detection operates on the entire image as a single field.
Detectors like `OtsuDetector` compute one global threshold and label
all connected foreground regions as separate objects.

**Advantages:**

- Simple, fast, no grid configuration needed
- Works for non-arrayed plates, single colonies, or arbitrary images

**Limitations:**

- No concept of wells or sections
- Cannot distinguish "two colonies in one well" from "one colony each
  in two wells"
- Refiners like `ReduceSectionsByLine` are not available

## Grid Detection (GridImage)

Grid detection partitions the image into a row × column layout. Each
grid section (well) is processed with awareness of its neighbors and
expected content.

**Advantages:**

- Per-well colony counts and measurements
- Grid-aware refiners that remove oversized objects, merge fragments
  within sections, and reduce multiple detections per well
- Grid alignment correction
- Section-level extraction (`image.grid[row, col]`)

**Limitations:**

- Requires correct `nrows` and `ncols` specification
- Assumes a regular rectangular grid
- Grid detection can fail on severely rotated or distorted plates

## When to Use Each

| Scenario                      | Image type  | Reason                                   |
|-------------------------------|-------------|------------------------------------------|
| 96-well pinned array          | `GridImage` | Grid structure enables per-well analysis |
| 384-well high-density plate   | `GridImage` | Essential for resolving dense arrays     |
| Streak plate                  | `Image`     | No grid structure                        |
| Single colony close-up        | `Image`     | One colony, no grid                      |
| Custom non-rectangular layout | `Image`     | Grid assumes rectangular layout          |

## The AutoGridFinder

When you create a `GridImage` without specifying a `grid_finder`,
PhenoTypic uses `AutoGridFinder`. It estimates row and column
boundaries from the spatial distribution of detected colonies. This
works well when colonies are present in most wells, but can fail on
sparse plates or when many wells are empty.

For sparse plates, consider `ManualGridPointDetector` to specify grid
positions explicitly.
