# How the Image Class Works

The `Image` class is PhenoTypic's central data structure. It stores pixel
data, metadata, detection results, and measurements in a single object,
accessed through a uniform accessor pattern.

## Architecture

`Image` is built from a linear chain of mixin classes, each adding a
layer of functionality:

```
ImageDataManager → ImageHandler → ImageObjectsHandler → ImagePlotHandler
→ ImagePanelHandler → ImageColorSpace → ImageIOHandler → Image
```

`GridImage` extends `Image` with grid-aware operations via
`ImageGridHandler`.

## The Accessor Pattern

Data is not accessed through direct attributes. Instead, **accessors**
provide a consistent interface with NumPy-style indexing:

```python
# Read
pixels = image.rgb[:]           # Full array
roi = image.rgb[100:200, 50:150]  # Slice
channel = image.rgb[:, :, 0]    # Single channel

# Write
image.detect_mat[:] = new_array
```

This pattern ensures:

- **Lazy evaluation** — Color space conversions compute only when accessed
- **Caching** — Repeated access returns the cached result
- **Consistency** — Write operations trigger downstream invalidation

## Primary Accessors

| Accessor | Type | Shape | Mutable | Description |
|----------|------|-------|---------|-------------|
| `rgb` | uint8/uint16 | (H, W, 3) | No* | Raw RGB color data |
| `gray` | float32 | (H, W) | No* | Luminance-weighted grayscale |
| `detect_mat` | float32 | (H, W) | Yes | Enhanced grayscale for detection |
| `objmask` | bool | (H, W) | Yes | Binary colony mask |
| `objmap` | uint16 | (H, W) | Yes | Labeled colony map |

\* Set via `Image()` constructor or `set_image()`; not directly writable
through the accessor.

## High-Level Accessors

- **`image.color`** — Color space conversions (Lab, HSV, XYZ, xy)
- **`image.grid`** — Grid layout and section operations (GridImage only)
- **`image.metadata`** — EXIF data and custom metadata
- **`image.plot`** — Visualization methods
- **`image.objects`** — Iterate over detected objects

## Image vs GridImage

`Image` is the base class for any plate image. `GridImage` adds:

- Grid dimensions (`nrows`, `ncols`)
- A `GridFinder` that computes row and column boundaries
- The `.grid` accessor for section-level operations
- Grid-aware detection and refinement

Use `GridImage` for arrayed colony plates (96-well, 384-well, etc.).
Use `Image` for non-arrayed plates, single colonies, or images where
grid structure is not relevant.

## Copy Semantics

`image.copy()` creates a deep copy with a new UUID. All pixel data,
masks, metadata, and grid state are duplicated. The copy is fully
independent of the original.

`image.reset()` clears detection results (`detect_mat`, `objmask`,
`objmap`) while preserving the original pixel data (`rgb`, `gray`)
and metadata.
