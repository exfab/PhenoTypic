# Core Module

`Image` and `GridImage` classes with accessor-based data access.

## Architecture

Linear MRO chain (bottom → top):
```
ImageDataManager → ImageHandler → ImageObjectsHandler → ImageVisualizationHandler
→ ImageColorSpace → ImageIOHandler → Image
```
`GridImage` extends `Image` via `ImageGridHandler`.

---

## Accessor Pattern

Data accessed through accessors (not direct attributes) — ensures consistency, lazy evaluation, and caching.

### Primary Accessors

- `image.rgb[:]` — raw RGB array (uint8/uint16)
- `image.gray[:]` — grayscale (weighted luminance), float32 in `[0, 1]` for every
  input. A single-channel integer array is divided by its dtype's maximum at
  construction (the divisor `rgb2gray` uses for RGB); a single-channel float
  array outside `[0, 1]` is refused. `_retain_original` still gives back the
  decoded integers, and stores/HDF/pickles written before this hold integer
  layers that are normalised on load.
- `image.detect_mat[:]` — enhanced grayscale for processing
- `image.objmask[:]` — binary mask of detected objects
- `image.objmap[:]` — labeled object map (integer labels)

### High-Level Accessors

- `image.objects` — iterate detected objects (`image.num_objects` for the count); per-object
  bounds/labels via `image.objects.info()`. Measure features with a `MeasureFeatures`
  operation, e.g. `MeasureSize().measure(image)` (not `image.objects.measure`)
- `image.color` — color space conversions
- `image.grid` — grid layout/alignment (**GridImage only**)
- `image.metadata` — EXIF, file info
- `image.napari()` — interactive visualization of all available image layers

### NumPy Interface

All accessors support NumPy indexing: `image.rgb[100:200, 50:150]`, `image.rgb[:, :, 0]`,
`image.rgb.shape`, `image.rgb.dtype`.

**Setter pattern:** Write via slice assignment: `image.detect_mat[:] = new_array`.
Direct attribute assignment will not work.

---

## Color Spaces

Via `image.color`: `Lab[:]` (CIELAB), `hsv[:]`, `XYZ[:]`, `XYZ_D65[:]`, `xy[:]` (chromaticity)

- Lazy-evaluated and cached
- sRGB gamma correction applied automatically
- D65 illuminant default; CIE 1931 2° Standard Observer
- A derived image — `copy()`, `Image(other)`, `GridImage(image)`, a crop, a grid
  section, `objects[i]` — inherits the source's `gamma`/`illuminant`/`_observer`;
  an argument passed explicitly to the constructor wins. The constructor defaults
  are the `UNSET` sentinel (`_image_color_handler.py`), not `"D65"`/`SRGB`, because
  `gamma=None` already means linear and an explicit `"D65"` must still override a
  D50 source.

---

