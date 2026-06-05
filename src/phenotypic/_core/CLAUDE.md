# Core Module

`Image` and `GridImage` classes with accessor-based data access.

## Architecture

Linear MRO chain (bottom → top):
```
ImageDataManager → ImageHandler → ImageObjectsHandler → ImagePlotHandler
→ ImageColorSpace → ImageIOHandler → Image
```
`GridImage` extends `Image` via `ImageGridHandler`.

---

## Accessor Pattern

Data accessed through accessors (not direct attributes) — ensures consistency, lazy evaluation, and caching.

### Primary Accessors

- `image.rgb[:]` — raw RGB array (uint8/uint16)
- `image.gray[:]` — grayscale (weighted luminance)
- `image.detect_mat[:]` — enhanced grayscale for processing
- `image.objmask[:]` — binary mask of detected objects
- `image.objmap[:]` — labeled object map (integer labels)

### High-Level Accessors

- `image.objects` — iterate detected objects
- `image.color` — color space conversions
- `image.grid` — grid layout/alignment
- `image.metadata` — EXIF, file info
- `image.measurements` — extracted features
- `image.plot` — visualization (interactive views via `image.plot.dash.<name>()`)

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

---

