# How `Image.imread` treats unusual inputs (plan Task 10 Step 1)

Settles spec open questions 2 and 5 by probe, so the header checks flag what
`imread` actually does rather than a guess. Run 2026-09-24 against the Task 9
head (`a18211a`), `uv run python` (Pillow via skimage/imageio; tifffile
2025.10.16). Probe scripts: the session scratchpad `p10/probe_headers*.py`;
each file below was written by Pillow or tifffile, then read with
`Image.imread(path, **kw)`.

| Input | Header says | `imread` result |
|---|---|---|
| RGBA PNG | PIL `RGBA`, 4 bands | OK: RGB `(16,16,3)` float64, alpha dropped |
| Palette PNG | PIL `P`, **1 band** | OK: RGB `(16,16,3)` uint8 (the palette is expanded) |
| `LA` PNG | PIL `LA`, 2 bands | **raises** `ValueError: Image with 2 channels (unknown format)` |
| 2-sample TIFF | spp=2 (Pillow cannot identify) | **raises** the same `ValueError` |
| 4-sample TIFF (RGB+alpha) | spp=4 | OK: RGB |
| CMYK TIFF | PIL `CMYK`, spp=4, photometric SEPARATED | OK: treated as 4 channels, RGB out |
| 5-sample TIFF | spp=5 | **raises** `ValueError: Image with 5 channels (unknown format)` |
| 3-page grayscale TIFF, three separate series | spp=1, 3 series | OK: **first series only**, grayscale |
| `(3,H,W)` single series (Fiji `CYX`, OME-TIFF `CYX`, minisblack stack) | spp=1, 1 series of 3 pages | OK: **RGB** `(H,W,3)`, skimage moves the leading axis (added after review D1) |
| `(4,H,W)` single series | spp=1, 1 series of 4 pages | OK: RGB |
| `(2,H,W)` or `(5,H,W)` single series | spp=1, 1 series | **raises** `ValueError: Image with W channels`: the leading axis is not moved, so the width is read as a channel count |
| `(Z,H,W,3)` stack | 4-D series | **raises** `ValueError: Unknown format (unsupported number of dimensions)` |
| JPEG, `bit_depth=16` | 8-bit | OK: `imread` forces `bit_depth=8` for JPEG (added after review D3) |
| 16-bit grayscale PNG | PIL `I;16` | OK: grayscale, `bit_depth` 16 |
| uint16 RGB TIFF, `bit_depth=8` | dtype uint16 | OK, **silently** `bit_depth=8` on uint16 data |
| uint8 RGB PNG, `bit_depth=16` | dtype uint8 | OK, **silently** `bit_depth=16` on uint8 data |
| zero-byte `.png` | Pillow `UnidentifiedImageError` | **raises** `OSError: Could not find a backend` |
| truncated PNG (first 60 bytes) | PIL header parses: `RGB` | **raises** `OSError: image file is truncated` on decode |

## What the checks therefore do

- **Decoded channel count, not header bands.** Palette -> 3; RGBA/RGBX/CMYK/4-sample -> 3;
  `L`/`I`/`I;16`/`F`/1-sample -> 1. A TIFF is predicted from its **first series'**
  shape, passed through skimage's axis move (`skimage/io/_io.py`: a leading axis of 3
  or 4 goes last when the last is not 3 or 4) and then `Image`'s own rule (2-D gray; a
  last axis of 1 gray, 3 or 4 RGB, anything else refused; more dimensions unknown).
  **Corrected after the Phase D review (D1):** this line first said "multi-page -> the
  first page's count", generalized from a probe that wrote three separate series; a
  channel stack is one series and decodes to RGB, so the old rule refused valid runs.
- **`PF-CHANNELS`** flags exactly the counts that raise: 2 channels (`LA`, 2-sample
  TIFF) and 5 or more samples.
- **`PF-BIT-DEPTH`** flags a `--bit-depth` that disagrees with the header dtype
  (8 with a 16-bit dtype, 16 with an 8-bit one): `imread` accepts it silently and
  records the wrong bit depth for the data. JPEG is exempt, because `imread` ignores
  the flag for it (review D3). Mixed dtypes across inputs with no
  `--bit-depth` are not flagged: each image then records its own dtype's depth.
- **`PF-HEADER-UNREADABLE`** catches a zero-byte or unidentifiable file. A truncated
  file whose header parses is **not** caught: only a decode finds it, and the CLI
  already isolates that as a per-image failure. The report says so rather than
  implying coverage it lacks.

## Same-stem inputs (open question 5, F28)

A dataset holding `a.png` and `a.tif` (`--image-type Image`, `--njobs 1`): the scan
reports "Found 2 images", both are written to **one** store
`results/plate1/zarr/a.ome.zarr`, and the run ends with `Unexpected error: Cannot
publish GUI local completion while current image outcomes remain incomplete`, exit 1,
after processing. No earlier stage refuses it, so `PF-STEM-COLLISION` is an error.
