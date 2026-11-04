# `tests/fixtures/frame_index/`

Parity data for `AddFrameIndex` (`src/phenotypic/post/_add_frame_index.py`),
consumed by `tests/unit/post/test_add_frame_index.py`.

**Synthetic input only.** Three plates, `PLATE01` to `PLATE03`, imaged every
4 hours from 1 January 2001, with each capture drifting by a few minutes. The
image names are built from the plate and the capture time
(`PLATE01_2001-01-01_07-59-00`), so they sort into capture order within a plate.
Nothing in the file is real.

**The answer comes from the reference, not from the op.**
`reference_frame_index` was recorded by running the reference implementation's
own frame-index derivation (a polars script kept outside this repository) on
these rows. The generator that built the plates and ran it is kept in the
project's planning files. The reference ranks each plate's distinct images by their parsed
capture time, 0-based. `AddFrameIndex` ranks them by image name, 1-based, so the
parity test asserts it gives exactly `reference_frame_index + 1` on every row.

## `synthetic_frame_index.parquet`

92 rows: 23 images, four rows each.

| Column | Meaning |
|---|---|
| `Metadata_PlateID` | `PLATE01`, `PLATE02`, `PLATE03` |
| `Metadata_ImageName` | `<plate>_<YYYY-MM-DD_HH-MM-SS>` |
| `Metadata_ImageDatetime` | the capture time the reference parses |
| `Object_Label` | 1 to 3, or null on the image's phantom row |
| `reference_frame_index` | the reference's 0-based frame index |

What it exercises:

* capture intervals that drift by minutes, which is why the index is a rank and
  not elapsed time divided by the cadence;
* `PLATE02` skips one capture, so every later image on it ranks one lower than
  on its sister plates — the known limitation the op's docstring states;
* several rows per image, a phantom row (null label) among them, all sharing
  the image's index;
* rows shuffled out of capture order, so position in the frame cannot stand in
  for rank.
