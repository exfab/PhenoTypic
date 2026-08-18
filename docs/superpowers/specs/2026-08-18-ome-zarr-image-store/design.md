# OME-Zarr per-image store

**Date:** 2026-08-18
**Status:** Draft — awaiting review
**Scope:** Per-image CLI image storage, the staged-GPU commit protocol, GUI tile
reads, the legacy-HDF migration utility, and the Python support floor

## Summary

PhenoTypic will replace the per-image HDF5 file with an **OME-Zarr (NGFF 0.5 /
Zarr format v3) store**, one per input image:

```text
results/<dataset>/hdf/<stem>.h5   ->   results/<dataset>/zarr/<stem>.ome.zarr/
```

Image layers become named sibling NGFF multiscale images inside a
`bioformats2raw.layout` collection; `objmap` becomes a first-class NGFF **label
image**; and all PhenoTypic-specific state — including grid `nrows`/`ncols`,
`Metadata_ImageType`, and the three metadata sections — lives in a namespaced
`attributes.phenotypic` JSON block beside the `ome` block.

The replacement is total. `save2hdf5` / `load_hdf5` / `load_layer_hdf5` are
removed from the write and read paths. Existing `.h5` runs are handled by a
one-shot migration utility, not by a permanent dual reader.

## Locked decisions

These were settled during design and are not reopened by implementation:

1. **Full replacement, migration by utility.** One writer, one reader. Legacy
   `.h5` is reachable only through `migrate_hdf_to_zarr` /
   `migrate_run_hdf_to_zarr`. No format flag, no runtime sniffing in the hot
   path.
2. **Layout is a named-series collection.** Root carries
   `bioformats2raw.layout: 3`; `OME/zarr.json` carries
   `series: ["rgb", "gray", "detect_mat"]`. Named paths, not integer indices.
3. **NGFF 0.5 on Zarr format v3.** This raises `requires-python` to
   `>=3.11, <3.13`. Python 3.10 is dropped.
4. **Commit is rename-promote plus last-write markers.** Stage 1 promotes a
   `.part` directory by `os.replace`; Stage 2 commits by writing the labels
   group's `zarr.json` last.
5. **OME projection is write-only.** `attributes.phenotypic` is the sole source
   of truth on read. Standard OME fields are derived on every write and never
   read back, so they cannot drift.
6. **Layout version and metadata-schema version stay separate.** Inherited from
   the approved flat-metadata design, decision #6.
7. **Pyramids are in scope**, and the GUI tile server reads them.
8. **The dead HDF DataFrame layer is retired** as part of this change.
9. **Per-colony measurements are NOT projected into `image-label.properties`.**
   Considered and deferred; parquet remains the only measurement surface.

## Context

### What exists today

Per-image `results/<dataset>/hdf/<stem>.h5`, `schema_version = 2`, written by
[`_save_image2hdfgroup`](../../../../src/phenotypic/_core/_image_parts/_image_io_handler.py):

```text
/                     attrs: version, schema_version, metadata_schema_version,
                             phenotypic_class, bit_depth, illuminant, gamma
                             (+ CLI-injected phenotypic_work_id)
/layers/rgb           (H,W,3) uint8|uint16, optional
/layers/gray          (H,W)
/layers/detect_mat    (H,W) float, attrs: detect_mode
/layers/objmap        (H,W) integer labels
/metadata/{protected,public,imported}   JSON-encoded attrs
/grid/                attrs: nrows, ncols; dataset: grid_finder_json
```

Three properties of that file are load-bearing beyond mere storage:

- **Atomicity.** `save_image_hdf` writes a `.part` file and `os.replace`s it.
  A single-file rename is the entire crash-safety guarantee.
- **Content-defined resume.** `valid_staged_hdf` opens the file and asserts
  `gray` / `detect_mat` / `objmap` exist with agreeing `(H, W)`;
  `staged_hdf_matches_work_id` reads a root attribute.
- **Read-only Stage 2.** The staged-GPU engine cannot write into the HDF while
  holding it open read-only, which is the sole reason the `.npy` objmap
  sidecar exists.

### Why change

- `objmap` is a segmentation label map with no standard representation today.
  NGFF models exactly this as a label image.
- The GUI decodes an **entire** layer to render one whole-plate tile
  (`_load_hdf_layer_rgb`). Multiscale pyramids reduce that to a small read.
- Stage 2's sidecar is a workaround for an HDF constraint that Zarr does not
  have.
- Interoperability: a PhenoTypic output directory becomes readable by napari,
  QuPath, Vizarr, and any NGFF tool without a PhenoTypic install.

### Prior art in-repo

[`2026-08-17-flat-metadata-namespace/design.md`](../2026-08-17-flat-metadata-namespace/design.md)
is approved and binding here. Its decisions #2 (copy-on-write migration), #3
(stored-data compatibility is permanent), #6 (separate version markers), and #7
(the `deliverables/metadata.csv` snapshot is immutable provenance) all constrain
this design and are carried forward explicitly below.

---

## 1. Store layout

```text
results/<dataset>/zarr/<stem>.ome.zarr/
├── zarr.json          ome:        {version:"0.5", "bioformats2raw.layout":3}
│                      phenotypic: {…}                       ← §2, source of truth
├── OME/
│   ├── zarr.json      ome: {version:"0.5", series:["rgb","gray","detect_mat"]}
│   └── METADATA.ome.xml            MetadataOnly OME-XML, imported tags + REMBI
├── rgb/                            omitted entirely when rgb is empty
│   ├── zarr.json      ome: {version, multiscales, omero}
│   ├── 0/ 1/ 2/ 3/                 uint8|uint16, axes (c,y,x)
│   └── labels/
│       ├── zarr.json  ome: {version, labels:["objmap"]}     ← commit marker, §3
│       └── objmap/
│           ├── zarr.json  ome: {version, multiscales, image-label}
│           └── 0/ 1/ 2/ 3/         uint16, nearest-neighbour downsampled
├── gray/              zarr.json + levels, axes (y,x)
└── detect_mat/        zarr.json + levels, axes (y,x)
```

### 1.1 Series and the primary image

`rgb` is optional. The **primary series** is `rgb` when present and `gray`
otherwise. It is always the first entry of `OME/zarr.json`'s `series` list.

**Labels attach to the primary series.** Attaching them unconditionally to
`gray` would give a fixed path but would render the segmentation overlay against
the wrong image in a generic viewer. The resolved path is recorded in
`phenotypic.layers.objmap`; readers MUST NOT hard-code `rgb/labels/objmap`.

### 1.2 Axes

| Series | Axes | `dimension_names` |
|---|---|---|
| `rgb` | `channel`, `space`, `space` | `["c", "y", "x"]` |
| `gray`, `detect_mat` | `space`, `space` | `["y", "x"]` |
| `objmap` | `space`, `space` | `["y", "x"]` |

NGFF 0.5 requires `dimension_names` on each level array's own `zarr.json`, and
requires ordering time → channel → space. Both hold.

### 1.3 Pyramid

Levels halve until `max(H, W) <= 512`. The count is

```text
levels = ceil(log2(max(H, W) / 512)) + 1        (1 when max(H, W) <= 512)
```

`ceil`, not `floor` — a draft of this spec used `floor`, which terminates one
level early and leaves a 4000×3000 plate's smallest level at 1000×750. The
error was caught by
[`ngff_store_geometry.py`](../../logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py),
claim C1, whose "terminal level ≤ 512px" assertion is retained precisely because
it has already failed once.

Verified level counts: 2048² → 3 levels; 4000×3000 → 4; 6000×4000 → 5.

Image layers downsample by **local mean**. `objmap` downsamples by
**nearest-neighbour** — mean-downsampling a label map fabricates label values
that exist at no level-0 pixel (script claim C5 demonstrates this concretely).
NGFF additionally requires a label image to carry the **same level count** as its
parent; because `objmap` shares the parent's `(y, x)` extent, the formula above
yields parity with no special case (claim C2).

### 1.4 Chunking, sharding, compression

- Chunks: `(1, 1024, 1024)` for `rgb`; `(1024, 1024)` for 2-D arrays.
- Shards: `(C, 4096, 4096)` — the Zarr v3 sharding codec requires the shard
  shape to be an exact multiple of the chunk shape, which `4096 = 4 × 1024`
  satisfies (claim C3).
- Codec: `zstd`, replacing the current `gzip-4`.

**File counts, verified.** For the 4000×3000 reference plate, against a baseline
of exactly 1 HDF file:

| | data files | metadata files | total |
|---|---|---|---|
| Unsharded | 108 | 24 | **132** |
| Sharded | 16 | 24 | **40** |

A 3.3× reduction, rising to 4.5× at 6000×4000. Two honest qualifications:

1. **Sharding does not restore HDF parity.** A sharded store is ~40 inodes per
   image, not ~1. At 10k images that is 400k inodes rather than 10k. It is a
   real improvement over 1.3M unsharded, and it is what makes the format viable
   on Lustre/GPFS, but it is not free.
2. **Metadata files dominate a sharded store** (24 of 40). Every group and every
   array level carries its own `zarr.json`. An earlier estimate of "15–20 files
   sharded" counted only data files and was wrong; the script now asserts that
   figure is refuted so the mistake cannot recur.

---

## 2. Metadata contract

### 2.1 `attributes.phenotypic` — the source of truth

```json
{"attributes": {
  "ome": {"version": "0.5", "bioformats2raw.layout": 3},
  "phenotypic": {
    "store_schema_version": 3,
    "metadata_schema_version": 2,
    "phenotypic_version": "…",
    "image_class": "GridImage",
    "work_id": "…",
    "layers": {"rgb": "rgb", "gray": "gray",
               "detect_mat": "detect_mat", "objmap": "rgb/labels/objmap"},
    "detect_mode": "…",
    "illuminant": "D65",
    "gamma": "sRGB",
    "grid": {"nrows": 8, "ncols": 12,
             "grid_finder": {"class": "…", "params": {…}}},
    "metadata": {
      "protected": {"Metadata_ImageName": "…",
                    "Metadata_ImageType": "Grid",
                    "Metadata_BitDepth": 16},
      "public":   {"Metadata_Strain": "…"},
      "imported": {"TIFF:XResolution": …}
    }
  }}}
```

**Two version markers, not one.** `store_schema_version` describes groups and
arrays; `metadata_schema_version` describes the header namespace. A header-only
migration advances the latter without pretending the layout changed. This is
decision #6 of the flat-metadata design, carried forward verbatim.

**`image_class` and `Metadata_ImageType` remain distinct facts.** `image_class`
(`Image` / `GridImage`) drives loader dispatch, mirroring
[`_io_constants.py:1971`](../../../../src/phenotypic/sdk_/_io_constants.py).
`Metadata_ImageType` (`Base` / `Grid` / `Crop` / `Object` / `GridSection`) is
user-visible schema metadata. They are correlated but not equal — a `GridSection`
is not a `GridImage` — and collapsing them would lose information.

**Metadata keys are canonical flat headers.** The three sections are serialized
under `Metadata_<Label>`, matching the flat-metadata namespace. Semantic
ownership is recovered through `metadata_owner_for_header()` /
`metadata_member_for_header()`, never by prefix parsing, per the project rule.

### 2.2 Write-only OME projection

Derived on every write, **never read back**:

| Source | Projected into |
|---|---|
| `TIFF:XResolution` / `YResolution` | `coordinateTransformations.scale` + `axes[].unit` |
| `Metadata_ImageName` | `multiscales[].name`, `omero.name` |
| `Metadata_BitDepth` | `omero.channels[].window.max` |
| layer identity | `omero.channels[].label` (R/G/B) |

When no resolution tag is available the `scale` is `1.0` per axis with `unit`
omitted, which the spec permits.

Because nothing reads these fields back, they cannot drift from
`attributes.phenotypic`. This is stated as a hard invariant so a future
"convenience" reader is recognised as a spec change, not a refactor.

### 2.3 `OME/METADATA.ome.xml`

A `MetadataOnly` OME-XML document carrying imported TIFF/EXIF tags and a
REMBI-module-grouped view built from the existing
`phenotypic.schema.header_to_module()` map. `bioformats2raw.layout` marks this
SHOULD, not MUST; a store without it remains conforming, so a failure to build
the XML degrades to a warning rather than failing the image.

### 2.4 What is deliberately not mapped

`grid.nrows` / `grid.ncols` are **not** expressed as NGFF `plate` metadata.
NGFF's HCS model requires each well to be a separate image group; PhenoTypic's
grid is a virtual partition of a single array. Forcing the mapping would
multiply the store's group count by `nrows × ncols` for no reader benefit.

`detect_mode`, `illuminant`, `gamma`, and `work_id` have no standard equivalent
and live only under `phenotypic`.

---

## 3. Write path and commit protocol

### 3.1 Modules

- **New** `src/phenotypic/sdk_/ngff_.py`, paralleling `hdf_.py`: layout
  constants, pyramid builder, chunk/shard policy, the `attributes.phenotypic`
  models, and the atomic-promote helper.
- `_image_io_handler.py` gains `save2zarr` / `load_zarr` / `load_layer_zarr`,
  replacing the three HDF equivalents.
- `_grid_image_handler.py` writes `phenotypic.grid` where it currently writes
  the `/grid/` subgroup.
- `_io_constants.py`: `DIR_HDF` → `DIR_ZARR`, `dataset_hdf_dir` →
  `dataset_zarr_dir`, and a `zarr_store_path(output_dir, dataset, stem)` helper
  so no caller hand-joins `f"{stem}.ome.zarr"`.

### 3.2 Stage 1 / single-pass commit

1. Build `.<stem>.ome.zarr.part/` as a **sibling** of the target, guaranteeing
   same-filesystem rename.
2. Write all arrays and chunks; then `OME/zarr.json`; then the root `zarr.json`
   **last**. The root carries `work_id` and the `layers` map, so an interrupted
   store has no valid root and reads as absent.
3. `os.replace(part, final)`. When `final` already exists it is first moved to
   `.<stem>.ome.zarr.trash.<pid>` and deleted after the promote succeeds.

A crash anywhere in that window leaves no valid `final`, and resume regenerates
the image. The operation is idempotent and never half-publishes.

### 3.3 Resume validity

`valid_staged_store(path)` replaces `valid_staged_hdf` with matching semantics:

- root `zarr.json` parses and carries `phenotypic.store_schema_version`;
- every series named in `phenotypic.layers` opens as a Zarr array group;
- level-0 `(y, x)` shapes agree across series;
- `phenotypic.work_id` equals the expected id, when one is expected
  (replacing `staged_hdf_matches_work_id`).

### 3.4 Stage 2 loses the sidecar

Stage 2 opens the promoted store, writes `<primary>/labels/objmap` and its
chunks, then writes `<primary>/labels/zarr.json` carrying
`ome.labels: ["objmap"]` **last**.

That list is the **only** discovery path for the label, so a partially written
`objmap` array is invisible until the list lands. This makes it an exact
functional replacement for today's `sidecar_exists()` check in the resume
planner. `_cli_sidecar.py`, the `.npy` sidecar, and the Stage-3 merge-and-delete
step are all removed.

**Invariant.** The root `zarr.json` is committed before Stage 2 runs, so
`phenotypic.layers.objmap` names a path that may not yet exist. **Label presence
is determined by `ome.labels`, never by the `layers` map.**

Stage 3 is unchanged: measure, write parquet, write the existing atomic
completion marker.

### 3.5 Concurrency

Zarr has no locking. Two properties keep this safe: Stage 2 writes only into a
group no other stage writes, and the labels list makes partial state
undiscoverable. Concurrent *readers* (a running GUI) see either no label or a
complete one.

---

## 4. Read paths

### 4.1 Image loaders

`Image.load_zarr` / `GridImage.load_zarr` read the root `zarr.json`, dispatch the
class from `phenotypic.image_class`, load level 0 of each series in `layers`,
restore the three metadata sections, then apply `phenotypic.grid`.
`load_layer_zarr(path, layer, level=0)` replaces `load_layer_hdf5`.

### 4.2 GUI tile server

[`_load_hdf_layer_rgb`](../../../../src/phenotypic/gui/_shared/tiles.py) becomes
`_load_zarr_layer_rgb(store, key, layer, target_px)`, selecting the smallest
pyramid level that still covers `target_px`. This is where the pyramid pays for
itself: a whole-plate tile stops decoding a full-resolution layer.

`_crop_window` keeps its shape — Zarr slices a window off level 0 exactly as
h5py does.

**Cache-key correctness.** The existing `lru_cache` keys on the `.h5` file's
`st_mtime_ns`. A directory's mtime does **not** change when a nested chunk file
is rewritten, so keying on the store directory would serve stale tiles. The key
MUST be the **root `zarr.json`'s `st_mtime_ns`** — which, because §3.2 writes it
last on every commit, is exactly the right invalidation token.

### 4.3 Unchanged externally

`--mode process --layer {rgb|gray|detect_mat|objmap}` keeps its current
behaviour and output formats; it reads level 0 and writes the same integer TIFF /
float TIFF / 16-bit label PNG.

### 4.4 Affected modules

24 files reference `.h5`, `h5py`, or `dataset_hdf_dir`. Beyond the loaders and
constants above, the substantive ones are:

| File | Change |
|---|---|
| `_cli_directory_scanner.py` | `*.h5` glob → `*.ome.zarr` directory scan |
| `_cli_recompile_slurm_scripts.py` | same |
| `_cli_staged_{resume,strategy,workers,controller,slurm_worker}.py` | store paths + `valid_staged_store` |
| `_cli_output_manager.py` | `save_image_hdf` → `save_image_store`, §3.2 promote |
| `_cli_process_single.py`, `_cli_execution_strategies.py` | loader swap |
| `gui/_shared/tiles.py`, `gui/builder/_preview_{cache,tiles}.py` | §4.2 |
| `gui/results_viewer/_output_root.py` | discovery |
| `_cli_readme_generator.py` | documents the new layout |
| `tune/_tune_cli/_run.py` | loader swap |

---

## 5. Migration utility and dead-code retirement

### 5.1 Public API

```python
migrate_hdf_to_zarr(src: Path, dst: Path | None = None) -> Path
migrate_run_hdf_to_zarr(output_dir: Path, *, keep_source: bool = True) -> MigrationReport
```

The run-level form walks `results/*/hdf/*.h5` → `results/*/zarr/*.ome.zarr`,
reusing the existing v1-flat and v2-grouped HDF loaders and applying the
metadata-schema migration in memory before writing. Sources are retained by
default.

Binding constraints inherited from the flat-metadata design:

- **#3, stored-data compatibility is permanent.** The migration keeps accepting
  historical per-topic metadata headers indefinitely.
- **#7, the metadata snapshot is immutable.** `deliverables/metadata.csv` is
  never rewritten by migration; legacy headers are normalized in memory only.

### 5.2 Copy-on-write gets cheaper

Decision #2 of the flat-metadata design requires header migration to build a
validated sibling and publish atomically. Today
[`_metadata_migration.py`](../../../../src/phenotypic/sdk_/_metadata_migration.py)
copies an entire multi-hundred-megabyte HDF to change header strings. Under Zarr
a header-only migration rewrites one small `zarr.json`. The guarantee survives
intact, applied to a few kilobytes rather than the whole image, and
`_cli_recompile_metadata_migration*.py` simplifies accordingly.

### 5.3 Retirement

Audit finding: **only `HDF.save_array2hdf5` is live.** The DataFrame half of
[`sdk_/hdf_.py`](../../../../src/phenotypic/sdk_/hdf_.py) —
`preallocate_series_layout`, `save_series_new` / `_update` / `_append`,
`load_series`, `preallocate_frame_layout`, `save_frame_new` / `_update` /
`_append`, `load_frame`, and the fixed-length-string encode/decode helpers that
exist only to serve them — has no caller in `src/`, no caller in `tests/`, and no
documentation reference. Measurements go to parquet.

`hdf_.py` retains only what the legacy reader needs: `_open_hdf_with_recovery`,
`_clear_hdf_consistency_flags`, and the group-navigation helpers. The remainder
(~1,700 of 1,984 lines) is deleted. `h5py` stays a dependency, read-only, for
migration.

`_cli_sidecar.py` is deleted per §3.4.

---

## 6. Packaging, Python floor, CI

- `requires-python = ">=3.11, <3.13"`.
- Add `zarr>=3.0`. Version markers resolve it to 3.1.6 on Python 3.11 and 3.3.x
  on 3.12 with no pinning.
- Retain `h5py` for migration.
- CI: drop `3.10` from `run-pytest.yml`, `run-pytest-full.yml`, and
  `package-integrity.ci.yml`; move `publish_to_pypi.yml` from 3.10 to 3.11.

**`ome-zarr-models` is a dev/test dependency, not a runtime one.** It is the
natural pydantic model for NGFF metadata, but it caps `pydantic<2.13` against
the project's `pydantic>=2.12.5` floor — a one-release-wide resolvable band that
would block adopting pydantic 2.13 until upstream widens it. The `zarr.json`
payloads are small and fully specified by §1 and §2, so they are hand-built at
runtime and validated by `ome-zarr-models` in the conformance tests (§7). This
keeps the validation guarantee, drops a runtime dependency, and leaves pydantic
unconstrained.

`ome-zarr` itself is **not** adopted: version 0.18 pulls dask, fsspec, aiohttp,
scikit-image, rangehttpserver, toolz, and Deprecated for functionality this
design does not use.

---

## 7. Testing

- **Round-trip.** `Image` and `GridImage` → store → back. All four layers
  bit-exact, all three metadata sections equal, `nrows` / `ncols` /
  `grid_finder` equal, and `image_class` and `Metadata_ImageType` both
  preserved independently.
- **NGFF conformance.** Every written store validates against
  `ome-zarr-models`. This is the gate that decides whether the word "OME-Zarr"
  in this document is true, so a validation failure fails the suite — it is
  never downgraded to a warning or skipped on a missing optional dependency.
- **Third-party read.** Open a written store with `napari-ome-zarr` (already a
  test extra) and assert three series plus one label are enumerated.
- **Commit protocol.** Interrupt after chunks but before the root `zarr.json`;
  the store must read invalid and resume must regenerate. Prove the test can
  fail by reversing the write order.
- **Label discovery.** Write `objmap` chunks without the labels list; assert the
  label is undiscoverable, then write the list and assert it appears.
- **Pyramid correctness.** Assert no label value at level *n* is absent from
  level 0. Mutate the label downsampler to `mean` and confirm the assertion
  fails — the mutation and its expected failure are already demonstrated by
  script claim C5.
- **Migration.** Golden legacy fixtures in both v1-flat and v2-grouped layouts
  migrate to stores equal to freshly written ones.
- **Cache invalidation.** Rewrite a store in place under a live tile cache and
  assert the served tile changes (guards the §4.2 directory-mtime trap).

Per project test-integrity rules, no check in this suite may skip on a missing
fixture or optional dependency; a check that cannot run must fail.

### Logic validation

[`ngff_store_geometry.py`](../../logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py)
re-derives the pyramid level count, label level parity, shard/chunk
divisibility, file counts, and the label-downsampling requirement from numpy
alone. It has already refuted two claims made during design (the `floor` level
formula and a "15–20 sharded files" estimate) and both refutations are retained
as regression assertions.

---

## 8. Non-goals

- **Ingesting third-party OME-Zarr as pipeline input.** The projection is
  write-only (decision #5). Bidirectional reading is a separate project.
- **Measurements in `image-label.properties`.** Considered and deferred
  (decision #9).
- **Changing the measurement storage format.** Parquet is unaffected.
- **HCS `plate` metadata for the colony grid** (§2.4).
- **Adopting NGFF 0.6.** The `scene` layout in 0.6rc0 is structurally this
  design's directory tree, so migration will mean adding a `scene` object to
  the root, not moving arrays. Not in scope now.

## 9. Open questions

**OQ1 — `bioformats2raw.layout` is transitional.** The 0.5 spec states outright
that a future version will replace this layout with explicit metadata. The
directory tree survives that transition (§8), but the root `zarr.json` will need
rewriting. Accept the sunset, or wait for 0.6's `scene` to stabilise?

**OQ2 — Does dropping Python 3.10 have a known downstream cost?** The CI matrix
change is mechanical, but if a target HPC environment pins 3.10 this becomes a
deployment blocker rather than a packaging edit. Needs confirmation against the
actual HPCC module set.

**OQ3 — Sharded stores are still ~40× HDF's inode count** (§1.4). Is that
acceptable on the target filesystem, or does it warrant a follow-up
investigation into fewer pyramid levels or consolidated metadata? Note that
Zarr v3 consolidated metadata speeds reads but does **not** reduce inode count,
since per-group `zarr.json` files still exist.

**OQ4 — `.ome.zarr` vs `.zarr` suffix, and `zarr/` vs reusing `hdf/`.** The
proposed `results/<dataset>/zarr/<stem>.ome.zarr/` changes two path segments at
once. Confirm this is wanted, given `_cli_readme_generator.py` and any external
scripts users have written against the current layout.

**OQ5 — Migration ergonomics.** Should `migrate_run_hdf_to_zarr` be exposed as a
CLI subcommand in addition to the `sdk_` function, and should a run whose output
directory contains only `.h5` results be auto-migrated on next invocation or
fail with a pointer to the utility?
