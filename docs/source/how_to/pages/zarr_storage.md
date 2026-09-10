# Store Results in OME-Zarr

Save processed images and their intermediate states as OME-Zarr (NGFF 0.5 /
Zarr format v3) stores — one directory per image — for efficient storage,
partial reads, and interoperability with third-party viewers.

## Save a Processed Image

```python
import phenotypic as pht

image = pht.Image.imread("plate.png")
# ... process image ...
image.save2zarr("processed_plate.ome.zarr")

restored = pht.Image.load_zarr("processed_plate.ome.zarr")
```

The store carries the image layers as named sibling multiscale series
(`rgb`, `gray`, `detect_mat`), the object map as a first-class NGFF label
image, and all PhenoTypic state in a namespaced `attributes.phenotypic`
block. `rgb` is omitted entirely when empty, in which case `gray` becomes the
primary series and the label attaches under it.

Reading one layer without reconstructing a whole `Image` — and at whatever
pyramid level you need — is a single call:

```python
full_res = pht.Image.load_layer_zarr("processed_plate.ome.zarr", "objmap")
thumbnail = pht.Image.load_layer_zarr("processed_plate.ome.zarr", "gray", level=3)
```

## Save Pipeline Intermediates

Capture the image state after each pipeline operation:

```python
pipeline = pht.ImagePipeline(ops=[...])
result = pipeline.apply_with_intermediates(
    image,
    output_dir="intermediates/"
)
```

Each intermediate is saved as a separate `*.ome.zarr` store in `output_dir/`,
named after the operation that produced it. To write a chosen subset of
layers directly — the GUI builder's node previews do exactly this — use
`save_intermediate_zarr`:

```python
image.save_intermediate_zarr("node_03.ome.zarr", layers=("detect_mat",))
```

## Why OME-Zarr

- **Pyramid levels:** every series is stored as a multiscale pyramid, so a
  thumbnail or a zoomed-out view reads a small array instead of decoding a
  full plate.
- **Partial reads:** the store is chunked and sharded, so a crop reads only
  the chunks it overlaps — no whole-layer decode.
- **Interoperable:** the store is a conformant NGFF 0.5 image, readable by
  napari (`napari-ome-zarr`), Vizarr, Fiji, and anything else that speaks
  OME-Zarr, without PhenoTypic installed.
- **Self-describing:** metadata, image arrays, masks, and object maps live
  together in one directory, with PhenoTypic state namespaced under
  `attributes.phenotypic` so it never collides with the OME metadata.
- **Crash-safe writes:** each store is built in a `.part` sibling and
  promoted by directory rename, with the root `zarr.json` written last — an
  interrupted write leaves no valid root and reads as absent, never as
  partial.

## What the store looks like on disk

One image is one directory. Everything about it — pixels, label image, metadata,
grid state — is inside:

```text
plate_01.ome.zarr/
├── zarr.json              <- root: bioformats2raw.layout + attributes.phenotypic
├── OME/
│   ├── METADATA.ome.xml   <- write-only OME projection; never read back
│   └── zarr.json
├── rgb/                   <- primary series (omitted entirely when empty)
│   ├── zarr.json          <- ome.multiscales for this series
│   ├── 0/                 <- level 0, full resolution
│   │   ├── zarr.json
│   │   └── c.0.0.0        <- one shard; chunks live inside it
│   ├── 1/                 <- level 1, half resolution
│   └── labels/
│       ├── zarr.json      <- ome.labels
│       └── objmap/        <- the object map, a first-class NGFF label image
│           ├── zarr.json
│           ├── 0/
│           └── 1/
├── gray/                  <- same shape as rgb
└── detect_mat/
```

Pyramid depth is not configurable: levels halve until the longest edge is
512 px or smaller, derived purely from the level-0 shape. Two stores in one
output tree therefore cannot disagree about geometry.

**`rgb` is the primary series only when present.** An enhancement-only or
delta store has no `rgb`, in which case `gray` becomes primary and
`labels/objmap` attaches under `gray/` instead. Resolve the label through
`attributes.phenotypic.labels`, never by hard-coding `rgb/labels/objmap`.

**`attributes.phenotypic` is the only thing PhenoTypic reads back.** The `ome`
blocks and `METADATA.ome.xml` are a write-only projection for other tools; on
load, the namespaced block is the sole source of truth. That is what keeps a
third-party tool rewriting OME metadata from changing how PhenoTypic reads the
store.

## Open a store in another tool

The store is a conformant NGFF 0.5 image, so it opens without PhenoTypic
installed:

```bash
# napari
pip install napari-ome-zarr && napari plate_01.ome.zarr

# Vizarr / any HTTP viewer — serve the directory and point the viewer at it
python -m http.server 8000    # then open the store's URL in the viewer
```

QuPath and Fiji read the same layout through their Bio-Formats/OME-Zarr
readers. In every case the pyramid is what makes a whole-plate view cheap: the
viewer reads a coarse level rather than decoding full resolution.

## In a CLI run

A forward run writes one store per input image:

```text
<output>/results/<dataset>/zarr/<stem>.ome.zarr/
```

Two flags govern storage behaviour:

`--durable-writes` / `--no-durable-writes`
: Whether each store is `fsync`ed before being promoted into place. Unset
  means auto-detect — **on under SLURM, off locally** — and the resolved mode
  is logged at run start. Reach for `--no-durable-writes` when a cluster job
  writes to fast local scratch and you accept losing stores to node loss or
  power failure. A walltime kill does **not** need `fsync`: the kernel
  survives it.

`--mode migrate`
: Accepts a full legacy run, a direct OME-Zarr store, or a process-output tree.
  Full runs convert legacy `.h5` images **in place** and recertify markers;
  direct stores and process trees perform only an explicit root-provenance
  upgrade. Add `--dry-run` to validate without scientific writes, `--njobs N`
  for local parallel root checks, or repeated `--slurm key=value` options for
  the native migration dispatcher. `--delete-sources` applies only to full-run
  HDF conversion and is refused for provenance-only targets. Running migration
  twice changes nothing the second time. See
  [Migrate legacy results and provenance](migrate_ome_zarr.md).

Schema-v2 provenance is an ordered application history. A process store used
as normal CLI input retains its process application and appends a distinct full
application, including both pipeline identities. Only a migrated `legacy`
application may have a null `phenotypic_version`, and only when the historical
version is unavailable; readers must preserve that null.

## Invariants, if you are writing code against a store

Three separate subsystems depend on these, and **none of them can detect a
violation on its own** — so breaking one is silent:

1. **A store is promoted by directory rename, with the root `zarr.json`
   written last.** An interrupted write therefore has no valid root and reads
   as absent, never as partial.
2. **A store is replaced wholesale, never merged into.** A re-publish — a
   re-measure included — builds a new `.part` and replaces the directory. The
   refreshed root is written last there too, so a store is never left
   describing content it does not have.

Because of (1) and (2), both the per-image completion marker and the results
viewer's staleness scan identify a store by its root `zarr.json` alone. Add a
code path that writes into a promoted store *without* rewriting that root and
both start reporting stale data as fresh, with nothing failing to say so. The
guard on the promote itself is
`tests/unit/sdk_/test_ngff_promote.py::test_promote_store_replaces_rather_than_merges`,
which asserts inode identity rather than content — a merge-in-place
implementation leaves the old directory in position with new bytes inside it,
which passes any content comparison.

## Removed

The HDF5 per-image API this page used to document has been removed:

- `Image.save2hdf5`, `Image.load_hdf5`, `Image.load_layer_hdf5`, and
  `Image.save_intermediate_layers` (and their `GridImage` counterparts). Per-image
  storage is now an OME-Zarr (NGFF 0.5 / Zarr v3) store; use `save2zarr`,
  `load_zarr`, and `load_layer_zarr`.
- The DataFrame half of `phenotypic.sdk_.HDF` (`save_series_*`, `load_series`,
  `save_frame_*`, `load_frame`, `preallocate_*`, and their fixed-length-string
  codecs), together with three unrelated statics on the same class —
  `HDF.assert_swmr_on`, `HDF.get_uncompressed_sizes_for_group`, and
  `HDF.close_handle`. These had no remaining call sites. The HDF **read**
  surface is unchanged: the writer/reader properties, the group accessors, and
  `HDF.save_array2hdf5` all survive, because `--mode migrate` is built on them.
- The HDF path constants `DIR_HDF`, `dataset_hdf_dir`, `HdfAttr`,
  `load_image_from_hdf`, and `BundleLayout.hdf_path`. The two that migration
  still needs live on as private helpers inside
  `phenotypic.sdk_._hdf_to_zarr`.

## Migration

Existing `.h5` output directories are converted with:

    uv run python -m phenotypic --mode migrate --output <previous-output-dir>

A run whose output contains only `.h5` results now fails with a pointer to this
command rather than converting as a side effect.

## Requires

- Python 3.11 or 3.12. Python 3.10 is no longer supported.
