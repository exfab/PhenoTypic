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

## Reading an image's figures

When a CLI run's pipeline includes a per-image plot (a `PlotImage`, such as
`MeasureSymZones` listed in `plots=`), the store also holds that plot's
rendered figures. `--mode process --process-format zarr` stores them too. A
dashboard can therefore show a plate's figures from the store alone:

```text
plate_01.ome.zarr/
├── zarr.json                      <- attributes.phenotypic.figures describes everything below
└── figures/
    ├── 2026-09-22-3f9a1c2b7e04/   <- one run: {UTC date}-{first 12 hex of the pipeline sha256}
    │   └── sym/                   <- one plot binding
    │       └── default.plotly.json
    └── 2026-10-03-a07bc5e91d22/   <- a later run with another pipeline
```

Each folder level is an empty Zarr group, so Zarr readers see an ordinary
group tree, and `figures` is not listed in `ome.series`. Nothing is read from
the folder names. Read the store as follows:

1. **Read the descriptor.** Open the root `zarr.json` and take
   `attributes.phenotypic.figures`. A store without the key has no figures:
   it was written before this feature, or by a pipeline with no per-image
   plot. `schema_version` is `1`. Refuse a version you do not know rather than
   guessing at its layout.
2. **Choose a run.** `runs` maps each run folder to its entry. Every entry
   records the `date` the run started and the `pipeline_sha256` of the pipeline
   file it ran. An entry written by any CLI mode except `--mode process` also
   records the run's initial call, as `initiated_at_utc` and `initiated_pid`.
   There is no "latest" pointer: choose by date, by pipeline, or both. A later
   run never deletes an earlier one, so older entries stay beside newer ones.
   The exception is `--overwrite`, which starts the output folder over.
3. **Pick a file by `media_type`.** Each binding lists `pages`, and each page
   lists `files`. Dispatch on `media_type`, never on the file extension:
   `application/vnd.plotly.v1+json` is a Plotly figure (`plotly.io.from_json`),
   and `image/png` is a PNG. Skip a type you cannot display.
4. **Verify the `sha256`.** The root records each file's digest, so a file
   that does not match has changed since the store was written.

A run entry also says what is missing. `failed` lists each figure, page or
format that could not be rendered, with its error. `unavailable` lists the
plots that can only be drawn in the process that applied their operation, such
as `CalibrateColorRpcc`'s tile overlay, and that this run could not draw. An
earlier run's folder may still hold that figure.

This needs only the standard library:

```python
import hashlib
import json
from pathlib import Path

store = Path("plate_01.ome.zarr")
root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
figures = root["attributes"].get("phenotypic", {}).get("figures")
if figures is not None and figures["schema_version"] != 1:
    raise ValueError(f"unknown figures schema_version {figures['schema_version']}")


def choose_run(runs, pipeline_sha256=None):
    """The most recent run, optionally only among one pipeline's runs.

    Pass hashlib.sha256(Path("pipeline.json").read_bytes()).hexdigest()
    to restrict the choice to runs of that pipeline file.
    """
    candidates = [
        run
        for run in runs.values()
        if pipeline_sha256 is None or run["pipeline_sha256"] == pipeline_sha256
    ]
    return max(candidates, key=lambda run: run["date"], default=None)


displayable = {"application/vnd.plotly.v1+json", "image/png"}
run = choose_run(figures["runs"]) if figures is not None else None
for binding_id, binding in (run["bindings"] if run is not None else {}).items():
    for page in binding["pages"]:
        for entry in page["files"]:
            if entry["media_type"] not in displayable:
                continue
            data = (store / entry["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                print("changed since written, skipped:", entry["path"])
                continue
            print(binding_id, page["key"], entry["media_type"], len(data))
```

Several runs can share a date when the pipeline changed within a day. Filter
by `pipeline_sha256` to tell them apart.

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
   describing content it does not have. Figure run folders from earlier runs
   are carried into that new `.part` (hard-linked, or copied where links are
   refused), never written into the old directory.

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
