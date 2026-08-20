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

## Removed

The HDF5 per-image API this page used to document has been removed:

- `Image.save2hdf5`, `Image.load_hdf5`, `Image.load_layer_hdf5`, and
  `Image.save_intermediate_layers` (and their `GridImage` counterparts). Per-image
  storage is now an OME-Zarr (NGFF 0.5 / Zarr v3) store; use `save2zarr`,
  `load_zarr`, and `load_layer_zarr`.
- The DataFrame half of `phenotypic.sdk_.HDF` (`save_series_*`, `load_series`,
  `save_frame_*`, `load_frame`, `preallocate_*`, and their fixed-length-string
  codecs). These had no remaining call sites.
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
