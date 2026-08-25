---
name: working-with-ome-zarr
description: Use when creating, reading, validating, or migrating PhenoTypic OME-Zarr 0.5 stores on Zarr v3, especially when metadata placement, image series, or multiscale axes are involved.
---

# Working with OME-Zarr

Released stores are interoperable data products, not internal cache formats. Keep
project state separate from OME semantics and make the Zarr v3 and OME-Zarr 0.5
views agree.

## Store contract

| Concern | Required representation |
| --- | --- |
| Project provenance | Root `zarr.json` → `attributes.phenotypic` |
| OME semantics | `OME/zarr.json` and `OME/METADATA.ome.xml` |
| Original pixels | A normal root-level multiscale image group, listed in `OME/zarr.json` → `ome.series` |
| Scale metadata | Each multiscales dataset path and its array `dimension_names` describe the same axes |

`attributes.ome` is reserved OME-Zarr metadata. Never put PhenoTypic-specific
provenance, pipeline history, or arbitrary application data there; add it under
`attributes.phenotypic` at the store root instead.

The original image is not a private staging artifact: write it as an ordinary
root image group with normal multiscales metadata, then register that group in
the collection descriptor's `ome.series`. Keep its path, series entry, dataset
paths, axes, and array dimension names mutually consistent.

`OME/METADATA.ome.xml` is the collection's semantic projection for OME-aware
readers. It may describe images, channels, axes, and acquisition semantics; it
is not a processing journal. Store execution logs, resumability state, and
arbitrary provenance only in the application namespace or other explicitly
owned project artifacts.

## Before publishing or changing a store

1. Check the project's OME-Zarr invariants, including root metadata ownership,
   `ome.series` registration, and every multiscale level's path/axis agreement.
2. Open the result with an independent reader or validator that supports the
   released OME-Zarr 0.5 / Zarr v3 combination. Do not treat the writer's own
   successful reopen as independent validation.
3. Record or communicate ecosystem compatibility: some otherwise useful
   viewers and libraries still lack complete OME-Zarr 0.5 or Zarr v3 support.
   Preserve the standard form; provide a compatibility path only when the
   consuming workflow requires it.

## Common mistakes

- Adding arbitrary keys to `attributes.ome` because they look OME-related.
- Leaving the original image unregistered, or registering a processing-derived
  object instead of its root image group.
- Declaring `multiscales` axes that disagree with the target array's
  `dimension_names` or dataset dimensionality.
- Treating `METADATA.ome.xml` as an append-only run log.
- Claiming interoperability after validation only with the same library that
  wrote the store.
