---
name: working-with-ome-zarr
description: Use when creating, reading, validating, remeasuring, recompiling, or migrating PhenoTypic OME-Zarr 0.5 stores on Zarr v3, especially when metadata placement, image series, multiscale axes, or embedded object-measurement tables are involved.
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
| Object measurements | `tables/measurements/table.parquet`, described by root `attributes.phenotypic.tables.measurements` |
| Per-image figures | `figures/<run_id>/<binding>/<page>.<ext>`, described by root `attributes.phenotypic.figures.runs`; optional, and not listed in `ome.series` |

`attributes.ome` is reserved OME-Zarr metadata. Never put PhenoTypic-specific
provenance, pipeline history, table descriptors, or arbitrary application data
there; add them under `attributes.phenotypic` at the store root instead.

The original image is not a private staging artifact: write it as an ordinary
root image group with normal multiscales metadata, then register that group in
the collection descriptor's `ome.series`. Keep its path, series entry, dataset
paths, axes, and array dimension names mutually consistent.

`OME/METADATA.ome.xml` is the collection's semantic projection for OME-aware
readers. It may describe images, channels, axes, and acquisition semantics; it
is not a processing journal. Store execution logs, resumability state, and
arbitrary provenance only in the application namespace or other explicitly
owned project artifacts.

## Embedded measurement table

A measured image has two Zarr v3 groups and one non-Zarr payload:

```text
tables/zarr.json
tables/measurements/zarr.json
tables/measurements/table.parquet
```

Both `zarr.json` files declare a Zarr v3 group. The Parquet is the authoritative
per-image object table; forward, staged Stage 3, and measure runs must not also
publish `results/<dataset>/measurements/<stem>.parquet`.

The root descriptor at
`attributes.phenotypic.tables.measurements` is stable and has:

- `schema_version: 1`
- `type: "object_measurements"`
- `format: "parquet"`
- `path: "tables/measurements/table.parquet"`
- `measurement_columns`: ordered columns before external metadata is joined
- `target.column: "Object_Label"`
- `target.path`: the store's objmap label path

The Parquet schema metadata records the join status
(`not_requested`, `joined`, or `no_common_keys`), right-join direction
(metadata left, measurements right), ordered join keys, metadata snapshot
SHA-256, and the ordered baseline measurement columns. Preserve every measured
row, exclude metadata-only rows from the embedded table, retain duplicate
metadata-key fan-out with a warning, and leave measurements unchanged with a
warning when no key is shared.

## Per-image figures

A store whose pipeline binds a `PlotImage` also holds that plot's rendered
figures, one folder per run: `figures/<run_id>/`, where `<run_id>` is
`{UTC run date}-{first 12 hex of the pipeline sha256}`. Every level is an empty
Zarr v3 group document, like `tables/`. The descriptor at
`attributes.phenotypic.figures` is `{"schema_version": 1, "runs": {run_id:
entry}}`. Each entry carries `date`, `pipeline_sha256`, `bindings` (pages, and
each page's `files`), `failed` and `unavailable`, plus `initiated_at_utc` and
`initiated_pid` except in a process-mode store. **The media type is the
contract**: a consumer dispatches on each file's `media_type`
(`application/vnd.plotly.v1+json` or `image/png`), never on its extension or
folder name. The root records each file's `sha256`, so the completion marker,
which digests the root, binds them. Continuation never re-reads figure bytes,
but the copy-out to `deliverables/plots/` verifies each file before it copies
it. **Run folders are never wiped.** Every rewrite carries the other runs'
folders and descriptor entries across unchanged: a full, Stage 1, Stage 3 or
process save over an existing store, and a measure rewrite. The one exception
is `--overwrite`, which deletes the whole output tree before the run. Neither
`runs` nor `bindings` has a meaningful key order: the measure rewrite serializes
the root with sorted keys. The key is additive and optional, so
`store_schema_version` is not bumped. Test for the key's presence. A measure
rewrite replaces only its own run folder. It first removes that folder from the
`.part`, because the part's files are hard links into the live store, and then
writes every file as a new inode. A descriptor whose figures `schema_version`
this writer does not know is carried untouched with its files, and no run is
added to it.

## Transaction and marker ordering

For a new or descriptor-incompatible store, build every array, group, table,
descriptor, and OME projection in a unique sibling `.part` directory. Validate
the temporary Parquet, write root `zarr.json` last, promote the directory, and
publish the image completion marker last.

For a descriptor-compatible measure or recompile update, validate a
same-directory temporary Parquet and atomically replace the payload. If the
descriptor changes, use the root-last store transaction. Do not recompute or
rewrite pixel arrays. Recompile projects the old table to the descriptor's
baseline columns before joining the effective metadata snapshot, replaces each
table, and refreshes that image's marker before aggregate publication.

A completion marker binds both the store root and embedded Parquet hashes.
Missing, unreadable, or changed table bytes invalidate measurement authority.
Finalization must reject a mixture of metadata snapshot digests or ordered join
keys.

## Aggregation and migration

`deliverables/master_measurements.*` is the exact concatenation of authorized
embedded tables: joined measured rows only, before post operations. The mirror
`deliverables/measurements.*` appends the external metadata anti-join once
using the recorded keys, sets `QC_MetadataOnly=false` on measured rows and
`true` on phantoms, then applies post operations.

`--mode migrate` imports legacy external per-image Parquets after their image
stores exist. Preserve HDF and Parquet sources by default. Delete a source only
after the embedded payload validates and its marker is published. Dry runs,
retries, HDF-only images, already embedded stores, and interrupted partial
migrations must remain safe and idempotent.

## Before publishing or changing a store

1. Check root metadata ownership, `ome.series` registration, every multiscale
   level's path/axis agreement, and the embedded table descriptor when present.
2. Read the Parquet independently with PyArrow or DuckDB and confirm its schema
   metadata, ordered baseline, and `Object_Label` target.
3. Verify the image completion marker rejects a missing or corrupted table.
4. Open the image result with an independent reader or validator that supports
   OME-Zarr 0.5 / Zarr v3. The writer's own successful reopen is not independent
   validation.
5. Record ecosystem compatibility; some viewers still lack complete OME-Zarr
   0.5, Zarr v3, or embedded Parquet support. Preserve the standard form and add
   a compatibility path only when a consuming workflow requires it.

## Common mistakes

- Adding arbitrary keys or the table descriptor to `attributes.ome`.
- Writing the embedded table and an external per-image Parquet.
- Recording post-join columns as `measurement_columns`.
- Re-discovering join keys from an already-joined master instead of reading the
  Parquet provenance.
- Rewriting pixel arrays for measure or recompile.
- Publishing the marker before the table/root transaction is complete.
- Leaving the original image unregistered, or registering a processing-derived
  object instead of its root image group.
- Declaring `multiscales` axes that disagree with the target array's
  `dimension_names` or dataset dimensionality.
- Treating `METADATA.ome.xml` as an append-only run log.
- Claiming interoperability after validation only with the library that wrote
  the store.
