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
| Object measurements | `tables/<n>-<sha12>/measurements/table.parquet`, one folder per measuring pipeline, described by root `attributes.phenotypic.tables.runs.<n>-<sha12>.measurements` (tables `schema_version: 2`) |
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
arbitrar## Embedded measurement table

Tables live in one folder per pipeline that measured the store (#281 spec
D1-D7):

```text
tables/zarr.json
tables/<n>-<sha12>/zarr.json
tables/<n>-<sha12>/measurements/zarr.json
tables/<n>-<sha12>/measurements/table.parquet
tables/<n>-<sha12>/metadata/pht-metadata.parquet       (when metadata was joined)
tables/<n>-<sha12>/embeddings/<model>/{objects,grid}[-<name>]/    (optional)
```

An embedding region is `objects` or `grid`, or `objects-<name>` / `grid-<name>`
for an embedder with a `name` (a lowercase slug, `^[a-z0-9][a-z0-9_]{0,31}$`),
so one run can hold several embedders of one model and region (rig1:
`grid-wells`, `grid-agar`).
`split_embedding_key` (`sdk_/_table_runs.py`) is the one rule for regions.

Every level is a Zarr v3 group document; every subtable is optional. The
Parquet is the authoritative per-image object table; forward, staged Stage 3,
and measure runs must not also publish
`results/<dataset>/measurements/<stem>.parquet`.

- **Folder name.** `<n>` is a store-local call-order counter: a new folder
  takes 1 + the highest number among the store's runs and `upstream` entries,
  and numbers are never reused. `<sha12>` is the first 12 hex of the pipeline
  file's sha256 the journal records; `<n>-unknown` (with
  `pipeline_sha256: null`) when none is known. Never fabricate a digest.
- **Which folder a writer owns.** The folder with its pipeline's sha256 is
  replaced, keeping its number, with every file a new inode; otherwise a new
  folder is added. An unknown sha replaces the highest `-unknown` folder.
  Every writer (save2zarr, process `_save_store`, the measure rewrite) carries
  every other folder unchanged; a write with no tables of its own carries all
  of them. Only `--overwrite` deletes folders. A save over a store whose root
  exists but cannot be parsed is **refused** rather than carrying nothing; only
  migrate's rebuild from HDF passes `save2zarr(replace_unreadable=True)`.
- **`upstream` is replaced, not merged.** A writer that read its input
  (`upstream=` given, even `{}`) records exactly that input's runs; Stage 3
  and the measure rewrite pass none and carry the block. A run id or order
  shared between `runs` and `upstream` is logged, never renumbered.
- **Which folder a reader uses.** Finalize, recompile, measure and migrate
  read the folder the image's record certifies (its `measurements` artifact
  path); finalize logs a warning when those folders span several pipelines.
  A reader without a record reads `latest_measurement_run(store)`: the
  highest-numbered folder that holds `measurements`. The journal-keyed resume
  probe (`journal_table_run`) reads a `legacy` last application as the folder
  migrate named (`v1_table_pipeline_sha`), not `-unknown`.
- **Reach folders through `phenotypic.sdk_._table_runs`** (`list_table_runs`,
  `latest_measurement_run`, `owned_table_run_id`, `TableRun.
  measurements_path()`), never by joining `tables/...` by hand.
  `MEASUREMENT_TABLE_RELATIVE_PATH` is the **v1** flat path, kept only for
  migrate and legacy reads.

The root descriptor `attributes.phenotypic.tables` is:

```json
{"schema_version": 2,
 "runs": {"<n>-<sha12>": {"order": n, "pipeline_sha256": "<64 hex> | null",
                          "measurements": {...}, "metadata": {...},
                          "metadata_table": {"snapshot_sha256": "..."},
                          "embeddings": {"<model>/<region>": {...}}}},
 "upstream": {"<n>-<sha12>": {"source": "...", "root_sha256": "...",
                              "pipeline_sha256": "..."}}}
```

The root-level `metadata_table` block of v1 moved into each run entry. Each
run's `measurements` descriptor has:

- `schema_version: 1`
- `type: "object_measurements"`
- `format: "parquet"`
- `path: "tables/<n>-<sha12>/measurements/table.parquet"`
- `measurement_columns`: ordered columns before external metadata is joined
- `target.column: "Object_Label"`
- `target.path`: the store's objmap label path

Each `measurements/` group's `zarr.json` carries ngio's feature-table
attributes (`type: "feature_table"`, `region.path` relative to the run
folder, `index_key: "Object_Label"`).

**What ngio actually reads** (ngio 1.1.5, measured 2026-11-18; check and result
in `docs/superpowers/plans/2026-11-12-dvc-embeddings-refactor/ngio_check/`):
- `ngio.open_table("<store>/tables/<run>/measurements")` returns a
  `FeatureTableV1`, and its `reference_label` resolves to `objmap`.
- ngio does **not** discover the tables. Opening the `rgb` image lists
  `tables: []`, because ngio looks under the image group for a `tables` list
  attribute. Opening the store root fails, because ngio expects a single image
  there, not a collection.

This is the limit spec D4 anticipated: keep the attributes and do not flatten
the layout to suit ngio. Point an ngio user at the run's `measurements` group.

A v1 store (flat `tables/measurements/`, no `schema_version` under `tables`)
is converted by `--mode migrate`, and forward modes refuse it (spec D5, D10).
Record-less readers still read one read-only, through
`_table_runs.declared_measurement_tables`, keyed strictly on the tables schema;
`record_less_measurement_table` is the one owner of which table such a reader
uses (its last element).

The Parquet schema metadata records the join status (`not_requested`,
`joined`, or `no_common_keys`), ordered join keys, the metadata snapshot
SHA-256 and the ordered baseline measurement columns. A forward store's
measurement table is **un-joined**: measured rows and `measurement_columns`
only, with the run's metadata in `tables/<run>/metadata/pht-metadata.parquet`
beside it (`TableRun.metadata_path()`). A
pre-inversion (migrated) table may still carry joined metadata; aggregation
projects every table onto its descriptor's `measurement_columns` at read
(`project_embedded_measurement_table`), which is the identity for an un-joined
one.

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

A measure update also uses the root-last store transaction: the part rewrites
that one run folder as new inodes and carries every other folder. There is no
in-place payload replacement; nothing writes into a promoted store without
rewriting its root last. Do not recompute or rewrite pixel arrays. Measure mode
then republishes the image's record so it certifies the folder it wrote.
`--mode recompile` is aggregate and finalize only and writes **no** store byte
(`tests/unit/cli/test_recompile_no_store_writes.py`); it reads the folder each
record certifies.

A completion marker binds both the store root and embedded Parquet hashes.
Missing, unreadable, or changed table bytes invalidate measurement authority.
Finalization refuses a tree mixing embedded tables with legacy external
Parquets. Mixed metadata snapshot digests or per-store join keys are not
refused: rolling input produces them on purpose and the join is global, so
they are provenance, not input.

## Aggregation and migration

`deliverables/master_measurements.parquet` is the concatenation of the
authorized embedded tables, each projected onto its own descriptor's
`measurement_columns`: **un-joined**, intrinsic identity only, before post
operations. The mirror `deliverables/measurements.*` joins the metadata once,
globally, on the columns the master and the metadata table share, appends the
metadata-only phantoms, sets `QC_MetadataOnly=false` on measured rows and
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
