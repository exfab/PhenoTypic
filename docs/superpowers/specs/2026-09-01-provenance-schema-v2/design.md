# Provenance schema v2

**Date:** 2026-09-01
**Status:** Approved for implementation
**Scope:** Image provenance journals, process/full/programmatic application
boundaries, and explicit migration of schema-v1 journals.

## Contract

The canonical journal is:

```json
{
  "schema_version": 2,
  "status": "complete",
  "original_filename": "plate.tiff",
  "applications": [
    {
      "sequence": 1,
      "kind": "process",
      "phenotypic_version": "0.19.0",
      "input_filename": "plate.tiff",
      "status": "complete",
      "pipeline": {"source_path": "pipeline.json", "sha256": "..."},
      "retry_base_length": 0,
      "operations": []
    }
  ]
}
```

- `kind` is one of `process`, `full`, `programmatic`, or `legacy`.
- Filenames are exact basenames captured from `Path.name`; paths are forbidden.
  `original_filename` is the earliest imported artifact PhenoTypic knows, while
  `input_filename` is the immediate input to one application.
- New applications always carry a non-empty installed PhenoTypic version.
  Explicit migrate mode alone may emit `phenotypic_version: null`, and only for
  a `legacy` application whose historical version cannot be recovered. It must
  never substitute the version performing the migration.
- Application sequences are contiguous from one. Operation sequences are
  globally contiguous across all applications. `retry_base_length` is local to
  its application. The root status mirrors the last application status.
- Canonical v2 state exists only inside `applications`; v1 root aliases are not
  retained. The OME-Zarr store schema remains version 3.

## Ownership and continuation

The CLI owns one application for one invocation: process mode uses `process`,
normal processing uses `full`, and all staged GPU phases continue the same
application. An outermost programmatic pipeline or direct operation creates a
single `programmatic` application only when no external owner is active.
Applying the same pipeline twice creates two applications.

Automatic continuation reopens the unfinished application from the output
checkpoint using the existing run/work identity. Equal pipeline digests alone
never establish retry identity. The application's starting version stays
fixed; an operation retried under another release records that release in its
existing per-operation `phenotypic_version` field.

Read-only opening and flattened `Image.provenance` remain compatible with v1.
Every mutation seam refuses v1, malformed v2, and unknown future versions.

## Publication and migration

Process publication copies the entire journal, removes nondeterministic timing
fields from every application's operations, and reduces every historical
pipeline source path to a basename. It never truncates history.

`phenotypic --mode migrate --output PATH` detects a full result run, a direct
OME-Zarr store, or a process-output tree using the existing output predicates.
Ambiguous layouts fail. V1 becomes one `legacy` application. Historical
versions are recovered from the first operation then the store-root version;
exact filenames are recovered only from exact durable evidence. Unknown values
remain null rather than being fabricated.

Migration inventories by name/stat only, reads each root `zarr.json` once, and
atomically rewrites only that root when it is v1. Full-run marker recertification
follows the rewrite. Local work uses the existing joblib path. Slurm work uses
the existing migration dispatcher/fencing: full runs keep their normal chain;
direct stores and process trees use store array -> seal -> finalizer. A direct
store's control state is a hashed sibling below `.phenotypic`, never inside the
store. Provenance-only migration rejects `--delete-sources` and active outputs.

## Compatibility

`Image.provenance` remains a flattened immutable operation view. Internal
normalized readers retain application boundaries for the CLI and GUI. The
process -> full CLI -> exported store path must retain two separately typed
applications and both pipeline identities.
