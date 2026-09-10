# Migrate legacy results and provenance

Use explicit migrate mode to bring an existing PhenoTypic output into the
current OME-Zarr storage and provenance schemas. Migration is resumable and
idempotent: rerun the same command after an interruption. Stores that are
already current are validated and left byte-for-byte unchanged.

## Choose the target

`--output` accepts one of three layouts:

- A **full result run** containing `results/`. Migration keeps its normal
  metadata → image → seal → finalizer workflow. It upgrades existing store
  provenance before converting legacy HDF images or recertifying completion
  markers.
- A **direct `*.ome.zarr` store**. Only its root provenance journal is
  considered. Migration state is written to a hashed sibling below
  `.phenotypic/migration_targets/`, never inside the store.
- A **process-output tree** containing one or more OME-Zarr image stores.
  Only store provenance is considered.

An ambiguous directory that looks like both a full run and a process-output
tree is refused. Migration does not guess.

## Validate before writing

Run a dry validation locally:

```bash
uv run python -m phenotypic \
  --mode migrate \
  --output /path/to/target \
  --dry-run
```

A dry run does not rewrite scientific files or create lifecycle state in the
target. Slurm dry runs keep manifests, scripts, logs, and status evidence in an
external cache control root.

## Run locally

For a small target:

```bash
uv run python -m phenotypic \
  --mode migrate \
  --output /path/to/target
```

On GPFS, root-file checks are often latency-bound. Parallelize them with the
native migrate option rather than a custom wrapper:

```bash
uv run python -m phenotypic \
  --mode migrate \
  --output /path/to/target \
  --njobs 32
```

Local parallelism uses the migration mode's existing joblib dispatch. The
inventory uses names and file metadata without descending into Zarr chunks.
Each store worker reads the root `zarr.json` once and writes only when it must
upgrade a schema-v1 provenance journal.

## Dispatch through Slurm

Use migrate mode's own Slurm dispatch on a large cluster target:

```bash
uv run python -m phenotypic \
  --mode migrate \
  --output /path/to/target \
  --slurm slurm_partition=short \
  --slurm time=30 \
  --wait
```

Do not combine an explicit `--njobs` with `--slurm`; the scheduler owns array
parallelism. Without `--wait`, the command returns after durable submission and
prints the generation, control root, manifest, and finalizer script. With
`--wait`, it reads the matching typed terminal report after the finalizer closes
the lifecycle.

Full runs retain the metadata → image → seal → optional reclaim → finalizer
chain. Direct stores and process trees use an indexed store array → provenance
seal → finalizer chain. Both go through the shared drip-feed dispatcher and
generation fence; migrate mode does not submit a custom wrapper or a parallel
scheduler sidecar.

## Source deletion

Legacy HDF sources in a full run are retained by default. Add
`--delete-sources` only when you want migrate mode to delete each source after
its converted store passes the value-level readback checks:

```bash
uv run python -m phenotypic \
  --mode migrate \
  --output /path/to/full-run \
  --delete-sources
```

Direct-store and process-tree migrations reject `--delete-sources`: they only
upgrade root provenance and have no legacy image source to reclaim.

## Provenance schema-v2 convention

Schema v2 keeps an ordered `applications` journal. Process mode, the normal
CLI, and programmatic use are separate applications, so a process output can be
used as input to the CLI or browse GUI without losing the earlier pipeline.
Each application records:

- its kind (`process`, `full`, `programmatic`, or `legacy`);
- the immediate input basename and pipeline basename when durably known;
- the installed `phenotypic_version`; and
- its globally ordered operation entries.

New applications always have a non-empty `phenotypic_version`. Explicit migrate
mode may write `phenotypic_version: null` **only** on the single converted
`legacy` application when neither an operation nor the store root contains a
recoverable historical version. The migration release must never be substituted
for that unknown historical value.

Migration also does not invent filenames. A legacy journal with no exact durable
filename evidence receives `original_filename: null` and
`input_filename: null`. Consumers must preserve those nulls rather than guessing
from the store directory name.

## Failure and recovery

Malformed journals and unknown future schema versions are reported per store
without rewriting their bytes. Independent array items continue, the seal
records every failure, and the finalizer closes the generation with failure
category `provenance`. Fix or inspect the reported store, then rerun the same
public migrate command; do not invoke an internal worker or hand-edit migration
status files.
