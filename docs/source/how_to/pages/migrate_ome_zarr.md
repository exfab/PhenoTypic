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

## Work on a copy when the original must stay as it is

A full run is converted **in place**. When the original tree has to stay
untouched — while you compare results, or because other analyses still read
it — migrate a copy instead.

The legacy `.h5` sources do not need a byte copy. Migration opens them
read-only, so hard links cost no space and cannot change the original files.
Copy everything else for real:

```bash
SRC=/path/to/run
DST=/path/to/run-ome-zarr
DS=<dataset>          # repeat the first two commands for each dataset

mkdir -p "$DST/results/$DS/hdf"
rsync -a --link-dest="$SRC/results/$DS/hdf/" \
  "$SRC/results/$DS/hdf/" "$DST/results/$DS/hdf/"
rsync -a --exclude="/results/$DS/hdf/" "$SRC/" "$DST/"
```

On a run of several thousand images the non-HDF part (deliverables, overlays,
external measurement Parquets) is tens of gigabytes, so run the copy as a Slurm
job rather than on a login node. The converted stores then need about as much
space again as the `.h5` files they replace.

If the original was **already migrated once** — it has `results/<ds>/zarr/` and
`.phenotypic/progress/migration/` — leave out the state that records the
original's absolute paths. Leave out `zarr/` as well if every store should be
rebuilt from HDF:

```bash
rsync -a \
  --exclude="/results/$DS/hdf/" \
  --exclude="/results/$DS/zarr/" \
  --exclude="/.phenotypic/progress/migration/" \
  --exclude="/.phenotypic/progress/recompile/" \
  --exclude="/.phenotypic/metadata_migration/" \
  --exclude="/.phenotypic/slurm_scripts/" \
  --exclude="/.phenotypic/.migration-attempt.lease" \
  "$SRC/" "$DST/"
```

A metadata-migration receipt that points outside the tree is refused anyway, so
that pass simply reruns against the copy. `--delete-sources` on the copy
removes only the copy's link to each `.h5`; the original keeps its file.

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

The generated scripts run the Python interpreter of the environment you
submitted from. Keep that checkout and its `.venv` in place until the finalizer
has finished.

### Size a large run

Every stage of the chain — metadata, each image chunk, seal and finalizer — is
written with the **same** `--slurm` profile, so size the profile for the longest
stage, not for one image task. The short example above suits a small target
only.

Measured on a 6,657-image full run of 16-bit plate images (converted stores
≈ 190 MB each):

| Stage | Wall time | Peak memory |
|---|---|---|
| metadata | 9 min | — |
| one image task (HDF → store, table, marker) | ≈ 90 s, at most ≈ 2 min | ≈ 1.6 GB |
| seal | 45 min | — |
| finalizer | ≈ 3 h | < 4 GB |

- **Walltime.** Earlier attempts on the same data with 1–1.5 h limits timed out
  in the seal and in the finalizer. A finalizer killed at its limit leaves the
  generation open until you rerun. Choose a partition that allows several hours,
  for example `--slurm slurm_partition=intel --slurm time=480`.
- **Memory.** A 4 GB request out-of-memory-killed image tasks on this data;
  12 GB left ample headroom. Memory, not CPUs, is usually what caps how many
  array tasks your account runs at once, so do not over-request it.
- **The finalizer is slow, not hung.** It re-verifies every image's recorded
  artifacts, which re-reads each overlay PNG in full several times over. While
  its CPU time keeps rising it is working.

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

## What a converted full run contains

- **Stores.** Each image becomes `results/<ds>/zarr/<stem>.ome.zarr`, and its
  external Parquet is embedded at `tables/measurements/table.parquet`. When the
  run has a `deliverables/metadata.csv`, that table is written in the older,
  metadata-joined shape and the store has no separate `tables/metadata/` table.
  The store's descriptor still lists only the measured columns, in
  `measurement_columns`.
- **Master.** Migration finalizes the run again. Each embedded table is cut back
  to its own `measurement_columns` before the tables are concatenated, so
  `deliverables/master_measurements.parquet` carries measurements and the
  image's intrinsic identity only — no user metadata — exactly like a run
  written by the current release. `master_measurements.csv` is deleted.
- **Mirror.** `deliverables/measurements.{csv,parquet}` joins `metadata.csv`
  once, on the columns the master and the CSV share — for a plate run typically
  `Metadata_ImageName`, `Metadata_Dataset`, `Grid_RowNum` and `Grid_ColNum`.
  Metadata is the left side of that join: a metadata row with no measured object
  is kept with `QC_MetadataOnly=true`, and a measured object whose image or grid
  cell appears in no metadata row is **dropped**. A converted run can therefore
  show fewer measured rows in the mirror than its old `measurements.csv` did.
  Those objects are still in the master; add them to `metadata.csv` if they
  belong to the experiment.
- **Metadata snapshot.** `deliverables/metadata.csv` is never rewritten. A
  header-normalized view is written beside it as `metadata.canonical.csv`.
- **Sources.** `.h5` files and external Parquets stay unless you passed
  `--delete-sources`.

## Check the result

A migration is finished when its generation's terminal status says so, not when
the image array drains:

- In the control root printed at submission (for a Slurm run,
  `.phenotypic/progress/migration/<generation>/`),
  `migration_generations/<generation>/terminal_status.json` reads
  `"status": "succeeded"`, its report's `failed` list is empty, and `converted`
  equals the number of `.h5` sources.
- `.phenotypic/progress/slurm_lifecycle.json` reads `"active": false`.
- The mirror has measured rows. Filter `deliverables/measurements.parquet` on
  `QC_MetadataOnly == false`. If the master has rows and that filter returns
  none, the metadata join matched nothing: check that `metadata.csv` names the
  converted images and grid cells.

Rerunning migrate on a tree that already succeeded is safe but not free. It
starts a new generation and repeats the whole chain, re-verifying every image,
before it finalizes again.

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
