# Track a Run with DVC

Make a PhenoTypic run a [DVC](https://dvc.org) pipeline stage, so that the
inputs, the pipeline parameters, the outputs and the run's summary metrics are
versioned together. This page covers a local stage, a SLURM stage, a
process-mode stage, how to change parameters, and what to do after you curate
results in the GUI.

DVC is not a dependency of PhenoTypic and PhenoTypic never imports it. Install
it in your own environment (for example `uvx --from 'dvc>=3,<4' dvc ...`). The
commands below were checked against DVC 3.67.

## The Three Outputs

A run writes three directories under `--output`, and nothing else at the root:

| Out | Holds |
|---|---|
| `out/deliverables` | measurements, analysis, dashboards, QC, `metrics.json`, `schema.json` |
| `out/results` | one OME-Zarr store per image |
| `out/.phenotypic` | machine state: progress, failure journal, event log, pipeline copy |

Declare all three with `persist: true`. Continuation, figure run folders and
embeddings depend on outputs surviving a re-run, and DVC deletes a stage's outs
before running it unless they are persisted. `--overwrite` remains the only
thing that wipes an output.

DVC treats a declared out that is missing after the command as a failure:

```text
ERROR: failed to reproduce 'run': output 'out/results' does not exist
```

(exit code 255). An existing but empty out only warns
(`WARNING: 'out/results' is empty.`). PhenoTypic creates all three outs in every
mode, even when every image fails, so a stage whose images all fail ends with
the run's own exit code (see {ref}`dvc-exit-codes`) and not with this error.

## A Local Stage

```yaml
# dvc.yaml
stages:
  run:
    cmd: python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --njobs 16
    deps:
      - data
    params:
      - pipeline.pht-pipe.json:
    outs:
      - out/deliverables:
          persist: true
      - out/results:
          persist: true
      - out/.phenotypic:
          persist: true

metrics:
  - out/deliverables/metrics.json

plots:
  - out/deliverables/measurements.tsv:
      x: Metadata_ImageFile
      y: Size_Area
```

- `params: - pipeline.pht-pipe.json:` with no keys tracks every key in the file.
- `metrics:` and `plots:` are top-level sections. They may point inside an out,
  so they do not create overlapping outputs.
- The plot columns are examples: `Size_Area` exists only when the pipeline
  has a `MeasureSize` measurer. Plot a column your own run produced.
  `dvc plots` reads CSV, TSV, JSON and YAML but not Parquet, so point it at
  the TSV tables.

The tables are TSV, and curated edits are **not** in the TSV yet: curation in
the GUI rewrites `measurements.csv` and `measurements.parquet` but not
`measurements.tsv` (this is fixed in a later change, #283). Plot the TSV splits
(`measurements_by_feature/`, `measurements_by_category/`) or the mirror
`measurements.tsv` knowing that caveat.

Run it:

```bash
dvc repro
```

### Metrics and the column schema

`deliverables/metrics.json` is written when the run finalizes. Its keys:

```json
{
  "images": {"accepted": 3, "failed": 1, "succeeded": 2},
  "metadata": {"phantoms": 0},
  "objects": {"measured": 180},
  "phenotypic_version": "0.21.0",
  "pipeline_sha256": null,
  "schema_version": 1
}
```

`pipeline_sha256` is the digest of the pipeline that produced every aggregated
table, and it **may be `null`**: it is `null` whenever the tables do not all
record the same pipeline digest or do not record one. Do not rely on a value.
When an embedding step ran, an `embeddings` block adds `{rows, stale_skipped}`
per entry.

`deliverables/schema.json` lists every column of each published table with its
dtype, producing measurer, and `MeasurementInfo` description.

Both files are byte-stable: no timestamps, sorted keys, so identical runs give
identical bytes and `dvc metrics diff` is meaningful.

## A SLURM Stage

A SLURM run returns exit code `4` at submission, long before it finishes. DVC
hashes a stage's outs the moment its command exits, and it fails a stage whose
command exits non-zero. So a SLURM stage is one command: `phenotypic-wait`
launches the submission itself, then waits for the run to end.

```yaml
stages:
  run:
    cmd: phenotypic-wait --output out -- python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --slurm slurm_partition=intel --slurm mem_gb=16 --slurm time=240
    deps:
      - data
    params:
      - pipeline.pht-pipe.json:
    outs:
      - out/deliverables:
          persist: true
      - out/results:
          persist: true
      - out/.phenotypic:
          persist: true
```

Everything after `--` is the submit command. `phenotypic-wait` runs it as a
child process (no shell) in its own working directory, environment and
terminal, then acts on the child's exit code:

- `4` (submitted): it waits for the run, as described below;
- `0` or `3` (the command ran locally and already finished): returned as-is,
  with no waiting;
- any other code is returned as-is, and nothing is waited for. A submit killed
  by signal N returns `128 + N`, the shell's convention; a submit command that
  cannot be found returns `127`.

It refuses to launch, with a usage error (exit `2`), when:

- nothing follows `--`;
- the submit command names an output (`-o`/`--output`) that resolves to a
  different directory from `--output` (a relative path resolves against the
  current directory, which the child shares);
- the submit command runs `--mode migrate`.

A command that names no output, such as a wrapper script, is launched
unchecked. `--timeout-seconds` bounds the wait, never the submit.

If you submit separately, `phenotypic-wait --output out` without `--` only
waits. In a `dvc.yaml` that form needs the submit to count as success, because
DVC stops a `cmd` list at the first command that exits non-zero:
`cmd: ["python -m phenotypic ... --slurm ...; test $? -eq 4", "phenotypic-wait --output out"]`.

Once the run is submitted, `phenotypic-wait` polls the scheduler and the run
state every `--poll-seconds` (default 60) and exits with the code the run would
have returned locally:

- `0` once this submission published and no image failed terminally;
- `3` once it published and some images failed terminally;
- `5` when no job is alive and this submission published nothing, for example a
  finalizer killed for running out of memory;
- `6` after `--timeout-seconds` (default: wait indefinitely). Nothing is
  cancelled.

Details that matter:

- "Published" is bound to **this submission's lifecycle generation**, read at the
  first poll. A re-submission whose finalizer dies does not read as published
  because an earlier run's outputs still match.
- Any live job keeps it waiting. A scheduler that cannot be queried also keeps it
  waiting; it never guesses.
- It never aggregates, publishes or cancels, and the only thing it submits is
  the command you give it after `--`. Its only write is
  refreshing `.phenotypic/verification_cache.json`, which the run-state check
  already does and which is harmless.
- It does **not** serve `--mode migrate`.

### Run `dvc repro` itself inside a Slurm job

Head nodes kill long-running processes, and the waiter may poll for days. Run the
whole `dvc repro` inside a small, long-walltime job:

```bash
sbatch --partition=intel --cpus-per-task=1 --mem=4G --time=7-00:00:00 \
    --output=dvc-repro-%j.log --wrap='dvc repro'
```

## A Process-Mode Stage

`--mode process` exports one layer per image and mirrors the input tree. It
writes no `deliverables/`, so the out is the whole `out` root:

```yaml
stages:
  export:
    cmd: python -m phenotypic --mode process --layer gray -i data -o out -p pipeline.pht-pipe.json --njobs 16
    deps:
      - data
    params:
      - pipeline.pht-pipe.json:
    outs:
      - out:
          persist: true
```

(There is no `metrics:` here: process mode produces no `metrics.json`.)

**An in-place process run is not a DVC stage.** When `--output` equals
`--input`, the input tree would also be the out, and a DVC stage must write new
stores rather than rewrite its own dependency.

## Cache Type

Keep DVC's default `cache.type`, `reflink,copy`. Reflinks need Btrfs, XFS or OCFS2,
so on GPFS this resolves to **a copy**: every out is copied into the cache once,
and checked out by copy again. That is the cost of the default.

`hardlink` and `symlink` are unsupported. DVC makes linked files read-only, and
`.phenotypic/` holds files appended in place by concurrent writers (the
failure journal, the event log, the SLURM ledger), as do some files under
`deliverables/` (the QC database, GUI curation). The run preflight warns
(`PF-DVC-LINKED-CACHE`) when the enclosing project selects either link type. It
reads only the project's `.dvc/config` and `.dvc/config.local`, not the global or
system DVC config, so set the type in the project.

## Should `results/` Be Cached?

A directory out is hashed file by file and copied into the cache. On a large run
that is the dominant cost. It was measured, not guessed: a bounded subset of a
real run (303 of 6,657 stores, 17,101 files, 60.16 GB; one cold node, DVC 3.67.1,
default copy cache) gave:

| Step | Measured (303 stores) | Extrapolated to 6,657 stores |
|---|---|---|
| `dvc add out/results` | 1,909.7 s | about 11.7 h |
| `dvc status`, cold | 13.7 s | about 5.0 min |
| `dvc status`, warm | 6.1 s | about 2.2 min |
| `dvc commit` after one modified file | 10.4 s | about 3.8 min |
| Cache size | 60.12 GB | about 1.32 TB |

The extrapolation is linear by store count and assumes uniform stores; the
`status` and `commit` figures are upper bounds because DVC re-hashes only what
changed.

For a tree of this size, set `cache: false` on `out/results`:

```yaml
    outs:
      - out/deliverables:
          persist: true
      - out/results:
          persist: true
          cache: false
      - out/.phenotypic:
          persist: true
```

DVC still records the out's hash in `dvc.lock` and still detects changes, but
the stores are not copied into the cache, saving about 1.3 TB. **This does not
make the first hash free.** The split of the measured `add` time between hashing
and copying was not measured, so `cache: false` saves the copy only. The first
hash of a full tree still costs part of that time. Note also that
`cache: false` on any out disables DVC's run cache for the stage, and an
uncached out cannot be restored by `dvc checkout`. For small runs, keep the
default. Keep `deliverables` and `.phenotypic` cached either way.

## Change Parameters

The pipeline JSON is the params file. Override a single value for one
experiment with `dvc exp run -S`:

```bash
dvc exp run -S 'pipeline.pht-pipe.json:pipe_cfgs.detect.params.ignore_zeros=true'
```

The path is `pipe_cfgs.<OpKey>.params.<field>`, where `<OpKey>` is the key the
operation has in the pipeline's `ops`. Two caveats:

- An op key that contains a `.` cannot be addressed, because DVC splits the path
  on dots.
- DVC writes JSON through `json.dump`, and likely rewrites the file as a single
  line. The content is the same; a diff of the file is not readable.

## After Curation in the GUI

Curating results in the GUI changes files inside `out/deliverables`, so
`dvc status` reports that out as modified. That is intended. Record the curated
state with:

```bash
dvc commit
```

(see the TSV caveat above: until #283, curated edits reach `measurements.csv`
and `measurements.parquet` but not `measurements.tsv`).

(dvc-exit-codes)=
## Exit Codes

The exit code is a contract: a DVC stage acts on it without parsing logs. The
values are `phenotypic.sdk_.RunExitCode`.

| Code | Name | Meaning |
|---|---|---|
| `0` | `COMPLETE` | finished, no image failed terminally |
| `1` | `ERROR` | a runtime or configuration error |
| `2` | none | a usage error: Click's own code, for a refused option combination |
| `3` | `COMPLETE_WITH_FAILURES` | finished, some images failed terminally |
| `4` | `SUBMITTED` | submitted to SLURM, not finished |
| `5` | `ENDED_UNPUBLISHED` | `phenotypic-wait`: no job alive and this submission published nothing |
| `6` | `WAIT_TIMEOUT` | `phenotypic-wait` gave up after `--timeout-seconds` |

`phenotypic-wait --output out -- <submit>` returns the submit's own code
whenever the submit does not exit `4`, so any code above can come from it, as
can `128 + N` for a submit killed by signal N.

DVC treats any non-zero code as a failed stage, so `3` fails the stage while the
outputs of the images that succeeded remain on disk.

## Reclaim Cache Space

Tracking `.phenotypic/` makes the cache grow: it changes on every run. Remove
cache entries no revision refers to with:

```bash
dvc gc --workspace
```

`--workspace` keeps only what the current workspace uses; add `--all-branches`
or `--all-tags` to keep what other revisions need. Read `dvc gc --help` before
running it on a shared cache.
