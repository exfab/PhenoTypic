# Track a Run with DVC

Make a PhenoTypic run a [DVC](https://dvc.org) pipeline stage, so that the
inputs, the pipeline parameters, the outputs and the run's summary metrics are
versioned together. This page covers a local stage, a SLURM stage, a
process-mode stage, how to change parameters, and what to do after you curate
results in the GUI.

DVC is not a dependency of PhenoTypic and PhenoTypic never imports it. Install
it in your own environment (for example `uvx --from 'dvc>=3,<4' dvc ...`). The
commands below were checked against DVC 3.67.1.

**Every stage command starts with `uv run`.** A stage runs with DVC's own
environment, and a DVC run through `uvx` puts its own Python first on `PATH`,
even when your project's environment is activated. A bare
`python -m phenotypic` then starts a Python without PhenoTypic
(`No module named 'phenotypic'`). A bare `phenotypic-wait` is found only when
your environment is activated, and even then the `python` it launches after
`--` is DVC's. `uv run` resolves both from the project's environment, and a
command `phenotypic-wait` launches inherits that `PATH`.
If you do not use uv, install DVC into the same environment as PhenoTypic and
drop the `uv run`.

## The Three Outputs

A run writes three directories under `--output`, and nothing else at the root
(process mode differs: see {ref}`dvc-process-mode`):

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
mode except process mode, even when every image fails, so a stage whose images
all fail ends with the run's own exit code (see {ref}`dvc-exit-codes`) and not
with this error. A process-mode stage declares a different out; see
{ref}`dvc-process-mode`.

## A Local Stage

```yaml
# dvc.yaml
stages:
  run:
    cmd: uv run python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --njobs 16
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
      x: Metadata_ImageName
      y: Size_Area
```

- `params: - pipeline.pht-pipe.json:` with no keys tracks every key in the file.
- `metrics:` and `plots:` are top-level sections. They may point inside an out,
  so they do not create overlapping outputs.
- The plot columns are examples. `Metadata_ImageName` is in every mirror;
  `Size_Area` exists only when the pipeline has a `MeasureSize` measurer. Plot
  a column your own run produced.
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

(dvc-metrics)=
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

After a `--mode measure` run, `images.failed` counts the images whose
re-measure failed in that run, so it agrees with the exit code `3`, and
`images.succeeded` leaves them out. A re-measure that fails before it replaces
the image's table leaves the store as it was, so that image's **previous**
measurements are still in the tables.

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
    cmd: uv run phenotypic-wait --output out -- python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --slurm slurm_partition=intel --slurm mem_gb=16 --slurm time=240
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
terminal, so the bare `python` after `--` is the one `uv run` put first on
`PATH`. It then acts on the child's exit code:

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

Only the options after the program `phenotypic` (`python -m phenotypic …` or the
`phenotypic` console script) are read, so a wrapper's own options before it,
such as `srun -o slurm.log`, are not taken as the run's. A command that runs no
`phenotypic` program by name, such as a wrapper script, is launched unchecked.
`--timeout-seconds` bounds the wait, never the submit.

**If a job of `--output` is already running**, `phenotypic-wait` does not launch
the submit. It prints that it is attaching, names the run's lifecycle
generation and active jobs, and waits for that run exactly as after a `4`. Only
one SLURM run per output can be live, so this is the run the stage started
earlier. The waiter cannot check that it uses the same pipeline; the message
names the generation so you can.

If you submit separately, `phenotypic-wait --output out` without `--` only
waits. In a `dvc.yaml` that form needs the submit to count as success, because
DVC stops a `cmd` list at the first command that exits non-zero:
`cmd: ["uv run python -m phenotypic ... --slurm ...; test $? -eq 4", "uv run phenotypic-wait --output out"]`.

Once the run is submitted, `phenotypic-wait` polls the scheduler and the run
state every `--poll-seconds` (default 60) and exits with the code the run would
have returned locally:

- `0` once this submission published and no image failed terminally;
- `3` once it published, some images failed terminally, and every accepted
  image has a result: a success or a terminal failure (see
  {ref}`dvc-accept-failures` to complete the stage anyway);
- `5` when no job is alive and the run **ended without finishing**: this
  submission published nothing (a finalizer killed for running out of memory, a
  cancelled run), or some images were never attempted (an array task killed
  before its images wrote a result). Re-run the same command;
- `6` after `--timeout-seconds` (default: wait indefinitely). Nothing is
  cancelled;
- `1` when the poll itself fails five times in a row (an unreadable state file,
  a scheduler command that raises); each failure is printed and retried first.
  `130` when the waiter is interrupted. Neither cancels anything, and the run
  goes on.

Details that matter:

- "Published" is bound to **this submission's lifecycle generation**, read at the
  first poll. A re-submission whose finalizer dies does not read as published
  because an earlier run's outputs still match.
- A `--mode measure` resubmission keeps no run state, so its terminal failures
  are counted from the failure journal rows its own jobs wrote. It exits `3`
  when one of its images failed, as a local `--mode measure` does.
- Any live job keeps it waiting. A scheduler that cannot be queried also keeps it
  waiting; it never guesses.
- It never aggregates, publishes or cancels, and the only thing it submits is
  the command you give it after `--`. Its only write is
  refreshing `.phenotypic/verification_cache.json`, which the run-state check
  already does and which is harmless.
- Ctrl-C (or DVC's interrupt) while the submit runs waits for the submit to
  exit, so it can record what it submitted, and returns the submit's code
  without waiting for the run. Run the same command again to attach to it.
- It does **not** serve `--mode migrate`.

### Run `dvc repro` itself inside a Slurm job

Head nodes kill long-running processes, and the waiter may poll for days. Run the
whole `dvc repro` inside a small, long-walltime job:

```bash
sbatch --partition=intel --cpus-per-task=1 --mem=4G --time=7-00:00:00 \
    --output=dvc-repro-%j.log --wrap='uvx --from dvc==3.67.1 dvc repro'
```

**If that job hits its walltime**, or DVC is killed, the SLURM run keeps going
without anything waiting for it, and DVC records nothing. Submit the same
`sbatch` line again. The stage's `phenotypic-wait` finds the run's jobs still
active, attaches to that run instead of submitting a second one, and the stage
completes when the run ends. If the run ended in the meantime, the stage
launches the submit again, which is an ordinary continuation: with every image
done it exits `0` or `3` at once, and otherwise it submits only what is left,
which the waiter then waits for.

(dvc-process-mode)=
## A Process-Mode Stage

`--mode process` exports one layer per image and mirrors the input tree. It
writes no `deliverables/` and no `results/`: the exported layers sit under the
output root itself, beside `.phenotypic/`. So the out is the whole `out` root:

```yaml
stages:
  export:
    cmd: uv run python -m phenotypic --mode process --layer gray -i data -o out -p pipeline.pht-pipe.json --njobs 16
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
| `5` | `ENDED_UNPUBLISHED` | `phenotypic-wait`: no job alive and the run ended without finishing (nothing published, or images never attempted); re-run the same command |
| `6` | `WAIT_TIMEOUT` | `phenotypic-wait` gave up after `--timeout-seconds` |

`phenotypic-wait --output out -- <submit>` returns the submit's own code
whenever the submit does not exit `4`, so any code above can come from it, as
can `128 + N` for a submit killed by signal N. `phenotypic-wait` itself also
exits `1` when its polls keep failing and `130` when interrupted.

DVC treats any non-zero code as a failed stage, so by default `3` fails the
stage while the outputs of the images that succeeded remain on disk. DVC writes
no `dvc.lock` entry, and the next `dvc repro` runs the stage again, which exits
`3` again, because a recorded terminal failure is not retried.

(dvc-accept-failures)=
### Completing a stage that has terminal failures

A dataset with one plate that cannot be measured, such as an empty plate that
raises `NoObjectsError`, exits `3` on every run. To let its stage complete, add
`--accept-failures` to `phenotypic-wait`:

```yaml
    cmd: uv run phenotypic-wait --output out --accept-failures -- python -m phenotypic -i data -o out -p pipeline.pht-pipe.json --slurm slurm_partition=intel
```

A final `3` becomes `0`, after the waiter prints how many images failed
terminally. That holds whether the `3` came from the wait or from a submit that
ran locally, so the launch form also serves a **local** stage: drop the
`--slurm` options and keep the rest. No other code changes; `1`, `2`, `5` and
`6` still fail the stage.

What it costs:

- **DVC records the run as successful.** `dvc.lock` is written, `dvc status`
  reports the stage up to date, and downstream stages run on the images that
  succeeded. An image that failed in a full run is absent from every table;
  one whose `--mode measure` re-measure failed usually keeps its previous
  measurements (see {ref}`dvc-metrics`).
- **The failures are visible only in the outputs.** `metrics.json`'s
  `images.failed` holds the count, and `dvc metrics diff` shows it changing.
  The failures themselves are in `.phenotypic/terminal_failures.jsonl`.
- **A clean `dvc repro` does not retry them.** The stage is up to date, so
  nothing runs. To retry, add `--retry-failures` to the submit command, which
  changes the stage and makes DVC run it.

Without the flag, the exit-code contract is unchanged.

## After `dvc checkout` of Another Revision

A `dvc checkout` restores `out/.phenotypic` from that revision along with the
other outs. What it restores is checked before it is trusted:

- A per-image record names the digests of the store it describes. A record
  restored beside a store it does not describe is rejected, and that image
  runs again. The outcome can be behind, never wrong.
- The lifecycle fence `out/.phenotypic/progress/slurm_lifecycle.json` may be
  restored as `active`, naming SLURM jobs from that revision's run. The next
  run asks the scheduler about those jobs rather than trusting the flag, so
  ended jobs do not block it.

That breaks down once SLURM has forgotten the jobs. After its `MinJobAge` and
`PurgeJobAfter` limits, `squeue` reports `Invalid job id` and `sacct` returns
no rows. PhenoTypic then cannot tell whether a job is running, and it never
calls an unknown job ended. Every run over that output refuses with
`Ledgered SLURM jobs are unverified: <ids>`, and `phenotypic-wait` waits
instead of attaching. There is no override flag. To recover by hand:

1. Confirm that none of the listed jobs is running, from a node where
   `squeue` works: `squeue --jobs <ids>` and `sacct -j <ids>`. Then confirm no
   unrecorded submission is queued:
   `squeue --noheader --format='%i|%k' | grep phenotypic:`.
2. Mark the restored fence inactive:

   ```bash
   uv run python - <<'PY'
   import json, pathlib
   fence = pathlib.Path("out/.phenotypic/progress/slurm_lifecycle.json")
   state = json.loads(fence.read_text(encoding="utf-8"))
   state["active"] = False
   fence.write_text(json.dumps(state, indent=2), encoding="utf-8")
   PY
   ```

Do **not** delete the file instead. Without a fence, every generation in the
job ledger is checked, so the same unverified jobs still refuse the run. The
next run replaces the inactive fence with its own.

## Reclaim Cache Space

Tracking `.phenotypic/` makes the cache grow: it changes on every run. Remove
cache entries no revision refers to with:

```bash
dvc gc --workspace
```

`--workspace` keeps only what the current workspace uses; add `--all-branches`
or `--all-tags` to keep what other revisions need. Read `dvc gc --help` before
running it on a shared cache.
