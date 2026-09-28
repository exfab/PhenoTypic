# Distributed Tuning on HPCC Clusters

Run one Optuna tuning study across many SLURM compute nodes without deploying a
database service. A distributed PhenoTypic tune run uses one GPFS-visible
append-only journal and a terminal publisher owned by the run lifecycle.

## Start a distributed study

Use an Optuna sampler and add `--slurm`. When no storage URL is supplied,
PhenoTypic records an absolute shared journal URL at
`<output>/.pht-tune-cache/journal.log`; every worker opens that same journal
with Optuna's symlink-lock backend.

```bash
uv run phenotypic-tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --n-workers 16 \
    --slurm slurm_partition=short --slurm slurm_mem=8G --slurm slurm_time=04:00:00
```

The output directory and its journal must be on storage visible to every
compute node (for example your `/bigdata` allocation), not a node-local
`/scratch` path. The command returns after submitting the worker array and its
one dependent terminal finalizer; it does not run a separate scheduler sidecar.

## Storage policy

For an Optuna strategy, storage resolves in this order:

1. `--storage-url` on the command line.
2. `strategy.storage_url` in the tuning spec.
3. `$PHENOTYPIC_TUNE_STORAGE_URL`.
4. The mode default: run-local SQLite for a local run, or the absolute
   run-local `journal://` URL for `--slurm`.

Do not use SQLite for a multi-node fleet. `--slurm` rejects an explicit SQLite
URL before it creates run artifacts. Password-bearing URLs are also rejected:
they would otherwise be copied into the run marker and generated worker script.
An explicitly configured password-less RDB URL remains available when your
environment already operates one, but it is not required for the normal HPCC
workflow.

## What SLURM submits

The submitter writes the resolved tuning spec, the recorded storage URL, a
held-out split, and one lifecycle generation before it submits work. It then
submits exactly these compute roles:

- One worker **array**. Each task reloads the resolved spec and calibration
  images, then claims trials from the shared journal until the common terminal
  budget is reached.
- One non-array terminal **finalizer** with an `afterany` dependency on that
  array. It reopens the recorded backend without create semantics, requires the
  expected number of completed-or-pruned terminal trials and a valid winner,
  then publishes the tuning artifacts.

The finalizer writes `trials.parquet`, `best_pipeline.json`, `best_params.json`,
importance/Pareto outputs where applicable, and the held-out generalization
report. `best_params.json` is the final completion signal. If the journal is
missing, corrupt, incomplete, or has no valid terminal winner, publication fails
closed and the lifecycle records the failed generation.

## Monitor and recover

The run marker at `.pht-tune-cache/run.json` records the selected backend and
expected terminal budget. The lifecycle state and append-only ledger live under
`.phenotypic/progress/`; use them together with your scheduler's normal job
tools to follow the worker and finalizer roles.

After a completed or interrupted distributed run is quiescent, republish from
the exact recorded backend:

```bash
uv run phenotypic-tune finalize ./out
```

Normal manual finalization refuses an active lifecycle generation. If a failed
generation remains active, `--force` first requests cancellation and proceeds
only after scheduler quiescence has been proven and no unresolved scheduler
tokens remain:

```bash
uv run phenotypic-tune finalize ./out --force
```

Publication is lifecycle-locked from ownership validation through all writes.
Running `finalize` again after a successful publication is safe and produces
byte-identical published artifacts. Re-running `run` against an output with an
active lifecycle is not a replacement for finalization; wait for the finalizer
or use the explicit recovery command after quiescence.
