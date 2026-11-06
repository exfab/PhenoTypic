# SLURM Pipelines

Run PhenoTypic batch processing on SLURM-managed clusters.

## Choose Local or SLURM Execution

The CLI runs locally unless you provide `--slurm`. Merely having SLURM
available does not submit a job. To run locally explicitly:

```bash
uv run python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --force-local
```

## SLURM Arguments

Pass SLURM parameters with repeated `--slurm` flags:

```bash
uv run python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ \
    --slurm time=240 \
    --slurm slurm_partition=gpu \
    --slurm mem_gb=16
```

## Check the Profile Before Submitting

Add `--dry-run` to the same command to check it without submitting anything.
The {ref}`run preflight <Run Preflight Checks>`
tests each profile the run would submit with `sbatch --test-only` (the CPU
profile, and the GPU profile of a staged GPU run), compares `time` with the
partition's `MaxTime`, checks that a GPU stage's partition has GPUs, and warns
when `--output`, `--input`, the pipeline, or the metadata CSV is on node-local
storage that other nodes cannot see. The same checks run before a real
submission, so a rejected profile stops the run before any job is queued. The
dry-run preview also prints the `#SBATCH` lines the profile produces.

## Monitor Completion

Every SLURM run returns after submission and prints the commands to monitor
its jobs. For example:

```bash
uv run python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ \
    --slurm slurm_partition=compute
```

The dependent finalizer publishes aggregated outputs and the completion marker.
Successful submission is not successful processing: follow the printed scheduler
commands and completion evidence before consuming those outputs. Migration
submissions print the path of their finalizer's terminal-status file. The
`--wait` option was removed in version 0.21.0.

## Continue on SLURM

Automatic continuation works the same way as local execution. Run the same
command again:

```bash
uv run python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ \
    --slurm slurm_partition=compute
```
