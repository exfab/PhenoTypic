# SLURM Pipelines

Run PhenoTypic batch processing on SLURM-managed clusters.

## Automatic SLURM Detection

When SLURM is available, the CLI automatically submits jobs to the scheduler.
To force local execution instead:

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --force-local
```

## SLURM Arguments

Pass SLURM parameters with repeated `--slurm` flags:

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ \
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

## Wait for Completion

By default, the CLI returns immediately after submitting SLURM jobs. To wait:

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --wait
```

## Continue on SLURM

Automatic continuation works the same way as local execution. Run the same
command again:

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/
```
