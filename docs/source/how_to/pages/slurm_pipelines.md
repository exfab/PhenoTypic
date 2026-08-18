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
