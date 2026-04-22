# SLURM Pipelines

Run PhenoTypic batch processing on SLURM-managed clusters.

## Automatic SLURM Detection

When SLURM is available, the CLI automatically submits jobs to the scheduler.
To force local execution instead:

```bash
python -m phenotypic pipeline.json /plates/ /output/ --force-local
```

## SLURM Arguments

Pass SLURM parameters with `--slurm-args`:

```bash
python -m phenotypic pipeline.json /plates/ /output/ \
    --slurm-args time=04:00:00 \
    --slurm-args partition=gpu \
    --slurm-args mem=16G
```

## Wait for Completion

By default, the CLI returns immediately after submitting SLURM jobs. To wait:

```bash
python -m phenotypic pipeline.json /plates/ /output/ --wait
```

## Resume on SLURM

Resume works the same way as local execution:

```bash
python -m phenotypic pipeline.json /plates/ /output/ --resume
```
