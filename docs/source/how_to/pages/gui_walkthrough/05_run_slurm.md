# Run on SLURM

The same Run console form submits to SLURM when you toggle `Mode` from
`Local` to `SLURM`. The hub shells out to the existing
`phenotypic._cli._cli_slurm_submission` pathway — the same code path used
by `phenotypic pipeline.json input/ -o output/ --slurm partition=general
--slurm time=04:00:00`.

## Switch to SLURM mode

Click the `SLURM` radio in the form. The `SLURM config` collapse opens
with the typed common fields:

![Run console with SLURM mode selected.](../../../_static/gui_images/run_slurm/01_slurm_mode.png)

| Field | Maps to | Notes |
|-------|---------|-------|
| `Partition` | `--slurm partition=…` | Required. Use the partition your cluster admin assigned. |
| `Time (HH:MM:SS)` | `--slurm time=…` | SLURM walltime cap. |
| `Memory` | `--slurm mem=…` | Per-node memory (e.g. `16G`, `64G`). |
| `CPUs / task` | `--slurm cpus_per_task=…` | Maps to `--cpus-per-task` in `sbatch`. |
| `GPUs` | `--slurm gpus=…` | Set to 0 unless your pipeline needs a GPU detector. |
| `Extra SLURM (one key=value per line)` | repeated `--slurm key=value` | Free-form pass-through for anything not in the common fields (`account`, `qos`, `mail-user`, etc.). The example screenshot uses `account=lab` and `qos=normal`. |

## Submit

Clicking `Run` while SLURM mode is active does **not** spawn a local
subprocess. Instead it hands the form values to a background thread that
calls `phenotypic … --slurm key=value …` against your SLURM cluster.
The hub waits for the submission to finish (typically <1 s for small
arrays), then registers the returned array primary job id with the run
registry. The Recent Runs panel surfaces the row as
`Mode: slurm-<job-id>` with status pulled from the manifest as the array
chunks finish.

```{warning}
This page captures the form **filled out**, not a successful submission.
Submitting requires `sbatch` on `PATH` and a real SLURM cluster — neither
exists on the workstation that captured these screenshots. To verify your
form values translate to the right CLI invocation, switch back to
`Local` mode and click `Validate (dry-run)`: the dry-run prints the argv
the hub would pass to `phenotypic`. Once you're satisfied, switch back
to `SLURM` and click `Run`.
```

## What SLURM submission writes

A successful submission writes:

- `<output_dir>/progress/job_metadata.json` with the array primary job id
  and per-chunk job ids. The hub reads this file to surface the
  `slurm-<id>` row in Recent Runs — it does **not** parse Rich-formatted
  stdout (locale and terminal-width fragile).
- `<output_dir>/dashboard.html` immediately, so the iframe panel can
  point at the dashboard before any chunk completes.

For deeper SLURM operational detail (chunk sizing, recompile-on-resume,
per-chunk cgroups), see [SLURM Pipelines](../slurm_pipelines.md).

Next: [View Results](06_view_results.md).
