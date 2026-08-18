# CLI Execution Modes

The PhenoTypic CLI has one entry point and one switch that decides what a run
actually does:

```bash
python -m phenotypic --mode {full,measure,recompile,process} ...
```

A colony-phenotyping run is expensive in the detection step and cheap in
everything downstream. Segmenting a plate costs seconds to minutes per image;
re-reading a segmentation to compute `Size_Area` costs milliseconds. The four
modes exist so that when you change your mind about a *measurement*, a
*dashboard*, or a *threshold*, you only pay for the part that actually changed.

| Mode | Re-runs detection? | Needs `--input` | Needs `--pipeline` | Produces |
|------|--------------------|-----------------|--------------------|----------|
| `full` (default) | Yes | Yes | Yes | Everything: HDFs, measurements, deliverables, QC, dashboard |
| `measure` | No | No | Yes | New measurements + deliverables from existing HDFs |
| `recompile` | No | No | No | Refreshed aggregate deliverables only |
| `process` | Yes | Yes | Yes | One exported image layer per input, mirroring the input tree |

`--output` is required in every mode.

## Setting up

Every command below is runnable. Create a pipeline and a small dataset first:

```python
from pathlib import Path

import tifffile

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize

plates = Path("plates/set_a")
plates.mkdir(parents=True, exist_ok=True)

plate = load_synth_yeast_plate()
for i in range(2):
    tifffile.imwrite(plates / f"plate_{i:02d}.tiff", plate.rgb[:])

pipeline = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()])
Path("pipe.json").write_text(pipeline.to_json())
```

The synthetic plate is not a grid layout, so pass `--image-type Image` to the
commands that follow. The CLI defaults to `GridImage` (8 × 12), which is the
right choice for real arrayed plates but would impose a grid on this image.

## `full` — the complete run

`full` applies the pipeline to every image, measures the detected objects, and
writes every deliverable. It is the default, so `--mode full` can be omitted.

```bash
python -m phenotypic --mode full \
    --pipeline pipe.json \
    --input ./plates \
    --output ./out \
    --image-type Image
```

Before committing to a long run, preview the plan. `--dry-run` resolves the
datasets, validates the pipeline, and prints the estimated output size without
processing anything:

```bash
python -m phenotypic --pipeline pipe.json --input ./plates --output ./out \
    --image-type Image --dry-run
```

A finished `full` run leaves three things under `--output`:

```text
out/
├── deliverables/            # what you read and share
│   ├── master_measurements.{csv,parquet}   # clean, pre-post archive
│   ├── measurements.{csv,parquet}          # post-applied mirror (the GUI reads this)
│   ├── measurements_by_feature/            # one file per measurer
│   ├── overlays/<dataset>/                 # detection overlay PNGs
│   ├── dashboard.html, processing_report.html
│   └── README.md                           # generated column documentation
├── results/<dataset>/
│   ├── hdf/                # one .h5 per image — the reusable segmentation
│   └── measurements/       # one parquet per image
└── .phenotypic/            # machine state: progress manifest, event log, pipeline copy
```

The `hdf/` directory is what makes the other modes cheap: it holds each image's
object map, so `measure` never has to detect again.

`dashboard.html` is the run-progress and failure surface. Local runs show
progress directly; SLURM runs add a Download tab. Explore measurements in the
Results Viewer or the GUI `/analysis/` app.

### Scoping a run while you iterate

Trying a new detector on a thousand plates to discover it splits every colony is
a bad way to spend an afternoon. Process a random subset first:

```bash
python -m phenotypic --pipeline pipe.json --input ./plates --output ./out \
    --image-type Image --sample 1 --random-seed 42
```

`--sample N` takes N random images **per dataset** (each first-level
subdirectory of `--input` is one dataset). Pair it with `--random-seed` so the
same subset is chosen on the next invocation — otherwise you are comparing two
pipelines on two different images.

### Surviving interruptions

Re-running the same compatible command skips completed work automatically.
Staged GPU pipelines infer the earliest required stage from their HDF, sidecar,
and Stage-3 marker. If everything is already done, the CLI says so and exits:

```bash
python -m phenotypic --pipeline pipe.json --input ./plates --output ./out \
    --image-type Image
# ✓ All images already processed!
```

Three flags control exceptional handling of prior state:

- `--retry-failures` — also re-process exact terminal failures recorded for the
  current computation.
- `--restart` — clear current machine state and start a new lifecycle.
- `--overwrite` — delete the output directory contents first.

Use `--retry-failures` after fixing a terminal scientific-processing failure to
reprocess only the matching failed computations. Infrastructure failures, such
as OOM or timeout, remain pending and are selected by an ordinary repeat call.

## `measure` — new numbers, same segmentation

You ran a detector over 500 plates, then realized you also want texture
measurements. `measure` re-runs the measurement stage against the HDFs already
sitting in `results/<dataset>/hdf/`, using a pipeline you supply:

```bash
python -m phenotypic --mode measure --pipeline pipe.json --output ./out
```

There is no `--input`: the images are discovered from the output root. Passing
one is an error.

Because `measure` re-reads a finished segmentation rather than producing one, it
refuses flags that imply a fresh detection pass or mutate run state. This applies
to `--restart`, `--retry-failures`, `--overwrite`, `--sample`, and `--dry-run`.
`measure` is all-or-nothing over every HDF it finds.

```{note}
Swap the *measurers* freely, but keep the detector consistent with the one that
wrote the HDFs. `measure` reads the stored object map; changing `OtsuDetector`
to `LiDetector` in the pipeline you pass will not re-segment anything.
```

## `recompile` — rebuild the deliverables

`recompile` takes neither `--input` nor `--pipeline`; both are reloaded from the
output root. It re-aggregates the per-image parquets into
`master_measurements.csv`, regenerates any missing overlay PNGs from their HDFs,
rebuilds the progress manifest, and regenerates the progress dashboard.

```bash
uv run python -m phenotypic --mode recompile --output ./out
```

Before aggregation, recompile preflights and automatically migrates
bundle-owned authoritative metadata to the flat `Metadata_<Label>` namespace.
The same ordering applies locally and on SLURM. A blocked or failed migration
aborts before aggregate outputs are published; a canonical bundle is an
idempotent no-op. The migration receipt printed by the CLI can be used for
rollback. HDF inputs are migrated copy-on-write, and an external file passed via
`--metadata` is copied byte-for-byte to `deliverables/metadata.csv` before work,
then normalized in memory. Neither the external file nor that provenance
snapshot is rewritten by migration.

Reach for it when the *numbers* are right but the *presentation* is not:

- You deleted or corrupted `dashboard.html`.
- A SLURM array finished its images but died before the aggregation step.
- You want to attach a plate-layout metadata CSV to a finished run:

```bash
python -m phenotypic --mode recompile --output ./out --metadata plate_layout.csv
```

`--metadata` left-joins the CSV onto the measurements mirror on shared columns.
Every CSV row survives — one that matches no measured object is kept with null
measurements and `QC_MetadataOnly` set to `true`, so a strain that was never
detected is visible rather than silently dropped.

The output directory must already exist, and passing `--pipeline` is an error —
`recompile` deliberately uses the pipeline the run was executed with, not
whatever is on disk now.

```{warning}
`--study` is **silently ignored** in `recompile` mode — the REMBI study fields
are folded into `deliverables/rembi.yaml` during `full` and `measure` runs only.
To attach a `study.yaml` to a finished run, use `--mode measure --study
study.yaml`, which rebuilds `rembi.yaml` without re-detecting.
```

## `process` — export a layer, skip the analysis

`process` is an **apply-only export**. It runs `pipeline.apply()` and writes one
image layer per input, mirroring the input tree under `--output`. No
measurement, no deliverables, no QC, no dashboard.

```bash
python -m phenotypic --mode process --pipeline pipe.json \
    --input ./plates --output ./proc --image-type Image --layer objmap
```

```text
proc/
├── set_a/
│   ├── plate_00.png        # mirrors plates/set_a/plate_00.tiff
│   └── plate_01.png
└── .phenotypic/
```

`--layer` is required in `process` mode and rejected in every other mode. Each
layer is written at its native dtype through the accessor's `imsave`, with
PhenoTypic metadata embedded:

| `--layer` | File | Dtype |
|-----------|------|-------|
| `rgb` | `.tiff` | integer, at the source bit depth |
| `gray` | `.tiff` | float, full precision |
| `detect_mat` | `.tiff` | float, full precision |
| `objmap` | `.png` | 16-bit raw labels |

This is the mode for feeding another tool. Export `detect_mat` to inspect what
your enhancement chain actually hands the detector; export `objmap` to import
segmentations into CellProfiler, Fiji, or a training set. Note that `objmap`
holds **raw label values**, not a rendered image — opening it in a viewer shows
a near-black frame, because label 3 is the pixel value 3.

Flags that only make sense for measurement output are ignored with a warning
rather than an error:

```console
$ python -m phenotypic --mode process ... --layer gray --no-qc
Warning: --no-qc is ignored in --mode process (no measurement/aggregation output).
```

The same applies to `--metadata` and `--no-dataset-column`.

## Parameters worth knowing

### Loading the images

`--image-type` defaults to `GridImage`, which imposes an arrayed layout. Use
`--nrows` / `--ncols` to override the pipeline's grid preset (the fallback is
8 × 12); pass `--image-type Image` for non-arrayed plates.

`--detect-mode` selects the channel the detector thresholds. It defaults to
`gray`, but colony contrast often lives elsewhere — `LabA` separates
red-pigmented colonies from agar far better than luminance does. The full set is
`gray`, `red`, `green`, `blue`, `MinRGB`, `HsvS`, `HsvV`, `InvS`, `LabL`,
`LabA`, `LabB`.

```{warning}
`--ext` no longer selects the output format. Forward runs write a single `.h5`
per image; the flag now only affects overlay PNG rendering and is deprecated.
```

### Scaling out

`--njobs` sets local parallelism and defaults to `-1` (all cores). Images are
large and operations copy — on a memory-constrained machine, capping `--njobs`
is often faster than swapping.

`--slurm` submits to a cluster instead. It takes repeated `KEY=VALUE` pairs:

```bash
python -m phenotypic --pipeline pipe.json --input ./plates --output ./out \
    --slurm slurm_partition=compute \
    --slurm slurm_account=myproj \
    --slurm mem_gb=16 \
    --slurm time=120
```

Use the `slurm_` prefix for standard SBATCH directives, or the convenience keys
`mem_gb` and `time`. **`time` is in minutes, as an integer** — `time=120`, not
`time=02:00:00`. The CLI returns as soon as the jobs are submitted; add `--wait`
to block and monitor them. A staged GPU run without `--wait` prints
`PROCESSING SUBMITTED` and leaves aggregation, reports, and README publication
to its dependent finalizer. It does not print a premature completion summary.
With `--wait`, the CLI follows the active orchestration epoch through the
finalizer completion marker. Ctrl+C detaches monitoring while the jobs continue.
`--force-local` overrides SLURM detection, which is
what you want when testing on a login node.

`--checkpoint-interval N` inserts checkpoint tasks every N images in a SLURM
array so a walltime kill loses at most N images of progress.

### GPU detectors

A pipeline containing a `GpuDetector` automatically splits into three stages —
CPU preprocess, resident-model GPU detect, CPU measure — reusing the per-image
HDF. You do not opt in; you only tune it:

- `--gpu-slurm KEY=VALUE` — SBATCH profile for the GPU stage. It **inherits and
  deltas over `--slurm`**, so put the GPU partition and account here and leave
  the CPU profile in `--slurm`. `slurm_gpus_per_node=1` is added automatically.
- `--gpu-shards N` — parallel whole-GPU tasks (SLURM only; ignored locally).
  Set it to your concurrent-GPU count.
- `--gpu-workers-per-gpu W` — reserved for future per-GPU replica packing. The
  current staged worker runs one resident model per GPU shard.

```bash
python -m phenotypic --pipeline gpu_pipe.json --input ./plates --output ./out \
    --slurm slurm_partition=cpu --slurm mem_gb=32 \
    --gpu-slurm slurm_partition=gpu --gpu-slurm slurm_account=myproj \
    --gpu-shards 4
```

GPU distribution uses `--gpu-shards` to spread work across whole GPUs and
nodes. `--gpu-workers-per-gpu` is currently reserved and does not add replicas. There is no
per-forward batching knob — the segmentation models PhenoTypic targets do not
broadly support batched inference.

Stage 2 survives walltime through dependent continuation rounds. After a GPU
array terminates, a controller checks which atomic sidecars remain absent and
submits another array only when work remains. Workers do not catch walltime
signals and do not call `scontrol requeue`. Cancellation fences the run epoch
before cancelling every active job in its ledger.

### Shaping the output

- `--overlay-alpha` (default `0.3`) — opacity of the label overlay in the
  saved overlay PNGs.
- `--no-dataset-column` — drop the `Metadata_Dataset` column, which is
  included by default and is what makes multi-dataset analysis possible.
- `--no-qc` — skip the QC compute step. QC otherwise runs whenever the pipeline
  has a non-empty `qc` section, and re-running it resets GUI review progress.
- `--skip-validation` — bypass pipeline validation. Only worth it for pipelines
  you have already run.

## A typical iteration loop

```bash
# 1. Preview the plan.
python -m phenotypic -p pipe.json -i ./plates -o ./out --dry-run

# 2. Tune the detector against a fixed random subset.
python -m phenotypic -p pipe.json -i ./plates -o ./trial --sample 5 --random-seed 42

# 3. Commit to the full dataset on the cluster.
python -m phenotypic -p pipe.json -i ./plates -o ./out \
    --slurm slurm_partition=compute --slurm mem_gb=16 --slurm time=240

# 4. Retry exact terminal scientific failures after fixing their cause.
python -m phenotypic -p pipe.json -i ./plates -o ./out --retry-failures

# 5. Add a texture measurer. No re-detection.
python -m phenotypic -m measure -p pipe_with_texture.json -o ./out

# 6. Attach the plate layout and rebuild the dashboard.
python -m phenotypic -m recompile -o ./out --metadata plate_layout.csv
```

Steps 5 and 6 are the payoff: neither one re-segments a single colony.

## Where to go next

- [CLI Batch Processing](cli_batch_processing.md) — the condensed recipe.
- [SLURM Pipelines](../../how_to/pages/slurm_pipelines.md) — cluster execution
  in depth.
- [GPU Detection Setup](../../how_to/pages/gpu_detection_setup.md) — staging
  model weights on offline compute nodes.
- [CLI Reference](../../api_reference/cli_reference.rst) — every flag.
