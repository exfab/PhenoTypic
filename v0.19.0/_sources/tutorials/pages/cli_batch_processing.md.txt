# CLI Batch Processing

Process an entire directory of plate images using the PhenoTypic command-line
interface.

This is the condensed recipe for the default `full` mode. For what the other
three modes (`measure`, `recompile`, `process`) produce and which flags each
one accepts, see [CLI Execution Modes](cli_modes.md).

## Basic Usage

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /path/to/plates/ --output /path/to/output/
```

**Required path options:**

1. `--pipeline pipeline.json` — Pipeline configuration (created with `pipeline.to_json()`)
2. `--input /path/to/plates/` — Folder containing plate images
3. `--output /path/to/output/` — Where results are saved

## Grid Plates

`--image-type` already defaults to `GridImage`; pass `--nrows` / `--ncols` to
override the pipeline's grid preset (which itself falls back to 8 × 12).

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ \
    --image-type GridImage --nrows 8 --ncols 12
```

## Parallelism

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --njobs 4
```

Omit `--njobs` to use all available CPU cores.

## Continue After Interruption

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/
```

Run the same command again to continue compatible unfinished work. Add
`--retry-failures` to also re-process images that previously failed, instead of
skipping them.

## Testing

```bash
# Dry run: validate pipeline and list images without processing
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --dry-run

# Process 5 random images per dataset as a test
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ \
    --sample 5 --random-seed 42
```

`--sample` draws N images from *each* dataset (each first-level subdirectory of
`--input`). Pass `--random-seed` to draw the same subset every time.

## Run Preflight Checks

Before any image is processed, and before anything under `--output` is created,
cleared, or submitted, the CLI runs a **run preflight**: read-only checks of
the pipeline, the environment, the cluster profile, the input files' headers,
the metadata CSV, and the output location. The preflight never decodes a
pixel, so a single bad image cannot make it fail.

Each problem is reported as a finding with a code, the affected paths, and a
remedy:

```text
✗ Error [PF-GRID-IMAGE]: grid operation(s) meas:spread require a GridImage, but the run uses --image-type Image; each such image would fail with GridImageInputError
    → Run with --image-type GridImage (and --nrows/--ncols for your plate layout), or remove the grid operations listed above.
```

- An **error** refuses the run. The CLI exits with status 1 and says that
  nothing under `--output` was changed.
- A **warning** is printed and the run continues.
- A finding about input files is an error when it affects every input, and a
  warning listing the affected files when it affects only some of them: the
  other images can still be processed.

`--dry-run` runs the same preflight, so it is the way to check a
configuration without starting a run. `--skip-validation` skips validation:
the execution-configuration check, the pipeline load check, and the run
preflight. It does not skip the checks that protect the output folder or that
the run cannot proceed without: the GPU placement refusal (a GPU detector in
a position the staged GPU engine cannot run), the refusal of a `--restart` or
`--overwrite` that would delete the run's own inputs, and the parsing of
`--metadata`, `--bit-depth`, and `--gpu-slurm` values.

| Code | Reported when | Severity | Remedy |
|---|---|---|---|
| `PF-PIPELINE-LOAD` | The pipeline file cannot be read: invalid JSON, a duplicated key, or invalid parameters | error | Fix the pipeline file |
| `PF-CUSTOM-OP` | The pipeline names a class that is not in the `phenotypic` namespace | error | List a self-registering module in `PHENOTYPIC_PRELOAD_MODULES` |
| `PF-GRID-IMAGE` | A grid operation or grid measurer runs on images read as plain `Image` | by reach | `--image-type GridImage`, or remove the grid operations |
| `PF-GRID-PRESET` | The pipeline's `nrows` and `ncols` preset adds a grid finder, but images are read as plain `Image` | by reach | `--image-type GridImage`, or remove the preset |
| `PF-NO-DETECTOR` | A `full` run has no object detector in the pipeline's operations | error | Add a detector, or use `--skip-validation` if a custom operation writes the object map |
| `PF-MISSING-MODULE` | An operation needs a package that is not installed | error | Install the extra the message names |
| `PF-LICENSE` | A detector needs gated weights whose license the run has not accepted | error | `export PHENOTYPIC_ACCEPT_MODEL_LICENSE=<name>` |
| `PF-WEIGHTS-UNCACHED` | A detector's weights are not in the local cache | warning | Pre-download them on a node with network access |
| `PF-SBATCH-REJECTED` | `sbatch --test-only` names a configuration fault in a profile the run submits: an invalid partition, account, QoS, GRES or option | error | Fix the option `sbatch` names |
| `PF-SBATCH-UNAVAILABLE` | A profile could not be confirmed: no `sbatch`, a timeout, a controller fault, or a failure a real submission may queue through (e.g. every node of the partition drained) | warning | Read `sbatch`'s message; a real submission still reports real failures |
| `PF-TIME-OVER-PARTITION` | The requested time exceeds the partition's `MaxTime` (for a partition list, per the cluster's `EnforcePartLimits`) | warning | Lower the time, or choose another partition |
| `PF-SLURM-LIMIT` | The staged GPU engine cannot fit the cluster's `MaxSubmitJobs` or array limit | error | Reduce `--gpu-shards`, or ask for a higher limit |
| `PF-GPU-PARTITION` | `sinfo` lists the GRES of the partition a GPU job runs in, and none is a GPU | error | `--gpu-slurm slurm_partition=<gpu-partition>` (or `--slurm` for `--mode process`) |
| `PF-HEADER-UNREADABLE` | An input file's header cannot be read | by reach | Replace or remove the files listed |
| `PF-CHANNELS` | An input has 2, or 5 or more, channels, which `Image.imread` refuses | by reach | Convert the files to grayscale or RGB(A) |
| `PF-DETECT-MODE-GRAY` | A color `--detect-mode` is set, but inputs are grayscale | by reach | `--detect-mode gray`, or supply RGB images |
| `PF-RGB-OP-GRAY` | A color operation or measurer runs on grayscale inputs | by reach | Supply RGB images, or remove the color operations |
| `PF-BIT-DEPTH` | `--bit-depth` contradicts the depth the files store (JPEG is exempt: it is always read as 8-bit) | by reach | Drop `--bit-depth`, or set it to the stored depth |
| `PF-RAW-NO-RAWPY` | RAW inputs, but `rawpy` is not installed (it is not installed on Windows) | by reach | Install `rawpy`, or convert the RAW files to TIFF |
| `PF-STEM-COLLISION` | Two inputs in one dataset share a name without extension (e.g. `a.png`, `a.tif`) | error | Rename or move one file of each pair |
| `PF-META-PARSE` | The `--metadata` CSV does not parse | error | Fix the CSV |
| `PF-META-ALIAS` | The CSV has a legacy and a current spelling of one column, with conflicting values | error | Keep one spelling |
| `PF-META-NO-KEYS` | The CSV shares no column that identifies an image | error; warning when its keys may be measurement columns such as `Grid_RowNum`, or metadata the images carry themselves (PhenoTypic exports, custom operations) | Add an `ImageName` column |
| `PF-META-DUP-KEYS` | A join key repeats in the CSV | as `PF-META-NO-KEYS` | Make each key unique |
| `PF-META-UNMATCHED` | Some images have no metadata row | warning | Add rows, or match `ImageName` to the file names |
| `PF-META-ORPHANS` | Some metadata rows match no image | warning | Expected for wells that grew nothing |
| `PF-META-UNVERIFIED` | The CSV joins on columns only measurements carry | warning | None needed; the join is checked at finalization |
| `PF-OUTPUT-UNWRITABLE` | `--output` (or its nearest existing parent) is not writable | error | Choose another `--output`, or fix permissions |
| `PF-OUTPUT-SPACE` | A `full` run's output disk has less free space than the inputs' size (a heuristic) | warning | Free space, or choose another `--output` |
| `PF-NODE-LOCAL` | On a SLURM run, a path workers read or write is on node-local storage | warning | Move it to shared storage |
| `PF-CHECK-CRASHED` | A check itself failed | warning | Report it; the run is not blocked |

"By reach" means an error when every input is affected, and a warning listing
the affected inputs otherwise.

### Changes that can refuse a run that used to start

These came with the run preflight. Each refusal replaces a run that would have
failed, or silently produced the wrong result, later:

- **SAM3 needs an explicit license acceptance.** Set
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3` (comma-separate several models, e.g.
  `sam3,dinov3`). No GPU detector prompts for a license during a run any more;
  an unaccepted license is reported as `PF-LICENSE` before the run starts.
- **Camera RAW files are decoded as sensor data through `rawpy`.** Where
  `rawpy` is not installed, which includes Windows, RAW inputs are refused
  (`PF-RAW-NO-RAWPY`) instead of being read as the embedded preview image.
  Convert them to TIFF first on such a machine. A resumed run re-processes its
  RAW images, because they now decode to different pixels; other images are
  reused as before.
- **A pipeline file with a duplicated JSON key no longer loads**, in every
  mode including `--mode recompile`, and in `ImagePipeline.from_json`. Such a
  file never ran as written: the last duplicate silently replaced the first and
  ran in the first one's position. Remove the duplicate.
- **Custom operations** defined outside PhenoTypic resolve in every process
  that loads the pipeline, including local parallel workers, once a module that
  registers them is listed in `PHENOTYPIC_PRELOAD_MODULES` (`PF-CUSTOM-OP`
  otherwise). A module name that cannot be imported stops the run at startup.
