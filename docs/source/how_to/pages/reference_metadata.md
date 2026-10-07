# Subtract a media blank with reference metadata

Some operations need a value from your experiment metadata while they run, and
sometimes that value names *another image*. `SubtractBlank` is the first: each
frame of a time-lapse plate names its media-blank frame in a metadata column,
and the blank's signal is subtracted from the frame's `detect_mat` before
detection. This page covers the metadata table, the Python, CLI and GUI
surfaces, where to place the operation, and the errors you can meet.

## What it is for

On a time-lapse plate series (for example, a *Ganoderma* plate imaged every few
hours) the first frame is taken before the colonies grow, so it shows only
agar, lid glare and scanner vignetting. Those are present in every later frame
of that plate. Subtracting frame 0 from frame *n* cancels them and leaves the
growth since frame 0, which matters most for faint, filamentous mycelium that an
estimate of the background taken from the image itself would partly absorb.

For a single image with no blank frame, use `SubtractGaussian` or
`SubtractRollingBall` instead.

## The table

One row per image, or one row per colony. Two columns matter:

| Column | Holds |
|---|---|
| `ImageName` (or `Metadata_ImageName`) | The image's file name without its extension. |
| `BlankImage` (or `Metadata_BlankImage`) | The blank frame's file name, with or without its extension. |

```text
ImageName,BlankImage
plate1_t00,plate1_t00
plate1_t04,plate1_t00
plate1_t08,plate1_t00
```

- Headers are canonicalized in memory, so `BlankImage` and
  `Metadata_BlankImage` are the same column. Your file is never rewritten.
- Every value is read as text, so a name such as `000123` stays `000123`.
- A per-colony table repeats each image's name once per colony. The values for
  one image must **agree**; a column that is empty or disagrees across an
  image's rows is an error, not a guess.
- Blanks are found by name in the **same input directory** as the image. A blank
  need not be one of your inputs, but it must live in that directory.
- A table with a `Metadata_Dataset` column also matches on the dataset when the
  run knows one.

## Python

Activate a `ReferenceContext` around the call. Without one, `SubtractBlank`
raises `RefMetadataUnavailableError`, whose message names both fixes.

```python
import phenotypic as pht
from phenotypic import Image, ImagePipeline, ReferenceContext
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank

pipe = ImagePipeline(ops={"sb": SubtractBlank(), "det": OtsuDetector()})

# Which table columns does this pipeline read, and where?
pipe.reference_columns()          # {'sb': ('Metadata_BlankImage',)}

with ReferenceContext("blank_map.csv", image_root="images/plate1"):
    colonies = pipe.apply(Image.imread("images/plate1/plate1_t04.tif"))
```

To see what an operation would be given, without running it, ask the context:

```python
ctx = ReferenceContext("blank_map.csv", image_root="images/plate1")
ctx.lookup("plate1_t04", ["Metadata_BlankImage"])   # {'Metadata_BlankImage': 'plate1_t00'}
ctx.resolve_image("plate1_t00")                      # the file it names
blank = ctx.load_image("plate1_t00")                 # read, cached per process
```

`ctx.has_column(...)`, `ctx.columns`, `ctx.table` and `ctx.table_sha256` describe
the table. `ctx.narrow(dataset=..., image_root=..., images=...)` returns a
context that shares the parsed table. A context built with
`images={"plate1_t00": blank_image}` takes in-memory images instead of files,
which is how the docstring examples run. Entering a second context **replaces**
the first for the duration of its block; contexts do not merge.

The table is validated when the context is built, so a bad table fails before
the first image.

## CLI

Pass the table with `--metadata`. Blank frames are themselves images, so leave
them out of the run with `--image-manifest`, or each one fails its preflight with
`PF-REF-SELF`.

```bash
uv run python -m phenotypic \
    --input images/plate1 --image-manifest later_frames.txt \
    --metadata blank_map.csv --output plate1_run
```

What happens:

- Startup resolves every image's blank once and writes
  `.phenotypic/reference_manifest.json` under `--output`. Local workers, SLURM
  workers and every staged GPU stage read that plan; none is handed the table.
- A per-image reference digest (the resolved values plus each blank file's
  SHA-256) joins the work-id **only for pipelines that read reference metadata**.
  Rerun the same command after editing one plate's blank and only that plate's
  frames are redone. A blank re-exported **while** the run is in progress is
  refused when a worker loads it (its bytes no longer match the plan); those
  frames are left pending, not failed, and the same command re-plans them.
- In full mode the table is also the measurement join, so
  `Metadata_BlankImage` appears in `measurements.csv` and every colony names its
  plate's blank.
- Each image's provenance journal records the table SHA-256, the blank's name
  and its file digest.

**Process mode** (`--mode process`) uses `--metadata` only when a reference
operation actually runs in that mode. It then copies the table, byte for byte,
to `.phenotypic/reference_metadata.csv`; a continuation may omit `--metadata`,
and the snapshot survives `--restart`. For a pipeline with no such operation
the usual "`--metadata` is ignored in `--mode process`" warning still prints.

**`--mode measure` refuses** a pipeline whose measurers read reference
metadata, with a message pointing at `--mode full`. Re-measuring such pipelines
is not supported.

**Early refusals** (before `--overwrite` deletes anything and before
`--dry-run` exits):

- `--overwrite` without `--metadata`: the snapshot the run would fall back to
  is the thing `--overwrite` deletes.
- No `--metadata` and no snapshot to fall back to (`PF-REF-NO-TABLE`).
- A `--metadata` that is not a readable `.csv`. The CLI snapshots and joins it
  as a CSV, so a `.parquet` table (which a notebook `ReferenceContext` accepts)
  must be exported to CSV first.
- A table that `ReferenceContext` cannot read.

**Preflight findings** (`PF-REF-*`). The run preflight reads the table and the
directory listings, and never opens an image:

| Code | Severity | Meaning |
|---|---|---|
| `PF-REF-NO-TABLE` | error | The pipeline reads reference columns, but there is no `--metadata` and no snapshot. |
| `PF-REF-TABLE` | error | The table cannot be read, or has no `Metadata_ImageName`. |
| `PF-REF-COLUMN` | error | The table lacks a column an operation names; the message lists operation path and column. |
| `PF-REF-UNMATCHED` | warning | Input images with no row in the table. |
| `PF-REF-AMBIGUOUS` | warning | Input images whose rows are empty or disagree for a needed column. |
| `PF-REF-SELF` | warning | Input images that name themselves as their blank. Leave them out with `--image-manifest`. |
| `PF-REF-UNRESOLVED` | warning | A named blank matches no file, or more than one, in its image's input directory. |

A warning that covers every input image is escalated to an error.

## GUI

- **Builder.** The *Image source* column has a **Reference metadata** field.
  Pick a table and a `SubtractBlank` node's `blank_column` becomes a dropdown
  of the table's headers, and previews run inside a `ReferenceContext` built
  from it, resolving blanks in the preview image's directory. The picked path is
  session state and is never saved into the pipeline. With no table, the node's
  preview shows the unavailable-context message instead of an image. Like the
  builder's other file pickers, the field is confined to the GUI's image root
  (`phenotypic-gui --root`): the table must be a `.csv` or `.parquet` file under
  that directory, a relative path is resolved against it, and a symlink is
  followed and judged by where it lands. Any other path is refused without
  reading the file, so put the table under the image root (beside the images
  is fine).
- **Run console.** When the loaded pipeline reads reference metadata, a warning
  names the columns and **Run is disabled** until a metadata CSV is included.
  Validate and Run are also refused at launch. An output that already holds a
  previous full run's `deliverables/metadata.csv` needs no CSV: the run falls
  back to that snapshot, as the CLI does. The check looks at the whole
  pipeline and does not model `--mode`, so the CLI's own refusals above still
  apply.

## Placement and polarity

`SubtractBlank` subtracts the blank *as read from disk*, so the target's
`detect_mat` must still be unmodified:

- Put it **before any enhancer**, or **directly after `SetDetectMode`**. Inside a
  `CompositeEnhance` or `CompositeDetector` branch is fine on the same terms.
- It must come after **no `ImageCorrector`**, since a corrector changes the
  target's pixels and the raw blank does not share that change.
- It refuses (`StaleDetectMatError`) when either rule is broken.
- A **later `SetDetectMode` discards the subtraction**, as it discards any
  enhancement. This is not detected.

The blank is taken in the target's detection mode (grey in grey mode, `LabL` in
`LabL` mode) and the difference is clipped to `[0, 1]`:

| `polarity` | Result | Use for |
|---|---|---|
| `"brighter"` (default) | `clip(target - blank, 0, 1)` | Colonies brighter than the media, such as white mycelium on darker agar. |
| `"darker"` | `clip(blank - target, 0, 1)` | Colonies darker than the media; the result is object-bright. |
| `"both"` | `abs(target - blank)` | Either direction counts as colony. |

## Limits and errors

- **Pixel-aligned frames.** The imager must not move between frames. A blank
  whose shape or bit depth differs from the frame is refused
  (`ReferenceImageError`), as is a pair where only one is RGB.
- **Single-channel integer scans** are normalized to float32 `[0, 1]` at load,
  so their `Size_IntegratedIntensity` is in normalized units. A single-channel
  run started before that change must be re-run with `--overwrite`. A
  single-channel float image outside `[0, 1]` is refused.
- **Stores with custom history.** If a store's history names an operation from a
  module that is not importable (not under `phenotypic.`, not already imported,
  not in `PHENOTYPIC_PRELOAD_MODULES`), `SubtractBlank` refuses, because it
  cannot rule out a corrector there.
- **Letter case on macOS.** A blank name that differs from its file only in
  letter case does not resolve; you get a clear "matches 0 files" error.
- **Nested errors.** Inside an `ImagePipeline`, each enclosing pipeline wraps the
  failure in a `RuntimeError`. Walk `__cause__` until you reach a
  `ReferenceContextError` (`ReferenceLookupError`, `ReferenceImageError`,
  `ReferenceTableError`, `RefMetadataUnavailableError` or
  `StaleDetectMatError`, all importable from `phenotypic.sdk_`).
- **Third-party OME-Zarr blanks** are identified by their root `zarr.json` only,
  so a change inside a chunk alone does not change the digest.
- **Digit-only blank names in the join.** The CLI joins `--metadata` onto
  `measurements.csv` with type inference, so a digit-only `Metadata_BlankImage`
  loses its leading zeros there, even though the subtraction used the right file.
- **Tuning.** `phenotypic-tune` refuses a pipeline that reads reference
  metadata.
