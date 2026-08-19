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
