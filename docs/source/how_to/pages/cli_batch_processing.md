# CLI Batch Processing

Process an entire directory of plate images using the PhenoTypic command-line
interface.

## Basic Usage

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /path/to/plates/ --output /path/to/output/
```

**Required path options:**

1. `--pipeline pipeline.json` — Pipeline configuration (created with `pipeline.to_json()`)
2. `--input /path/to/plates/` — Folder containing plate images
3. `--output /path/to/output/` — Where results are saved

## Grid Plates

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ \
    --image-type GridImage --nrows 8 --ncols 12 --ext .png
```

## Parallelism

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --njobs 4
```

Omit `--njobs` to use all available CPU cores.

## Resume After Interruption

```bash
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --resume
```

Add `--retry-failures` to reprocess only failed images.

## Testing

```bash
# Dry run: validate pipeline and list images without processing
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --dry-run

# Process 5 random images as a test
python -m phenotypic --mode full --pipeline pipeline.json --input /plates/ --output /output/ --sample 5
```
