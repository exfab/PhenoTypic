# CLI Batch Processing

Process an entire directory of plate images using the PhenoTypic command-line
interface.

## Basic Usage

```bash
python -m phenotypic --pipeline pipeline.json --input /path/to/plates/ -o /path/to/output/
```

**Required path options:**

1. `--pipeline pipeline.json` — Pipeline configuration (created with `pipeline.to_json()`)
2. `--input /path/to/plates/` — Folder containing plate images
3. `-o /path/to/output/` — Where results are saved

## Grid Plates

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ \
    --image-type GridImage --nrows 8 --ncols 12 --ext .png
```

## Parallelism

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --n-jobs 4
```

Omit `--n-jobs` to use all available CPU cores.

## Resume After Interruption

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --resume
```

Add `--retry-failures` to reprocess only failed images.

## Testing

```bash
# Dry run: validate pipeline and list images without processing
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --dry-run

# Process 5 random images as a test
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/ --sample 5
```
