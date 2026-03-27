# CLI Batch Processing

Process an entire directory of plate images using the PhenoTypic command-line
interface.

## Basic Usage

```bash
python -m phenotypic pipeline.json /path/to/plates/ /path/to/output/
```

**Arguments:**

1. `pipeline.json` — Pipeline configuration (created with `pipeline.to_json()`)
2. Input directory — Folder containing plate images
3. Output directory — Where results are saved

## Grid Plates

```bash
python -m phenotypic pipeline.json /plates/ /output/ \
    --image-type GridImage --nrows 8 --ncols 12 --ext .png
```

## Parallelism

```bash
python -m phenotypic pipeline.json /plates/ /output/ --n-jobs 4
```

Omit `--n-jobs` to use all available CPU cores.

## Resume After Interruption

```bash
python -m phenotypic pipeline.json /plates/ /output/ --resume
```

Add `--retry-failures` to reprocess only failed images.

## Testing

```bash
# Dry run: validate pipeline and list images without processing
python -m phenotypic pipeline.json /plates/ /output/ --dry-run

# Process 5 random images as a test
python -m phenotypic pipeline.json /plates/ /output/ --sample 5
```
