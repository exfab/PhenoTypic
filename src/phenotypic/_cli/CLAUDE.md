# CLI Module

## Basic Usage

```bash
pixi run python -m phenotypic pipeline.json ./images
pixi run python -m phenotypic pipeline.json ./images -o ./results
pixi run python -m phenotypic pipeline.json ./plates \
    --image-type GridImage --nrows 16 --ncols 24
pixi run python -m phenotypic pipeline.json ./images --n-jobs -1
```

---

## Output Structure

Dataset-first hierarchy under `results/`:

```
output_dir/
├── results/
│   ├── dataset_1/
│   │   ├── measurements/ (*.parquet)
│   │   ├── overlays/ (*.png)
│   │   └── rgb/, gray/, detect_mat/, objmap/  # Always saved
│   └── dataset_2/
├── logs/ (+ slurm/)
├── pipeline.json              # Copy of input pipeline for reproducibility
├── master_measurements.csv
├── processing_state.json
├── processing_report.html
└── README.md
```

- Input directory structure preserved (1 level deep)
- Single images create `single_image/` folder

---

## Dashboard & Apache Serving

The generated `dashboard.html` is designed for **Apache HTTP Server** file sharing (e.g., HPCC `public_html` or `~user/` directories):

- All fetch paths are **relative** — no absolute URLs or CDN dependencies
- `marked.min.js` is vendored inline (works behind firewalls without CDN access)
- Dashboard at output root fetches from `progress/manifest.json` and `progress/failures.jsonl`
- Apache must serve `.json` and `.jsonl` files (verify MIME types if dashboard shows errors)
- The Download tab with `wget` commands only appears in SLURM execution mode

---

## CLI Options

### Required

| Argument | Description |
|----------|-------------|
| `PIPELINE_JSON` | Pipeline config file (.json/.yaml) |
| `INPUT_PATH` | Image file or directory |

### Output

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | Auto-generated | Output directory |
| `--no-dataset-column` | False | Exclude `Metadata_Dataset` from master CSV |

### Image Type

| Option | Default | Description |
|--------|---------|-------------|
| `--image-type` | `GridImage` | `GridImage` or `Image` |
| `--nrows` | `8` | Grid rows |
| `--ncols` | `12` | Grid columns |
| `--bit-depth` | Auto | Input bit depth (8/16) |

### Execution

| Option | Default | Description |
|--------|---------|-------------|
| `--n-jobs` | `-1` | Parallel jobs (-1 = all cores) |
| `--force-local` | False | Force local even if SLURM available |
| `--skip-validation` | False | Skip pipeline validation |

### Save Layers (always saved)

| Layer | Directory | Format |
|-------|-----------|--------|
| RGB | `rgb/` | `--ext` (default: tiff) |
| Grayscale | `gray/` | `--ext` (default: tiff) |
| Detection matrix | `detect_mat/` | `--ext` (default: tiff) |
| Object map | `objmap/` | PNG (always) |
| Overlays | `overlays/` | PNG (always) |
| Measurements | `measurements/` | Parquet (zstd) |

| Option | Default | Description |
|--------|---------|-------------|
| `--ext` | `tiff` | Extension for rgb, gray, detect_mat |
| `--overlay-alpha` | `0.3` | Alpha for overlay compositing |

### Processing Modes

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview without executing |
| `--sample N` | Process N random images per dataset |
| `--random-seed` | Seed for `--sample` reproducibility |
| `--resume` | Resume from checkpoint |
| `--retry-failures` | Include failures when resuming (requires `--resume`) |
| `--restart` | Restart, clearing previous state |
| `--overwrite` | Delete existing output directory before processing |

---

## SLURM Cluster Execution

```bash
pixi run python -m phenotypic pipeline.json ./images \
    --slurm slurm_partition=compute --slurm slurm_account=lab_proj \
    --slurm mem_gb=32 --slurm time=120 --wait
```

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| `slurm_partition` | Partition name |
| `slurm_account` | Billing account |
| `slurm_qos` | QoS tier |
| `time` | Wall time (minutes, auto-converts to HH:MM:SS) |
| `mem_gb` | Memory per node (GB) |
| `slurm_cpus_per_task` | CPUs per task |
| `slurm_mail_type` / `slurm_mail_user` | Email notifications |

### Advanced Parameters

`slurm_nodes`, `slurm_mem`, `slurm_mem_per_cpu`, `slurm_gpus_per_node`, `slurm_constraint`

---

## Best Practices

- Test first: `--sample 5 --dry-run`
- Always check `overlays/` for visual QC
- Use `--n-jobs -1` for max parallelism
