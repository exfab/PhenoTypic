# PhenoTypic Processing Results

Generated: 2026-11-13 16:51:22

Pipeline: `tiny_run.pht-pipe.json`

Image Type: GridImage
Grid Dimensions: 8 rows x 12 columns

## Output Structure

```
output_folder/
+-- deliverables/                     # User-facing run outputs
|   +-- master_measurements.parquet   # Clean archive: every measured object, un-joined (Parquet only)
|   +-- measurements.tsv              # Metadata-joined mirror, tab-separated (refreshed on every run)
|   +-- measurements.csv              # Same mirror, comma-separated: the file the GUI results viewer reads and curates (kept until #283)
|   +-- measurements.parquet          # Parquet companion of the mirror
|   +-- measurements_by_feature/      # One TSV + Parquet per measurer: shared context columns + that measurer's columns
|   +-- measurements_by_category/     # One TSV + Parquet per measurement category (see Measurement Categories)
|   +-- metrics.json                  # Run summary: image/object/phantom counts, PhenoTypic version, pipeline sha256
|   +-- schema.json                   # Every column of each published table: dtype, unit, measurer, schema member, description
|   +-- pipeline.pht-pipe.json        # Reproducibility spec (operations + filters + model); seeded from --pipeline at start
|   +-- <AnalysisClass>.tsv           # Class-named model-fit output
|   +-- <AnalysisClass>.parquet       # Parquet companion
|   +-- analysis_manifest.json        # Named-analysis artifact index and checksums
|   +-- plots/                        # Configured plot outputs and page manifests
|   +-- dashboard.html                # Live processing dashboard
|   +-- overlays/                     # Detection overlay PNGs (per-dataset subfolders, one per input image)
|   +-- README.md                     # This file
+-- results/                          # All dataset results
|   +-- ds/
|       +-- zarr/                     # Processed images as one OME-Zarr store per input (<stem>.ome.zarr: layers + objmap label image + grid state, plus the authoritative object measurements at tables/measurements/table.parquet)
|       +-- measurements/             # Legacy per-image Parquets — only in a tree written by an older release, never created by this run
+-- .phenotypic/                      # Hidden machine-state cache
|   +-- processing_state.json         # Continuation/state tracking
|   +-- processing_events.log         # Append-only event log
|   +-- logs/                         # Execution logs
|   |   +-- slurm/                    # SLURM job logs (if applicable)
|   +-- slurm_scripts/                # Generated SLURM scripts (if applicable)
```

**Generated tables are tab-separated (`.tsv`).** Releases before this change
wrote `.csv` for the per-feature and per-category splits and for
`<AnalysisClass>`; re-running finalize into such a folder replaces those with
`.tsv` and deletes the old `.csv` files it replaces. Two CSVs stay: `metadata.csv`
(or `metadata.tsv`; the byte-for-byte copy of the table you passed to
`--metadata`) and `error_analysis.csv` (written by the results viewer).
`measurements.csv` is kept because the results viewer's curation writes to it
and to `measurements.parquet`, **not** to `measurements.tsv`: until #283 lands,
curated edits reach `measurements.csv` and `measurements.parquet` but not
`measurements.tsv`. `--metadata` must now be a `.csv` or `.tsv` table (the
delimiter follows the suffix); any other suffix, such as a comma-separated
`.txt` that earlier releases read, is refused, so rename it.

**`metrics.json` and `schema.json` are for automation** (for example
`dvc metrics show`). `metrics.json` counts the images the run accepted,
succeeded on and failed terminally, the objects measured and the
metadata-only phantom rows, and names the PhenoTypic version and the sha256 of
the pipeline every aggregated table came from (`null` when they differ or it
is not recorded). `schema.json` lists, for every published table (keyed by
its path under `deliverables/` without the suffix), each column's Parquet
dtype, unit, producing measurer, schema member and description. Both are
rewritten each time a full, measure or recompile run publishes its tables
(`--mode migrate` does not write them), and they hold no timestamps, so
identical runs give byte-identical files.

## Saved Layers

Each dataset directory contains the following:

| Path | Format | Description |
|------|--------|-------------|
| `zarr/` | OME-Zarr (NGFF 0.5) | Processed image (layers + `objmap` label image + metadata + grid state) saved as a single `<stem>.ome.zarr` store directory per input image, reloadable via `Image.load_zarr` / `GridImage.load_zarr`. |
| `zarr/<stem>.ome.zarr/tables/measurements/table.parquet` | Parquet | Authoritative per-object measurements for that image, embedded in its store and described by the store's root `attributes.phenotypic.tables.measurements`. This run writes no separate per-image `measurements/` folder. |

Each `.ome.zarr` store is a standard OME-NGFF image: **napari, QuPath, and
Vizarr open it directly, with no PhenoTypic install** — drag the store
directory in, or point the viewer at its path. It carries a resolution
pyramid, so a viewer reads only the levels and tiles it needs instead of
decoding the whole plate.

Detection overlay PNGs are written per input image under
`deliverables/overlays/<dataset>/` (always written for forward runs).

Use the Results Viewer and its `/analysis/` workspace for interactive result
exploration.

## Measurements

The following measurements are extracted for each detected object:

### Object

| Column | Description |
|--------|-------------|
| `Object_Label` | Unique integer label identifying each detected object. |

### Bbox

| Column | Description |
|--------|-------------|
| `Bbox_CenterRR` | Row coordinate of the colony's centroid, the mean position of its pixels (skimage ``centroid``). |
| `Bbox_MinRR` | The smallest row coordinate of the bounding box. |
| `Bbox_MaxRR` | The largest row coordinate of the bounding box. It is one past the colony's last pixel row (exclusive, as in Python slicing), so rows MinRR to MaxRR - 1 hold the colony and ``image[MinRR:MaxRR]`` c... |
| `Bbox_CenterCC` | Column coordinate of the colony's centroid, the mean position of its pixels (skimage ``centroid``). |
| `Bbox_MinCC` | The smallest column coordinate of the bounding box. |
| `Bbox_MaxCC` | The largest column coordinate of the bounding box. It is one past the colony's last pixel column (exclusive, as in Python slicing), so columns MinCC to MaxCC - 1 hold the colony and ``image[:, MinC... |
| `Bbox_IntensityWeightedCenterRR` | The intensity-weighted center row coordinate of the object (skimage ``centroid_weighted``). |
| `Bbox_IntensityWeightedCenterCC` | The intensity-weighted center column coordinate of the object (skimage ``centroid_weighted``). |
| `Bbox_DistWeightedCenterRR` | Row coordinate of the colony's distance-weighted centroid: the mean position of its pixels, each weighted by its distance to the colony's edge. It is Σ dt·row / Σ dt over the Euclidean distance tra... |
| `Bbox_DistWeightedCenterCC` | Column coordinate of the colony's distance-weighted centroid: the mean position of its pixels, each weighted by its distance to the colony's edge. It is Σ dt·col / Σ dt over the Euclidean distance ... |


### Grid

| Column | Description |
|--------|-------------|
| `Grid_RowNum` | Zero-based row number, from 0 at the top, of the grid cell that holds the colony's centroid (the mean position of its pixels). A colony whose centroid lies outside the grid is assigned to the neare... |
| `Grid_RowIntervalStart` | Reserved for the pixel row where the colony's grid row starts; no PhenoTypic operation writes this column. |
| `Grid_RowIntervalEnd` | Reserved for the pixel row where the colony's grid row ends; no PhenoTypic operation writes this column. |
| `Grid_ColNum` | Zero-based column number, from 0 at the left, of the grid cell that holds the colony's centroid (the mean position of its pixels). A colony whose centroid lies outside the grid is assigned to the n... |
| `Grid_ColIntervalStart` | Reserved for the pixel column where the colony's grid column starts; no PhenoTypic operation writes this column. |
| `Grid_ColIntervalEnd` | Reserved for the pixel column where the colony's grid column ends; no PhenoTypic operation writes this column. |
| `Grid_RowMajorIdx` | Zero-based number of the colony's grid cell, counted left to right along each row and then down the rows. It is RowNum × (number of grid columns) + ColNum. Row major is the standard in most program... |
| `Grid_ColMajorIdx` | Zero-based number of the colony's grid cell, counted top to bottom down each column and then across the columns. It is ColNum × (number of grid rows) + RowNum. Lab automation logic uses column-majo... |


### Size

| Column | Description | Categories |
|--------|-------------|------------|
| `Size_Area` | Total number of pixels the colony occupies. | Starting Metrics |
| `Size_IntegratedIntensity` | The sum of the object's grayscale pixel values. |  |
| `Size_Perimeter` | Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to... |  |
| `Size_ConvexArea` | Area of the smallest convex polygon that completely contains the colony, computed from the convex hull of its pixel centers. Represents the colony's "filled-in" appearance if all indentations and h... |  |
| `Size_BboxArea` | Area of the smallest rectangle that completely contains the colony. Represents the total spatial shape of the colony including any empty space. In high-throughput assays, this helps assess colony p... |  |
| `Size_MajorAxisLength` | Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have g... |  |
| `Size_MinorAxisLength` | Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and... |  |
| `Size_MinFeretDiameter` | Minimum caliper diameter - the shortest distance between two parallel tangent lines touching opposite sides of the colony. Represents the narrowest dimension of the colony regardless of orientation... |  |
| `Size_MaxFeretDiameter` | Maximum caliper diameter - the longest distance between two parallel tangent lines touching opposite sides of the colony. Represents the maximum dimension of the colony regardless of orientation. O... |  |
| `Size_InscribedRadius` | Radius of the largest circle that fits entirely inside the colony, equal to the maximum of the colony's Euclidean distance transform: the distance from the colony's deepest interior point to its ne... |  |
| `Size_MedianRadius` | Median distance from the colony center to its boundary over the radial signature: the boundary distance sampled in equal angular directions (360 by default), keeping the outermost boundary crossing... |  |
| `Size_MeanRadius` | Mean distance from the colony center to its boundary over the radial signature (see MedianRadius for the center and sampling). For an ideal disk it equals the disk radius; for an ideal 100 x 20 pix... |  |
| `Size_RobustMeanRadius` | Symmetrically trimmed mean (20% from each end by default) of the radial signature (see MedianRadius). Because the signature is sampled by angle, a narrow runner contributes only its angular width a... | Starting Metrics |
| `Size_MaxRadius` | Largest distance from the colony center to its boundary over the radial signature (see MedianRadius): the colony's furthest reach. For an ideal disk it equals the disk radius; for an ideal 100 x 20... |  |


## Measurement Categories

### Starting Metrics

Per-colony values to examine first: area and robust mean radius, circularity and eccentricity, mean grayscale intensity, and the CIELAB medoid colour.

Written to `deliverables/measurements_by_category/StartingMetrics.tsv` (and `.parquet`), alongside every context column.

- `Size_Area`
- `Size_RobustMeanRadius`

---

## About

Documentation: https://exfab.github.io/PhenoTypic/ (source:
https://github.com/exfab/PhenoTypic)

This table was written by PhenoTypic v0.21.0. In the docs' version menu, choose that version, or the nearest earlier one listed.