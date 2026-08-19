# Describe a run's metadata with REMBI

Attach experiment metadata to a batch run and get a
[REMBI](https://doi.org/10.1038/s41592-021-01166-8)-structured manifest
(`deliverables/rembi.yaml`) alongside your measurements — so a dataset ships
with machine-readable provenance grouped by REMBI module (Study, Biosample,
SpecimenPreparation, ImageAcquisition, ImageData, AnalyzedData).

## The two metadata inputs

A run takes metadata from two optional inputs, and folds both into the
manifest:

| Input | Flag | Scope | Format | Varies per |
|---|---|---|---|---|
| **Sample metadata** | `--metadata` | per image / per colony | CSV | row (image or grid cell) |
| **Study profile** | `--study` | one set per run | YAML | nothing — constant |

Neither is required. With no metadata at all you still get a manifest — its
`image_data` and `analyzed_data` sections are always populated from the run
itself. The two flags just enrich the biological sections.

```bash
uv run python -m phenotypic ./images -o ./out \
    --metadata metadata.csv \
    --study study.yaml
```

```{note}
`--mode process` (apply-only layer export) writes no `deliverables/` bundle, so
it emits **no manifest**. The manifest is produced by full measure runs only.
```

## 1. Sample metadata — the `--metadata` CSV

This CSV carries the metadata that *changes* across your images or colonies:
strain, media, timepoint, plate, replicate. Example:

```{literalinclude} ../../_static/rembi/metadata.csv
:language: text
:caption: metadata.csv — per-image sample metadata
```

**How it joins.** The CSV is **left-joined onto the measurements on every
column whose name it shares with them**. So a column only participates in the
join if its header *exactly* matches a measurement column:

- Join per **image**: give the CSV a `Metadata_ImageName` column (the
  image filename, matching the framework's image-name column). Each row then
  broadcasts to every colony of that image — the pattern shown above.
- Join per **colony**: additionally include `Grid_RowNum` and `Grid_ColNum`.
  The row then applies to one grid cell.

**Every row of your CSV survives the join.** A row that matches no measured
object — a strain that failed to grow, or that detection missed — is kept in
`deliverables/measurements.csv` with its metadata intact, every measurement
column null, and `QC_MetadataOnly` set to `true`. Absence of a colony is data,
so the sample stays visible instead of silently disappearing. To list what was
never detected, filter on that column:

```python
import polars as pl

mirror = pl.read_parquet("out/deliverables/measurements.parquet")
missing = mirror.filter(pl.col("QC_MetadataOnly"))
```

The run also logs a `WARNING` naming how many CSV rows matched nothing.

```{warning}
The join key must match after metadata normalization. The canonical image-name
column is `Metadata_ImageName`; a bare `ImageName` or exact historical
`MetadataImage_ImageName` header is accepted and normalized in memory. Unknown
headers are treated as metadata and receive the `Metadata_` prefix.
Duplicate keys in the CSV multiply rows (also warned). Note the join is
directional: *measurement* rows with no matching CSV row **are** dropped, so a
CSV that omits an imaged plate silently removes it from the mirror.
```

**Schema ownership decides the REMBI module.** Use the recommended flat
`Metadata_<Label>` vocabulary. PhenoTypic resolves each known label to its enum
owner and routes that member to the right manifest section:

| Enum owner | Example fields | REMBI module → manifest section |
|---|---|
| `GENETIC`, `SAMPLE` | `Metadata_Strain`, `Metadata_BioReplicate` | Biosample → `biosample` |
| `CONDITION`, `PLATE` | `Metadata_Media`, `Metadata_PlateID` | SpecimenPreparation → `specimen_preparation` |
| `CULTURE` preparation members | `Metadata_Temperature`, `Metadata_Humidity` | SpecimenPreparation → `specimen_preparation` |
| `CULTURE` time members | `Metadata_Time`, `Metadata_TimeUnit`, `Metadata_Timepoint`, `Metadata_FrameIndex` | **Biosample** → `biosample` |
| `ACQUISITION` | `Metadata_Instrument` | ImageAcquisition → `image_acquisition` |
| `STUDY`, `EXPERIMENT` | `Metadata_Title`, `Metadata_Dataset` | Study → `study` |

The `CULTURE` owner **straddles two modules**: temperature and atmosphere
describe specimen preparation, while elapsed *time* is a biosample variable.
The owner member carries that distinction even though both fields share the
same `Metadata_` prefix.

This is a **recommended vocabulary, not a validator** — any column is accepted.
A label outside the vocabulary keeps a generic `Metadata_<Label>` header and
routes to the manifest's `uncategorized` section. The full prefix vocabulary is
documented in the {doc}`metadata namespace explanation
</explanation/metadata_namespace>`; per-enum column lists are in the
{doc}`metadata schema reference </measurements_ref/metadata/index>`.

## 2. Study profile — the `--study` YAML

This YAML carries the *static* study-level fields — the ones that are constant
for the whole run (title, authors, license, funding). It is the REMBI **Study**
module as a small file you keep beside the dataset:

```{literalinclude} ../../_static/rembi/study.yaml
:language: yaml
:caption: study.yaml — static REMBI Study profile
```

**Keys are the bare labels** — `Title`, `Author`, `License`, … — *without* the
`Metadata_` prefix. A key given here **overrides**
a same-named constant `Metadata*` column that came in via `--metadata`, so the
study file is the single source of truth for run-level fields even if the CSV
also carries them. Every field is optional. The available keys mirror the
{ref}`STUDY <measurement-info-study>` and `EXPERIMENT` schema pages.

## The output: `deliverables/rembi.yaml`

The manifest folds the per-colony metadata mirror up to each REMBI module. From
the two inputs above it looks like this:

```yaml
study:
  Title: Yeast deletion-collection colony-size time course
  Author: Nguyen, A.; Example Lab
  License: CC-BY-4.0
  Project: SGA-Phenomics
  # ... remaining study.yaml fields ...
biosample:
  Strain: [BY4741, BY4742]        # distinct values collapse to a list
  MatingType: [a, alpha]
  BioReplicate: 1
  Time: [24, 48]                  # time-course variable (from Metadata_Time)
  TimeUnit: h
specimen_preparation:
  Media: YPD
  Temperature: 30
  PlateID: [P1, P2]
image_acquisition: {}             # empty here — no ACQUISITION-owned columns
image_data:                       # ALWAYS present, from the run itself
  n_images: 4
  bit_depth: [8]
  files:
    - {name: plate_001_t24.tif, uuid: ..., bit_depth: 8, image_type: RGB}
    # ... one entry per image ...
analyzed_data:                    # ALWAYS present, the measured feature catalog
  features:
    Size: [Area, Perimeter]
    Shape: [Circularity, ...]
    Intensity: [MeanIntensity, ...]
```

Within a section, a field with a single value across the run collapses to a
scalar; multiple distinct values collapse to a sorted list. `image_data` and
`analyzed_data` are derived from the run and are present even with no metadata
inputs.

## Refreshing a pre-migration run

Old output folders remain readable. Exact historical per-topic headers are
normalized in memory and retain their schema ownership. To rewrite
bundle-owned authoritative sources to flat headers and rebuild derived output,
run `--mode recompile`. Local and SLURM recompiles migrate before aggregation;
a conflict stops publication. External `--metadata` files are never modified,
while the regenerated bundle-owned metadata copy uses canonical headers.
Background, standalone migration, receipts, and rollback are described in the
{doc}`metadata namespace explanation </explanation/metadata_namespace>`.
