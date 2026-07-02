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

**How it joins.** The CSV is **inner-joined onto the measurements on every
column whose name it shares with them**. So a column only participates in the
join if its header *exactly* matches a measurement column:

- Join per **image**: give the CSV a `MetadataImage_ImageName` column (the
  image filename, matching the framework's image-name column). Each row then
  broadcasts to every colony of that image — the pattern shown above.
- Join per **colony**: additionally include `Grid_RowNum` and `Grid_ColNum`.
  The row then applies to one grid cell.

```{warning}
The join key must match the measurement column name **exactly**. After the
REMBI namespace migration the image-name column is `MetadataImage_ImageName`
(not the old `Metadata_ImageName`). A CSV that still uses the old header shares
no column with the measurements and the join is skipped with a warning. Rows
with no match are dropped (also warned); duplicate keys multiply rows.
```

**Column names decide the REMBI module.** Use the recommended
`Metadata<Topic>_<Label>` vocabulary and each column routes itself to the right
manifest section:

| CSV prefix | REMBI module → manifest section |
|---|---|
| `MetadataGenetic_*`, `MetadataSample_*` | Biosample → `biosample` |
| `MetadataCondition_*`, `MetadataPlate_*` | SpecimenPreparation → `specimen_preparation` |
| `MetadataCulture_Temperature`/`_Day`/`_Humidity`/… | SpecimenPreparation → `specimen_preparation` |
| `MetadataCulture_Time`/`_TimeUnit`/`_Timepoint`/`_FrameIndex` | **Biosample** → `biosample` (time-course modeled as a Biosample variable, per REMBI) |
| `MetadataAcquisition_*` | ImageAcquisition → `image_acquisition` |
| `MetadataStudy_*`, `MetadataExperiment_*` | Study → `study` |

Note the `MetadataCulture_` prefix **straddles two modules**: temperature and
atmosphere describe how the specimen was prepared, but elapsed *time* is a
biosample variable in REMBI, so `MetadataCulture_Time` lands in `biosample`.

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
`MetadataStudy_` / `MetadataExperiment_` prefix. A key given here **overrides**
a same-named constant `Metadata*` column that came in via `--metadata`, so the
study file is the single source of truth for run-level fields even if the CSV
also carries them. Every field is optional. The available keys mirror the
{doc}`STUDY_METADATA </measurements_ref/metadata/study_metadata>` and
`EXPERIMENT_METADATA` schema pages.

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
  Time: [24, 48]                  # time-course variable (from MetadataCulture_Time)
  TimeUnit: h
specimen_preparation:
  Media: YPD
  Temperature: 30
  PlateID: [P1, P2]
image_acquisition: {}             # empty here — no MetadataAcquisition_* columns
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

Old output folders still open, but their metadata columns predate the
per-module namespace and read as `uncategorized`. To remap an existing run's
measurement parquets to the current REMBI namespace, re-run with
`--mode recompile`. Background on the rename is in the
{doc}`metadata namespace explanation </explanation/metadata_namespace>`.
