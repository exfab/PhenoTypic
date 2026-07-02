# Metadata column namespace (REMBI modules)

PhenoTypic's metadata columns are **self-describing, per-topic** headers of the
form `Metadata<Topic>_<Label>` — for example `MetadataGenetic_Strain`,
`MetadataCulture_Time`, `MetadataImage_ImageName`. Each metadata enum in
`phenotypic.schema` owns one category prefix:

| Enum | Category prefix | REMBI module |
|---|---|---|
| `METADATA` (framework) | `MetadataImage` | ImageData |
| `STUDY_METADATA` | `MetadataStudy` | Study |
| `EXPERIMENT_METADATA` | `MetadataExperiment` | Study |
| `GENETIC_METADATA` | `MetadataGenetic` | Biosample |
| `SAMPLE_METADATA` | `MetadataSample` | Biosample |
| `CONDITION_METADATA` | `MetadataCondition` | SpecimenPreparation |
| `CULTURE_METADATA` | `MetadataCulture` | SpecimenPreparation |
| `PLATE_METADATA` | `MetadataPlate` | SpecimenPreparation |
| `ACQUISITION_METADATA` | `MetadataAcquisition` | ImageAcquisition |

All prefixes share the `Metadata` **column family**. Code recognizes any metadata
column with {func}`phenotypic.sdk_.is_metadata_header` (which matches every
`Metadata<Topic>_` prefix *and* the generic `Metadata_` fallback) rather than a
bare `"Metadata_"` string literal. The REMBI module is a separate axis exposed by
`MeasurementInfo.rembi_module()` and folded into the run's `deliverables/rembi.yaml`
manifest; the per-topic category prefix and the REMBI module are independent (the
time straddler `MetadataCulture_Time` carries `rembi_module=Biosample`).

This is a **recommended vocabulary, not a validator**: arbitrary metadata columns
are always accepted. A user-supplied label that is not in the vocabulary keeps a
generic `Metadata_<Label>` header and routes to the REMBI `Uncategorized` module.

```{seealso}
To attach metadata to a run and emit the `deliverables/rembi.yaml` manifest, see
the how-to guide {doc}`/how_to/pages/rembi_metadata` — it covers the `--metadata`
sample CSV, the `--study` profile, and how columns fold into REMBI modules, with
downloadable example files.
```

## Migration note (namespace rename)

Metadata columns are now per-REMBI-module prefixed (`MetadataGenetic_Strain`
instead of the previous shared `Metadata_Strain`). The ad-hoc `Metadata_ImageFile`
column was consolidated into `MetadataImage_ImageName` (plus
`MetadataImage_FileSuffix`). This is a **clean break**:

- **Old output folders** still open in the GUI: curation state auto-migrates on
  load (the legacy `Metadata_ImageFile` curation column is renamed on read), and
  old `master_measurements.parquet` archives still load (the legacy
  `Metadata_ImageName` column is recognized as the image-stem key).
- **Old metadata columns will not auto-map** to their REMBI module (they read as
  `Uncategorized`). To refresh a pre-migration run's measurement parquets to the
  new namespace, re-run with `--mode recompile`.
