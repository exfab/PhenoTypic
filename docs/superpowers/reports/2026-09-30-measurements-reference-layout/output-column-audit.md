# Output-column audit: which `phenotypic.schema` enums reach output tables

Reviewer: Explore subagent, 2026-09-30. The subagent ran read-only and could not
write this file, so the orchestrating session saved its report here verbatim in
substance. File:line references are relative to `src/phenotypic/` and were
taken while the schema reorganization (`eb40075`) was in progress; schema enums
are named by their public `phenotypic.schema.X` name.

**[exec]** = run with `uv run python -c ...` on synthetic data.
**[read]** = inferred from reading the code.

## Summary

| Enum | In an output table? | Produced by | Table / file | Real header form |
|---|---|---|---|---|
| QUALITY_CHECK | Yes, all 3 labels, never under the enum's own values | `QualityCheck.analyze` | `.analyze()`/`.results()`; `deliverables/qc/qc.duckdb`, one table per QC module | `QC_<name>_Metric`, `QC_<name>_Flag`, `QC_<name>_Status` |
| QUALITY_COUNT | Yes, 3/3 | ExpectedVsDetectedCount | same | values as declared |
| QUALITY_OCCUPANCY | Yes, 3/3 | GridOccupancy | same (one row per group) | values as declared |
| QUALITY_ICC | Yes, 3/3 | ICC | same | values as declared |
| QUALITY_ZMAX | Yes, 3/3 | MaxModifiedZScore | same | values as declared |
| QUALITY_MAD | Yes, 3/3 | RelativeMAD | same | values as declared |
| QUALITY_SE | Yes, 4/4 | ReplicateAgreement | same | values as declared |
| QUALITY_TUKEY | Yes, 4/4 | TukeyOutlierFraction | same | values as declared |
| LOG_GROWTH_MODEL | Yes, 7/7 | LogGrowthModel | `.analyze()`; `deliverables/LogGrowthModel.{csv,parquet}` when it is the pipeline model | `LogGrowthModel_<token>_<label>` (enum value never appears) |
| LINEAR_LAG_MODEL | Yes, 4/4 | LinearLagModel | `deliverables/LinearLagModel.{csv,parquet}` | `LinearLagModel_<token>_<label>` |
| LINEAR_CAP_AND_LAG_MODEL | Yes, 7/7 | LinearCapAndLagModel | `deliverables/LinearCapAndLagModel.{csv,parquet}` | `LinearCapAndLagModel_<token>_<label>` |
| MODEL_METRICS | Yes, 7/7 on every model | ModelFitter base | same file as the model | `ModelMetrics_<token>_<label>` |
| EDGE_CORRECTION | Yes, 2/2, in memory only; the CLI never writes it | EdgeCorrector | `.analyze()`/`.results()` | `EdgeCorrection_NewVal-<on>`, `EdgeCorrection_Cap-<on>` |
| (MADOutlierRemover, TukeyOutlierRemover) | Add no columns | | | |
| OBJECT | Yes, every measurement row | pipeline info block | `pipeline.measure()`; master and `deliverables/measurements.{csv,parquet}` | `Object_Label` |
| BBOX | Yes, 10/10 on every row, with or without MeasureBounds configured | pipeline info block calls `MeasureBounds()` | same | `Bbox_*` |
| GRID | 4 of 8 members, GridImage rows only | pipeline info block via `GridFinder` | same | `Grid_RowNum`, `Grid_ColNum`, `Grid_RowMajorIdx`, `Grid_ColMajorIdx`; the four `*IntervalStart/End` never |
| IMAGE | 4 of 8 | image metadata + CLI chunk writer | same | `Metadata_ImageName`, `Metadata_ImageType`, `Metadata_BitDepth`, `Metadata_FileSuffix`; never `UUID`, `ParentImageName`, `ParentUUID`, `ImageFormat` |
| METADATA_MATCH | Only with `--metadata` | CLI `join_metadata(how="left")` | `deliverables/measurements.{csv,parquet}` and derived | `QC_MetadataOnly` |
| CURATION | Only after curation | GUI `CurationLabels`, re-emitted by CLI | `deliverables/errors/<category>.parquet`, `deliverables/qc/curation_labels.parquet` | `Curation_Category` |
| ErrorCategory | No columns; labels are values of `Curation_Category` and `errors/` filenames | | | never a header |
| RADIAL_EXPANSION | No; nothing produces it | | | |
| ColorComposition | No pipeline or CLI output (`MeasureColorComposition` is commented out of `measure/__init__.py`) | | | |

## Part 1: SetAnalyzer subclasses

### Execution [exec]

Synthetic long-format frame: 2 plates (`Metadata_SourcePlate`), 12 wells
(`Grid_RowMajorIdx` 0-11, `Object_Label` 1-12), 6 timepoints (`Metadata_Time`),
`Metadata_BioReplicate` = well % 2, `Metadata_Clone` = `S{well//4}`, logistic
`Size_Area`; 144 rows.

```
ExpectedVsDetectedCount NEW: QC_Count_Detected, QC_Count_Expected, QC_Count_Delta, QC_Count_Metric, QC_Count_Flag, QC_Count_Status
GridOccupancy NEW:           QC_Occupancy_Filled, _Expected, _Vacant, QC_Occupancy_Metric, _Flag, _Status
ICC NEW:                     QC_ICC_NumSubjects, _NumRaters, _NumMembers, QC_ICC_Metric, _Flag, _Status
MaxModifiedZScore NEW:       QC_ZMax_Median, _MAD, _NumMembers, QC_ZMax_Metric, _Flag, _Status
RelativeMAD NEW:             QC_MAD_Median, _MAD, _NumMembers, QC_MAD_Metric, _Flag, _Status
ReplicateAgreement NEW:      QC_SE_Value, _Mean, _CV, _NumReplicates, QC_SE_Metric, _Flag, _Status
TukeyOutlierFraction NEW:    QC_Tukey_LowerFence, _UpperFence, _NumOutliers, _NumMembers, QC_Tukey_Metric, _Flag, _Status
LogGrowthModel:     LogGrowthModel_Area_{r,K,N0,µmax,Kmax,lambda,beta} + ModelMetrics_Area_{MAE,MSE,RMSE,R2,OptimizerLoss,OptimizerStatus,NumSamples}
LinearLagModel:     LinearLagModel_Area_{v,s0,lambda,alpha} + 7 ModelMetrics_Area_*
LinearCapAndLagModel: LinearCapAndLagModel_Area_{v,s0,lambda,alpha,smax,beta,mode} + 7 ModelMetrics_Area_*
EdgeCorrector NEW:  EdgeCorrection_NewVal-Size_Area, EdgeCorrection_Cap-Size_Area (Size_Area unchanged)
MADOutlierRemover / TukeyOutlierRemover NEW: [] (rows 10 -> 9 with one outlier, columns unchanged)
```

For every QC check and model, `results()` equals the `analyze()` output.

### QC checks [read, exec]

- Shared trio: `QualityCheck.analyze` (`analysis/abc_/_quality_check.py:178-252`)
  writes `QC_<name>_Metric/Flag/Status` (247-249), named by `metric_col` /
  `flag_col` / `status_col` (367-380) as `f"QC_{cls.name}_..."`, not from
  `QUALITY_CHECK` members, which are used only to generate docstrings (493-511).
- Every member of each per-check enum is emitted on every row; the columns are
  pre-initialised before any guard path: `_expected_vs_detected.py:593-596`,
  `_grid_occupancy.py:194-197`, `_icc.py:216-224`, `_max_modz.py:154-163`,
  `_relative_mad.py:173-182`, `_replicate_agreement.py:164-176`,
  `_tukey_fraction.py:149-159`.
- CLI: `run_qc` (`sdk_/_qc_recipe/_runner.py:125`) writes `check.to_table()` to
  `<output>/deliverables/qc/qc.duckdb` as table `<instance_id>` (352-360), plus
  `<instance_id>__summary` and a `qc_modules` catalog; called from
  `_cli/_cli_output_manager.py:1263`. `to_table()` (`_quality_check.py:386-418`)
  keeps groupby columns, member keys (`Metadata_ImageName`, `Object_Label`),
  `on`, `Metadata_Dataset`, `time_label` and every `QC_<name>_*` column.
  GridOccupancy overrides it to one row per group (`_grid_occupancy.py:200-239`).
  Summary tables use generic names (`n_members`, `metric`, ...). QC columns are
  not written into `measurements.parquet`.

### Growth models [read, exec]

- `ModelFitter._apply2group_func` (`analysis/abc_/_model_fitter.py:262-323`)
  emits every enum member and all of MODEL_METRICS, NaN-filled on failure
  (`_nan_fit_columns`, 236-254). `analyze` (334-377) renames through
  `qualified_header` (`{Family}_{token}_{label}`), token = `metric_token(on)`
  (`util/_measurement_outputs.py:313-324`; `Size_Area` -> `Area`).
- CLI writes only the pipeline *model*: `_emit_analysis_outputs`
  (`_cli/_cli_output_manager.py:650`) -> `deliverables/<ModelClass>.{csv,parquet}`
  plus `deliverables/analysis_manifest.json`.
- `ModelFitter` declares `_measurement_infoclass: ClassVar[type]`; MODEL_METRICS
  is used directly, not declared.

### EdgeCorrector [read, exec]

- Adds `f"{EDGE_CORRECTION.NEW_VAL}-{on}"` and `f"{EDGE_CORRECTION.CORRECTED_CAP}-{on}"`
  (`analysis/edge/_edge_correction.py:771, 774, 850-853`); `on` is untouched. Its
  `results()` docstring examples (`'Size-Area'`, `'Cap-Area'`, 685-688) are stale.
- `EdgeCorrection.analyze` (`analysis/abc_/_edge_correction.py:228-277`) first
  aggregates to one row per (groupby, `Grid_RowMajorIdx`, time).
- Only usable in the pipeline `filters` slot, whose outputs are not persisted, so
  EDGE_CORRECTION columns exist only in `.analyze()` / `.results()` frames.

### Declarations on the abstract bases

`SetAnalyzer` has no schema accessor. `QualityCheck` declares
`_measurement_infoclass = None`; `ModelFitter` declares it as a ClassVar;
`EdgeCorrector` sets it. `util/_measurement_outputs.py:192-236`
(`_discover_measurement_producers`) already discovers every analyzer's declared
schema and treats MODEL_METRICS as shared for ModelFitters; it knows nothing of
QUALITY_CHECK.

## Part 2: enums no MeasureFeatures declares

BBOX is declared by `MeasureBounds` (`measure/_measure_bounds.py:62`) and is also
on every row regardless.

`ImagePipeline.measure` (`_core/_pipeline_parts/_image_pipeline_core.py:1101-1266`)
always appends `_get_image_info` (1228/1244): `image.grid.info()` for a
GridImage, else `image.objects.info()` (1269-1272), merged on `Object_Label`.
`ObjectsAccessor.info` is `MeasureBounds().measure(image)`
(`_objects_accessor.py:701-703`); `GridFinder._get_grid_info`
(`abc_/_grid_finder.py:384-420`) adds `Grid_RowNum/ColNum/RowMajorIdx/ColMajorIdx`.

```
Image cols (no measurers): Metadata_ImageName, Metadata_ImageType, Metadata_BitDepth, Object_Label, Bbox_* (10)
GridImage cols (no measurers): the same + Grid_RowNum, Grid_ColNum, Grid_RowMajorIdx, Grid_ColMajorIdx
```

- OBJECT: `Object_Label` on every row [exec].
- GRID: the four interval members are never emitted (no reference outside
  `schema/`) [read].
- IMAGE: emitted `Metadata_ImageName`, `Metadata_ImageType`, `Metadata_BitDepth`
  (`_image_data_manager.py:163-168`), `Metadata_FileSuffix`
  (`_image_io_handler.py:826, 921`; `_cli/_cli_chunk_writer.py:353-357`). Never:
  `UUID` (private), `ParentImageName`, `ParentUUID`, `ImageFormat`.
- METADATA_MATCH: `QC_MetadataOnly`, only with CLI `--metadata`
  (`join_metadata`, `_cli_output_manager.py:285-, 344, 366-372`), then in
  `measurements.{csv,parquet}` and everything derived.
- CURATION: `Curation_Category` in `deliverables/qc/curation_labels.parquet`
  (`_gui/results_viewer/_curation_labels.py:816-837`) and
  `deliverables/errors/<category>.parquet` (853-874), re-emitted by the CLI
  (`_cli/_cli_error_outputs.py:29-63`) when labels exist.
- ErrorCategory: never a header; bare labels only.
- RADIAL_EXPANSION: nothing produces it.
- ColorComposition: `MeasureColorComposition` is commented out of
  `measure/__init__.py:22, 27`; pipeline JSON cannot resolve it.

## Notes for the docs reference

1. Document the emitted header forms, which differ from enum values for
   QUALITY_CHECK, the growth models, MODEL_METRICS and EDGE_CORRECTION.
2. Never emitted: GRID interval members, IMAGE `UUID`/`ParentImageName`/
   `ParentUUID`/`ImageFormat`, RADIAL_EXPANSION, ColorComposition, ErrorCategory
   as columns.
3. Conditional: METADATA_MATCH (`--metadata`), CURATION (after curation),
   EDGE_CORRECTION (in memory), GRID (GridImage only).
