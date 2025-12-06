# Pandas → Polars Migration Todo

Ordered for root pandas dependencies first (break upstream contracts before touching downstream consumers). No code edits done yet—this is the plan.

## Value Snapshot
- Speed: Polars groupby/join/agg typically 2–10× faster on multi-GB batch merges (pipelines, image sets).
- Memory: Arrow columnar and predicate pushdown lower peak RAM; better for batch runs.
- Parallelism: Native multithreading reduces wall-clock without extra joblib layers.
- Serialization: Prefer Parquet/IPC over pandas+HDF; smaller, cross-platform.
- Caveats: API differences (no implicit index semantics) require doc/test rewrites.

## Ordered Todo Checklist
- [ ] 1) Redefine measurement data model  
  - Files: `src/phenotypic/core/_image_parts/accessors/_measurement_accessor.py`, `src/phenotypic/abc_/_measure_features.py`, `src/phenotypic/abc_/_grid_measure.py`.  
  - Tasks: Replace pd.Series/DataFrame storage/validation with Polars; decide eager vs lazy frames and copy semantics; adjust helper expectations.
- [ ] 2) Update accessor layer to shed pandas semantics  
  - Files: `src/phenotypic/core/_image_parts/accessors/_objects_accessor.py`, `_grid_accessor.py`, `_metadata_accessor.py`, `_array_accessor.py`, `src/phenotypic/core/_image_parts/_image_grid_handler.py`, `src/phenotypic/core/_image_parts/accessor_abstracts/_image_accessor_base.py`.  
  - Tasks: Rework loc/iloc-like APIs, Series conversions, grid/metadata exports to Polars; align type validation.
- [ ] 3) Refactor pipeline serialization and queues  
  - Files: `src/phenotypic/core/_pipeline_parts/_serializable_pipeline.py`, `_image_pipeline_core.py`, `_image_pipeline_batch.py`.  
  - Tasks: Swap pandas detection/merge/concat paths for Polars; adopt Arrow-friendly serialization.
- [ ] 4) Replace pandas HDF handling  
  - File: `src/phenotypic/tools/hdf_.py`.  
  - Tasks: Drop pandas HDF encoders; choose Polars-friendly persistence (likely Parquet/IPC) and update callers’ contracts.
- [ ] 5) Convert image-set containers  
  - Files: `src/phenotypic/core/_image_set_parts/_image_set_measurements.py`, `_image_set_metadata.py`, `_image_set_status.py`, `_image_set_accessors/_image_set_measurements_accessor.py`, `_image_set_accessors/_image_set_metadata_accessor.py`.  
  - Tasks: Migrate stored frames, dtype checks, and update semantics to Polars.
- [ ] 6) Align grid abstractions  
  - Files: `src/phenotypic/abc_/_grid_finder.py`, `src/phenotypic/abc_/_grid_corrector.py`, `src/phenotypic/grid/_manual_grid_finder.py`, `src/phenotypic/grid/_auto_grid_finder.py`.  
  - Tasks: Refactor DataFrame outputs (cuts/binning) and grid alignment helpers to Polars ops.
- [ ] 7) Update refinement operations  
  - Files: `src/phenotypic/refine/_transitive_distance_merger.py`, `_nearest_neighbor_merger.py`, `_small_to_large_merger.py`, `_circularity_modifier.py`.  
  - Tasks: Replace pd merges/groupbys with Polars joins/aggregations.
- [ ] 8) Port measurement implementations  
  - Files: `src/phenotypic/measure/_measure_intensity.py`, `_measure_shape.py`, `_measure_color_composition.py`, `_measure_bounds.py`, `_measure_texture.py`, `_measure_color.py`, `_measure_grid_linreg_stats.py`, `_measure_grid_spread.py`, `_measure_size.py`, doc line in `src/phenotypic/measure/__init__.py`.  
  - Tasks: Emit Polars DataFrames; switch stats helpers to Polars API.
- [ ] 9) Convert analysis layer  
  - Files: `src/phenotypic/analysis/_log_growth_model.py`, `_edge_correction.py`, `_tukey_outlier.py`, `src/phenotypic/analysis/abc_/_set_analyzer.py`, `src/phenotypic/analysis/abc_/_model_fitter.py`.  
  - Tasks: Swap groupby/agg/resample-like logic to Polars equivalents.
- [ ] 10) Adjust CLI and utilities  
  - Files: `src/phenotypic/phenotypicCLI.py`, `src/phenotypic/data/_sample_image_data.py`, `debug/checkmem.py`.  
  - Tasks: Ensure input/output aggregation and logging use Polars frames.
- [ ] 11) Update docs/examples/notebooks and diagrams  
  - Files: `docs/source/conf.py`, `docs/diagrams/phenotypic-abc-classes.mmd`, `docs/source/_downloads/phenotypic-slurm.sh`, notebooks in `docs/source/user_guide/tutorial/notebooks/`.  
  - Tasks: Rewrite examples and intersphinx mappings to Polars.
- [ ] 12) Update tests to new Polars expectations  
  - Files: `tests/test_image.py`, `tests/test_image_pipeline.py`, `tests/test_image_pipeline_batch.py`, `tests/test_pipeline_serialization.py`, `tests/test_phenotypic_cli.py`, `tests/test_measurement.py`, `tests/test_measure_color_composition.py`, `tests/test_merge_on_object_label.py`, `tests/test_hdf_pandas.py`, `tests/test_edge_correction.py`, `tests/test_tukey_outlier.py`, `tests/test_log_growth_model.py`.  
  - Tasks: Replace fixtures/assertions built on pandas; add Polars-specific checks.
- [ ] 13) Clean dependencies  
  - Files: `pyproject.toml`, `uv.lock`.  
  - Tasks: Remove pandas; add Polars (and features needed); ensure extras cleaned up.

## Triage Notes
- Highest risk/effort: measurement accessors/ABCs, pipeline serialization/HDF, image-set containers—tackle first.
- After API shifts, tests and docs must be updated in lockstep to reflect Polars semantics (no implicit index).
