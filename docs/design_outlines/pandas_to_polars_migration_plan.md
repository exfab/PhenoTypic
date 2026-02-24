# PhenoTypic: Pandas to Polars Migration Plan

**Generated**: 2025-12-13
**Objective**: Migrate PhenoTypic codebase from pandas to polars with minimal risk and maximum incremental progress
**Scope**: 60 Python files with pandas dependencies across source code and tests

---

## Executive Summary

The PhenoTypic codebase has **extensive pandas dependencies** across 60 files, organized into 3 primary dependency trees:

1. **Measurement System** (30+ files): Core data generation and processing
2. **Analysis System** (4 files): Statistical analysis and modeling
3. **Storage/IO System** (5+ files): HDF5 persistence and CSV I/O

The migration must proceed **bottom-up** through the dependency tree, starting with abstract base classes (ABCs) that define DataFrame interfaces, then moving through concrete implementations, accessors, pipelines, and finally CLI/batch processing.

**Critical Success Factor**: The `MeasureFeatures` ABC defines the DataFrame contract for the entire measurement system. This must be migrated first, followed by all concrete measurers that implement it.

---

## Dependency Graph

### Tree 1: Measurement System (Primary Migration Path)

```
TIER 0 (Foundation - No local pandas dependencies)
├── abc_/_measure_features.py
│   ├── Imports: pandas (extensive)
│   ├── Returns: pd.DataFrame
│   ├── Dependencies: None (root ABC)
│   └── Downstream: 16 measurers + 2 grid ABCs
│
├── abc_/_set_analyzer.py
│   ├── Imports: pandas (extensive)
│   ├── Returns: pd.DataFrame
│   ├── Dependencies: None (root ABC)
│   └── Downstream: 3 analyzers
│
└── tools/hdf_.py
    ├── Imports: pandas (very extensive)
    ├── Functions: DataFrame ↔ HDF5 serialization
    ├── Dependencies: None
    └── Downstream: ImageSet, batch processing

TIER 1 (Secondary ABCs - Depend on Tier 0)
├── abc_/_grid_measure.py
│   ├── Imports: pandas
│   ├── Depends on: MeasureFeatures
│   └── Downstream: 2 grid measurers
│
├── abc_/_grid_finder.py
│   ├── Imports: pandas (pd.cut, categorical)
│   ├── Depends on: GridMeasureFeatures → MeasureFeatures
│   └── Downstream: 2 grid finders
│
├── abc_/_model_fitter.py
│   ├── Imports: pandas
│   ├── Depends on: SetAnalyzer
│   └── Downstream: 1 model fitter
│
└── _core/_image_parts/accessor_abstracts/_image_accessor_base.py
    ├── Imports: pandas (pd.Interval, is_scalar)
    ├── Dependencies: None
    └── Downstream: All accessors

TIER 2 (Concrete Measurers - Depend on Tier 0-1)
├── measure/_measure_shape.py → MeasureFeatures
├── measure/_measure_size.py → MeasureFeatures
├── measure/_measure_bounds.py → MeasureFeatures
├── measure/_measure_intensity.py → MeasureFeatures
├── measure/_measure_color.py → MeasureFeatures
├── measure/_measure_color_composition.py → MeasureFeatures
├── measure/_measure_texture.py → MeasureFeatures (+ merging)
├── measure/_measure_grid_spread.py → GridMeasureFeatures
└── measure/_measure_grid_linreg_stats.py → GridMeasureFeatures (complex)

TIER 3 (Refiners - Create DataFrames from regionprops)
├── refine/_small_to_large_merger.py → ObjectRefiner
├── refine/_nearest_neighbor_merger.py → ObjectRefiner
├── refine/_transitive_distance_merger.py → ObjectRefiner
└── refine/_circularity_modifier.py → ObjectRefiner

TIER 4 (Analysis Classes - Receive DataFrames)
├── analysis/_tukey_outlier.py → SetAnalyzer
├── analysis/_edge_correction.py → SetAnalyzer (very complex)
└── analysis/_log_growth_model.py → ModelFitter

TIER 5 (Grid Operations - Use DataFrames internally)
├── grid/_auto_grid_finder.py → GridFinder
└── grid/_manual_grid_finder.py → GridFinder

TIER 6 (Accessors - Store/Pass DataFrames)
├── _core/_image_parts/_image_grid_handler.py
├── _core/_image_parts/accessors/_grid_accessor.py
├── _core/_image_parts/accessors/_rgb_accessor.py (minimal)
├── _core/_image_parts/accessors/_objects_accessor.py
├── _core/_image_parts/accessors/_metadata_accessor.py
└── _core/_image_parts/accessors/_measurement_accessor.py

TIER 7 (ImageSet Components - Aggregate DataFrames)
├── _core/_image_set_parts/_image_set_measurements.py
├── _core/_image_set_parts/_image_set_metadata.py
├── _core/_image_set_parts/_image_set_status.py
├── _core/_image_set_parts/_image_set_accessors/_image_set_measurements_accessor.py
└── _core/_image_set_parts/_image_set_accessors/_image_set_metadata_accessor.py

TIER 8 (Pipelines - Orchestrate Measurements)
├── _core/_pipeline_parts/_image_pipeline_core.py
├── _core/_pipeline_parts/_image_pipeline_batch.py
└── _core/_pipeline_parts/_serializable_pipeline.py

TIER 9 (CLI & Data - Top-level consumers)
├── phenotypicCLI.py
└── data/_sample_image_data.py
```

### Tree 2: Standalone Components (Can migrate independently)

```
├── tests/test_*.py (27 test files)
│   ├── Most consume DataFrames from operations
│   ├── Some create test DataFrames
│   └── Migrate after corresponding src/ files
│
└── docs/source/user_guide/tutorial/notebooks/*.ipynb
    └── Migrate after API is stable
```

---

## Migration Priority List

### Priority 1: Foundation ABCs (MUST MIGRATE FIRST)

#### **1.1 `abc_/_measure_features.py`**
- **Pandas usage**: Extensive
  - `pd.DataFrame()` creation
  - `DataFrame.insert()` for Label column
  - `DataFrame.merge()` for combining measurements
  - `pd.isna()` for NA validation
  - `pandas.api.types.is_scalar` for validation
- **Upstream dependencies**: None (root ABC)
- **Downstream dependents**: 16 measurers + GridMeasureFeatures + GridFinder
- **Migration impact**: **CRITICAL** - Affects entire measurement system
- **Rationale**: Root of measurement tree; defines DataFrame contract for all measurers
- **Polars equivalent**:
  - `pl.DataFrame()`
  - `df.with_columns()` instead of `.insert()`
  - `df.join()` instead of `.merge()`
  - `pl.col().is_null()` instead of `pd.isna()`
- **Complexity**: **HIGH** - 20+ static helper methods, complex merge logic

#### **1.2 `abc_/_set_analyzer.py`**
- **Pandas usage**: Extensive
  - `pd.DataFrame()` storage attributes
  - Abstract method signature: `analyze(data: pd.DataFrame) -> pd.DataFrame`
  - `pd.Series`, `.isin()`, `.isna()`, `.eq()`, `.any()`, `.iloc[]`, `.copy()`
- **Upstream dependencies**: None (root ABC)
- **Downstream dependents**: ModelFitter + 3 analyzers
- **Migration impact**: **HIGH** - Affects all analysis operations
- **Rationale**: Root of analysis tree; defines interface for statistical operations
- **Polars equivalent**:
  - `pl.DataFrame()` for storage
  - Change method signatures to accept/return polars DataFrames
  - `df.filter()`, `pl.col().is_null()`, `df.clone()`
- **Complexity**: **MEDIUM** - Complex filtering logic in `_filter_by()`

#### **1.3 `tools/hdf_.py`**
- **Pandas usage**: **VERY EXTENSIVE** (700+ lines of DataFrame serialization)
  - Schema detection: `pd.api.types.is_numeric_dtype()`, `is_bool_dtype()`, `is_categorical_dtype()`
  - Index/MultiIndex handling: `pd.Index()`, `pd.MultiIndex.from_arrays()`
  - DataFrame I/O: `preallocate_frame_layout()`, `save_frame_*()`, `load_frame()`
  - Series I/O: `save_series_*()`, `load_series()`
- **Upstream dependencies**: None
- **Downstream dependents**: ImageSet components, batch processing
- **Migration impact**: **CRITICAL** - Required for large-scale data persistence
- **Rationale**: Standalone utility; can migrate independently but affects storage layer
- **Polars equivalent**:
  - Polars has native schema inspection
  - May need custom encoding for HDF5 compatibility
  - Consider using Polars' native serialization where possible
- **Complexity**: **VERY HIGH** - Complex type system integration, extensive validation

---

### Priority 2: Secondary ABCs (Depend on Priority 1)

#### **2.1 `abc_/_grid_measure.py`**
- **Pandas usage**: Validation only
  - `isinstance(output, pd.DataFrame)` type checking
  - Inherits all DataFrame operations from MeasureFeatures
- **Upstream dependencies**: MeasureFeatures (Priority 1.1)
- **Downstream dependents**: 2 grid measurers
- **Migration impact**: **MEDIUM**
- **Rationale**: Simple wrapper; migrates easily after parent ABC
- **Complexity**: **LOW** - Just type validation updates

#### **2.2 `abc_/_grid_finder.py`**
- **Pandas usage**: Medium-complex
  - **`pd.cut()`** for binning centroids into grid cells (lines 247-253, 264-270)
  - `DataFrame.loc[]` for column assignment
  - `.astype("category")`, `.astype("Int64")` for dtype conversion
  - `pd.notna()`, `pd.Series`
- **Upstream dependencies**: GridMeasureFeatures → MeasureFeatures
- **Downstream dependents**: 2 grid finders
- **Migration impact**: **MEDIUM-HIGH** - Critical for grid-based workflows
- **Rationale**: Requires custom binning logic; no direct polars equivalent for pd.cut()
- **Polars equivalent**:
  - **Custom binning**: Use `pl.when().then().otherwise()` chains or UDF
  - Categorical: `pl.col().cast(pl.Categorical)`
  - Nullable integers: Native in polars (simpler)
- **Complexity**: **MEDIUM** - pd.cut() needs custom implementation

#### **2.3 `abc_/_model_fitter.py`**
- **Pandas usage**: Minimal
  - `self._latest_model_scores: pd.DataFrame = pd.DataFrame()`
  - Inherits from SetAnalyzer
- **Upstream dependencies**: SetAnalyzer (Priority 1.2)
- **Downstream dependents**: 1 model fitter
- **Migration impact**: **LOW**
- **Rationale**: Simple storage; migrates with parent
- **Complexity**: **LOW**

#### **2.4 `_core/_image_parts/accessor_abstracts/_image_accessor_base.py`**
- **Pandas usage**: Limited
  - `pd.Interval(left, right, closed="both")` for value ranges
  - `pd.api.types.is_scalar()` for validation
- **Upstream dependencies**: None
- **Downstream dependents**: All accessors
- **Migration impact**: **MEDIUM** - Affects accessor interface
- **Rationale**: Foundation for accessor pattern
- **Polars equivalent**:
  - No direct Interval type; may use tuple `(min, max)` or custom class
  - Polars has builtin scalar detection
- **Complexity**: **LOW** - Minimal pandas usage

---

### Priority 3: Concrete Measurers (Depend on Priority 1-2)

**Migration order**: Simplest to most complex

#### **3.1 `measure/_measure_size.py`** (Simplest)
- **Pandas usage**: Basic
  - `pd.DataFrame(measurements)`, `.insert()`
- **Complexity**: **LOW**

#### **3.2 `measure/_measure_shape.py`**
- **Pandas usage**: Basic
  - `pd.DataFrame(measurements)`, `.insert()`
- **Complexity**: **LOW**

#### **3.3 `measure/_measure_bounds.py`**
- **Pandas usage**: Basic + rename
  - `pd.DataFrame()`, `.rename(columns={...})`
- **Complexity**: **LOW**

#### **3.4 `measure/_measure_intensity.py`**
- **Pandas usage**: Basic
  - `pd.DataFrame()`, `.insert()`
- **Complexity**: **LOW**

#### **3.5 `measure/_measure_color_composition.py`**
- **Pandas usage**: Basic
  - `pd.DataFrame()`, `.insert()`, `pd.Series`
- **Complexity**: **LOW**

#### **3.6 `measure/_measure_color.py`**
- **Pandas usage**: Medium
  - `pd.DataFrame()`, `.insert()`, `.loc[]` for computed columns
- **Complexity**: **MEDIUM** - Vectorized color calculations

#### **3.7 `measure/_measure_grid_spread.py`**
- **Pandas usage**: Medium
  - `.value_counts()`, `.sort_values()`, `pd.Series()`, `.insert()`
- **Upstream dependencies**: GridMeasureFeatures (Priority 2.1)
- **Complexity**: **MEDIUM** - Aggregation operations

#### **3.8 `measure/_measure_texture.py`**
- **Pandas usage**: Medium
  - `pd.DataFrame()`, `.insert()`, `.merge()`
- **Complexity**: **MEDIUM** - Merges multi-scale results

#### **3.9 `measure/_measure_grid_linreg_stats.py`** (Most complex)
- **Pandas usage**: **Complex**
  - `.reset_index()`, `.loc[]` (extensive), `pd.DataFrame()` with custom index
  - `pd.merge()`, `pd.Index()`, `.apply()` with lambdas, `.set_index()`
- **Upstream dependencies**: GridMeasureFeatures (Priority 2.1)
- **Complexity**: **HIGH** - Multi-step transformations, row-wise apply

**Collective impact**: After migrating all 9 measurers, measurement system is complete

---

### Priority 4: Refiners (Depend on Priority 3 indirectly)

All refiners create temporary DataFrames from `regionprops_table()` for internal processing. They don't inherit from measurement ABCs but follow similar patterns.

#### **4.1 `refine/_nearest_neighbor_merger.py`** (Simplest)
- **Pandas usage**: Basic
  - `pd.DataFrame(props)`, column selection, `.values`
- **Complexity**: **LOW**

#### **4.2 `refine/_transitive_distance_merger.py`**
- **Pandas usage**: Basic
  - `pd.DataFrame(props)`, column selection
- **Complexity**: **LOW**

#### **4.3 `refine/_small_to_large_merger.py`**
- **Pandas usage**: Medium
  - `pd.DataFrame()`, boolean indexing, multi-column selection
- **Complexity**: **LOW**

#### **4.4 `refine/_circularity_modifier.py`** (Most complex)
- **Pandas usage**: Medium-complex
  - `pd.DataFrame()`, `.rename()`, `.set_index()`, column arithmetic, boolean indexing, `.index.to_numpy()`
- **Complexity**: **MEDIUM** - Index manipulation

---

### Priority 5: Analysis Classes (Depend on Priority 1-2)

#### **5.1 `analysis/_tukey_outlier.py`** (Simplest)
- **Pandas usage**: Medium
  - `.groupby()`, `pd.concat()`, `.copy()`, boolean filtering
- **Upstream dependencies**: SetAnalyzer (Priority 1.2)
- **Complexity**: **MEDIUM** - GroupBy operations

#### **5.2 `analysis/_log_growth_model.py`**
- **Pandas usage**: **Extensive**
  - `.groupby().agg()` with dict, `pd.concat()`, `.insert()`
  - `pd.MultiIndex.from_tuples()`, `.reset_index()`, `.unique()`, `.mean()`, `.std()`, `.count()`
- **Upstream dependencies**: ModelFitter (Priority 2.3) → SetAnalyzer (Priority 1.2)
- **Complexity**: **HIGH** - MultiIndex, time-series GroupBy

#### **5.3 `analysis/_edge_correction.py`** (Most complex)
- **Pandas usage**: **VERY EXTENSIVE**
  - `.groupby().agg()`, `pd.concat()`, `.loc[]` (extensive), `.isin()`, `.max()`, `.mean()`
  - String aggregation: `.astype(str).agg(" | ".join, axis=1)`
  - Parallel processing with joblib, conditional filtering
- **Upstream dependencies**: SetAnalyzer (Priority 1.2)
- **Complexity**: **VERY HIGH** - Most complex pandas usage in codebase

---

### Priority 6: Grid Operations (Depend on Priority 2.2)

#### **6.1 `grid/_manual_grid_finder.py`**
- **Pandas usage**: None (import only)
- **Upstream dependencies**: GridFinder (Priority 2.2)
- **Complexity**: **TRIVIAL**

#### **6.2 `grid/_auto_grid_finder.py`**
- **Pandas usage**: Limited
  - `.groupby(observed=False).mean()`
- **Upstream dependencies**: GridFinder (Priority 2.2)
- **Complexity**: **LOW**

---

### Priority 7: Accessors (Depend on Priority 2.4 + earlier tiers)

#### **7.1 `_core/_image_parts/accessors/_rgb_accessor.py`** (Minimal)
- **Pandas usage**: `pd.api.types.is_scalar()`
- **Complexity**: **TRIVIAL**

#### **7.2 `_core/_image_parts/_image_grid_handler.py`**
- **Pandas usage**: Pass-through
  - `info()` returns DataFrame from GridAccessor
  - `pd.Interval` import
- **Complexity**: **LOW**

#### **7.3 `_core/_image_parts/accessors/_metadata_accessor.py`**
- **Pandas usage**: Basic
  - `DataFrame.insert()`, `pd.Series()` for metadata
- **Complexity**: **LOW**

#### **7.4 `_core/_image_parts/accessors/_objects_accessor.py`**
- **Pandas usage**: Medium
  - `pd.DataFrame(regionprops_table())`, `.rename()`, `pd.Series()`
- **Complexity**: **MEDIUM** - regionprops integration

#### **7.5 `_core/_image_parts/accessors/_grid_accessor.py`**
- **Pandas usage**: Medium-complex
  - `.loc[]` filtering, `.groupby()`, `.value_counts()`, `.to_numpy()`, `.unique()`
- **Complexity**: **MEDIUM** - Grid position filtering

#### **7.6 `_core/_image_parts/accessors/_measurement_accessor.py`**
- **Pandas usage**: Medium
  - `pd.concat()`, `.to_records()`
- **Complexity**: **MEDIUM** - Measurement aggregation

---

### Priority 8: ImageSet Components (Depend on Priority 7 + HDF5)

#### **8.1 `_core/_image_set_parts/_image_set_status.py`**
- **Pandas usage**: Basic
  - `pd.DataFrame()` for status flags
- **Upstream dependencies**: hdf_.py (Priority 1.3)
- **Complexity**: **LOW**

#### **8.2 `_core/_image_set_parts/_image_set_metadata.py`**
- **Pandas usage**: Pass-through
- **Complexity**: **TRIVIAL**

#### **8.3 `_core/_image_set_parts/_image_set_accessors/_image_set_measurements_accessor.py`**
- **Pandas usage**: Basic
  - `pd.concat()`, type validation
- **Complexity**: **LOW**

#### **8.4 `_core/_image_set_parts/_image_set_accessors/_image_set_metadata_accessor.py`**
- **Pandas usage**: Medium
  - `pd.DataFrame()`, `.set_index()`, Series processing
- **Complexity**: **MEDIUM**

#### **8.5 `_core/_image_set_parts/_image_set_measurements.py`**
- **Pandas usage**: Medium
  - `pd.concat()`, DataFrame insertion from HDF5
- **Upstream dependencies**: hdf_.py (Priority 1.3), measurement accessors
- **Complexity**: **MEDIUM** - HDF5 aggregation

---

### Priority 9: Pipelines (Depend on Priority 3-8)

#### **9.1 `_core/_pipeline_parts/_serializable_pipeline.py`** (Simplest)
- **Pandas usage**: Filtering only
  - `isinstance(value, pd.DataFrame)` check to skip serialization
- **Complexity**: **TRIVIAL**

#### **9.2 `_core/_pipeline_parts/_image_pipeline_core.py`**
- **Pandas usage**: Medium-complex
  - `pd.DataFrame()` for benchmarks, `pd.concat()`, `df.sum()`
  - `_merge_on_object_labels()`: DataFrame merging with validation
- **Upstream dependencies**: All measurers (Priority 3), accessors (Priority 7)
- **Complexity**: **MEDIUM-HIGH** - Orchestrates measurement aggregation

#### **9.3 `_core/_pipeline_parts/_image_pipeline_batch.py`**
- **Pandas usage**: Medium
  - Returns DataFrames from batch processing
  - Schema detection for HDF5
  - Pickle serialization of DataFrames for multiprocessing
- **Upstream dependencies**: ImageSet (Priority 8), hdf_.py (Priority 1.3)
- **Complexity**: **MEDIUM** - Multiprocessing, SWMR coordination

---

### Priority 10: CLI & Top-Level (Depend on all previous)

#### **10.1 `data/_sample_image_data.py`**
- **Pandas usage**: CSV loading
  - `pd.read_csv()` for example data
- **Upstream dependencies**: None (standalone)
- **Complexity**: **TRIVIAL** - Just change to `pl.read_csv()`

#### **10.2 `phenotypicCLI.py`**
- **Pandas usage**: CSV I/O, aggregation
  - `.to_csv()`, `pd.concat()`
- **Upstream dependencies**: Pipelines (Priority 9)
- **Complexity**: **LOW** - Simple I/O operations

---

### Priority 11: Tests (Migrate after corresponding src/)

**Strategy**: Migrate tests after their corresponding source files are complete

- `tests/test_measurement.py` → After Priority 3
- `tests/test_tukey_outlier.py` → After Priority 5.1
- `tests/test_log_growth_model.py` → After Priority 5.2
- `tests/test_edge_correction.py` → After Priority 5.3
- `tests/test_image_pipeline.py` → After Priority 9.2
- `tests/test_image_pipeline_batch.py` → After Priority 9.3
- `tests/test_phenotypic_cli.py` → After Priority 10.2
- `tests/test_hdf_pandas.py` → After Priority 1.3
- `tests/test_merge_on_object_label.py` → After Priority 9.2
- (17 additional test files) → After corresponding features

**Complexity**: Varies by test file (LOW to MEDIUM)

---

## Summary Table

| Priority | File | Pandas | Upstream Deps | Downstream Deps | Impact | Complexity |
|----------|------|--------|---------------|-----------------|--------|------------|
| **1.1** | `abc_/_measure_features.py` | Extensive | None | 16 measurers + 2 ABCs | **CRITICAL** | **HIGH** |
| **1.2** | `abc_/_set_analyzer.py` | Extensive | None | 3 analyzers + 1 ABC | **HIGH** | **MEDIUM** |
| **1.3** | `tools/hdf_.py` | Very Extensive | None | ImageSet, batch | **CRITICAL** | **VERY HIGH** |
| **2.1** | `abc_/_grid_measure.py` | Validation | 1.1 | 2 grid measurers | MEDIUM | LOW |
| **2.2** | `abc_/_grid_finder.py` | Medium | 2.1 → 1.1 | 2 grid finders | MEDIUM-HIGH | MEDIUM |
| **2.3** | `abc_/_model_fitter.py` | Minimal | 1.2 | 1 model fitter | LOW | LOW |
| **2.4** | `accessor_abstracts/_image_accessor_base.py` | Limited | None | All accessors | MEDIUM | LOW |
| **3.1** | `measure/_measure_size.py` | Basic | 1.1 | Pipelines | LOW | LOW |
| **3.2** | `measure/_measure_shape.py` | Basic | 1.1 | Pipelines | LOW | LOW |
| **3.3** | `measure/_measure_bounds.py` | Basic | 1.1 | Pipelines | LOW | LOW |
| **3.4** | `measure/_measure_intensity.py` | Basic | 1.1 | Pipelines | LOW | LOW |
| **3.5** | `measure/_measure_color_composition.py` | Basic | 1.1 | Pipelines | LOW | LOW |
| **3.6** | `measure/_measure_color.py` | Medium | 1.1 | Pipelines | LOW | MEDIUM |
| **3.7** | `measure/_measure_grid_spread.py` | Medium | 2.1 → 1.1 | Pipelines | LOW | MEDIUM |
| **3.8** | `measure/_measure_texture.py` | Medium | 1.1 | Pipelines | LOW | MEDIUM |
| **3.9** | `measure/_measure_grid_linreg_stats.py` | Complex | 2.1 → 1.1 | Pipelines | LOW | **HIGH** |
| **4.1** | `refine/_nearest_neighbor_merger.py` | Basic | None | Pipelines | LOW | LOW |
| **4.2** | `refine/_transitive_distance_merger.py` | Basic | None | Pipelines | LOW | LOW |
| **4.3** | `refine/_small_to_large_merger.py` | Medium | None | Pipelines | LOW | LOW |
| **4.4** | `refine/_circularity_modifier.py` | Medium | None | Pipelines | LOW | MEDIUM |
| **5.1** | `analysis/_tukey_outlier.py` | Medium | 1.2 | User code | MEDIUM | MEDIUM |
| **5.2** | `analysis/_log_growth_model.py` | Extensive | 2.3 → 1.2 | User code | MEDIUM | **HIGH** |
| **5.3** | `analysis/_edge_correction.py` | Very Extensive | 1.2 | User code | MEDIUM | **VERY HIGH** |
| **6.1** | `grid/_manual_grid_finder.py` | None | 2.2 | Pipelines | LOW | TRIVIAL |
| **6.2** | `grid/_auto_grid_finder.py` | Limited | 2.2 | Pipelines | LOW | LOW |
| **7.1** | `accessors/_rgb_accessor.py` | Minimal | 2.4 | Image class | LOW | TRIVIAL |
| **7.2** | `_image_grid_handler.py` | Pass-through | 2.4 | GridImage | LOW | LOW |
| **7.3** | `accessors/_metadata_accessor.py` | Basic | 2.4 | Image class | LOW | LOW |
| **7.4** | `accessors/_objects_accessor.py` | Medium | 2.4 | Image class | MEDIUM | MEDIUM |
| **7.5** | `accessors/_grid_accessor.py` | Medium | 2.4 | GridImage | MEDIUM | MEDIUM |
| **7.6** | `accessors/_measurement_accessor.py` | Medium | 2.4 | Image class | MEDIUM | MEDIUM |
| **8.1** | `_image_set_status.py` | Basic | 1.3 | ImageSet | LOW | LOW |
| **8.2** | `_image_set_metadata.py` | Pass-through | - | ImageSet | LOW | TRIVIAL |
| **8.3** | `_image_set_measurements_accessor.py` | Basic | - | ImageSet | LOW | LOW |
| **8.4** | `_image_set_metadata_accessor.py` | Medium | - | ImageSet | MEDIUM | MEDIUM |
| **8.5** | `_image_set_measurements.py` | Medium | 1.3, 7.6 | ImageSet | MEDIUM | MEDIUM |
| **9.1** | `_serializable_pipeline.py` | Trivial | - | Pipeline | LOW | TRIVIAL |
| **9.2** | `_image_pipeline_core.py` | Medium | 3.*, 7.* | User code | **HIGH** | MEDIUM-HIGH |
| **9.3** | `_image_pipeline_batch.py` | Medium | 8.*, 1.3 | CLI, user | **HIGH** | MEDIUM |
| **10.1** | `data/_sample_image_data.py` | Basic | None | Examples | LOW | TRIVIAL |
| **10.2** | `phenotypicCLI.py` | Basic | 9.* | User CLI | MEDIUM | LOW |
| **11** | `tests/*.py` (27 files) | Varies | Corresponding src | Testing | MEDIUM | LOW-MEDIUM |

**Total files**: 60+ (38 source files + 22+ test/doc files)

---

## Implementation Notes

### Circular Dependencies
**None detected**. The codebase has a clean hierarchical structure with ABCs at the root and implementations as leaves.

### Drop-in Replacement Candidates

Files where polars can be nearly drop-in (minimal API changes):

1. **CSV I/O** (Priority 10.1, 10.2):
   - `pd.read_csv()` → `pl.read_csv()`
   - `df.to_csv()` → `df.write_csv()`

2. **Simple DataFrame creation** (Priority 3.1-3.5):
   - `pd.DataFrame(dict)` → `pl.DataFrame(dict)`
   - Column names remain the same

3. **Basic filtering** (Priority 4.1-4.3):
   - Boolean indexing → `df.filter()`
   - Column selection → `df.select()`

### API Changes Required

Files requiring significant API refactoring:

1. **`pd.cut()` usage** (Priority 2.2):
   - **No polars equivalent**
   - Need custom binning implementation using `pl.when().then().otherwise()` chains
   - Or write a UDF: `df.with_columns(pl.col("x").map_elements(custom_bin_func))`

2. **DataFrame.insert()** (Throughout):
   - Pandas: `df.insert(loc=0, column="name", value=values)`
   - Polars: `df = df.with_columns(pl.lit(value).alias("name"))` then reorder

3. **DataFrame.merge()** (Priority 1.1, 3.8, 9.2):
   - Pandas: `df1.merge(df2, on=cols, suffixes=("", "_merged"))`
   - Polars: `df1.join(df2, on=cols, suffix="_merged")` (left table gets no suffix by default)

4. **Index operations** (Priority 3.9, 5.2):
   - Pandas: `.set_index()`, `.reset_index()`, `pd.MultiIndex`
   - Polars: **No index concept** - use regular columns
   - May need to materialize index as column before operations

5. **Groupby().agg() with dict** (Priority 5.2, 5.3):
   - Pandas: `df.groupby("col").agg({"col1": "mean", "col2": "sum"})`
   - Polars: `df.group_by("col").agg([pl.col("col1").mean(), pl.col("col2").sum()])`

6. **pd.concat()** (Throughout):
   - Pandas: `pd.concat([df1, df2], axis=0, ignore_index=True)`
   - Polars: `pl.concat([df1, df2], how="vertical")`

7. **HDF5 integration** (Priority 1.3):
   - Pandas has native HDF5 support via PyTables
   - Polars does not - may need to:
     - Convert to Arrow format for HDF5 storage
     - Use polars → pandas → HDF5 bridge temporarily
     - Implement custom HDF5 serialization for polars DataFrames

### Pandas Patterns Requiring Special Handling

#### 1. **GroupBy chains** (Priority 5.2, 5.3)
```python
# Pandas
result = (
    df.groupby(by=groupby_cols, as_index=False)
      .agg(agg_dict)
      .groupby(by=self.groupby, as_index=False)
)

# Polars
result = (
    df.group_by(groupby_cols, maintain_order=True)
      .agg([...])  # Need to expand agg_dict
      .group_by(self.groupby, maintain_order=True)
)
```

#### 2. **Window operations** (Priority 5.2)
```python
# Pandas
df["rolling_mean"] = df.groupby("group")["value"].transform(lambda x: x.rolling(3).mean())

# Polars
df = df.with_columns(
    pl.col("value").rolling_mean(window_size=3).over("group").alias("rolling_mean")
)
```

#### 3. **Custom iterators with apply()** (Priority 3.9)
```python
# Pandas
df["result"] = df.apply(lambda row: custom_func(row["col1"], row["col2"]), axis=1)

# Polars (prefer vectorized)
df = df.with_columns(
    pl.struct(["col1", "col2"])
      .map_elements(lambda x: custom_func(x["col1"], x["col2"]))
      .alias("result")
)
# Or use pl.when().then() chains if logic is simple
```

#### 4. **String aggregation** (Priority 5.3)
```python
# Pandas
data["_group_key"] = data[self.groupby].astype(str).agg(" | ".join, axis=1)

# Polars
data = data.with_columns(
    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in self.groupby], separator=" | ")
      .alias("_group_key")
)
```

#### 5. **Categorical dtypes** (Priority 2.2)
```python
# Pandas
df["grid_col"] = df["grid_col"].astype("category")

# Polars
df = df.with_columns(pl.col("grid_col").cast(pl.Categorical))
```

#### 6. **Nullable integers** (Priority 2.2)
```python
# Pandas
df["section_num"] = df["section_num"].astype("Int64")  # Nullable

# Polars (simpler - integers are nullable by default)
df = df.with_columns(pl.col("section_num").cast(pl.Int64))
```

### Testing Strategy

1. **Unit tests first**: Migrate tests for each source file immediately after migrating the source
2. **Integration tests**: Update after each tier is complete
3. **Regression testing**: Compare outputs (polars vs pandas) on sample data
4. **Performance benchmarking**: Track speed improvements from polars

### Compatibility Shim (Optional)

Consider creating a temporary compatibility layer during migration:

```python
# phenotypic/compat.py
try:
    import polars as pl
    DataFrame = pl.DataFrame
    Series = pl.Series
    read_csv = pl.read_csv
    concat = pl.concat
except ImportError:
    import pandas as pd
    DataFrame = pd.DataFrame
    Series = pd.Series
    read_csv = pd.read_csv
    concat = pd.concat
```

This allows incremental migration with fallback to pandas if needed.

### Performance Expectations

**Polars advantages**:
- **Lazy evaluation**: Optimizes query plans automatically
- **Parallel execution**: Uses all CPU cores by default
- **Memory efficiency**: Better memory layout, less copying
- **Type safety**: Strict schema enforcement

**Expected speedups** (based on typical polars benchmarks):
- **CSV I/O**: 2-5x faster
- **GroupBy/aggregation**: 3-10x faster
- **Joins**: 2-5x faster
- **Filtering**: 2-4x faster

**Potential slowdowns**:
- **Small datasets** (&lt;1000 rows): Pandas may be faster due to lower overhead
- **UDFs/apply**: Python callbacks break polars' vectorization

### Breaking Changes for Users

#### API Changes:
1. **Type annotations**: `pd.DataFrame` → `pl.DataFrame` in function signatures
2. **Return types**: All measurement/analysis functions return polars DataFrames
3. **Index handling**: Users relying on pandas index must use regular columns
4. **Categorical behavior**: Polars categoricals are more strict (no automatic casting)

#### Backwards Compatibility:
- **Major version bump required**: This is a breaking change (v1.x → v2.0)
- **Migration guide**: Provide comprehensive documentation with examples
- **Pandas bridge**: Consider providing `to_pandas()` converters for gradual migration

### Deprioritized Files

Files with minimal pandas usage that can be deprioritized:

1. **`accessor_abstracts/_image_accessor_base.py`** (Priority 2.4):
   - Only uses `pd.Interval` and `is_scalar`
   - Can be migrated late with minimal impact

2. **`refine/_circularity_modifier.py`** (Priority 4.4):
   - Internal DataFrame usage only
   - Not on critical path

3. **Test files** (Priority 11):
   - Can migrate incrementally
   - Not blocking for source code migration

---

## Migration Phases

### Phase 1: Foundation (Weeks 1-3)
- **Files**: Priority 1.1, 1.2, 1.3 (3 files)
- **Goal**: Migrate core ABCs and HDF5 utilities
- **Risk**: **HIGH** - Breaking changes to fundamental interfaces
- **Testing**: Comprehensive unit + integration tests

### Phase 2: Secondary ABCs (Week 4)
- **Files**: Priority 2.1-2.4 (4 files)
- **Goal**: Complete ABC layer
- **Risk**: **MEDIUM**
- **Testing**: Validate all downstream dependencies

### Phase 3: Concrete Implementations (Weeks 5-7)
- **Files**: Priority 3.1-3.9, 4.1-4.4 (13 files)
- **Goal**: Migrate all measurers and refiners
- **Risk**: **LOW** - Well-defined interfaces
- **Testing**: Regression tests on sample data

### Phase 4: Analysis & Specialized Operations (Weeks 8-9)
- **Files**: Priority 5.1-5.3, 6.1-6.2 (5 files)
- **Goal**: Migrate analysis and grid operations
- **Risk**: **MEDIUM-HIGH** (edge correction complexity)
- **Testing**: Statistical validation

### Phase 5: Accessors & Data Layer (Weeks 10-11)
- **Files**: Priority 7.1-7.6, 8.1-8.5 (11 files)
- **Goal**: Migrate accessors and ImageSet
- **Risk**: **MEDIUM**
- **Testing**: Integration tests with pipelines

### Phase 6: Pipelines & CLI (Week 12)
- **Files**: Priority 9.1-9.3, 10.1-10.2 (5 files)
- **Goal**: Complete top-level orchestration
- **Risk**: **LOW** - Depends on all previous work
- **Testing**: End-to-end workflows

### Phase 7: Tests & Documentation (Weeks 13-14)
- **Files**: Priority 11 (27+ files)
- **Goal**: Migrate all tests, update docs
- **Risk**: **LOW**
- **Testing**: Full test suite passing

**Total estimated time**: **14 weeks** for complete migration

---

## Success Criteria

1. ✅ All tests passing with polars DataFrames
2. ✅ No pandas imports in src/phenotypic/ (except compatibility shim if used)
3. ✅ Performance benchmarks show &gt;2x speedup on typical workflows
4. ✅ Memory usage reduced by &gt;20% on large datasets
5. ✅ Documentation updated with polars examples
6. ✅ Migration guide published for users
7. ✅ CI/CD pipeline validates polars-only dependencies

---

## Risk Mitigation

1. **Branching strategy**: Create `polars-migration` branch for all work
2. **Feature flags**: Use `PHENOTYPIC_USE_POLARS=1` env var to toggle during development
3. **Regression testing**: Maintain pandas test suite in parallel until migration complete
4. **Performance monitoring**: Track benchmarks at each phase
5. **Rollback plan**: Keep pandas-compatible v1.x branch maintained for 6 months post-migration

---

## Appendix: File Locations

All files referenced in this plan:

**ABC files** (8 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_measure_features.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_grid_measure.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_grid_finder.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_set_analyzer.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_model_fitter.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_grid_corrector.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_object_refiner.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/abc_/_grid_object_refiner.py`

**Measurement files** (9 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_shape.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_size.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_bounds.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_intensity.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_color.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_color_composition.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_texture.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_grid_spread.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/measure/_measure_grid_linreg_stats.py`

**Refiner files** (4 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/refine/_small_to_large_merger.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/refine/_nearest_neighbor_merger.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/refine/_transitive_distance_merger.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/refine/_circularity_modifier.py`

**Analysis files** (3 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/analysis/_tukey_outlier.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/analysis/_edge_correction.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/analysis/_log_growth_model.py`

**Grid operation files** (2 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/grid/_auto_grid_finder.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/grid/_manual_grid_finder.py`

**Core image files** (13 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/_image_grid_handler.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessors/_grid_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessors/_rgb_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessors/_objects_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_parts/accessors/_measurement_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_set_parts/_image_set_measurements.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_set_parts/_image_set_metadata.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_set_parts/_image_set_status.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_set_parts/_image_set_accessors/_image_set_measurements_accessor.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_image_set_parts/_image_set_accessors/_image_set_metadata_accessor.py`

**Pipeline files** (3 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_pipeline_parts/_image_pipeline_batch.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py`

**Utility files** (3 files):
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/tools/hdf_.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/data/_sample_image_data.py`
- `/Users/alex/Projects/PhenoTypic/src/phenotypic/phenotypicCLI.py`

**Test files** (27+ files in `/Users/alex/Projects/PhenoTypic/tests/`)

---

**Document version**: 1.0
**Last updated**: 2025-12-13
