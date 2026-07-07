# Design: High-Priority Dedupe Cleanup

- **Date:** 2026-07-03
- **Status:** Draft for discussion
- **Related report:** `docs/superpowers/reports/2026-07-02-dedupe-simplification-audit.md`
- **Topic:** Behavior-safe extraction of the highest-priority duplication found in
  detection, staged CLI state, CLI measurement-source discovery, and GUI URL
  prefix handling.

## 1. Problem and Goal

The dedupe audit found several duplicated implementation contracts that are not
just cosmetic repetition. They affect scientific behavior, durable run state,
output artifact discovery, and mounted GUI routing.

This spec covers five high-priority cleanup targets:

1. Standardize threshold method dispatch and invalid-method errors.
2. Centralize CLI measurement parquet discovery and image-name normalization.
3. Replace raw staged-GPU stage strings with validated internal constants.
4. Extract Dash URL-prefix `index_string` injection.
5. Simplify the shared grid-peak detector workflow after threshold behavior is
   centralized.

The goal is to reduce drift while keeping public operation names, serialized
pipeline JSON, output file layout, event-log string values, and Dash mount
semantics stable.

## 2. Scope

### In Scope

- Private helper modules and tests for the five P1 audit findings.
- One intentional behavior change: invalid threshold method names fail
  consistently instead of silently falling back to Otsu in grid detectors.
- Characterization tests before each behavior-sensitive extraction.
- Focused edits in detection, CLI output discovery, staged event handling, and
  GUI app factories.

### Out of Scope

- No public API additions.
- No output directory layout changes.
- No event-log value migration.
- No broad Dash app factory consolidation beyond URL-prefix injection.
- No OpenSeadragon asset consolidation.
- No SLURM script renderer rewrite.
- No pydantic field migration from concrete grid detectors into a private base
  class in this pass.

## 3. Proposed File Plan

### Files Added

| File | Purpose |
|---|---|
| `src/phenotypic/detect/_thresholding_registry.py` | Private `ThresholdingRegistry` for method validation, threshold computation, and reusable mask helpers. |
| `src/phenotypic/detect/_grid_peak_common.py` | Private grid-peak workflow helpers with no pydantic fields. |
| `src/phenotypic/_cli/_measurement_sources.py` | Deterministic measurement parquet discovery and filename-to-image-name normalization. |
| `src/phenotypic/_cli/_stages.py` | Internal staged-GPU stage tag constants, type alias, and validation. |
| `tests/unit/detect/test_thresholding_registry.py` | Unit tests for `ThresholdingRegistry` and invalid-method behavior. |
| `tests/unit/detect/test_grid_peak_common.py` | Unit tests for grid-peak helper behavior where direct helper tests are useful. |
| `tests/unit/cli/test_measurement_sources.py` | Unit tests for measurement parquet source discovery and normalization. |
| `tests/unit/cli/test_stage_tags.py` | Unit tests for stage tag validation and event parsing. |
| `tests/unit/gui/test_url_prefix_index.py` | Unit tests for Dash index-string prefix injection and escaping. |

### Files Touched

| Area | Files |
|---|---|
| Detection threshold dispatch | `src/phenotypic/detect/_otsu_detector.py`, `_minimum_detector.py`, `_hysteresis_detector.py`, `_round_peaks_detector.py`, `_sine_peak_detector.py`, `_inoculum_detector.py` |
| Grid-peak workflow | `src/phenotypic/detect/_round_peaks_detector.py`, `_sine_peak_detector.py`, `_inoculum_detector.py` |
| CLI measurement sources | `src/phenotypic/_cli/_cli_output_manager.py`, `_dashboard/_analysis_data.py`, `_cli_recompile_slurm_scripts.py` |
| Staged-GPU events | `src/phenotypic/_cli/_cli_staged_strategy.py`, `_cli_staged_slurm_worker.py`, `_cli_staged_workers.py`, `_cli_update_state.py` |
| GUI URL prefix | `src/phenotypic/gui/_url_prefix.py`, `gui/builder/_app.py`, `gui/results_viewer/_app.py`, `gui/browse/_app.py` |
| Tests | Existing related tests under `tests/unit/detect/`, `tests/integration/cli/`, and `tests/unit/gui/` |

## 4. Design

### 4.1 ThresholdingRegistry

Add `src/phenotypic/detect/_thresholding_registry.py` as the private owner of
threshold method dispatch. The central type is `ThresholdingRegistry`.

Proposed internal surface:

```python
ThresholdMethodName = Literal[
    "otsu",
    "mean",
    "local",
    "triangle",
    "minimum",
    "isodata",
    "li",
    "yen",
]

class ThresholdingRegistry:
    GRID_METHODS: ClassVar[frozenset[str]]
    SCALAR_METHODS: ClassVar[frozenset[str]]
    NBINS_METHODS: ClassVar[frozenset[str]]

    @classmethod
    def validate_method(
        cls,
        method: str,
        *,
        allowed_methods: Collection[str] | None = None,
    ) -> str: ...

    @classmethod
    def threshold_value(
        cls,
        threshold_spec: str | int | float,
        data: np.ndarray,
        *,
        bit_depth: int | None = None,
        allowed_methods: Collection[str] | None = None,
    ) -> float: ...

    @classmethod
    def threshold_mask(
        cls,
        matrix: np.ndarray,
        *,
        method: str,
        bit_depth: int | None = None,
        local_block_size: int | None = None,
        allowed_methods: Collection[str] | None = None,
        inclusive: bool = True,
    ) -> np.ndarray: ...
```

Rules:

- Unknown string methods raise `ValueError`.
- Numeric threshold specs are accepted only by APIs that currently support manual
  thresholds, such as `HysteresisDetector.low` and `HysteresisDetector.high`.
- `local` returns a threshold map internally and produces a boolean mask through
  `ThresholdingRegistry.threshold_mask`; it is not valid for scalar-only
  threshold value calls.
- Methods that support `nbins` receive `nbins=2 ** bit_depth` when `bit_depth`
  is provided.
- Methods that do not support `nbins` never receive it.
- Grid detectors continue to expose their current `Literal[...]` fields.
- Existing pydantic `ValidationError` behavior remains for invalid constructor
  values. The helper still raises `ValueError` if an invalid value reaches
  runtime code through deserialization or internal calls.

Affected behavior:

- The current grid-detector Otsu catch-all is removed intentionally.
- Tests must assert invalid grid threshold names fail consistently.

### 4.2 CLI Measurement Sources

Add `src/phenotypic/_cli/_measurement_sources.py` as the private owner of
measurement parquet source discovery.

Proposed internal surface:

```python
@dataclass(frozen=True)
class MeasurementSource:
    path: Path
    dataset: str

def discover_measurement_sources(
    output_dir: Path,
    dataset_names: Iterable[str] | None = None,
) -> list[MeasurementSource]: ...

def measurement_sources_by_path(
    sources: Iterable[MeasurementSource],
) -> dict[Path, str]: ...

def add_metadata_image_name_from_filename(
    frame: pl.DataFrame,
) -> pl.DataFrame: ...
```

Rules:

- Discovery always looks under `DIR_RESULTS / dataset / DIR_MEASUREMENTS`.
- If `_dataset_aggregated.parquet` exists for a dataset, it is the only source
  for that dataset.
- Otherwise, use sorted `*.parquet` files excluding names that start with `_`.
- If `dataset_names` is `None`, discover dataset directories under `DIR_RESULTS`.
- Preserve deterministic order by sorting datasets and per-dataset parquet files.
- `add_metadata_image_name_from_filename` derives `str(METADATA.IMAGE_NAME)`
  from the source-path basename only when that column is absent and `filename`
  exists. It then drops `filename`, matching current aggregation behavior.

Consumers:

- `aggregate_measurements(...)` in `_cli_output_manager.py`
- dashboard analysis data loading in `_dashboard/_analysis_data.py`
- recompile source discovery in `_cli_recompile_slurm_scripts.py`

### 4.3 Staged GPU Stage Tags

Add `src/phenotypic/_cli/_stages.py` as the private owner of staged-GPU stage
tags.

Proposed internal surface:

```python
StageTag = Literal["stage1", "stage2", "stage3"]

STAGE_PREPROCESS: StageTag = "stage1"
STAGE_GPU_DETECT: StageTag = "stage2"
STAGE_MEASURE: StageTag = "stage3"
STAGED_TERMINAL_STAGE: StageTag = STAGE_MEASURE
VALID_STAGE_TAGS: frozenset[str] = frozenset(...)

def validate_stage_tag(stage: str | None) -> StageTag | None: ...
```

Rules:

- Persisted event-log strings remain exactly `stage1`, `stage2`, and `stage3`.
- Legacy event rows with no stage field remain valid.
- Non-empty unknown stage tags raise `ValueError` in `parse_event_line`.
- Existing event aggregators already skip malformed lines after parse errors;
  keep that behavior.
- Overall completion uses `STAGED_TERMINAL_STAGE`, not a repeated string literal.
- Stage producers call `stage_event(..., STAGE_PREPROCESS)` etc.

Consumers:

- local staged strategy
- staged SLURM worker
- shared staged worker event helpers
- update-state parsing and aggregation

### 4.4 Dash URL-Prefix Index Injection

Extend `src/phenotypic/gui/_url_prefix.py` with a helper that owns Dash
`index_string` injection.

Proposed internal surface:

```python
def dash_index_string_with_app_prefix(url_prefix: str) -> str: ...
```

Rules:

- The helper escapes backslashes and double quotes before embedding the prefix
  in a JavaScript string literal.
- The injected value remains `window.__phenotypicAppPrefix`.
- The returned template preserves the standard Dash placeholders:
  `{%metas%}`, `{%favicon%}`, `{%css%}`, `{%app_entry%}`, `{%config%}`,
  `{%scripts%}`, and `{%renderer%}`.
- The helper does not install middleware and does not normalize routing by
  itself. It only returns the template string.

Consumers:

- `gui/builder/_app.py`
- `gui/results_viewer/_app.py`
- `gui/browse/_app.py`

The broader repeated Dash app factory setup remains out of scope for this spec.

### 4.5 Grid-Peak Workflow Helpers

Add `src/phenotypic/detect/_grid_peak_common.py` only after threshold dispatch is
centralized.

Preferred shape:

- Use private helper functions or a no-field mixin.
- Do not move pydantic fields into a private base class in this pass.
- Keep public `RoundPeaksDetector`, `SinePeakDetector`, and `InoculumDetector`
  schemas stable.
- Keep class-specific edge estimation local.

Proposed helper responsibilities:

```python
def grid_peak_background_kernel(
    matrix_shape: tuple[int, int],
    *,
    footprint_width: int,
    nrows: int | None,
    ncols: int | None,
) -> np.ndarray: ...

def grid_peak_threshold_mask(
    matrix: np.ndarray,
    *,
    thresh_method: str,
    subtract_background: bool,
    footprint_width: int,
    nrows: int | None,
    ncols: int | None,
) -> np.ndarray: ...
```

Potential second-step helpers, only if the tests make behavior parity clear:

```python
def assign_labeled_regions_to_grid_cells(...): ...
def finalize_grid_peak_objmap(...): ...
```

Rules:

- First extraction target is the exact duplicated `_thresholding(...)` logic in
  `RoundPeaksDetector` and `SinePeakDetector`.
- Larger workflow extraction is allowed only after detector parity tests pass.
- `InoculumDetector` should benefit through its use of `RoundPeaksDetector`; do
  not add a parallel threshold implementation there.

## 5. Phase Plan

### Phase 0: Characterization Tests

Add tests that describe current intended behavior before code extraction:

- Threshold methods compute expected scalar or local masks.
- Invalid threshold method names fail consistently.
- Current Round/Sine detector outputs on synthetic plates are captured at the
  object-count and objmap-shape level.
- Measurement source discovery prefers `_dataset_aggregated.parquet`.
- Stage aggregation treats stage 1 and stage 2 completion as in-progress, and
  stage 3 completion as overall completion.
- URL-prefix injection preserves escaping and Dash placeholders.

### Phase 1: Dash URL-Prefix Helper

Add `dash_index_string_with_app_prefix(...)` and replace the three local
`_index_string_with_prefix(...)` functions.

Risk: low. This is string extraction with direct tests.

### Phase 2: Stage Tags

Add `_cli/_stages.py`, replace raw stage literals at producer and aggregation
sites, and validate non-empty stage tags in `parse_event_line`.

Risk: medium. Event logs are durable state. Legacy no-stage rows must continue
to parse.

### Phase 3: Measurement Sources

Add `_cli/_measurement_sources.py`, use it from aggregation, dashboard analysis
data loading, and recompile planning.

Risk: medium. File ordering and `Metadata_ImageName` derivation are user-visible
in output artifacts.

### Phase 4: Threshold Registry

Add `_thresholding_registry.py`, migrate Hysteresis and grid detector threshold
dispatch, then migrate simple threshold detectors where it reduces duplication
without changing comparator semantics.

Risk: medium to high. Thresholding is scientific behavior. The invalid-method
strictness change is intentional and must be called out in release notes or PR
description.

### Phase 5: Grid-Peak Workflow

Add `_grid_peak_common.py` and remove duplicated grid-peak threshold/background
logic from Round/Sine detectors. Consider larger workflow helpers only if parity
tests remain stable and the helper boundary stays simple.

Risk: medium. The detectors are user-facing segmentation operations.

## 6. Testing and Acceptance Criteria

### Unit Tests

- `uv run pytest tests/unit/gui/test_url_prefix_index.py`
- `uv run pytest tests/unit/cli/test_stage_tags.py`
- `uv run pytest tests/unit/cli/test_measurement_sources.py`
- `uv run pytest tests/unit/detect/test_thresholding_registry.py`
- `uv run pytest tests/unit/detect/test_grid_peak_common.py`

### Existing Tests to Re-run

- `uv run pytest tests/unit/gui/test_url_prefix_middleware.py`
- `uv run pytest tests/integration/cli/test_staged_gpu_local.py`
- `uv run pytest tests/unit/detect/test_round_peaks_detector.py`
- `uv run pytest tests/unit/detect/test_sine_peak_detector.py`
- `uv run pytest tests/unit/detect/test_inoculum_detector.py`
- `uv run pytest tests/unit/refine/test_sine_alignment_refiner.py`

### Acceptance Criteria

- Invalid threshold method names raise consistently.
- No public operation field names are changed.
- Existing serialized pipelines using valid threshold values still load.
- Event logs with no stage field still parse.
- Event logs with valid stage strings preserve current aggregation semantics.
- Unknown non-empty stage tags are treated as malformed event lines.
- Aggregated measurement outputs use the same source files and image names as
  before, except through the shared helper.
- Mounted GUI prefixes still populate `window.__phenotypicAppPrefix` correctly.
- Round/Sine detector output parity is maintained except for the intentional
  invalid-threshold strictness change.

## 7. Rollout and Review Notes

- Implement as separate commits or PR phases in the order above.
- Each phase should be reviewable independently and include its focused tests.
- Do not combine this cleanup with broader P2 work from the audit report.
- The PR description should explicitly call out the strict invalid-threshold
  behavior change.
- If Phase 5 starts expanding into field migration or a private pydantic base
  class, stop and split that into a follow-up spec.

## 8. Open Questions

None. The stricter invalid-threshold behavior is locked for this spec.
