# Dedupe and Simplification Audit

Date: 2026-07-02

## Status

Tracking report. No production code changes were made as part of this audit.

## Provenance

This report summarizes a read-only codebase audit for extractable duplication,
stringly typed contracts, repeated IO/access patterns, and simplification
opportunities. The audit used local search, AST-oriented clone scans, targeted
file inspection, and four focused subagent passes:

- Operations, image-processing, detection, refinement, measurement, and SDK code.
- CLI, staged GPU, output layout, schema, analysis, post-processing, and tuning.
- GUI, dashboard, Dash app factories, JavaScript assets, CSS, and docs templates.
- Tests, scripts, fixtures, and generated-reference tooling.

The goal is not to eliminate every repeated line. The useful targets are places
where duplicated behavior can drift, where constants represent public or durable
contracts, or where extraction would make future changes safer.

## Tracking

| ID | Priority | Area | Status | Follow-up | Verification focus |
|---|---:|---|---|---|---|
| DEDUPE-001 | P1 | Detection threshold registry | Spec drafted | `docs/superpowers/specs/2026-07-03-high-priority-dedupe-cleanup/design.md` | Detector output parity, method validation, nbins handling |
| DEDUPE-002 | P1 | CLI measurement-source discovery | Spec drafted | `docs/superpowers/specs/2026-07-03-high-priority-dedupe-cleanup/design.md` | Parquet discovery order, aggregated-file preference, image-name normalization |
| DEDUPE-003 | P1 | Staged GPU stage tags | Spec drafted | `docs/superpowers/specs/2026-07-03-high-priority-dedupe-cleanup/design.md` | Legacy events, invalid stages, terminal-stage aggregation |
| DEDUPE-004 | P1 | Dash URL-prefix injection | Spec drafted | `docs/superpowers/specs/2026-07-03-high-priority-dedupe-cleanup/design.md` | Escaping, mounted prefixes, app factory parity |
| DEDUPE-005 | P1 | Grid peak detector workflow | Spec drafted | `docs/superpowers/specs/2026-07-03-high-priority-dedupe-cleanup/design.md` | Round/Sine detector parity on synthetic plates |
| DEDUPE-006 | P2 | Sine grid edge estimator | Tracked | Later or with DEDUPE-005 | Edge arrays and NCC parity |
| DEDUPE-007 | P2 | SLURM script rendering | Tracked | Later spec | Generated script snapshots and live smoke tests |
| DEDUPE-008 | P2 | Atomic writes and parquet policy | Tracked | Later spec | Partial-write safety, compression options |
| DEDUPE-009 | P2 | GUI/design tokens | Tracked | Later spec | Generated CSS and visual parity |
| DEDUPE-010 | P2 | OpenSeadragon vendor assets and loader | Tracked | Later spec | Offline asset loading and prefixUrl behavior |
| DEDUPE-011 | P2 | Dash app bootstrap | Tracked | Later spec | App factory config and shared static registration |
| DEDUPE-012 | P2 | Wavelet denoise operation pattern | Tracked | Later spec | Operation schema and numerical output parity |
| DEDUPE-013 | P2 | QC status labels/ranks/colors | Tracked | Later spec | QC summary and Plotly color parity |
| DEDUPE-014 | P2 | Linear growth model heuristics | Tracked | Later spec | Fitted parameter arrays and bounds parity |
| DEDUPE-015 | P2 | HDF writer retry lifecycle | Tracked | Later spec | Lock recovery, SWMR behavior, cleanup |
| DEDUPE-016 | P2 | Morphological footprint resolution | Tracked | Later spec | Footprint shape/default parity |
| DEDUPE-017 | P2 | ColorChecker defaults | Tracked | Later spec | Helper/profile default consistency |
| DEDUPE-018 | P2 | Symmetric-zone parameter pattern | Tracked | Later spec | Refiner/measurer defaults and diagnostics |
| DEDUPE-019 | P2 | pytest xdist worker hook | Tracked | Later spec | `pytest -n auto` behavior on local and SLURM envs |
| DEDUPE-020 | P2 | GUI ledger validators | Tracked | Later spec | Escaped-pipe Markdown parsing and CI checks |
| DEDUPE-021 | P2 | Reference generator runners | Tracked | Later spec | `--check` failure output and stale-output messaging |
| DEDUPE-022 | P3 | Image layer closed set | Tracked | Later spec | CLI choices, HDF validation, process-only exports |
| DEDUPE-023 | P3 | Dashboard analysis CSS fragments | Tracked | Later spec | Static dashboard rendering |
| DEDUPE-024 | P3 | Docs templates | Tracked | Later spec | Sphinx output parity |
| DEDUPE-025 | P3 | CLI test fixtures | Tracked | Opportunistic | Test readability only |
| DEDUPE-026 | P3 | GUI output-root test seeding | Tracked | Opportunistic | Test fixture parity |
| DEDUPE-027 | P3 | Builder DAG fixture paths | Tracked | Opportunistic | Test fixture path correctness |
| DEDUPE-028 | P3 | Profiling script logs | Tracked | Opportunistic | Benchmark log output parity |
| DEDUPE-029 | P3 | Plot style token re-spelling | Tracked | Opportunistic | Diagnostic figure color parity |
| DEDUPE-030 | P3 | Small repeated helpers | Tracked | Opportunistic | Local unit tests |

## P1 Findings

### DEDUPE-001: Detection Threshold Registry

Threshold selection is hand-rolled across threshold detectors and grid detectors.
The duplicated logic covers method dispatch, zero filtering, optional `nbins`,
border clearing, local threshold maps, and fallback behavior. There is already
semantic drift: some paths validate unknown methods, while grid detectors fall
back to Otsu.

Evidence:

- `src/phenotypic/detect/_otsu_detector.py`
- `src/phenotypic/detect/_minimum_detector.py`
- `src/phenotypic/detect/_hysteresis_detector.py`
- `src/phenotypic/detect/_round_peaks_detector.py`
- `src/phenotypic/detect/_sine_peak_detector.py`
- `tests/unit/detect/test_inoculum_detector.py`
- `tests/unit/detect/test_round_peaks_detector.py`
- `tests/unit/detect/test_sine_peak_detector.py`

Proposal:

Extract a private `ThresholdingRegistry` in
`src/phenotypic/detect/_thresholding_registry.py` for method registry, `nbins`
capability, zero exclusion, local threshold handling, and optional binary-mask
postprocessing. Preserve all current public detector defaults and serialized
operation fields, but standardize invalid threshold names to fail instead of
silently falling back to Otsu.

### DEDUPE-002: CLI Measurement-Source Discovery

Master aggregation, dashboard sidecar loading, and recompile planning rediscover
`results/<dataset>/measurements`, prefer `_dataset_aggregated.parquet`, skip
underscore-prefixed files, and derive `Metadata_ImageName` from source filenames
independently.

Evidence:

- `src/phenotypic/_cli/_cli_output_manager.py`
- `src/phenotypic/_cli/_dashboard/_analysis_data.py`
- `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py`

Proposal:

Extract one CLI output-access helper for deterministic measurement source
discovery and image-name normalization. Use it from aggregation, dashboard
analysis data loading, and recompile task selection.

### DEDUPE-003: Staged GPU Stage Tags

Staged execution emits raw `"stage1"`, `"stage2"`, and `"stage3"` strings, while
overall aggregation treats only exact `"stage3"` as terminal. `parse_event_line`
validates processing status but not the stage tag.

Evidence:

- `src/phenotypic/_cli/_cli_staged_strategy.py`
- `src/phenotypic/_cli/_cli_staged_slurm_worker.py`
- `src/phenotypic/_cli/_cli_staged_workers.py`
- `src/phenotypic/_cli/_cli_update_state.py`
- `tests/integration/cli/test_staged_gpu_local.py`

Proposal:

Introduce a closed `StageTag` type and constants for `stage1`, `stage2`,
`stage3`, and terminal `stage3`. Validate staged event tags at parse/append
boundaries while preserving durable string values in existing event logs.

### DEDUPE-004: Dash URL-Prefix Injection

Three app factories duplicate Dash `index_string` injection for
`window.__phenotypicAppPrefix`, including JavaScript string escaping. This is a
routing-critical value for mounted GUI deployments and Open OnDemand-style
prefixes.

Evidence:

- `src/phenotypic/gui/builder/_app.py`
- `src/phenotypic/gui/results_viewer/_app.py`
- `src/phenotypic/gui/browse/_app.py`
- `src/phenotypic/gui/_url_prefix.py`

Proposal:

Move the prefix-injected Dash index template into `phenotypic.gui._url_prefix`
as a reusable helper. Keep app factories responsible only for assigning the
returned `index_string`.

### DEDUPE-005: Grid Peak Detector Workflow

`RoundPeaksDetector` and `SinePeakDetector` duplicate most of their workflow:
parameter blocks, adaptive background subtraction, thresholding, noise removal,
labeling, grid-cell assignment, fallback behavior, relabeling, and cleanup. The
main difference is the grid-edge estimator.

Evidence:

- `src/phenotypic/detect/_round_peaks_detector.py`
- `src/phenotypic/detect/_sine_peak_detector.py`
- `src/phenotypic/detect/_inoculum_detector.py`
- `tests/unit/detect/test_round_peaks_detector.py`
- `tests/unit/detect/test_sine_peak_detector.py`

Proposal:

Extract shared grid-peak helpers only after `ThresholdingRegistry` is in place.
Use a private mixin or base class with an edge-estimation hook, leaving public
operation classes and schemas unchanged.

## P2 Findings

### DEDUPE-006: Sine Grid Edge Estimator

The sine/rank cross-correlation edge estimator and normalized
cross-correlation implementation are duplicated between detection and
refinement, including denominator guards and peak-selection logic.

Evidence:

- `src/phenotypic/detect/_sine_peak_detector.py`
- `src/phenotypic/refine/_refine_by_sine_fit.py`

Proposal:

Move sine edge inference and normalized cross-correlation into a shared private
grid inference helper or mixin with explicit parameters and named epsilon
constants.

### DEDUPE-007: SLURM Script Rendering

Array jobs, recompile jobs, staged GPU jobs, and tuning SLURM execution repeat
directive rendering, array directives, prologues, metadata echoes, exit-code
logging, `write_text`, and `chmod`.

Evidence:

- `src/phenotypic/_cli/_cli_slurm_array_scripts.py`
- `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py`
- `src/phenotypic/_cli/_cli_staged_slurm.py`
- `src/phenotypic/_execution/_slurm.py`

Proposal:

Add a shared `SlurmArrayScriptSpec` or rendering helper parameterized by job
name, logs, array shape, optional signal/requeue directives, and script body.

### DEDUPE-008: Atomic Writes and Parquet Policy

Atomic temp-write plus `os.replace` is implemented several ways. Cleanup,
`fsync`, and parquet compression settings are inconsistent across CLI
deliverables, dashboard manifests, error outputs, chunk writing, and tune
journal export.

Evidence:

- `src/phenotypic/_cli/_cli_output_manager.py`
- `src/phenotypic/_cli/_dashboard/_analysis_helpers.py`
- `src/phenotypic/_cli/_dashboard/_manifest_builder.py`
- `src/phenotypic/_cli/_cli_error_outputs.py`
- `src/phenotypic/_cli/_cli_chunk_writer.py`
- `src/phenotypic/tune/_study_store.py`

Proposal:

Centralize atomic JSON/text/parquet writes and define one parquet write policy,
for example `PARQUET_WRITE_OPTIONS = {"compression": "zstd",
"compression_level": 3}`.

### DEDUPE-009: GUI and Dashboard Design Tokens

GUI design tokens are centralized in `gui/_design.py`, but CLI dashboard,
CLI report generation, docs CSS, and Plotly diagnostics re-declare palettes
and fonts. Drift is already visible in background color and font choices.

Evidence:

- `src/phenotypic/gui/_design.py`
- `src/phenotypic/_cli/_dashboard/_generator.py`
- `src/phenotypic/_cli/_cli_report_generator.py`
- `src/phenotypic/sdk_/viz/figures/_theme.py`
- `docs/source/_static/custom.css`

Proposal:

Choose one neutral token source or export a shared CSS/token map. Keep rendering
layers free to map tokens into their own format, but avoid re-authoring values.

### DEDUPE-010: OpenSeadragon Vendor Assets and Loader

OpenSeadragon assets are duplicated under browse and results viewer, while
several JS files repeat loader and `prefixUrl` construction logic.

Evidence:

- `src/phenotypic/gui/browse/_assets/openseadragon/`
- `src/phenotypic/gui/results_viewer/_assets/openseadragon/`
- `src/phenotypic/gui/results_viewer/_assets/results_viewer.js`
- `src/phenotypic/gui/browse/_assets/browse.js`
- `src/phenotypic/gui/builder/assets/point_picker.js`
- `src/phenotypic/gui/builder/assets/preview.js`

Proposal:

Serve OpenSeadragon from one shared static route and factor a small JS loader
with configurable CDN/offline policy and image-prefix resolution.

### DEDUPE-011: Dash App Bootstrap

Dash app factories repeat Bootstrap stylesheet setup,
`suppress_callback_exceptions`, `requests_pathname_prefix`, `routes_pathname_prefix`,
design-token injection, and shared-static registration.

Evidence:

- `src/phenotypic/gui/builder/_app.py`
- `src/phenotypic/gui/results_viewer/_app.py`
- `src/phenotypic/gui/run_console/_app.py`
- `src/phenotypic/gui/analysis/_app.py`
- `src/phenotypic/gui/tune/_app.py`
- `src/phenotypic/gui/browse/_app.py`
- `src/phenotypic/gui/shell/_app.py`

Proposal:

Introduce a small `create_gui_dash_app` helper after the URL-prefix injection
helper lands. Keep app-specific layout, stores, routes, and callbacks local.

### DEDUPE-012: Wavelet Denoise Operation Pattern

Four wavelet operations repeat generalized Anscombe transform support, wavelet
fields, `denoise_wavelet` kwargs, clip handling, and layer-specific assignment.

Evidence:

- `src/phenotypic/enhance/_visushrink_enhancer.py`
- `src/phenotypic/enhance/_bayesshrink_enhancer.py`
- `src/phenotypic/correction/_visushrink_corrector.py`
- `src/phenotypic/correction/_bayesshrink_corrector.py`

Proposal:

Extract a private wavelet-denoise mixin/helper for common fields and layer
application. Preserve public operation class names and JSON schemas.

### DEDUPE-013: QC Status Labels, Ranks, and Colors

QC pass/warn/fail labels, severity ranking, unknown color, and plot colors are
redefined in base summary logic and multiple QC plotters.

Evidence:

- `src/phenotypic/analysis/abc_/_quality_check.py`
- `src/phenotypic/analysis/qc/_grid_occupancy.py`
- `src/phenotypic/analysis/qc/_expected_vs_detected.py`
- `src/phenotypic/analysis/qc/_replicate_agreement.py`

Proposal:

Extract QC status constants and a `worst_qc_status` helper in the QC analysis
layer.

### DEDUPE-014: Linear Growth Model Heuristics

Linear lag and linear cap-and-lag models duplicate initial-guess and bounds
heuristics, including slope floors, range floors, tail-fit fractions, crossing
thresholds, alpha defaults, and alpha bounds.

Evidence:

- `src/phenotypic/analysis/_linear_lag_model.py`
- `src/phenotypic/analysis/_linear_cap_and_lag_model.py`

Proposal:

Move shared constants and helper methods into the existing linear softplus base,
with cap-and-lag adding beta-specific parameters.

### DEDUPE-015: HDF Writer Retry Lifecycle

Safe writer and SWMR writer paths duplicate retry/backoff, HDF lock-error
matching, `h5clear` calls, and final error construction.

Evidence:

- `src/phenotypic/sdk_/hdf_.py`

Proposal:

Extract a private HDF open-with-recovery helper or context manager parameterized
by mode, SWMR enablement, and `h5clear` force behavior.

### DEDUPE-016: Morphological Footprint Resolution

Mask morphology operations repeatedly resolve auto, custom, named, and `None`
footprints, with small differences in auto fractions, default shapes, and `None`
handling.

Evidence:

- `src/phenotypic/refine/_mask_erosion.py`
- `src/phenotypic/refine/_mask_dilation.py`
- `src/phenotypic/refine/_mask_opening.py`
- `src/phenotypic/refine/_mask_closing.py`
- `src/phenotypic/refine/_mask_white_tophat.py`

Proposal:

Extend `FootprintMixin` with a `_resolve_footprint(...)` helper and keep each
operation responsible only for its morphology function and defaults.

### DEDUPE-017: ColorChecker Defaults

`ColorCheckerProfile` fields duplicate helper defaults for median filter size,
standard-deviation threshold, swatch-area fraction, and core fraction.

Evidence:

- `src/phenotypic/correction/_color_correction/_color_checker_profile.py`
- `src/phenotypic/correction/_color_correction/_helpers.py`

Proposal:

Centralize defaults as module constants or a segmentation-params value object
shared by profile fields and helper signatures.

### DEDUPE-018: Symmetric-Zone Parameter Pattern

`TrimAsymmetry` and `MeasureSymmetricZones` share radial-density, PELT,
angular-coverage, smoothing, and method parameters. Some defaults have already
diverged.

Evidence:

- `src/phenotypic/refine/_trim_asymmetry.py`
- `src/phenotypic/measure/_measure_symmetric_zones.py`

Proposal:

Extract shared symmetry-analysis parameter constants and compute-intermediate
helpers. Preserve intentional per-operation default differences by naming them.

### DEDUPE-019: pytest xdist Auto-Worker Hook

`pytest_xdist_auto_num_workers` is implemented twice with different semantics:
the root hook returns affinity or `None`, while the test hook special-cases
`SLURM_CPUS_PER_TASK` and falls back to `os.cpu_count()`.

Evidence:

- `conftest.py`
- `tests/conftest.py`

Proposal:

Keep one hook or extract `resolve_xdist_auto_workers()` to a shared test config
helper.

### DEDUPE-020: GUI Ledger Validators

Feature and workflow ledger validators both parse Markdown tables and carry
ledger status/path logic. One handles escaped pipes and the other uses a plain
split.

Evidence:

- `scripts/check_features_md.py`
- `scripts/check_workflows_md.py`

Proposal:

Extract a small Markdown table parser and shared GUI ledger constants.

### DEDUPE-021: Reference Generator Runners

Validation and dispatch RST generators duplicate coverage checks, `--check`
comparison, write behavior, and stale-output messaging.

Evidence:

- `scripts/generate_validation_reference.py`
- `scripts/generate_dispatch_reference.py`

Proposal:

Extract a generic reference-generator runner while leaving curated rule and
dispatch tables local.

## P3 Findings

The following are lower-risk or test/docs-only cleanup candidates. They are
worth addressing opportunistically but should not block the high-priority
cleanup spec.

| ID | Area | Evidence | Suggested extraction |
|---|---|---|---|
| DEDUPE-022 | Image layer closed set | CLI choices, process-only maps, HDF validation, layer saving | Runtime tuple and layer maps near `ProcessOnlyLayer` |
| DEDUPE-023 | Dashboard analysis CSS | Dashboard analysis plugins | Shared analysis-control/table CSS |
| DEDUPE-024 | Docs templates | Sphinx class and image-accessor templates | Canonical template or Jinja include |
| DEDUPE-025 | CLI test fixtures | CLI unit/integration conftests and integration tests | Shared CLI fixture and argv builder |
| DEDUPE-026 | GUI output-root test seeding | Viewer and output-root tests | `seed_output_root(...)` helper |
| DEDUPE-027 | Builder DAG fixture paths | Builder unit/integration/e2e tests | `tests/fixtures/paths.py` helper |
| DEDUPE-028 | Profiling script logs | `profile_tests.py`, `profile_docs.py` | Shared benchmark log emitter |
| DEDUPE-029 | Plot style token re-spelling | Diagnostic modules and Plotly theme | Import sanctioned tokens |
| DEDUPE-030 | Small repeated helpers | BM3D stage converter, negative margin validators, lock caches, NN image readers | Local shared helpers only when touching nearby code |

## High-Priority Spec Seed

The high-priority work should be designed as a phased cleanup bundle. Each phase
must be independently reviewable and behavior-preserving.

### Candidate Files Added

- `src/phenotypic/detect/_thresholding_registry.py`
- `src/phenotypic/detect/_grid_peak_base.py` or `src/phenotypic/detect/_grid_peak_common.py`
- `src/phenotypic/_cli/_measurement_sources.py`
- `src/phenotypic/_cli/_stages.py`
- `tests/unit/detect/test_thresholding_helpers.py`
- `tests/unit/cli/test_measurement_sources.py`
- `tests/unit/cli/test_stage_tags.py`
- `tests/unit/gui/test_url_prefix_index.py`

### Candidate Files Touched

- `src/phenotypic/detect/_otsu_detector.py`
- `src/phenotypic/detect/_minimum_detector.py`
- `src/phenotypic/detect/_hysteresis_detector.py`
- `src/phenotypic/detect/_round_peaks_detector.py`
- `src/phenotypic/detect/_sine_peak_detector.py`
- `src/phenotypic/detect/_inoculum_detector.py`
- `src/phenotypic/_cli/_cli_output_manager.py`
- `src/phenotypic/_cli/_dashboard/_analysis_data.py`
- `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py`
- `src/phenotypic/_cli/_cli_staged_strategy.py`
- `src/phenotypic/_cli/_cli_staged_slurm_worker.py`
- `src/phenotypic/_cli/_cli_staged_workers.py`
- `src/phenotypic/_cli/_cli_update_state.py`
- `src/phenotypic/gui/_url_prefix.py`
- `src/phenotypic/gui/builder/_app.py`
- `src/phenotypic/gui/results_viewer/_app.py`
- `src/phenotypic/gui/browse/_app.py`
- Existing tests under `tests/unit/detect/`, `tests/integration/cli/`, and
  `tests/unit/gui/`.

### Design Constraints

- Keep public operation names, pydantic fields, serialized pipeline JSON, output
  paths, event-log string values, and Dash mount semantics stable.
- Prefer private helpers over new public API.
- Start with characterization tests before extraction in behavior-sensitive
  areas.
- Treat threshold behavior as scientific behavior. Invalid threshold methods
  should raise consistently across detectors; the existing grid-detector
  fallback to Otsu is removed intentionally and covered by tests.
- Treat staged event logs as durable state. New validation must not break old
  event logs that omit the stage field.
- Keep GUI helper extraction narrow before attempting broader Dash app bootstrap
  consolidation.

### Recommended Phase Order

1. Add characterization tests for P1 behavior.
2. Extract Dash URL-prefix injection helper.
3. Extract staged GPU stage constants and validation.
4. Extract CLI measurement-source discovery and normalization.
5. Extract `ThresholdingRegistry`.
6. Extract grid-peak common workflow only after threshold behavior is centralized.

This order starts with lower-risk routing/state helpers before touching
detection behavior.
