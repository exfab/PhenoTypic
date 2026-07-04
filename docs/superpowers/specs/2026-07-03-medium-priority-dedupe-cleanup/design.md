# Medium-Priority Dedupe Cleanup Design

Date: 2026-07-03

## Status

Draft spec for the P2 findings in
`docs/superpowers/reports/2026-07-02-dedupe-simplification-audit.md`.
No production code changes are part of this spec.

## Goal

Turn the P2 dedupe findings into implementation-sized cleanup bundles that can
be executed independently, reviewed independently, and verified with behavior
parity tests before any extraction changes scientific or runtime behavior.

## Scope

In scope:

- DEDUPE-006: Sine grid edge estimator.
- DEDUPE-007: SLURM script rendering.
- DEDUPE-008: Atomic writes and parquet policy.
- DEDUPE-009: GUI and dashboard design tokens.
- DEDUPE-010: OpenSeadragon vendor assets and loader.
- DEDUPE-011: Dash app bootstrap.
- DEDUPE-012: Wavelet denoise operation pattern.
- DEDUPE-013: QC status labels, ranks, and colors.
- DEDUPE-014: Linear growth model heuristics.
- DEDUPE-015: HDF writer retry lifecycle.
- DEDUPE-016: Morphological footprint resolution.
- DEDUPE-017: ColorChecker defaults.
- DEDUPE-018: Symmetric-zone parameter pattern.
- DEDUPE-019: pytest xdist auto-worker hook.
- DEDUPE-020: GUI ledger validators.
- DEDUPE-021: Reference generator runners.

Out of scope:

- P3 cleanup items.
- Public operation renames, public schema changes, output layout changes, and
  serialized pipeline field changes.
- Changing numerical defaults unless the spec explicitly names an intentional
  standardization.
- Replacing existing vendor libraries or adding runtime dependencies.

## Design Options

### Option A: One P2 Mega-Refactor

Put all medium-priority findings into one implementation pass. This reduces
coordination overhead, but it combines scientific behavior, GUI bootstrap,
script generation, HDF lifecycle, and test infrastructure in one diff. The
failure mode is a hard-to-review PR where a small regression in one area blocks
everything else.

### Option B: One Finding Per PR

Treat each DEDUPE item as its own implementation. This is easy to review, but
it creates too many tiny PRs and misses natural shared boundaries, for example
OpenSeadragon static serving and Dash app bootstrap.

### Option C: Clustered Bundles

Group findings by behavior surface and risk: test/script utilities, runtime
IO and SLURM, GUI/static bootstrap, and scientific helper extraction. This is
the recommended approach. It keeps each implementation reviewable while still
letting related call sites share one helper at the right layer.

## Recommended Phase Order

### Phase 1: Test and Script Utilities

Findings:

- DEDUPE-019: pytest xdist auto-worker hook.
- DEDUPE-020: GUI ledger validators.
- DEDUPE-021: Reference generator runners.

Rationale:

This phase is low risk and improves CI/test maintenance before broader
behavior-preserving extractions. It also gives the next phases shared test
helpers for generated-file checks.

Files to add:

- `tests/_support/__init__.py`
- `tests/_support/xdist_workers.py`
- `scripts/_markdown_table.py`
- `scripts/_reference_generator.py`
- `tests/unit/test_xdist_workers.py`
- `tests/unit/gui/test_check_features_md.py`
- `tests/unit/gui/test_reference_generators.py`

Files to modify:

- `conftest.py`
- `tests/conftest.py`
- `scripts/check_features_md.py`
- `scripts/check_workflows_md.py`
- `scripts/generate_validation_reference.py`
- `scripts/generate_dispatch_reference.py`
- `tests/unit/gui/test_check_workflows_md.py`

Design:

- Add `resolve_xdist_auto_workers(env, affinity_count, cpu_count)` as a pure
  helper. Both conftest hooks call it and standardize on this order:
  `SLURM_CPUS_PER_TASK` when present, then affinity count, then a caller-chosen
  fallback. The root hook passes `fallback=None` to preserve non-xdist default
  behavior on platforms without affinity; `tests/conftest.py` passes
  `fallback=os.cpu_count() or 1`.
- Extract escaped-pipe Markdown row splitting and table extraction into
  `scripts/_markdown_table.py`. Both ledger validators use the same parser,
  with each script retaining its own ledger-specific rules.
- Extract reference-generation check/write mechanics into
  `scripts/_reference_generator.py`. The validation and dispatch scripts keep
  their local rule tables and renderers, but share stale-output messaging,
  `--check` behavior, and write behavior.

Verification:

- `uv run pytest tests/unit/test_xdist_workers.py`
- `uv run pytest tests/unit/gui/test_check_features_md.py tests/unit/gui/test_check_workflows_md.py`
- `uv run pytest tests/unit/gui/test_reference_generators.py`
- `uv run python scripts/generate_validation_reference.py --check`
- `uv run python scripts/generate_dispatch_reference.py --check`

### Phase 2: Runtime IO, SLURM, and HDF Lifecycle

Findings:

- DEDUPE-007: SLURM script rendering.
- DEDUPE-008: Atomic writes and parquet policy.
- DEDUPE-015: HDF writer retry lifecycle.

Rationale:

These are all runtime-safety concerns. They should not be mixed with detector
or GUI behavior because verification needs script snapshots, atomic-write
failure tests, and HDF lock-recovery tests.

Files to add:

- `src/phenotypic/sdk_/_atomic_writes.py`
- `src/phenotypic/sdk_/slurm/_script_rendering.py`
- `tests/unit/sdk_/test_atomic_writes.py`
- `tests/unit/sdk_/test_slurm_script_rendering.py`
- `tests/unit/sdk_/test_hdf_open_recovery.py`

Files to modify:

- `src/phenotypic/_cli/_cli_output_manager.py`
- `src/phenotypic/_cli/_dashboard/_analysis_helpers.py`
- `src/phenotypic/_cli/_dashboard/_manifest_builder.py`
- `src/phenotypic/_cli/_cli_error_outputs.py`
- `src/phenotypic/_cli/_cli_chunk_writer.py`
- `src/phenotypic/tune/_study_store.py`
- `src/phenotypic/_cli/_cli_slurm_array_scripts.py`
- `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py`
- `src/phenotypic/_cli/_cli_staged_slurm.py`
- `src/phenotypic/_execution/_slurm.py`
- `src/phenotypic/sdk_/hdf_.py`
- Existing SLURM tests under `tests/unit/cli/`, `tests/unit/tune/`, and
  `tests/unit/sdk_/`.

Design:

- Add private atomic write helpers for text, JSON, bytes, and parquet. The
  helpers own temp-path creation, cleanup after writer failure, and `os.replace`.
  The default parquet policy is one named constant:
  `PARQUET_WRITE_OPTIONS = {"compression": "zstd", "compression_level": 3}`.
- Keep caller-specific serialization local. Callers pass either a payload or a
  writer callback, so this helper does not learn dashboard, tune, or CLI domain
  concepts.
- Add a private `SlurmArrayScriptSpec` that owns common SBATCH array rendering,
  optional signal/requeue directives, exit-code logging, and executable script
  writes. CLI image arrays, recompile arrays, staged GPU arrays, and tune worker
  arrays provide only their job-specific body.
- Add a private HDF open-recovery helper in `sdk_/hdf_.py` for lock-error
  matching, backoff, optional `h5clear -s`, optional `h5clear -f`, and final
  error construction. `safe_writer()` and `swmr_writer()` become policy calls
  into that helper.

Verification:

- Atomic writer tests simulate writer exceptions and assert no partial final
  file is left behind.
- SLURM script tests compare generated scripts for array directives, logs,
  signal/requeue handling, executable mode, and job-specific bodies.
- HDF tests monkeypatch `h5py.File`, `subprocess.run`, and `time.sleep` to
  verify retry, `h5clear`, and final error behavior without requiring a live
  lock conflict.
- Run existing targeted suites:
  `uv run pytest tests/unit/cli/test_cli_slurm_array.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_staged_slurm_scripts.py tests/unit/tune/test_slurm_executor.py tests/unit/sdk_/test_slurm_dispatcher.py tests/integration/cli/test_cli_hdf_output.py`

### Phase 3: GUI Static Assets and App Bootstrap

Findings:

- DEDUPE-009: GUI and dashboard design tokens.
- DEDUPE-010: OpenSeadragon vendor assets and loader.
- DEDUPE-011: Dash app bootstrap.

Rationale:

These items share URL-prefix and static-route concerns. They should build on
the completed URL-prefix helper from the high-priority cleanup.

Files to add:

- `src/phenotypic/gui/_dash_app.py`
- `src/phenotypic/gui/_shared/_openseadragon.py`
- `src/phenotypic/gui/_shared/assets/openseadragon/`
- `src/phenotypic/gui/_assets/tokens.json`
- `src/phenotypic/gui/_assets/tokens.css`
- `tests/unit/gui/test_dash_app_factory.py`
- `tests/unit/gui/test_shared_openseadragon.py`
- `tests/unit/gui/test_design_tokens.py`

Files to modify:

- `src/phenotypic/gui/_shared/_blueprint.py`
- `src/phenotypic/gui/_shared/__init__.py`
- `src/phenotypic/gui/_design.py`
- `src/phenotypic/gui/builder/_app.py`
- `src/phenotypic/gui/results_viewer/_app.py`
- `src/phenotypic/gui/browse/_app.py`
- `src/phenotypic/gui/run_console/_app.py`
- `src/phenotypic/gui/analysis/_app.py`
- `src/phenotypic/gui/tune/_app.py`
- `src/phenotypic/gui/shell/_app.py`
- `src/phenotypic/gui/results_viewer/_assets/results_viewer.js`
- `src/phenotypic/gui/browse/_assets/browse.js`
- `src/phenotypic/gui/builder/assets/point_picker.js`
- `src/phenotypic/gui/builder/assets/preview.js`
- `src/phenotypic/_cli/_dashboard/_generator.py`
- `src/phenotypic/_cli/_cli_report_generator.py`
- `src/phenotypic/sdk_/viz/figures/_theme.py`
- `docs/source/_static/custom.css`

Design:

- Add `create_gui_dash_app(...)` for common Dash construction:
  Bootstrap stylesheet, `suppress_callback_exceptions`, URL-prefix routing,
  optional app-prefix index injection, and shared-static registration. App
  factories keep local layout, stores, callbacks, and Flask routes.
- Move one canonical OpenSeadragon vendor tree under the shared GUI static
  assets. Browse, results viewer, and builder JS construct viewer URLs through
  a tiny shared JS loader contract rather than each app knowing the vendor
  subdirectory.
- Treat `gui/_design.py` as the canonical Python token source. Add a generated
  or checked-in `tokens.css`/`tokens.json` bridge only for consumers that cannot
  import Python. Keep Plotly's existing chart template, but make drift tests
  compare it against sanctioned GUI token values instead of duplicating literals
  in assertions.

Verification:

- Unit tests assert every Dash app factory still sets the same request and
  route prefixes as today.
- Shared-static tests request the logo and OpenSeadragon assets through the
  registered Flask blueprint.
- JS-loader tests or string tests verify `prefixUrl` is derived from the
  app prefix and shared-static path.
- Token tests assert dashboard/report/Plotly/docs token values match the
  canonical source where they are intended to match.
- Optional manual GUI smoke: launch `uv run phenotypic-gui --root ./images`
  and verify builder, browse, results viewer, run console, analysis, and tune
  tabs render static assets under a non-root `--url-prefix`.

### Phase 4: Scientific Helper Extraction

Findings:

- DEDUPE-006: Sine grid edge estimator.
- DEDUPE-012: Wavelet denoise operation pattern.
- DEDUPE-013: QC status labels, ranks, and colors.
- DEDUPE-014: Linear growth model heuristics.
- DEDUPE-016: Morphological footprint resolution.
- DEDUPE-017: ColorChecker defaults.
- DEDUPE-018: Symmetric-zone parameter pattern.

Rationale:

These are medium priority because the duplication is real, but output parity is
more important than reducing lines. This phase should be split into small
commits with characterization tests before each extraction.

Files to add:

- `src/phenotypic/detect/_sine_grid_inference.py`
- `src/phenotypic/sdk_/mixin/_wavelet_denoise_mixin.py`
- `src/phenotypic/analysis/qc/_status.py`
- `src/phenotypic/analysis/abc_/_linear_softplus_helpers.py`
- `src/phenotypic/correction/_color_correction/_defaults.py`
- `src/phenotypic/measure/_symmetric_zone_common.py`
- `tests/unit/detect/test_sine_grid_inference.py`
- `tests/unit/sdk_/test_wavelet_denoise_mixin.py`
- `tests/unit/analysis/test_qc_status.py`
- `tests/unit/analysis/test_linear_softplus_helpers.py`
- `tests/unit/refine/test_mask_footprint_resolution.py`
- `tests/unit/correction/test_color_checker_defaults.py`
- `tests/unit/measure/test_symmetric_zone_common.py`

Files to modify:

- `src/phenotypic/detect/_sine_peak_detector.py`
- `src/phenotypic/refine/_refine_by_sine_fit.py`
- `src/phenotypic/enhance/_visushrink_enhancer.py`
- `src/phenotypic/enhance/_bayesshrink_enhancer.py`
- `src/phenotypic/correction/_visushrink_corrector.py`
- `src/phenotypic/correction/_bayesshrink_corrector.py`
- `src/phenotypic/analysis/abc_/_quality_check.py`
- `src/phenotypic/analysis/qc/_grid_occupancy.py`
- `src/phenotypic/analysis/qc/_expected_vs_detected.py`
- `src/phenotypic/analysis/qc/_replicate_agreement.py`
- `src/phenotypic/analysis/_linear_lag_model.py`
- `src/phenotypic/analysis/_linear_cap_and_lag_model.py`
- `src/phenotypic/analysis/abc_/_linear_softplus_base.py`
- `src/phenotypic/sdk_/mixin/_footprint_mixin.py`
- `src/phenotypic/refine/_mask_erosion.py`
- `src/phenotypic/refine/_mask_dilation.py`
- `src/phenotypic/refine/_mask_opening.py`
- `src/phenotypic/refine/_mask_closing.py`
- `src/phenotypic/refine/_mask_white_tophat.py`
- `src/phenotypic/correction/_color_correction/_color_checker_profile.py`
- `src/phenotypic/correction/_color_correction/_helpers.py`
- `src/phenotypic/refine/_trim_asymmetry.py`
- `src/phenotypic/measure/_measure_symmetric_zones.py`

Design:

- Sine grid inference: extract normalized cross-correlation, rank-template
  construction, peak filtering, fallback peak generation, and edge derivation.
  Both `SinePeakDetector` and `RefineBySineFit` pass their existing fields into
  the helper. Keep object assignment and refinement local.
- Wavelet denoise: extract common denoise kwargs and layer-application helpers,
  but leave public operation fields and docstrings on each operation class.
  Enhancement ops still write only `detect_mat`; correction ops still write
  RGB, gray, and detect_mat.
- QC status: define private constants for pass/warn/fail labels, ranks, and
  plot colors. Add `worst_qc_status(statuses)` and `qc_status_color(status)`.
  Keep status column names and DataFrame values as strings.
- Linear growth: move common initial guess and bounds helpers into the existing
  linear softplus base layer. Cap-and-lag appends beta-specific guess/bounds;
  lag-only remains four-parameter.
- Footprints: extend `FootprintMixin` with a `_resolve_footprint(...)` helper
  that handles `"auto"`, named shapes, array fields, and `None`. Each operation
  still supplies its auto width rule, default shape, and morphology function.
- ColorChecker: centralize segmentation defaults as constants or a frozen
  value object shared by profile fields and helper signatures. Validators stay
  on `ColorCheckerProfile`.
- Symmetric zones: extract shared radial-density, PELT core-radius,
  angular-profile, and symmetric-radius helpers into `_symmetric_zone_common`.
  Keep `MeasureSymmetricZones` as the richer measurer and `TrimAsymmetry` as a
  refiner. Name intentional default differences explicitly.

Verification:

- Characterization tests must run before extraction and compare arrays or
  frames before and after each helper change.
- Use fixed synthetic inputs from existing tests where possible.
- Required targeted suites:
  `uv run pytest tests/unit/detect/test_sine_peak_detector.py tests/unit/refine/test_refine_by_sine_fit.py tests/unit/correction/test_wavelet_correctors.py tests/unit/analysis/test_qc_risk_scenarios.py tests/unit/analysis/test_linear_softplus.py tests/unit/refine/test_asymmetric_spur_trimmer.py tests/unit/measure/test_measure_symmetric_zones.py tests/unit/measure/test_symmetric_zones_figure.py`

## Implementation Rules

- Keep helpers private unless a public API already exists at that layer.
- Prefer characterization tests before extraction in Phases 2 through 4.
- Preserve serialized operation fields and JSON round-trips.
- Preserve output file names, dashboard static paths, SLURM script purpose, and
  HDF open modes.
- Use `uv run pytest`, `uv run ruff`, and targeted snapshot/parity tests for
  each phase before moving to the next.
- Run an independent code-review subagent after each implemented phase, matching
  the high-priority cleanup workflow.

## Tracking Plan

Recommended execution order:

1. Phase 1: Test and script utilities.
2. Phase 3a: Dash app bootstrap only.
3. Phase 3b: OpenSeadragon shared assets and token drift guards.
4. Phase 2a: Atomic write and parquet policy.
5. Phase 2b: SLURM script rendering.
6. Phase 2c: HDF open recovery.
7. Phase 4a: QC status and ColorChecker defaults.
8. Phase 4b: Footprint resolution.
9. Phase 4c: Sine grid inference.
10. Phase 4d: Wavelet denoise and linear/symmetric-zone helpers.

This order front-loads low-risk utility wins, then GUI/static consolidation,
then runtime safety, and leaves scientific-output extractions for the end after
the testing pattern is established.

## Open Questions

- Whether to commit each phase separately or use one commit per sub-phase.
  The recommended default is one commit per sub-phase above.
- Whether design tokens should be generated from `gui/_design.py` or checked in
  as a mirrored JSON/CSS bridge. The recommended default is checked-in bridge
  files with drift tests, because the repo does not currently have a token
  generation step.
- Whether the SLURM rendering helper belongs in `sdk_.slurm` or `_cli`.
  The recommended default is `sdk_.slurm` because tune execution and CLI
  execution both need it.
