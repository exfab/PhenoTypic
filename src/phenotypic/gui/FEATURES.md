# PhenoTypic GUI v1 — feature & test ledger

This file is the source of truth for every user-facing affordance the unified
GUI ships in v1. Each row maps to either an automated test or a manual smoke
step. The pre-commit hook + CI gate verify ``Test ref`` existence for any row
that reaches ``✅ shipping`` status, and the merge gate rejects any rows still
in ``🚧 in progress``.

## Status legend

| Status         | Meaning                                                  |
| -------------- | -------------------------------------------------------- |
| 🔭 planned     | Specced; not yet implemented.                            |
| 🚧 in progress | Partially built; do not ship — CI rejects at merge time. |
| ✅ shipping    | Implemented + tested + documented.                       |

``Test layer`` values: ``unit``, ``integration``, ``e2e``, ``manual``.
``Test ref`` values: ``path/to/test::test_name`` for automated, or
``n/a (manual)`` for manual-only smoke steps. The validator skips ref
resolution for non-``✅`` rows.

See ``GUI_SPEC_V1.md`` for the canonical design.

## Top bar (shell chrome)

| Feature        | Element                                  | Expected behaviour                                                  | Status     | Test layer  | Test ref                                                                  |
| -------------- | ---------------------------------------- | ------------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------------- |
| Tab navigation | Top-bar tabs (Home/Builder/Viewer/Run)   | Click navigates to mount; active tab is highlighted                 | ✅ shipping | e2e         | tests/e2e/gui/test_topbar_nav.py::test_tab_navigation_active_class_tracks |
| Sandbox label  | Top-bar root display                     | Shows resolved root path; truncates long paths                      | ✅ shipping | e2e         | tests/e2e/gui/test_topbar_nav.py::test_sandbox_label_renders_root         |
| RSS readout    | Top-bar memory readout                   | Updates on `dcc.Interval` tick using `psutil`                       | ✅ shipping | integration | tests/integration/gui/test_lifecycle.py::test_rss_readout_returns_string  |
| Help modal     | "?" button                               | Opens modal with cheatsheet + cloud-deploy non-goal note            | ✅ shipping | e2e         | tests/e2e/gui/test_topbar_nav.py::test_help_modal_opens_and_contains_copy |

## Sidebar (file browser)

| Feature           | Element                            | Expected behaviour                                                          | Status     | Test layer  | Test ref                                                                   |
| ----------------- | ---------------------------------- | --------------------------------------------------------------------------- | ---------- | ----------- | -------------------------------------------------------------------------- |
| Lazy tree expand  | Folder twisty                      | One-level expansion per click; second click collapses; icon swaps 📁↔📂   | ✅ shipping | e2e         | tests/e2e/gui/test_lazy_expand_handoff.py::test_lazy_expand_collapse_state_machine |
| Capability badges | Per-row badge (img/cfg/out)        | Reflects `_classifier.classify(path)` output                                | ✅ shipping | e2e         | tests/e2e/gui/test_sidebar.py::test_image_dir_carries_image_count_badge    |
| Hidden toggle     | "Show hidden" checkbox             | Toggles dotfile visibility; persists via store                              | ✅ shipping | e2e         | tests/e2e/gui/test_sidebar.py::test_sidebar_toggle_changes_state           |
| Symlink toggle    | "Show external symlinks" checkbox  | Off by default; reveals out-of-root symlinks when on                        | ✅ shipping | e2e         | tests/e2e/gui/test_sidebar.py::test_sidebar_toggle_changes_state           |
| Refresh button    | Refresh icon                       | Re-runs sidebar tree query, busts classifier cache                          | ✅ shipping | integration | tests/integration/gui/test_lifecycle.py::test_refresh_callback_flushes_cache |
| Hand-off store    | "↩ from sidebar" button            | Click an entry; run console banner offers `Set as pipeline / input dir / output dir` (contextual)  | ✅ shipping | e2e         | tests/e2e/gui/test_lazy_expand_handoff.py::test_handoff_use_input_writes_to_input_store |

## Home page

| Feature                    | Element              | Expected behaviour                                                | Status     | Test layer  | Test ref                                                            |
| -------------------------- | -------------------- | ----------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------- |
| Welcome card               | Landing pane         | Shows project name + sandbox root + tutorial links                | ✅ shipping | e2e         | tests/e2e/gui/test_topbar_nav.py::test_home_loads_with_chrome       |
| Sandbox capability summary | Capability counts    | Shows ``n images / n outputs / n pipelines`` discovered           | ✅ shipping | integration | tests/integration/gui/test_app.py::test_home_capability_summary_renders |

## Builder integration

| Feature              | Element                                | Expected behaviour                                                     | Status     | Test layer  | Test ref                                                          |
| -------------------- | -------------------------------------- | ---------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------- |
| URL prefix support   | `create_app(url_prefix=...)` kwarg     | Dash constructed with `requests_pathname_prefix=url_prefix`            | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_builder_logo_uses_prefix |
| Mounted under /builder/ | Shell mount                         | Reachable at `/builder/`; assets resolve under prefix                  | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_builder_mount_routes |
| Standalone parity    | `python -m phenotypic.gui.builder`     | Continues to work with default `url_prefix="/"`                        | ✅ shipping | manual      | n/a (manual)                                                      |

## Builder pipeline editor

| Feature                    | Element                    | Expected behaviour                                                                      | Status     | Test layer  | Test ref                                                                                          |
| -------------------------- | -------------------------- | --------------------------------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------------------------------------- |
| Breadcrumb update payload  | Breadcrumb nav             | Callback updates replace the existing nav's children without nesting another breadcrumb | ✅ shipping | integration | tests/gui/builder/test_callbacks.py::test_render_views_returns_breadcrumb_children_not_nested_nav |
| Optional numeric params    | Parameter form number input | PEP 604 optional numeric operation parameters render as number inputs and parse as numbers | ✅ shipping | integration | tests/gui/builder/test_point_picker_param_form.py::test_pep604_optional_int_param_uses_numeric_widget_and_parser |

## Builder point picker

| Feature              | Element                                  | Expected behaviour                                                                                       | Status     | Test layer  | Test ref                                                                                          |
| -------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------------------------------------- |
| Mixin marker         | `PointPickerMixin`                       | `OperationInfo.is_point_pickable` true for ManualPointDetector + ManualSelector, false elsewhere         | ✅ shipping | unit        | tests/unit/gui/test_operation_registry.py::TestPointPickerMarker::test_point_picker_marker_propagates |
| Pickable badge       | Palette buttons                          | "PICK" badge + left-border accent on `ManualPointDetector` and `ManualSelector`                          | ✅ shipping | integration | tests/gui/builder/test_point_picker_palette.py::test_badge_renders_for_pickable_ops               |
| Picker widget        | Param form for `centers`                 | Renders Pick button + count label + hidden store; default list/tuple/str input for `centers` suppressed  | ✅ shipping | integration | tests/gui/builder/test_point_picker_param_form.py::test_picker_widget_replaces_input_for_manual_point_detector |
| Tile blueprint       | `/builder/tiles/<sid>/<src>.dzi`         | 200 + valid DZI XML for `rgb` / `intermediate`; rejects bad source / unsafe sid / missing PNG            | ✅ shipping | integration | tests/gui/builder/test_tile_blueprint.py::test_dzi_manifest_served                                |
| Modal open + pick    | Picker button → modal → set staged store | Modal opens; OSD canvas mounts; pushing three points yields a count of 3                                 | ✅ shipping | e2e         | tests/e2e/gui/test_point_picker_modal.py::test_pick_three_points_updates_count                    |
| Channel toggle       | RGB / Input radio                        | Toggling between channels does not lose staged points (image-coords reanchor)                            | ✅ shipping | e2e         | tests/e2e/gui/test_point_picker_modal.py::test_channel_toggle_preserves_points                    |
| Confirm round trip   | Confirm button                           | Modal closes; param-form count label outside the modal reflects the picks                                | ✅ shipping | e2e         | tests/e2e/gui/test_point_picker_modal.py::test_confirm_writes_centers_to_node                     |
| Picker JS surface    | `assets/point_picker.js`                 | Exposes `mountViewer` / `redrawOverlay` / `disposeViewer` under `window.__phenotypicBuilderPointPicker`  | ✅ shipping | integration | tests/gui/builder/test_point_picker_js_loads.py::test_point_picker_js_exists                      |

## Results Viewer integration

| Feature                  | Element                              | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                |
| ------------------------ | ------------------------------------ | ------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------- |
| URL prefix support       | `create_app(url_prefix=...)` kwarg   | Dash constructed with `requests_pathname_prefix=url_prefix`                     | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_results_assets_are_prefix_aware |
| Optional output_root     | `create_app(output_root=None)`       | Skips tile/colony/measurements load; renders empty state                        | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_results_layout_is_empty_state |
| Mounted via _ViewerProxy | `/results/` route                    | Per-request resolves `session.get().server`; survives release+rebuild           | ✅ shipping | integration | tests/integration/gui/test_viewer_session.py::test_release_rebuilds_on_next_request |
| Release button           | "Release loaded data"                | Drops in-memory state; subsequent access re-loads from disk (RSS may stay high) | ✅ shipping | integration | tests/integration/gui/test_viewer_session.py::test_release_rebuilds_on_next_request |
| Idle auto-release        | Daemon thread                        | Calls `release()` after `idle_seconds > N`                                      | ✅ shipping | integration | tests/integration/gui/test_viewer_session.py::test_idle_release_thread_releases_built_session |
| Sidebar hand-off         | "↩ Open in viewer" empty-state button | Selecting a `is_cli_output` folder enables the banner button; clicking POSTs to `/sandbox/api/viewer/output-root`, which validates via `OutputRoot.discover`, swaps `viewer_state["output_root"]`, releases the session; next page-load mounts the loaded viewer | ✅ shipping | integration | tests/integration/gui/test_viewer_handoff.py::test_post_swaps_output_root_and_rebuilds |
| Curation on-disk format  | `<root>/measurements.{csv,parquet}`  | The viewer's "remove colony" toggle reads/writes `measurements.parquet` (CSV mirror kept in lockstep). The CLI seeds both as a fresh full copy of `master_measurements.*` on every run (forward / `--measure` / `--recompile`); curation is intentionally wiped on re-run. Stale `filtered_measurements.*` files from earlier versions are silently ignored. | ✅ shipping | unit | tests/gui/results_viewer/test_filtered_state.py::test_remove_then_load_round_trip |
| Stale-seed write guard   | `FilteredMeasurements._save_locked` mtime check | If a CLI re-run rewrites `measurements.parquet` while the viewer holds a stale `_master_df`, the next remove/restore is a no-op (logged at WARNING). User must release the viewer (sidebar button or process restart) to reload the fresh master before further curating. | ✅ shipping | unit | tests/gui/results_viewer/test_filtered_state.py::test_save_refuses_when_seed_was_externally_rewritten |

## Run console

| Feature              | Element                | Expected behaviour                                                                   | Status     | Test layer  | Test ref                                                                       |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------ | ---------- | ----------- | ------------------------------------------------------------------------------ |
| Pipeline picker      | Modal browser          | Opens sandboxed dir picker; selects pipeline.json                                    | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_picker_modal_opens                     |
| Input picker         | Modal browser          | Opens sandboxed dir picker; selects image dir                                        | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_picker_modal_opens                     |
| Output picker        | Modal browser          | Defaults to `output_<timestamp>` adjacent to input                                   | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_picker_modal_opens                     |
| Mode toggle          | Local / SLURM radio    | Switches advanced sections + log/iframe behaviour                                    | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_mode_toggle_switches_state             |
| Dry-run checkbox     | Inline                 | `--dry-run` flag added to subprocess args                                            | ✅ shipping | unit        | tests/unit/gui/run_console/test_state.py::test_to_argv_includes_dry_run_flag   |
| Resume checkbox      | Inline                 | `--resume` flag added to subprocess args                                             | ✅ shipping | unit        | tests/unit/gui/run_console/test_state.py::test_to_argv_includes_resume_flag    |
| Run (Local)          | Run button             | Spawns Popen, polls dashboard.html, sets iframe src                                  | ✅ shipping | unit        | tests/unit/gui/run_console/test_runner.py::test_start_spawns_subprocess_and_streams_stdout |
| Run (SLURM)          | Run button             | Submits via `_cli_slurm_submission`; reads job-id from `progress/job_metadata.json`  | ✅ shipping | unit        | tests/unit/gui/run_console/test_slurm.py::test_submit_slurm_returns_array_job_id |
| Cancel               | Cancel button          | LocalRunner SIGTERMs; SIGKILL after 10s                                              | ✅ shipping | unit        | tests/unit/gui/run_console/test_runner.py::test_stop_sigterm_then_sigkill      |
| Validate (dry-run)   | Validate button        | Runs with `--dry-run`; logs only; no iframe                                          | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_validate_button_is_present_and_enabled |
| Save preset          | Save preset button     | Writes form to `<root>/.phenotypic-gui/presets/<name>.json`                          | ✅ shipping | e2e         | tests/e2e/gui/test_save_preset.py::test_save_preset_writes_file                |
| Log tail             | Log panel              | Streams Popen stdout via `dcc.Interval`; deque ring-buffered                         | ✅ shipping | unit        | tests/unit/gui/run_console/test_runner.py::test_ring_buffer_drops_oldest_under_flood |
| Recent Runs list     | Side panel             | Rehydrated from sandbox scan; row click re-points iframe                             | ✅ shipping | e2e         | tests/e2e/gui/test_run_console.py::test_recent_runs_row_click_sets_iframe_src |
| Max-local-runs cap   | Run button disabled    | `--max-local-runs` (default 1) gates new local runs                                  | ✅ shipping | integration | tests/integration/gui/test_run_console_callbacks.py::test_local_run_active_excludes_validate_records |

## CLI dashboard iframe integration

| Feature                          | Element                          | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                  |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------------- |
| `dashboard.html` iframe          | Run console main pane            | Iframes `/runs/<rel>/dashboard.html` for in-progress runs                       | ✅ shipping | e2e         | tests/e2e/gui/test_iframe_postmessage.py::test_iframe_loads_runs_blueprint_url |
| postMessage (dashboard → parent) | JS in `_cli/_dashboard/_generator.py` | Guarded by `window.parent !== window`; standalone path silent              | ✅ shipping | integration | tests/integration/gui/test_postmessage_listener.py::test_postshell_event_guarded_by_parent_check |

## Cross-cutting infrastructure

| Feature                       | Element                                    | Expected behaviour                                                                 | Status     | Test layer  | Test ref                                                                      |
| ----------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------------- |
| SandboxRoot containment       | `resolve()`                                | Rejects out-of-root and out-of-root symlinks with `ValueError`                     | ✅ shipping | unit        | tests/unit/gui/shell/test_sandbox.py::test_resolve_symlink_escape_rejected    |
| Capability classifier         | `classify(path)`                           | Stat-only; LRU-cached on `(path, mtime)`                                           | ✅ shipping | unit        | tests/unit/gui/shell/test_classifier.py::test_cache_invalidates_on_mtime_change |
| ToolSession lifecycle         | `get`/`touch`/`idle_seconds`/`release`     | Lazy build; idempotent release; thread-safe                                        | ✅ shipping | unit        | tests/unit/gui/shell/test_session.py::test_concurrent_get_runs_build_once     |
| RunRegistry concurrency       | Locked register/get/list/update_status     | Concurrent updates serialize via `threading.Lock`                                  | ✅ shipping | unit        | tests/unit/gui/shell/test_runs_registry.py::test_concurrent_update_status_is_serialised |
| LocalRunner ring buffer       | `collections.deque(maxlen=5000)`           | No subprocess pipe deadlock under flood                                            | ✅ shipping | unit        | tests/unit/gui/run_console/test_runner.py::test_ring_buffer_drops_oldest_under_flood |
| `/sandbox/api/*` blueprint    | Flask blueprint on `shell_app.server`      | Returns JSON for root/children/classify; respects sandbox                          | ✅ shipping | integration | tests/integration/gui/test_routes.py::test_sandbox_api_shapes                 |
| `/runs/<rel>/<path:file>` BP  | Flask blueprint on `shell_app.server`      | Serves files under sandbox; rejects path traversal; touches viewer session         | ✅ shipping | integration | tests/integration/gui/test_runs_blueprint.py::test_dashboard_html_served      |
| ID-collision check            | `_assert_no_id_collisions` per app         | Intra-app duplicates raise; cross-app collisions are legitimate                    | ✅ shipping | integration | tests/integration/gui/test_no_id_collisions.py::test_shell_layout_no_duplicate_ids |
| Shared launcher defaults      | `gui/_config.py` (`DEFAULT_HOST/PORT`, `add_launcher_args`, `print_launcher_banner`) | Single source of truth for `--host` / `--port` / `--debug` and the launcher banner across all four entry points. | ✅ shipping | unit | tests/unit/gui/test_config_and_design.py::TestAddLauncherArgs::test_adds_three_flags_with_defaults |
| Shared mount + config keys    | `gui/_config.py` (`MOUNT_*`, `CFG_*`, `SANDBOX_*`) | Mount prefixes, Flask `app.server.config` keys (with type-distinct `CFG_OPERATION_REGISTRY` / `CFG_RUN_REGISTRY`), and sandbox/cache dirnames are imported from one module so writers and readers stay in sync. | ✅ shipping | unit | tests/unit/gui/test_config_and_design.py::TestConfigConstants::test_registry_keys_are_distinct |
| Design-token injection        | `gui/_design.py::inject_design_tokens`     | Splices the full `:root` design token block (font + colour + type + spacing + radius + shadow + motion) into every Dash app's `index_string`; idempotent via marker comment. | ✅ shipping | unit | tests/unit/gui/test_config_and_design.py::TestInjectDesignTokens::test_all_token_groups_appear_in_index_string |

## Analysis sub-app (`/analysis/`)

| Feature                       | Element                                  | Expected behaviour                                                                                            | Status     | Test layer  | Test ref                                                                                  |
| ----------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------------------------- |
| Analysis tab in top bar       | `#shell-tab-analysis`                    | New tab on the shell top bar; clicking navigates to `/analysis/`; active class flips on navigation.            | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_tab_in_top_bar                          |
| Analysis empty-state mount    | `#analysis-page` with `.analysis-empty`  | Hub mounts `/analysis/` before any output is bound; renders an empty-state placeholder pointing at the sidebar.| ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_mount_renders_empty_state               |
| Pipeline header summary       | `#analysis-pipeline-header`              | Reads `<output>/pipeline.json` via `RecipeState.load`; shows op/meas/post/filter/model summary chip.            | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header      |
| Recompile banner              | `#analysis-recompile-banner`             | Persistent reminder above the post stack: post edits require a CLI re-run to land in `master_measurements.parquet`. | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header      |
| Post section stack            | `#analysis-post-stack` + `_POST_CHOICES`  | Authors `pipeline.post`; "Add post…" dropdown adds a section card; "×" removes; preview UX defers to v2.       | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header      |
| Filter section stack          | `#analysis-filter-stack` + `_FILTER_CHOICES` | Authors `pipeline.filters`; same add/remove pattern; plot preview deferred to v2.                          | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header      |
| Model dropdown                | `#analysis-model-dropdown` + `_MODEL_CHOICES` | Authors `pipeline.model` (single endpoint); "(no model)" clears; selection enables `#analysis-run-button`. | ✅ shipping | e2e         | tests/e2e/gui/test_analysis_app.py::test_analysis_standalone_renders_pipeline_header      |
| Run analysis button           | `#analysis-run-button`                   | Inline `pipeline.analyze(measurements.parquet)` on click; writes `analysis.{csv,parquet}` next to the master.   | ✅ shipping | unit        | tests/unit/cli/test_cli_analysis.py::TestEmitAnalysisOutputs::test_writes_csv_and_parquet_when_model_configured |
| Recipe persistence            | `RecipeState.save` (`<output>/pipeline.json`) | Atomic write on every section mutation; mtime-staleness detection refuses to clobber a fresh CLI seed.        | ✅ shipping | unit        | tests/unit/cli/test_cli_analysis.py::TestPersistPipelineJson::test_round_trip_via_load_pipeline_from_output_dir |
| Plot autodetection            | `_render.render_plot`                    | Tries `.dash()` (plotly) first; on `NotImplementedError` falls back to `.show()` rendered to PNG `<img>`.       | ✅ shipping | unit        | tests/unit/core/test_pipeline_analyze.py::TestAnalyzeContract::test_analyze_runs_filter_then_model |
| Post preview table            | `_post_preview.render_post_preview`       | Renders col-name + top-5 before/after for each affected metadata column on a post op.                          | ✅ shipping | unit        | tests/unit/core/test_pipeline_analyze.py::TestSetters::test_set_filters_list_dedupes      |
| `pipeline.analyze` library API| `ImagePipeline.analyze` / `_analyze_steps` | Runs filters then model on aggregate frame; raises ValueError when no model configured.                       | ✅ shipping | unit        | tests/unit/core/test_pipeline_analyze.py::TestAnalyzeContract::test_analyze_runs_filter_then_model |
| `filters` / `model` round-trip| `SerializablePipeline.to_json/from_json`  | Persists analysis chain in `pipeline.json`; legacy JSONs without these keys load with empty defaults.          | ✅ shipping | unit        | tests/unit/core/test_pipeline_analyze.py::TestJSONRoundTrip::test_filters_and_model_round_trip |
| CLI `pipeline.json` persist   | `_persist_pipeline_to_output_dir`        | Aggregate finalize writes `<output>/pipeline.json`; `_load_pipeline_from_output_dir` prefers it over legacy.    | ✅ shipping | unit        | tests/unit/cli/test_cli_analysis.py::TestLoadPipelinePrefersCanonical::test_canonical_pipeline_json_wins |
| CLI auto-emit `analysis.*`    | `_emit_analysis_outputs`                  | Aggregate + recompile finalize call `pipeline.analyze` and write `analysis.{csv,parquet}` when model is set.   | ✅ shipping | unit        | tests/unit/cli/test_cli_analysis.py::TestEmitAnalysisOutputs::test_writes_csv_and_parquet_when_model_configured |

## Entry points

| Feature                                | Element                | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                |
| -------------------------------------- | ---------------------- | ------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------- |
| `python -m phenotypic.gui.analysis`    | Module entry           | Standalone launcher: `--root <output-dir>` boots analysis sub-app at `/`.       | ✅ shipping | manual      | n/a (manual)                                                            |
| `python -m phenotypic.gui`             | Module entry           | Argparse `--root`/`--port`/...; calls `launch_gui`                              | ✅ shipping | integration | tests/integration/gui/test_console_script.py::test_phenotypic_gui_help_succeeds |
| `phenotypic-gui` console script        | `[project.scripts]`    | Same launcher; `--help` + `--root` work                                         | ✅ shipping | integration | tests/integration/gui/test_console_script.py::test_phenotypic_gui_help_succeeds |
| Standalone `python -m ....run_console` | Standalone parity      | Boots Run console only at `url_prefix="/"`                                      | ✅ shipping | manual      | n/a (manual)                                                            |
| Existing CLI parity                    | `phenotypic pipeline.json input/ --dry-run` | Untouched; no click refactor                                       | ✅ shipping | integration | tests/integration/gui/test_console_script.py::test_phenotypic_cli_still_works |

## Documentation

| Feature                            | Element                            | Expected behaviour                                                       | Status     | Test layer | Test ref     |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------------------ | ---------- | ---------- | ------------ |
| User guide                         | `docs/source/how_to/pages/gui_hub.md` | Walks through hub home, sandbox, run modes, iframe, release, SSH-tunnel  | ✅ shipping | manual     | n/a (manual) |
| README "Launch the GUI"            | README.md section                  | One-screen quick-start                                                   | ✅ shipping | manual     | n/a (manual) |
| `phenotypic gui` non-support note  | gui_hub.md                         | Explicitly states the no-hyphen subcommand is unsupported                | ✅ shipping | manual     | n/a (manual) |
| CLAUDE.md update                   | Quick Start section                | Mentions `python -m phenotypic.gui` and `phenotypic-gui`                 | ✅ shipping | manual     | n/a (manual) |
| Point-picker walkthrough           | docs/source/how_to/pages/gui_walkthrough/07_pick_points.md | Manual-curation tutorial with screenshots; linked from gui_walkthrough/index.md  | ✅ shipping | manual     | n/a (manual) |
