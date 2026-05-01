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
| Tab navigation | Top-bar tabs (Home/Builder/Viewer/Run)   | Click navigates to mount; active tab is highlighted                 | 🔭 planned | e2e         | tests/e2e/gui/test_topbar.py::test_tab_active_highlight                   |
| Sandbox label  | Top-bar root display                     | Shows resolved root path; truncates long paths                      | 🔭 planned | e2e         | tests/e2e/gui/test_topbar.py::test_root_label                             |
| RSS readout    | Top-bar memory readout                   | Updates on `dcc.Interval` tick using `psutil`                       | ✅ shipping | integration | tests/integration/gui/test_lifecycle.py::test_rss_readout_returns_string  |
| Help modal     | "?" button                               | Opens modal with cheatsheet + cloud-deploy non-goal note            | 🔭 planned | e2e         | tests/e2e/gui/test_topbar.py::test_help_modal_opens                       |

## Sidebar (file browser)

| Feature           | Element                            | Expected behaviour                                                          | Status     | Test layer  | Test ref                                                                   |
| ----------------- | ---------------------------------- | --------------------------------------------------------------------------- | ---------- | ----------- | -------------------------------------------------------------------------- |
| Lazy tree expand  | Folder twisty                      | One-level expansion per click                                               | 🔭 planned | e2e         | tests/e2e/gui/test_sidebar_classifier_badges.py::test_lazy_expand          |
| Capability badges | Per-row badge (img/cfg/out)        | Reflects `_classifier.classify(path)` output                                | 🔭 planned | e2e         | tests/e2e/gui/test_sidebar_classifier_badges.py::test_badges_render        |
| Hidden toggle     | "Show hidden" checkbox             | Toggles dotfile visibility; persists via store                              | 🔭 planned | integration | tests/integration/gui/test_app.py::test_hidden_toggle                      |
| Symlink toggle    | "Show external symlinks" checkbox  | Off by default; reveals out-of-root symlinks when on                        | 🔭 planned | integration | tests/integration/gui/test_app.py::test_symlink_toggle                     |
| Refresh button    | Refresh icon                       | Re-runs sidebar tree query, busts classifier cache                          | ✅ shipping | integration | tests/integration/gui/test_lifecycle.py::test_refresh_callback_flushes_cache |
| Hand-off store    | "↩ from sidebar" button            | Stamps store; active tab reads it when picking from sidebar                 | 🔭 planned | e2e         | tests/e2e/gui/test_hub_navigation.py::test_sidebar_handoff                 |

## Home page

| Feature                    | Element              | Expected behaviour                                                | Status     | Test layer  | Test ref                                                            |
| -------------------------- | -------------------- | ----------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------- |
| Welcome card               | Landing pane         | Shows project name + sandbox root + tutorial links                | 🔭 planned | e2e         | tests/e2e/gui/test_topbar.py::test_home_renders                     |
| Sandbox capability summary | Capability counts    | Shows ``n images / n outputs / n pipelines`` discovered           | ✅ shipping | integration | tests/integration/gui/test_app.py::test_home_capability_summary_renders |

## Builder integration

| Feature              | Element                                | Expected behaviour                                                     | Status     | Test layer  | Test ref                                                          |
| -------------------- | -------------------------------------- | ---------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------- |
| URL prefix support   | `create_app(url_prefix=...)` kwarg     | Dash constructed with `requests_pathname_prefix=url_prefix`            | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_builder_logo_uses_prefix |
| Mounted under /builder/ | Shell mount                         | Reachable at `/builder/`; assets resolve under prefix                  | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_builder_mount_routes |
| Standalone parity    | `python -m phenotypic.gui.builder`     | Continues to work with default `url_prefix="/"`                        | 🔭 planned | manual      | n/a (manual)                                                      |

## Results Viewer integration

| Feature                  | Element                              | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                |
| ------------------------ | ------------------------------------ | ------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------- |
| URL prefix support       | `create_app(url_prefix=...)` kwarg   | Dash constructed with `requests_pathname_prefix=url_prefix`                     | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_results_assets_are_prefix_aware |
| Optional output_root     | `create_app(output_root=None)`       | Skips tile/colony/measurements load; renders empty state                        | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_results_layout_is_empty_state |
| Mounted via _ViewerProxy | `/results/` route                    | Per-request resolves `session.get().server`; survives release+rebuild           | ✅ shipping | integration | tests/integration/gui/test_viewer_session.py::test_release_rebuilds_on_next_request |
| Release button           | "Release loaded data"                | Drops in-memory state; subsequent access re-loads from disk (RSS may stay high) | 🔭 planned | integration | tests/integration/gui/test_lifecycle.py::test_release_drops_state       |
| Idle auto-release        | Daemon thread                        | Calls `release()` after `idle_seconds > N`                                      | 🔭 planned | integration | tests/integration/gui/test_lifecycle.py::test_idle_release_fires        |

## Run console

| Feature              | Element                | Expected behaviour                                                                   | Status     | Test layer  | Test ref                                                                       |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------ | ---------- | ----------- | ------------------------------------------------------------------------------ |
| Pipeline picker      | Modal browser          | Opens sandboxed dir picker; selects pipeline.json                                    | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_pick_pipeline                        |
| Input picker         | Modal browser          | Opens sandboxed dir picker; selects image dir                                        | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_pick_input                           |
| Output picker        | Modal browser          | Defaults to `output_<timestamp>` adjacent to input                                   | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_pick_output                          |
| Mode toggle          | Local / SLURM radio    | Switches advanced sections + log/iframe behaviour                                    | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_mode_toggle                          |
| Dry-run checkbox     | Inline                 | `--dry-run` flag added to subprocess args                                            | 🔭 planned | unit        | tests/unit/gui/run_console/test_callbacks.py::test_dry_run_flag                |
| Resume checkbox      | Inline                 | `--resume` flag added to subprocess args                                             | 🔭 planned | unit        | tests/unit/gui/run_console/test_callbacks.py::test_resume_flag                 |
| Run (Local)          | Run button             | Spawns Popen, polls dashboard.html, sets iframe src                                  | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_run_local                            |
| Run (SLURM)          | Run button             | Submits via `_cli_slurm_submission`; reads job-id from `progress/job_metadata.json`  | 🔭 planned | unit        | tests/unit/gui/run_console/test_slurm.py::test_slurm_job_id_from_metadata      |
| Cancel               | Cancel button          | LocalRunner SIGTERMs; SIGKILL after 10s                                              | 🔭 planned | unit        | tests/unit/gui/run_console/test_runner.py::test_stop_sigterm_then_sigkill      |
| Validate (dry-run)   | Validate button        | Runs with `--dry-run`; logs only; no iframe                                          | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_validate_no_iframe                   |
| Save preset          | Save preset button     | Writes form to `<root>/.phenotypic-gui/presets/<name>.json`                          | 🔭 planned | unit        | tests/unit/gui/run_console/test_callbacks.py::test_save_preset                 |
| Log tail             | Log panel              | Streams Popen stdout via `dcc.Interval`; deque ring-buffered                         | 🔭 planned | unit        | tests/unit/gui/run_console/test_runner.py::test_log_tail_streams               |
| Recent Runs list     | Side panel             | Rehydrated from sandbox scan; row click re-points iframe                             | 🔭 planned | integration | tests/integration/gui/test_recent_runs_rehydrate.py::test_rehydrate            |
| Max-local-runs cap   | Run button disabled    | `--max-local-runs` (default 1) gates new local runs                                  | 🔭 planned | unit        | tests/unit/gui/run_console/test_callbacks.py::test_max_local_runs_cap          |

## CLI dashboard iframe integration

| Feature                          | Element                          | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                  |
| -------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- | ---------- | ----------- | ------------------------------------------------------------------------- |
| `dashboard.html` iframe          | Run console main pane            | Iframes `/runs/<rel>/dashboard.html` for in-progress runs                       | 🔭 planned | e2e         | tests/e2e/gui/test_run_local_e2e.py::test_dashboard_iframe                |
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
| ID-collision check            | `_assert_no_id_collisions` per app         | Intra-app duplicates raise; cross-app collisions are legitimate                    | 🔭 planned | integration | tests/integration/gui/test_no_id_collisions.py::test_intra_app_duplicates     |

## Entry points

| Feature                                | Element                | Expected behaviour                                                              | Status     | Test layer  | Test ref                                                                |
| -------------------------------------- | ---------------------- | ------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------- |
| `python -m phenotypic.gui`             | Module entry           | Argparse `--root`/`--port`/...; calls `launch_gui`                              | 🔭 planned | integration | tests/integration/gui/test_console_script.py::test_module_entry         |
| `phenotypic-gui` console script        | `[project.scripts]`    | Same launcher; `--help` + `--root` work                                         | 🔭 planned | integration | tests/integration/gui/test_console_script.py::test_console_script       |
| Standalone `python -m ....run_console` | Standalone parity      | Boots Run console only at `url_prefix="/"`                                      | 🔭 planned | manual      | n/a (manual)                                                            |
| Existing CLI parity                    | `phenotypic pipeline.json input/ --dry-run` | Untouched; no click refactor                                       | 🔭 planned | unit        | tests/unit/cli/test_cli_invocation.py::test_existing_invocation_unchanged |

## Documentation

| Feature                            | Element                            | Expected behaviour                                                       | Status     | Test layer | Test ref     |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------------------ | ---------- | ---------- | ------------ |
| User guide                         | `docs/source/user_guide/gui.rst`   | Walks through hub home, sandbox, run modes, iframe, release, SSH-tunnel  | 🔭 planned | manual     | n/a (manual) |
| README "Launch the GUI"            | README.md section                  | One-screen quick-start                                                   | 🔭 planned | manual     | n/a (manual) |
| `phenotypic gui` non-support note  | gui.rst                            | Explicitly states the no-hyphen subcommand is unsupported                | 🔭 planned | manual     | n/a (manual) |
| CLAUDE.md update                   | Quick Start section                | Mentions `python -m phenotypic.gui` and `phenotypic-gui`                 | 🔭 planned | manual     | n/a (manual) |
