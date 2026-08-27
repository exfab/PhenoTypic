# PhenoTypic GUI v1 — workflow tutorial registry

This file is the source of truth for which **end-to-end user flows** the
GUI hub ships a tutorial for. For tracking individual user-visible
affordances (a button, a badge, a callback, a store), see
[`FEATURES.md`](FEATURES.md) — different ledger, different gate.

Each row below maps to:

* A folder under `docs/source/_static/gui_images/<id>/` containing the
  workflow's screenshots.
* One or more `_capture_<id>` capture functions in
  `scripts/capture_gui_tutorial_screenshots.py`, dispatched from
  `capture_workflow_screenshots` or
  `capture_standalone_viewer_screenshots`.
* A walkthrough page at
  `docs/source/tutorials/gui/<NN>_<id>.md`.

The pre-commit hook + the `workflows-md-gate` job in the `gui-checks`
CI workflow run `scripts/check_workflows_md.py`, which rejects
mismatches: a row here without a matching defined+dispatched
`_capture_<id>` (or a `_capture_*` function with no row) fails the
build.

## Status legend

| Status         | Meaning                                                  |
| -------------- | -------------------------------------------------------- |
| 🔭 planned     | Specced; not yet implemented.                            |
| 🚧 in progress | Partially built; do not ship — CI rejects at merge time. |
| ✅ shipping    | Capture function defined + dispatched, screenshots committed, tutorial page live. |

## Workflows

| ID             | Title                | Description                                                                | Capture function          | Tutorial page                            | Status     |
| -------------- | -------------------- | -------------------------------------------------------------------------- | ------------------------- | ---------------------------------------- | ---------- |
| setup          | Setup & landing      | First-launch home page tour: shell chrome, grouped tab nav (Home + Pipeline/Results dropdowns), sandbox label. | `_capture_setup`          | `gui/01_setup.md`            | ✅ shipping |
| file_explorer  | File explorer        | Sidebar tree, capability badges, hidden + symlink toggles, and one Refresh revision that retires stale Browse state.        | `_capture_file_explorer`  | `gui/02_file_explorer.md`    | ✅ shipping |
| build_pipeline | Build a pipeline     | Fixed linear port map with view-only zoom/fit controls, click-only palette insertion, green continuation target, side loader, Save / Load, and embedded-pipeline breadcrumbs. | `_capture_build_pipeline` | `gui/03_build_pipeline.md`   | ✅ shipping |
| run_local      | Run locally          | Run console form, pipeline / input / output pickers, automatic compatible continuation and terminal-retry controls, direct progress-only dashboard, generation receipts, uncertain acknowledgement handling, terminal Cancel gating, and retained logs. | `_capture_run_local`      | `gui/04_run_local.md`        | ✅ shipping |
| run_slurm      | Run on SLURM         | SLURM mode toggle, automatic compatible continuation and terminal-retry controls, advanced config, Progress/Download dashboard tabs, `sbatch --export=ALL`, and durable generation-bound ordinary/staged lifecycle observation, logs, cancellation, publication, restart guidance. | `_capture_run_slurm`      | `gui/05_run_slurm.md`        | ✅ shipping |
| view_results   | View results         | Empty-state hub mount, asynchronous sidebar binding with coherent read-only Results/Analysis snapshots and consistency diagnostics, populated hub/standalone viewers, and measurement table. | `_capture_view_results`   | `gui/06_view_results.md`     | ✅ shipping |
| pick_points    | Manual point picker  | Pickable badge, picker modal, RGB / Input channel toggle, confirm round trip. | `_capture_pick_points`    | `gui/07_pick_points.md`      | ✅ shipping |
| analysis       | Analysis sub-app     | `/analysis/` mount, pipeline header, post / filter / edge / model section authoring, opaque-analyzer preservation, and atomic guarded publication of class-named artifacts plus configured `PlotAnalysis` refresh. | `_capture_analysis` | `gui/08_analysis.md` | ✅ shipping |
| aux_ports      | Fill aux op params | Op-typed side-loader rows render gold port buttons on the left; selecting a side target and clicking a compatible operation fills/replaces the hidden aux value, with clear/drill/help actions in the value row. | `_capture_aux_ports` | `gui/09_aux_ports.md`     | ✅ shipping |
| aux-wire-in-dag | Fill a scalar aux target | Select a consumer side port, watch it turn green, choose a compatible detector from the palette, and manage the resulting hidden aux value from the side loader. This replaces the development DAG wire gesture. | `_capture_aux_wire_in_dag` | `gui/12_aux_wire_in_dag.md` | ✅ shipping |
| wire-pipeline-as-aux | Edit an embedded Pipeline aux | Fill an op-typed side parameter with `+ Pipeline`, drill into the embedded pipeline through breadcrumbs, add a nested chain, and drill back out to the parent side loader. | `_capture_wire_pipeline_as_aux` | `gui/13_wire_pipeline_as_aux.md` | ✅ shipping |
| fix-validation-issues | Read validation and unsupported states | Show whole-pipeline validation gates and required side-value affordances; document the defensive unsupported-state panel used when an arbitrary non-linear development DAG is loaded. | `_capture_fix_validation_issues` | `gui/14_fix_validation_issues.md` | ✅ shipping |
| browse | Browse source images | Open the top-level **Browse** tab, set a source root, and navigate with the dropdown, ‹/› controls, or J/K and shifted ten-image jumps. A progressive preview remains behind the reused OpenSeadragon viewer until the revision-addressed DZI opens; an opt-in Keep position preference restores the viewport only across equal decoded dimensions. The centered filmstrip mounts at most four neighbours per side and reports preparation state. Selected and directional work outrank explicit **Prepare dataset** work through one bounded worker; **Stop** cancels queued work after the current native stage, and **Clear prepared images** protects displayed and in-flight revisions. Prepared previews and tiles persist in a revision-correct 10 GiB/8 GiB cache. Bundled libvips accelerates macOS/Windows, system libvips is optional on Linux/HPC, and Pillow remains supported. | `_capture_browse` | `gui/18_browse.md` | ✅ shipping |

> **`view_results` note.** The empty-state screenshot
> (`view_results/01_viewer_empty.png`) is captured by
> `_capture_view_results` while the hub is mounted with no
> `output_root`. That same capture then binds the coherent synthetic output
> asynchronously and records `05_hub_bound_snapshot.png`. The populated
> standalone screenshots (`02_viewer_loaded.png`, `03_measurement_table.png`)
> come from `capture_standalone_viewer_screenshots`, which spins up the
> standalone viewer against the same real CLI output. All contribute to the
> same `view_results/` folder; the validator only requires the primary
> `_capture_view_results` to be defined + dispatched.

## How to add a workflow

When introducing a new end-to-end user flow:

1. **Pick a slug** (`<id>`) — lowercase, snake_case, unique. Mirrors
   the screenshot folder name and the tutorial page suffix.
2. **Add a row** to the table above. Status starts at `🔭 planned`.
3. **Add a `_capture_<id>` function** in
   `scripts/capture_gui_tutorial_screenshots.py` and register it in
   `capture_workflow_screenshots` (or the standalone helper if it
   needs a separately-booted Dash app). The function should call
   `_save(page, "<id>", "NN_step.png")` for each screenshot.
4. **Run the capture script** —
   `uv run python scripts/capture_gui_tutorial_screenshots.py` — to
   write PNGs into `docs/source/_static/gui_images/<id>/`.
5. **Add a tutorial page** at
   `docs/source/tutorials/gui/<NN>_<id>.md` and link it from
   `tutorials/gui/index.md`. Use the existing pages as a
   structural template (Google-style intros, MyST `:::{figure}`
   blocks for screenshots).
6. **Add corresponding rows to [`FEATURES.md`](FEATURES.md)** for any
   new user-visible affordance the workflow exercises.
7. **Flip the row to `✅ shipping`** once steps 3–6 are merged. The
   merge gate rejects rows still in `🚧 in progress`.

## When NOT to add a workflow

Use `FEATURES.md` (not this file) when the change is:

* a new button or toggle inside an existing workflow,
* a new capability badge,
* a new Flask blueprint route,
* a new callback or store,
* an empty-state pathway tweak,
* a styling change.

Workflows are reserved for flows broad enough to warrant a dedicated
tutorial page with its own screenshots — typically 3+ user-driven
interactions chained together.
