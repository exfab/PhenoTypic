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

The pre-commit hook + `gui-docs` CI workflow run
`scripts/check_workflows_md.py`, which rejects mismatches: a row here
without a matching defined+dispatched `_capture_<id>` (or a
`_capture_*` function with no row) fails the build.

## Status legend

| Status         | Meaning                                                  |
| -------------- | -------------------------------------------------------- |
| 🔭 planned     | Specced; not yet implemented.                            |
| 🚧 in progress | Partially built; do not ship — CI rejects at merge time. |
| ✅ shipping    | Capture function defined + dispatched, screenshots committed, tutorial page live. |

## Workflows

| ID             | Title                | Description                                                                | Capture function          | Tutorial page                            | Status     |
| -------------- | -------------------- | -------------------------------------------------------------------------- | ------------------------- | ---------------------------------------- | ---------- |
| setup          | Setup & landing      | First-launch home page tour: shell chrome, tab strip, sandbox label.       | `_capture_setup`          | `gui/01_setup.md`            | ✅ shipping |
| file_explorer  | File explorer        | Sidebar tree, capability badges, hidden + symlink toggles, refresh.        | `_capture_file_explorer`  | `gui/02_file_explorer.md`    | ✅ shipping |
| build_pipeline | Build a pipeline     | Builder palette, canvas, Save / Load, `+ Pipeline`, breadcrumb, inspector. | `_capture_build_pipeline` | `gui/03_build_pipeline.md`   | ✅ shipping |
| run_local      | Run locally          | Run console form, pipeline / input / output pickers, validate, run, log.   | `_capture_run_local`      | `gui/04_run_local.md`        | ✅ shipping |
| run_slurm      | Run on SLURM         | SLURM mode toggle, advanced + SLURM config, sbatch handoff guidance.       | `_capture_run_slurm`      | `gui/05_run_slurm.md`        | ✅ shipping |
| view_results   | View results         | Empty-state hub mount + populated standalone viewer + measurement table.   | `_capture_view_results`   | `gui/06_view_results.md`     | ✅ shipping |
| pick_points    | Manual point picker  | Pickable badge, picker modal, RGB / Input channel toggle, confirm round trip. | `_capture_pick_points`    | `gui/07_pick_points.md`      | ✅ shipping |
| analysis       | Analysis sub-app     | `/analysis/` mount, pipeline header, post / filter / model section authoring, run button emitting `analysis.{csv,parquet}`. | `_capture_analysis` | `gui/08_analysis.md` | ✅ shipping |
| aux_ports      | Wiring aux op params | Aux input ports on the bottom edge of consumer nodes; click an aux port to open a canvas-anchored popover showing the class palette (empty slot) or wired-row with `Edit / Drill in / Disconnect` actions (wired slot). `Drill in →` swaps the main canvas to the aux's scope (every aux is treated as a 1-step pipeline minimum). Inspector mirrors the wired aux's params when the popover is open. Main I/O ports are now explicit blue circles on every operation node. | `_capture_aux_ports` | `gui/09_aux_ports.md`     | ✅ shipping |
| qc_curation_loop | QC curation loop | Configure Count + SE checks; watch metrics improve as you curate flagged colonies | `_capture_qc_curation_loop` | `gui/10_qc_curation_loop.md` | ✅ shipping |
| heatmap_exploration | Heatmap exploration | Pick a measurement and walk through time on a plate; spot edge/contamination patterns | `_capture_heatmap_exploration` | `gui/11_heatmap_exploration.md` | ✅ shipping |

> **`view_results` note.** The empty-state screenshot
> (`view_results/01_viewer_empty.png`) is captured by
> `_capture_view_results` while the hub is mounted with no
> `output_root`. The populated screenshots
> (`02_viewer_loaded.png`, `03_measurement_table.png`) come from
> `capture_standalone_viewer_screenshots`, which spins up the
> standalone viewer against a real CLI output. Both contribute to the
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
