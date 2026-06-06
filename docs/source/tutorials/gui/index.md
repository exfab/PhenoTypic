# GUI Walkthrough

A screenshot-driven tour through the PhenoTypic hub — from a fresh sandbox to a
finished run with viewable results. Each page is short and focused on a single
workflow so you can jump to whichever step you need.

The walkthrough uses a small synthetic yeast plate dataset (3 plates, 8 × 12
grid each, generated with `phenotypic.data.make_synthetic_plate`) so you can
follow along without your own data. The accompanying capture script
[`scripts/capture_gui_tutorial_screenshots.py`](https://github.com/exfab/PhenoTypic/blob/main/scripts/capture_gui_tutorial_screenshots.py)
regenerates the synthetic plates, runs the CLI to produce real output, and
takes every screenshot in this guide — re-run it after a GUI change to keep
the images aligned with the implementation.

## What's covered

| Page | Goal |
|------|------|
| [Setup](01_setup.md) | Generate the synthetic dataset and launch the hub. |
| [File Explorer](02_file_explorer.md) | Navigate the sandbox sidebar and read capability badges. |
| [Build a Pipeline](03_build_pipeline.md) | Compose a pipeline.json from the operations palette. |
| [Run Locally](04_run_local.md) | Submit a local subprocess run and tail the dashboard. |
| [Run on SLURM](05_run_slurm.md) | Fill in SLURM resources and ship a submission to a cluster. |
| [View Results](06_view_results.md) | Open the results viewer against a finished CLI output. |
| [Pick points](07_pick_points.md) | Curate detected colonies manually with the in-builder point picker. |
| [Analysis](08_analysis.md) | Compose `phenotypic.analysis` filters + endpoint model and emit `analysis.{csv,parquet}`. |
| [Aux ports](09_aux_ports.md) | Wire operation-typed parameters (e.g. `FilamentousFungiDetector.inoculum_detector`) via Galaxy-style aux input ports. |
| [QC curation loop](10_qc_curation_loop.md) | Configure `QualityCheck` analyzers and curate flagged colonies in a tight feedback loop. |
| [Heatmap exploration](11_heatmap_exploration.md) | Pick a measurement and walk through time on a plate to spot edge / contamination patterns. |
| [Wire an aux in the DAG](12_aux_wire_in_dag.md) | Drag a detector block onto the canvas, draw a purple aux wire to a consumer's bottom-edge aux port (post-redesign primary aux flow). |
| [Wire a Pipeline as aux](13_wire_pipeline_as_aux.md) | Wrap a multi-step chain inside a `Pipeline` container, then wire the container as a single aux producer. |
| [Fix validation issues](14_fix_validation_issues.md) | Triage the toolbar issue badge — pan to the offending block, see the validator's explanation, resolve, re-enable preview. |
| [QC review walkthrough](15_qc_review.md) | Switch the QC tab to Review; walk worst-first groups for a module, curate in the tile gallery, mark reviewed, and watch the metric recompute in place. |
| [Tune co-pilot](16_tune_copilot.md) | Open the `/tune/` read-only co-pilot over a tune output: monitor trials, curate A/B overlays, review the search space, and launch the next run. |

## Prerequisites

- PhenoTypic installed via `uv sync --group dev` (see [Getting Started](../getting_started.rst)).
- Playwright with Chromium installed if you want to regenerate the screenshots
  yourself: `uv run playwright install chromium`.

For a comprehensive reference of every panel, store, and admonition in the
hub, see the [GUI hub guide](../../how_to/pages/gui_hub.md). This walkthrough is the
"happy path"; the reference is the manual.

```{toctree}
:maxdepth: 1
:hidden:

01_setup
02_file_explorer
03_build_pipeline
04_run_local
05_run_slurm
06_view_results
07_pick_points
08_analysis
09_aux_ports
10_qc_curation_loop
11_heatmap_exploration
12_aux_wire_in_dag
13_wire_pipeline_as_aux
14_fix_validation_issues
15_qc_review
16_tune_copilot
```
