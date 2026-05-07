# Analysis sub-app

The `Analysis` tab composes the `phenotypic.analysis` chain — filters
plus an endpoint model — over the curated measurements produced by a
CLI run. Recipes are persisted as fields on the pipeline itself
(`pipeline.json` next to `master_measurements.parquet`), so the same
chain re-runs deterministically from the CLI when you `--recompile`.

```{note}
This page is a placeholder while the screenshot capture lands. The
sub-app is implemented and shipping (see
[`FEATURES.md`](../../../../src/phenotypic/gui/FEATURES.md) for the
authoritative list of affordances and tests). Once
`_capture_analysis` is fleshed out with the loaded-state screenshots,
this page will be flipped to `✅ shipping` in
[`WORKFLOWS.md`](../../../../src/phenotypic/gui/WORKFLOWS.md).
```

## Hub mount (empty state)

Open the `Analysis` tab in the hub:

![Analysis tab in empty state.](../../../_static/gui_images/analysis/01_analysis_empty.png)

Like the Viewer, the hub-mounted Analysis sub-app starts empty. Pick a
CLI output directory in the sidebar to bind the page to a
`pipeline.json`. Until the rebuild-on-select wiring lands, the
recommended path is the standalone launcher:

```bash
uv run python -m phenotypic.gui.analysis \
    --root <path-to-cli-output> --port 8051
```

## What you can do

- **Author the post stack** (metadata transforms): pick a class from the
  "Add post…" dropdown (`PrependString`, `AppendString`,
  `ExpandMetadata`, `MergeMetadata`). The recompile banner reminds you
  that post edits change per-image measurement and require a CLI re-run
  (`python -m phenotypic --recompile <output>`) to reach
  `master_measurements.parquet`.
- **Author the filter chain**: pick a class from the "Add filter…"
  dropdown (`EdgeCorrector`, `TukeyOutlierRemover`). Filters reshape the
  aggregate measurements during analysis — they don't touch the master.
- **Pick the endpoint model**: `LogGrowthModel` or `LinearSoftplusModel`.
  Only one model can be configured at a time; selecting `(no model)`
  clears it and disables the run button.
- **Run analysis**: click `Run analysis`. The sub-app reads
  `<output>/measurements.parquet` (the curated mirror), runs the chain
  via `pipeline.analyze(...)`, and writes `<output>/analysis.csv` and
  `<output>/analysis.parquet` next to the master.

## CLI parity

Every section you author from the GUI is persisted to
`<output>/pipeline.json`. A subsequent `python -m phenotypic --recompile
<output>` run reads that file and emits the same `analysis.{csv,parquet}`
without booting the GUI — so `pipeline.json` is the single
reproducibility surface.

## Where to next

- [GUI hub guide](../gui_hub.md) — the full reference for the hub.
- [Run Locally](04_run_local.md) — produce a CLI output before opening
  the analysis sub-app.
- [View Results](06_view_results.md) — curate `measurements.parquet`
  before running analysis.
