# Build a Pipeline

The Builder is a node-graph editor for `ImagePipeline`. You compose a pipeline
by dragging operations from the palette onto the canvas, wiring them up, and
saving the result as JSON. The walkthrough's `pipeline.json` was hand-written;
this page shows the workflow you'd use to make your own.

## Open the builder

Click the `Builder` tab in the top bar (or navigate to `/builder/`). The
chrome stays in place — same sidebar, same RSS readout — and the main pane
swaps to the builder UI:

![Empty builder canvas.](../../../_static/gui_images/build_pipeline/01_builder_empty.png)

The builder has four regions:

| Region | Contents |
|--------|----------|
| **Palette** (left) | Operations grouped by role: `Corrector`, `Detector`, `Enhancer`, `Refiner`, `Measure`, and `Post`. Each group's count is visible in the section header. |
| **Canvas** (center) | The pipeline graph. Empty by default; drag operations into it. |
| **Inspector** (lower) | Parameters of the currently-selected node. Empty when nothing is selected. |
| **Pipeline I/O** (top right) | `Save` writes the current graph to a pipeline JSON; `Load` reads one. |
| **Structure** (top right, below Pipeline I/O) | `+ Pipeline` starts a new pipeline graph; `Delete selected` removes the currently-selected node. |

## Compose a small pipeline

The walkthrough's pipeline (`gui_tutorial`) consists of:

1. `GaussianBlur` — smooths each plate before thresholding.
2. `OtsuDetector` — produces the binary colony mask.
3. `MeasureShape` and `MeasureSize` — extract per-colony shape and area
   measurements.

To reproduce it in the builder:

1. **Drag `GaussianBlur`** from the `Corrector` group onto the canvas. It
   appears as a node. Click it; the inspector shows its `sigma` parameter.
2. **Drag `OtsuDetector`** from the `Detector` group. Drop it next to the
   blur node and connect the blur node's output socket to the detector's
   input.
3. **Add `MeasureShape` and `MeasureSize`** from the `Measure` palette.
   Measurements consume the detector's labelled output map (objmap).
4. **Click `Save`** in the Pipeline I/O card. A modal browser opens; pick
   the directory you want to save to and confirm. The resulting JSON is
   what the run console will execute.

## Save format

The on-disk JSON is the same format the walkthrough used in
[Setup](01_setup.md):

```json
{
  "version": "0.1.0",
  "name": "gui_tutorial",
  "pipe_cfgs": {
    "GaussianBlur": {"class": "GaussianBlur", "params": {"sigma": 2}},
    "OtsuDetector": {"class": "OtsuDetector", "params": {"ignore_zeros": true}}
  },
  "meas": {
    "MeasureShape": {"class": "MeasureShape", "params": {}},
    "MeasureSize": {"class": "MeasureSize", "params": {}}
  },
  "post": {},
  "nrows": 8,
  "ncols": 12
}
```

`pipe_cfgs` are the operations that prepare the image (correctors,
enhancers, detectors, refiners). `meas` are the measurements that read the
detected objects. `post` is reserved for post-measurement transforms (e.g.
edge correction, growth-curve fitting).

The classifier picks up any JSON containing `"pipe_cfgs"` (or the legacy
`"operations"` key) in its first 4 KB and stamps it with the `cfg` badge in
the sidebar — that's why your saved pipeline appears as a `cfg`-badged
file the next time the sidebar refreshes.

Next: [Run Locally](04_run_local.md).
