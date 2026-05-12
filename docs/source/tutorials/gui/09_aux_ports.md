# Wiring aux op params

Some operations accept another operation as a constructor parameter —
e.g. `FilamentousFungiDetector.inoculum_detector` takes an
`ObjectDetector` or `ImagePipeline`. The builder surfaces these slots
as small **aux ports** on the consumer node. Clicking an aux port
opens a **popover** to pick a class, edit a wired aux's params, or
drill into a multi-step aux pipeline.

This tutorial walks a 4-step main pipeline that wires a single-op aux
into a detector.

## Step 1 — Empty canvas

Open the `Builder` tab (or navigate to `/builder/`):

![Empty builder canvas — palette on the left, nothing on the ribbon.](../../_static/gui_images/aux_ports/01_initial.png)

The palette groups ops by `Corrector` / `Detector` / `Enhancer` /
`Refiner`. No aux UI yet — it only appears once a consumer with
op-typed params lands on the canvas.

## Step 2 — Add the main pipeline

Drop the following four ops in order:

1. `GaussianBlur`
2. `ContrastStretching`
3. `FilamentousFungiDetector`
4. `MeasureColonySize`

![Four-node main pipeline ribbon with image-flow edges between blue I/O ports; FilamentousFungiDetector shows a purple aux port on its bottom edge.](../../_static/gui_images/aux_ports/02_main_pipeline.png)

Four ribbon nodes wired left-to-right by solid image-flow edges
between their **blue I/O ports** (input left, output right).
`FilamentousFungiDetector` carries one **purple aux port** on its
bottom edge — the `inoculum_detector` op-typed param, rendered
**hollow** since nothing is wired yet.

## Step 3 — Open the aux popover

Click the aux port on `FilamentousFungiDetector`:

![Popover anchored to the aux port showing a palette of compatible ObjectDetector and ImagePipeline classes.](../../_static/gui_images/aux_ports/03_popover_empty.png)

A popover appears anchored to the port. Since the slot is empty, it
shows a **class palette** filtered to types that satisfy the port —
every `ObjectDetector` subclass plus `ImagePipeline`. The inspector
stays on `FilamentousFungiDetector` until something is wired.

## Step 4 — Pick a class

Click `OtsuDetector` in the popover palette:

![Popover in the wired state showing OtsuDetector with Edit, Drill in, and Disconnect actions; inspector swaps to OtsuDetector params with a breadcrumb header.](../../_static/gui_images/aux_ports/04_popover_wired.png)

The popover flips to **wired** state — class name plus three actions:
`✎ Edit`, `Drill in →`, `⨯ Disconnect`. The inspector swaps to
`OtsuDetector`'s param form with a breadcrumb header
`← FilamentousFungiDetector.inoculum_detector / slot 0` so the
edited slot is unambiguous. The aux port on the consumer flips to
**filled purple**.

## Step 5 — Drill in and extend

Click `Drill in →` in the popover:

![Canvas drilled into the aux scope — one-node OtsuDetector ribbon; top-of-builder breadcrumb shows Pipeline / FilamentousFungiDetector / inoculum_detector / slot 0.](../../_static/gui_images/aux_ports/05_drill_in.png)

The popover dismisses and the canvas swaps to a one-step ribbon of
just `OtsuDetector`. Every aux is a 1-step pipeline by default, so
drilling in lets you extend it like any other pipeline. The builder
breadcrumb shows the drilled path:
`Pipeline / FilamentousFungiDetector / inoculum_detector / slot 0`.

Add more ops before or after `OtsuDetector` to extend the aux into a
multi-step pipeline — e.g. prepend a `GaussianBlur` to smooth the
inoculum spot before Otsu thresholds it:

![Two-step aux pipeline with GaussianBlur feeding OtsuDetector.](../../_static/gui_images/aux_ports/05b_drill_extended.png)

## Step 6 — Drill out and save

Click `FilamentousFungiDetector` in the breadcrumb to drill back out:

![Main 4-step ribbon restored; the aux port on FilamentousFungiDetector is filled with a "1" badge indicating one wired slot.](../../_static/gui_images/aux_ports/06_drill_out.png)

The canvas restores to the 4-step main ribbon. The aux port stays
filled purple and now carries a small `1` badge for the one wired
slot. Saving (`Pipeline I/O → Save`) preserves the wired aux as a
standard `__op_param_scope__` marker inside the consumer's params —
the JSON round-trips byte-identically.

## Reference

- **Aux port indicator** — purple square on the consumer's bottom
  edge. Hollow = empty, filled = wired (badge shows slot count).
- **Popover dismissal** — click outside, press `Esc`, or click a
  different aux port to swap focus.
- **List-typed params** — for `list[ObjectDetector | ImagePipeline]`
  params like `CompositeDetector.detectors`, the popover stacks every
  slot vertically with per-slot `✎ / → / ⨯` actions plus a `+ Add
  slot` button. Empty slots are folded out on save.
- **Nested aux of aux** — drill-in recurses. An aux's own op-typed
  params get aux ports inside the drilled scope; the breadcrumb
  grows another segment.

## Where to next

- [Build a Pipeline](03_build_pipeline.md) — the basics of the main
  image-flow ribbon.
- [Pick Points](07_pick_points.md) — another consumer-with-config
  pattern (manual curation).
- [GUI hub guide](../../how_to/pages/gui_hub.md) — full reference for
  every panel, store, and admonition in the hub.
