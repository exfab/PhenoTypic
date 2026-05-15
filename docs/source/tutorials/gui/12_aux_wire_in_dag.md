# Wire an aux in the DAG

The DAG redesign retires the popover class palette: every operation —
including the ones used as auxiliary parameters — is a **first-class
block on the canvas**. Wiring an aux is now exactly what it sounds
like: drag a producer block onto the canvas and drag a wire from its
output port to the consumer's bottom-edge aux port.

This tutorial walks through the minimal authoring trip: a main
image-flow ribbon plus a `FilamentousFungiDetector` that needs an
`inoculum_detector` wired into it.

## Step 1 — Open the builder and add the consumer

Open the `Builder` tab (or navigate to `/builder/`). Drop the main
image-flow ribbon: `GaussianBlur → ContrastStretching →
FilamentousFungiDetector`. The detector node carries a hollow purple
**aux port** on its bottom edge — that's the `inoculum_detector`
slot waiting for a producer.

![Main ribbon with FilamentousFungiDetector showing a hollow purple aux port on its bottom edge.](../../_static/gui_images/aux-wire-in-dag/01_main_with_consumer.png)

In the post-redesign builder, every operation is a draggable block;
there is no popover class palette for picking aux types. The hollow
purple port is the affordance — drop a producer next and wire it in.

## Step 2 — Drop the aux producer block

Drag `OtsuDetector` from the `Detector` palette onto the canvas. It
lands as a free-floating block — no incoming wire yet (it has no
image input), and a right-edge **image-output port** ready to be
wired into either the main flow or an aux slot.

![Canvas with OtsuDetector dropped to the right of the main ribbon. Its right-edge output port is highlighted.](../../_static/gui_images/aux-wire-in-dag/02_detector_dropped.png)

While the producer is unwired, the toolbar's issue badge surfaces a
"stranded block" warning. That's expected — the next step clears it
by wiring the producer's output into the consumer's aux port.

## Step 3 — Draw the aux wire

Drag from `OtsuDetector`'s right-edge output port to
`FilamentousFungiDetector`'s bottom-edge aux port. The wire renders
as a **purple aux edge** to distinguish it from blue image-flow
edges. The consumer's aux port flips from hollow to filled-purple and
the validation badge clears.

![Purple aux wire between the OtsuDetector output port and the FilamentousFungiDetector aux port; the consumer's aux port is now filled purple and the toolbar badge has cleared.](../../_static/gui_images/aux-wire-in-dag/03_aux_wired.png)

Clicking `Run preview` now enables the inspector preview thumbnail —
the pipeline is valid, all consumers have their inputs (image + aux),
and the runner has everything it needs to produce a labelled `objmap`.

## Reference

- **Aux port colour code** — purple for aux, blue for image-flow.
  Hollow when empty, filled when wired.
- **Producer placement** — auxiliary detectors don't need to live
  near their consumer; the wire spans the canvas. Move them where
  the layout reads cleanest.
- **Wire deletion** — click the wire to select it, then press
  `Delete` or click `Delete selected` in the toolbar. The consumer's
  aux port returns to hollow.
- **No popover** — the v1 popover class palette is gone. The DAG
  redesign treats aux producers as ordinary blocks; their parameters
  are edited in the inspector exactly like any other block.

## Where to next

- [Wire a Pipeline as aux](13_wire_pipeline_as_aux.md) — wrap a
  multi-step chain in a Pipeline container and wire it in as a single
  aux producer.
- [Fix validation issues](14_fix_validation_issues.md) — what to do
  when the toolbar issue badge lights up.
- [Build a Pipeline](03_build_pipeline.md) — the main image-flow
  basics.
