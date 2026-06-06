# Fill aux op params

Some operations accept another operation as a constructor parameter. For
example, `FilamentousFungiDetector.inoculum_detector` can take an
`ObjectDetector` or an `ImagePipeline`. The linear builder surfaces those
parameters as gold port buttons on the node card and in the side loader.

The default flow is click-only:

1. Select the consumer node.
2. Click the parameter port in the side loader.
3. Click a compatible operation in the palette.

The aux source is stored in the internal DAG state as a hidden side value. It is
not placed as a movable node on the main map.

## Step 1 - Empty canvas

Open the `Builder` tab, or navigate to `/builder/`:

![Empty builder canvas with palette, InputImage source, and floating continuation port.](../../_static/gui_images/aux_ports/01_initial.png)

Every scope starts with one `InputImage` source and a floating continuation
target. No side-value rows appear until you select a node with operation-typed
parameters.

## Step 2 - Add the main pipeline

Click `GaussianBlur`, then `FilamentousFungiDetector`. Each palette click adds
to the current green continuation target, so the main chain becomes
`InputImage -> GaussianBlur -> FilamentousFungiDetector`.

![Main linear chain with FilamentousFungiDetector selected and an empty inoculum_detector side value.](../../_static/gui_images/aux_ports/02_main_pipeline.png)

`FilamentousFungiDetector` exposes the required `inoculum_detector` side value.
While it is empty, the validation badge reports a blocking issue and global
preview/save stay disabled.

## Step 3 - Fill the side value

In the side loader, click the gold port button beside `inoculum_detector`. The
active target strip updates and the selected port turns green. Then click
`OtsuDetector` in the palette.

![Side loader showing inoculum_detector filled with OtsuDetector.](../../_static/gui_images/aux_ports/03_aux_wired.png)

The side value row now shows `OtsuDetector` with `Replace`, `Clear`, and `?`
actions. Replacing follows the same pattern: click `Replace`, then click a
compatible palette operation.

## Step 4 - Inspect the consumer

Select `FilamentousFungiDetector` again to inspect the filled consumer:

![Side loader focused on FilamentousFungiDetector with filled side value controls.](../../_static/gui_images/aux_ports/04_inspector_aux.png)

Scalar fields for the selected node remain editable above `Side values`.
Operation-valued parameters stay in the side-value section with their own
left-aligned port buttons and value-row actions.

## Reference

- **Gold parameter port** - selects a side-value target.
- **Green selected target** - the next palette click will add or fill here.
- **Filled value row** - hidden aux value with `Replace`, `Clear`, and help
  actions.
- **Embedded pipeline value** - use `+ New Pipeline` while a side target is
  selected, then edit the nested scope through breadcrumbs.
- **List-typed side values** - compatible values append to the next compact
  slot; clearing or replacing a slot compacts the list so there are no empty
  gaps.

## Where to next

- [Build a Pipeline](03_build_pipeline.md) - the basics of the main image-flow
  chain.
- [Fill a scalar aux target](12_aux_wire_in_dag.md) - a focused version of this
  flow with the green target highlighted.
- [Open an embedded Pipeline aux](13_wire_pipeline_as_aux.md) - fill a side
  value with a nested `ImagePipeline`.
