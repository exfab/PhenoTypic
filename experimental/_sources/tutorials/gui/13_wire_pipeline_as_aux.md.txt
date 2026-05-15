# Wire a Pipeline container as aux

Some aux slots want more than a single detector — you might need a
preprocessing step before the detector, or a refinement step after
it. The DAG redesign exposes the `Pipeline` block as a **container**:
drop one on the canvas, drag a chain of ops into its nested scope,
then wire the container itself into a consumer's aux port. The
container serialises as a nested `ImagePipeline` inside the
consumer's `aux_ports`.

This tutorial walks through wiring a 3-step preprocessing chain
(`GaussianBlur → OtsuDetector`) into a
`FilamentousFungiDetector.inoculum_detector` slot.

## Step 1 — Create an empty container

Click `+ New Pipeline` in the toolbar's Structure card. An empty
`Pipeline` container appears on the canvas — its inner scope already
holds the auto-seeded `Input Image` sentinel block (so the container
has somewhere for its first wire to start).

![Empty Pipeline container on the canvas — outer frame with the Input Image sentinel visible inside.](../../_static/gui_images/wire-pipeline-as-aux/01_empty_container.png)

The container has two outward-facing ports: a **right-edge
output port** for connecting its overall result to the outer scope,
and a **bottom-edge aux input** for accepting an aux feed from a
parent. The inner-left **consumer-fed dot** lights up when the
container is wired as aux (telling you the nested `Input Image`
sentinel is being fed by the parent consumer rather than the outer
ribbon).

## Step 2 — Drop a chain inside the container

Double-click the container (or click its `Drill in →` button) to
descend into its nested scope. Drop `GaussianBlur → OtsuDetector →
OtsuDetector` and wire them left-to-right starting from the
`Input Image` sentinel. The breadcrumb at the top of the builder
shows the drill path (e.g. `Pipeline / nested_pipeline_xyz`).

![Container drilled in; nested scope holds GaussianBlur → OtsuDetector wired left-to-right.](../../_static/gui_images/wire-pipeline-as-aux/02_chain_in_container.png)

Operations inside a container behave exactly like operations in the
outer scope: they have I/O ports, can themselves carry aux slots, and
validate against the same DAG rules. Drill back out by clicking the
first breadcrumb crumb.

## Step 3 — Wire the container as aux

Outside the container, drop a `FilamentousFungiDetector` and drag a
wire from the container's **right-edge output port** to the FFD's
**bottom-edge aux port**. The wire is purple (aux), the container's
consumer-fed dot lights up, and the FFD's aux port flips from hollow
to filled-purple.

![Pipeline container wired as aux to FilamentousFungiDetector; purple aux wire runs from the container's right edge to the FFD's bottom-edge aux port; container's consumer-fed dot is lit.](../../_static/gui_images/wire-pipeline-as-aux/03_pipeline_wired_as_aux.png)

When the pipeline saves, the wired container serialises as a nested
`ImagePipeline` inside the consumer's `aux_ports`:

```json
{
  "FilamentousFungiDetector": {
    "class": "FilamentousFungiDetector",
    "aux_ports": {
      "inoculum_detector": {
        "class": "ImagePipeline",
        "params": {
          "pipe_cfgs": {
            "GaussianBlur": {"class": "GaussianBlur", "params": {"sigma": 2}},
            "OtsuDetector": {"class": "OtsuDetector", "params": {}},
            "OtsuDetector": {"class": "OtsuDetector", "params": {}}
          }
        }
      }
    }
  }
}
```

The JSON round-trips byte-identically — loading it back into the
builder recreates the container + chain + wire.

## Reference

- **Containers vs. blocks** — a container *contains* its own scope;
  a block does not. Either can be wired as aux.
- **Drag-in adoption** — dragging a block over a container's bounds
  and releasing causes the container to *adopt* the block (move it
  from the outer scope into the inner scope). Useful for refactoring
  a long ribbon into a reusable sub-pipeline.
- **Drag-out snap-back** — dragging a block from inside a container
  to outside its bounds moves the block back to the outer scope.
  Wires reroute automatically.
- **Consumer-fed dot** — purple = aux mode; grey = image-flow mode.
  Only one wire can target the container's aux input at a time.

## Where to next

- [Wire an aux in the DAG](12_aux_wire_in_dag.md) — the simpler
  single-block aux flow.
- [Fix validation issues](14_fix_validation_issues.md) — when the
  toolbar issue badge lights up because a container is mis-wired.
- [Build a Pipeline](03_build_pipeline.md) — the main image-flow
  basics.
