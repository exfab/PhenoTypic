# Aux-port UX redesign — popover-anchored, port-attached

**Status**: design, awaiting user review
**Date**: 2026-05-11
**Related**: PR #83 (Galaxy-style aux input ports — the shipped state this redesign supersedes)

---

## Context

PR #83 shipped a first cut of Galaxy-style aux input ports for op-typed parameters (`FilamentousFungiDetector.inoculum_detector`, `CompositeDetector.detectors`, etc.). The shipped version puts aux source nodes in a horizontal dock to the LEFT of their consumer, with smooth bezier wires from aux's right edge into a port handle attached to the consumer's left edge. A persistent "AUX OPERATIONS · op-as-config" lane label sits above the dock.

User feedback after shipping:

- The aux lane title is always visible even when no aux is present (visual noise).
- The aux nodes hanging off-left of the consumer clutter the main flow and crowd the visible canvas.
- The "aux as canvas nodes" model overweights configuration: an aux op is conceptually a *parameter value*, not a peer node in the data flow.
- Main image-flow input/output ports are not visible at all — image-flow edges connect node centers, so users can't tell where data enters/exits each operation.

This spec replaces the inline aux-dock model with a **popover-anchored design**: aux ports remain visible on consumer nodes (as visually-distinct bottom-edge markers), but aux configuration is edited through a canvas-anchored popover that opens on port click. The main canvas reverts to showing only the image flow, with main I/O ports rendered explicitly on left/right edges.

---

## Locked-in decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Configuration surface | Popover anchored to the aux port (Option C from brainstorming) |
| Escape hatch for complex aux | "Drill in →" button inside the popover |
| Port handle multiplicity | One port handle per param — popover handles all slots for list-typed params |
| Edit behavior | Inspector swaps to the wired aux's params; popover stays open in parallel |
| Visual style for port distinction | Style 2: main I/O on left/right (circles), aux on bottom (squares) |
| Popover anchoring | Canvas-tracking — pans/zooms with cytoscape via `cytoscape-popper` |
| Popover dismissal | Click outside or Escape; clicking another port replaces with new popover |
| Drill-in visibility | Always available when an aux is wired |
| Drill-in canvas semantics | Every drilled-in aux renders as a pipeline canvas (1-step ribbon for single-op aux, N-step ribbon for `ImagePipeline` aux) — recursive consistency |
| Disconnect behavior | Drops the wired aux's configuration (aux only exists while wired) |
| Data model | Refactor: embed aux `StepNode` directly inside the slot (no separate `aux_nodes` list) |
| Test coverage | New e2e Playwright tests; existing dispatch + state-roundtrip tests updated |
| Tutorial demo | Multi-step main pipeline (≥3 ribbon nodes) to validate aesthetic across density |

---

## Scope

### What goes away (versus PR #83's shipped state)

- The "AUX OPERATIONS · op-as-config" HTML lane chrome (label + divider above the canvas).
- All aux nodes rendered as cytoscape elements on the main canvas (the lavender boxes below-and-left of consumers).
- Aux wires (purple dashed bezier curves) on the main canvas — `edge.aux-wire` cytoscape style is removed.
- The inspector "Aux palette" section with per-empty-slot buttons (`_build_aux_palette_section` in `_layout.py`).
- The "MAIN IMAGE FLOW" lane label (becomes redundant — the canvas IS the main flow now; no need to name it).
- The free-floating aux-orphan styling (`.aux-orphan`, dashed border) — orphans can no longer be created in the new model.
- The `BuilderScope.aux_nodes` field and ID-based slot references.
- The `aux_add`, `aux_delete` dispatch kinds (aux lifecycle is now tied to slot lifecycle).
- `_collect_orphan_aux_ids` + `_aux_id_in_current_scope` + `_last_aux_id_for_class` helpers in `_callbacks.py`.

### What stays (same model, same data shape on disk)

- `pipeline.json` schema — **unchanged**. Op-typed param markers (`{"__type__": "operation", ...}`) are the durable contract. Existing saved files round-trip byte-identically.
- Drill-in / breadcrumb infrastructure — extended to drill into aux slots via a new `drill_in_aux` segment shape `{"target_node_id": ..., "param": ..., "slot": ...}`.
- Click-then-click wire creation flow — gone; replaced by direct palette-click within the popover.
- Type-aware compatibility filtering — same logic, applied to popover palette filtering.

### What's new

- **Main I/O ports**: small blue circles on left (input) and right (output) edges of every operation node. Image-flow edges connect output port → next input port.
- **Aux ports**: small purple squares on the bottom edge of nodes that have op-typed params. One square per param (regardless of slot cardinality).
- **Canvas-anchored popover** via `cytoscape-popper` extension + `@popperjs/core`, vendored in `src/phenotypic/gui/builder/assets/`. Popover pans/zooms with the canvas.
- **Embedded aux StepNode** in `StepNode.aux_ports[param][slot]` (replaces the `aux_nodes`-list + ID-ref model).
- **E2E Playwright tests** (`tests/gui/builder/test_aux_port_e2e.py`) that boot the builder and drive the full flow with `dash[testing]` or vanilla Playwright.
- **Multi-step tutorial walkthrough** (rewrite of `docs/source/tutorials/gui/09_aux_ports.md`) with a 4-step main pipeline to demonstrate aesthetic and interaction at density.

---

## Visual design

### Node anatomy

Every operation node renders with:

- **Body**: 180px × 54px rounded rectangle, stage-tinted background (existing `_STAGE_COLORS`).
- **Class-name label**: centered, `text-wrap: ellipsis`, `text-max-width: 160`.
- **Main input port** (image-flow): blue circle, radius 5px, on the LEFT edge at vertical center. Always rendered filled (indicating the port exists and can accept image data); the *connected* state is communicated by the edge itself, not by port fill. The leftmost ribbon node's input port has no edge but renders identically — symmetry is the priority.
- **Main output port** (image-flow): same, on the RIGHT edge at vertical center. Same fill semantics.
- **Aux ports** (op-typed params): purple squares, 10px × 10px, on the BOTTOM edge, evenly spaced. One square per op-typed param. Filled when at least one slot in that param is wired; hollow outline when all slots empty. The wired/empty distinction matters here because there is no edge to convey it.

For list-typed aux ports, the single square may render a small numeric badge in its top-right corner showing the count of wired slots (e.g. `2/3` for a port with 2 wired and 1 empty slot).

### Popover anatomy

Anchored to the aux port square via `cytoscape-popper`. Renders as a small DOM card with:

- **Header**: param name (e.g., `inoculum_detector`) + a close button (×).
- **Body**: varies by state (see below).
- **Footer** (only when at least one slot wired): inline action row with `[✎ Edit]`, `[Drill in →]`, `[⨯ Disconnect]` per wired slot.

#### State variants

**Scalar param, empty** (`FilamentousFungiDetector.inoculum_detector` with no aux):

```
┌─────────────────────────────────┐
│ inoculum_detector            × │
├─────────────────────────────────┤
│ Pick an operation:              │
│ ┌─────────┐ ┌─────────┐ ...     │
│ │ Otsu... │ │ Hyster..│         │
│ └─────────┘ └─────────┘         │
└─────────────────────────────────┘
```

Compatible classes from registry (filtered by `is_operation`/`is_pipeline`) shown as small outline buttons in a wrapping grid. Click → wires that class into the slot, popover updates to "wired" variant.

**Scalar param, wired** (after wiring `OtsuDetector`):

```
┌─────────────────────────────────┐
│ inoculum_detector            × │
├─────────────────────────────────┤
│ ● OtsuDetector                  │
│   [✎ Edit] [Drill in →] [⨯]    │
└─────────────────────────────────┘
```

Clicking `✎ Edit` swaps the inspector to OtsuDetector's params (popover stays open). Clicking `[⨯]` drops the aux. Clicking `Drill in →` pushes the breadcrumb and the main canvas becomes the aux-as-pipeline editor.

**List param, mixed** (`CompositeDetector.detectors[*]` with 2 wired, 1 empty):

```
┌─────────────────────────────────┐
│ detectors[*]                 × │
├─────────────────────────────────┤
│ Slot 0  ● OtsuDetector          │
│         [✎] [↘] [⨯]             │
│ Slot 1  ● RoundPeaksDetector    │
│         [✎] [↘] [⨯]             │
│ Slot 2  ○ (empty)               │
│         Pick: Otsu | Hyst | ... │
│ [+ Add slot]                    │
└─────────────────────────────────┘
```

Each slot row is independently editable. `+ Add slot` appends a new empty slot. Per-slot `⨯` shrinks the list (slot 0 wired + slot 1 wired + slot 2 empty: clicking slot 2's `⨯` removes slot 2 entirely; clicking slot 0's `⨯` drops the wired aux and shifts remaining slots up — slot 1 becomes slot 0).

### Lane chrome

Removed entirely. The canvas IS the main flow; no labeling needed.

---

## Interaction flow

1. **User clicks an aux port** on the bottom edge of a consumer node.
   - `cytoscape-popper` anchors the popover DOM element to the port and tracks its screen position.
   - Inspector behavior:
     - If the port is wired (≥1 slot), inspector swaps to show the **first wired slot's** aux params. A small breadcrumb header appears: `← FilamentousFungiDetector.inoculum_detector`.
     - If the port is empty, inspector stays on the currently-selected consumer node.

2. **User picks a class from the popover palette** (empty slot path):
   - Dispatches `wire_create({target_node_id, param, slot, class_name})` — creates the embedded aux `StepNode` in the target slot with default params, in a single mutation.
   - Popover re-renders in "wired" variant.
   - Inspector swaps to show the new aux's params.

3. **User clicks `✎ Edit`** on a wired slot:
   - Inspector swaps to that aux's params (or refocuses if it's already showing them).
   - Popover stays open.

4. **User clicks `Drill in →`** on a wired slot:
   - Popover dismisses.
   - Dispatches `drill_in_aux({target_node_id, param, slot})` — pushes a new breadcrumb segment of shape `{"target_node_id": ..., "param": ..., "slot": ...}`.
   - Main canvas re-renders as the aux's drilled-in scope:
     - If the aux is a single op: 1-node ribbon with that op centered. User can add steps before/after (palette works normally), upgrading the aux from a single-op marker into a multi-step pipeline.
     - If the aux is already an `ImagePipeline`: N-node ribbon with the existing steps.
   - Breadcrumb shows the drill path; clicking a parent crumb drills back out.
   - The drilled canvas supports recursive aux: any op inside it with op-typed params renders its own aux ports + popover.

5. **User clicks `⨯ Disconnect`** on a wired slot:
   - Dispatches `wire_delete({target_node_id, param, slot})`.
   - The aux `StepNode` in that slot is dropped (set to `None`).
   - For list-typed ports, the slot itself remains (as an empty placeholder).
   - For scalar ports, the slot is permanently length 1; it just becomes `None`.
   - Popover re-renders.

6. **User clicks outside the popover** or **presses Escape**:
   - Popover dismisses.
   - Inspector reverts to showing the canvas-selected node (the consumer, or whatever was selected before the port click).

7. **User clicks a different aux port** while a popover is open:
   - Current popover dismisses.
   - New popover opens for the clicked port.
   - Inspector swaps to the new port's wired aux (or stays on consumer if empty).

---

## Data model

### New shape

```python
@dataclass
class BuilderScope:
    nodes: List[StepNode]              # main ribbon — unchanged
    # aux_nodes removed entirely

@dataclass
class StepNode:
    node_id: str
    class_name: str
    label: str
    params: Dict[str, Any]
    nested: Optional["BuilderScope"]
    # NEW shape: slot value IS the embedded StepNode, not an ID reference.
    aux_ports: Dict[str, List[Optional["StepNode"]]]
```

### Serialization invariants

- **`pipeline.json` schema is unchanged.** `to_pipeline` walks `consumer.aux_ports`, for each non-None slot it serializes the embedded `StepNode` directly into a marker dict (`{"__type__": "operation", "class_name": ..., "params": ...}`) and folds the marker into `consumer.params[<param_name>]`. Same output shape as today.
- **`from_pipeline` walks marker dicts** in each op's params, materializes each as an embedded aux `StepNode` in the corresponding slot. Same input shape as today.
- **Builder-state JSON** (`state_to_json` / `json_to_state`) — internal representation, not user-visible. Updated to serialize the new aux_ports shape. No backwards-compat concern: builder state is session-scoped.

### Recursive aux

Embedded aux nodes can themselves have `aux_ports`. So `FilamentousFungiDetector.aux_ports["inoculum_detector"][0]` could be a `FilamentousFungiDetector` StepNode whose own `aux_ports["inoculum_detector"][0]` is an `OtsuDetector` — etc. No depth limit. The marker schema naturally supports this (markers nest).

### What this simplifies

- **Orphan detection disappears.** Aux cannot exist without a slot; orphans are impossible by construction. `_collect_orphan_aux_ids` and friends are deleted.
- **`aux_add` and `aux_delete` dispatch kinds collapse into `wire_create` and `wire_delete`**. There's no "add aux first, wire later" path anymore.
- **No ID-resolution step** in `to_pipeline` / `from_pipeline`. Direct walk.

### Dispatch kinds (final list)

| Kind | Params | Effect |
|---|---|---|
| `wire_create` | `{target_node_id, param, slot, class_name}` | Materialize an embedded aux `StepNode` with the given class + default params; assign to the slot. Type-validates the class against the port type. **Breaking change from PR #83**: payload no longer carries `source_aux_id` (free-floating aux nodes don't exist anymore); takes `class_name` directly. |
| `wire_delete` | `{target_node_id, param, slot}` | Set slot value to `None`. Aux config is dropped. |
| `port_slot_add` | `{target_node_id, param}` | Append `None` to a list-typed param's slot list. No-op on scalar params. |
| `port_slot_remove` | `{target_node_id, param, slot}` | Remove the slot at index `slot`. List-typed only; scalar slots stay at length 1 (use `wire_delete` instead). |
| `drill_in_aux` | `{target_node_id, param, slot}` | Push breadcrumb segment, return canvas focus to the aux's scope. |
| `set_inspector_focus` | `{focus: "consumer" \| "aux", target_node_id, param?, slot?}` | Switch the inspector pane between the consumer's params and a wired aux's params. Dispatched when the popover opens with a wired slot (focus="aux"), when the user picks a class to wire (focus="aux"), when the popover dismisses (focus="consumer"), when the wired aux is disconnected (focus="consumer"), and when `drill_in_aux` fires (focus reverts to the new canvas scope's selection). |

The breadcrumb segment for aux drill-in uses the shape `{"target_node_id": str, "param": str, "slot": int}` — distinct from the existing `{"node_id": ...}` (nested-pipeline drill) and `{"aux_id": ...}` (old aux-node drill, removed). `_walk_to_scope` updated to navigate into the aux's embedded `StepNode`'s `nested` field if the aux is a pipeline, or treat the aux StepNode itself as a single-node ribbon scope.

---

## Backwards compatibility

### Saved `pipeline.json` files

**No migration required.** The on-disk schema is unchanged. Existing files load via `from_pipeline`, which materializes the marker dicts into the new embedded-aux shape. Saving via `to_pipeline` produces byte-identical output for the same logical pipeline.

### In-memory `BuilderState` from prior session

Not persisted across sessions in any release. No concern.

### Existing builder-state tests

Round-trip tests in `tests/gui/builder/test_state_roundtrip.py` updated to reflect the new shape. The byte-identical `pipeline.json` invariant still holds and is preserved as a test.

### Existing dispatch tests

Tests in `tests/gui/builder/test_aux_ports.py` rewritten for the new dispatch kinds. The 23 existing tests reduce to ~12 (since `aux_add`, `aux_delete`, and the click-then-click pending-wire state machine are gone). New e2e tests added.

---

## Testing

### Unit tests (state, dispatch, registry)

- `tests/gui/builder/test_state_roundtrip.py` — extend with:
  - Embedded-aux round-trip (`FilamentousFungiDetector` with wired `OtsuDetector`)
  - Recursive aux round-trip (`FilamentousFungiDetector` wired with another `FilamentousFungiDetector` wired with `OtsuDetector`)
  - `CompositeDetector` with 3-slot `detectors` list (mixed wired/empty)
- `tests/gui/builder/test_aux_ports.py` — rewrite for new dispatch kinds:
  - `wire_create` materializes embedded aux in slot
  - `wire_create` rejects type-incompatible class
  - `wire_delete` sets slot to None, drops aux config
  - `port_slot_add` / `port_slot_remove` for list-typed only
  - `drill_in_aux` pushes correct breadcrumb segment
- `tests/unit/gui/test_param_forms.py::TestWiredSlots` — rewritten as `TestPopoverSlotRendering`; same per-slot data structure, assertions retargeted at the popover renderer's output
- `tests/unit/gui/test_operation_registry.py::TestIsListDetection` — unchanged (registry detection is unchanged)

### E2E tests (new)

`tests/gui/builder/test_aux_port_e2e.py` (new file) — uses `dash[testing]` or Playwright directly to:

1. **Empty port click** — boot builder, add `FilamentousFungiDetector`, click `inoculum_detector` aux port, assert popover opens with compatible classes palette.
2. **Wire via palette** — from the open popover, click `OtsuDetector` button, assert popover transitions to wired variant and inspector shows OtsuDetector params.
3. **Drill-in flow** — click `Drill in →`, assert breadcrumb pushes and canvas shows OtsuDetector as a 1-step ribbon.
4. **Drill-out flow** — click parent breadcrumb, assert canvas restores to main flow.
5. **Disconnect** — click `⨯`, assert popover transitions to empty variant and inspector reverts to FilamentousFungiDetector.
6. **List port mixed state** — add `CompositeDetector`, click `detectors[*]` port, add 2 detectors via popover, leave slot 2 empty. Assert popover shows 3 rows.
7. **Click outside dismisses popover** — open popover, click empty canvas, assert popover closes.
8. **Pipeline.json round-trip** — wire an aux, save, reload, assert canvas + popover state match.

E2E tests use the existing `boot_gui` helper from `scripts/capture_gui_tutorial_screenshots.py` (refactored slightly to be reusable from tests).

### Manual smoke (rewrite of `docs/source/tutorials/gui/09_aux_ports.md`)

Tutorial walks through a **4-step main pipeline** to demonstrate aesthetic at density:

1. **Step 0**: Empty canvas with main I/O ports visible on every operation in palette.
2. **Step 1**: Add `GaussianBlur` → main I/O ports visible on left/right; no aux ports (no op-typed params).
3. **Step 2**: Add `ContrastStretching`, then `FilamentousFungiDetector`, then `MeasureColonySize`. Image-flow edges connect output→input ports clearly. `FilamentousFungiDetector` shows one aux port (purple square) on its bottom edge.
4. **Step 3**: Click the aux port → popover opens. Pick `OtsuDetector` from palette. Popover transitions to wired variant. Inspector shows OtsuDetector params.
5. **Step 4**: Click `Drill in →`. Canvas shows OtsuDetector as a 1-step ribbon. Add `GaussianBlur` before it (extending the aux into a 2-step pipeline). Breadcrumb shows `Pipeline / FilamentousFungiDetector / inoculum_detector / [slot 0]`.
6. **Step 5**: Drill back out. Save pipeline.json. Reload from disk. All wiring + drill-in scope intact.

Screenshot capture script `_capture_aux_ports` updated to mirror these 6 steps.

---

## File-by-file impact

### Modified

| File | Why |
|---|---|
| `src/phenotypic/gui/builder/_state.py` | Refactor `BuilderScope` (drop `aux_nodes`), `StepNode.aux_ports` (embed StepNode), `from_pipeline` / `to_pipeline` (direct walk), `_walk_to_scope` (new breadcrumb segment). |
| `src/phenotypic/gui/builder/_layout.py` | Drop aux-dock rendering, aux-wire stylesheet, lane chrome HTML. Add main I/O port cytoscape elements + bottom-aux port elements. Add `cytoscape-popper` mounting. Drop `_build_aux_palette_section`. |
| `src/phenotypic/gui/builder/_callbacks.py` | Drop `aux_add` / `aux_delete` / click-then-click pending-wire state. Update `wire_create` to create embedded aux. Update `drill_in_aux` for new segment shape. Drop orphan helpers. Add popover-dismiss + class-pick callbacks. |
| `src/phenotypic/gui/builder/_ids.py` | Drop `STORE_PENDING_WIRE` + `STORE_AUX_PALETTE_TARGET`. Add `PORT_CLICK_STORE` (clientside-to-server bridge for popover). Update pattern-matching helpers for main I/O ports + bottom-edge aux ports. |
| `src/phenotypic/gui/builder/_param_form.py` + `src/phenotypic/gui/_param_forms.py` | Drop `wired_slots` kwarg (popover replaces this surface). Inspector renders consumer params OR aux params based on `inspector_focus_aux` state. |
| `src/phenotypic/gui/builder/assets/builder.css` | Drop `.aux-orphan`, `.pheno-canvas-lane-chrome`, `.inspector-aux-palette*`. Add `.cy-popover`, `.cy-popover-header`, `.cy-popover-palette`, `.cy-popover-wired-row`. |
| `tests/gui/builder/test_state_roundtrip.py` | Update for embedded-aux shape; add recursive-aux test. |
| `tests/gui/builder/test_aux_ports.py` | Rewrite for new dispatch kinds. |
| `tests/unit/gui/test_param_forms.py` | Rewrite `TestWiredSlots` as `TestPopoverSlotRendering` — same per-slot data structure, assertions retargeted at the popover renderer's output (port dot, class name, action buttons). |
| `docs/source/tutorials/gui/09_aux_ports.md` | Rewrite for 4-step pipeline demo. |
| `scripts/capture_gui_tutorial_screenshots.py::_capture_aux_ports` | Updated for new 6-step flow. |
| `src/phenotypic/gui/FEATURES.md` | Rows updated to reflect popover-based affordances. |

### New

| File | Purpose |
|---|---|
| `src/phenotypic/gui/builder/assets/cytoscape-popper.min.js` | Vendored. ~3KB. |
| `src/phenotypic/gui/builder/assets/popperjs-core.min.js` | Vendored. ~9KB. |
| `src/phenotypic/gui/builder/assets/aux_popover.js` | Clientside glue: listens for cytoscape tap on `node.aux-port`, mounts a Popper instance pointing at a DOM popover element. ~80 lines. |
| `tests/gui/builder/test_aux_port_e2e.py` | E2E suite (above). |

### Removed

- `_collect_orphan_aux_ids`, `_aux_id_in_current_scope`, `_last_aux_id_for_class`, `_resolve_pending_wire_tap` from `_callbacks.py`.
- `_build_aux_palette_section`, `_compatible_classes_for_port` (moved to popover renderer module) from `_layout.py`.

---

## Risks

1. **`cytoscape-popper` extension integration risk** — third-party JS extension. Vendoring is straightforward but the clientside glue (computing popper position, dismiss on click-outside) needs care. Mitigation: write the glue first as a small standalone proof-of-concept before refactoring the rest.

2. **Inspector "aux focus" override state** — adds a new piece of UI state (inspector_focus_aux). Need clear rules for when it gets cleared (popover close, aux disconnect, drill-in, etc.). Mitigation: dedicate one dispatch kind (`set_inspector_focus`) and route all transitions through it.

3. **Drill-in scope rendering for single-op aux** — "every aux is a 1-step pipeline" is elegant but means the drilled scope is a near-empty canvas for the common case. Risk: feels visually thin. Mitigation: add a "drilled-in" lane chrome label naming the parent + param + slot (e.g., `Editing: FilamentousFungiDetector.inoculum_detector / slot 0`).

4. **E2E test flakiness** — Dash callbacks + cytoscape + popper.js render asynchronously. Mitigation: use `dash[testing]`'s built-in waits + explicit selectors with timeouts.

5. **Multi-step demo screenshot stability** — capture script must produce consistent screenshots across runs. Mitigation: snapshot test (image diff) gated behind a manual `--update-snapshots` flag.

---

## Out of scope (deferred to v2+)

- Drag-and-drop wire creation (with cytoscape-edgehandles).
- Aux port keyboard navigation / focus trap inside the popover (accessibility).
- Drag-to-reorder slots inside a list-typed popover.
- Popover-side mini-preview of the wired aux's effect on the image.
- Multi-edit (selecting multiple aux ports and editing in bulk).
- Free-floating aux nodes (intentionally removed; if a use case emerges, revisit).
