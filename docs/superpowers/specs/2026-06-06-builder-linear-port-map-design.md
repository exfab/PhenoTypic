# Builder canvas -- linear port map redesign

**Status:** Draft for user review
**Author:** Alexander Nguyen (with Codex)
**Date:** 2026-06-06
**Branch context:** current Codex worktree
**Mockup:** `docs/design-mockups/builder-linear-port-map.html`
**Scope:** `src/phenotypic/gui/builder/` state presentation, layout, callbacks, browser assets, tests, and GUI docs ledgers.
**Out of scope:** `ImagePipeline` runtime behavior, operation ABCs, results viewer, run console, analysis sub-app.

---

## 1. Problem statement

The current builder canvas exposes the full authoring model as a draggable DAG. That is more expressive than the product needs for the common case. A PhenoTypic image pipeline is still primarily a linear image-processing spine, with typed side inputs for operation-valued or pipeline-valued parameters.

The full DAG gives useful validation and data modeling, but it makes the visible UI harder to read:

1. Linear image flow competes visually with aux wires and container nesting.
2. Free node movement implies layout freedom that users do not need.
3. Visible ports are useful, but drag-to-wire is heavier than the desired "select target, click operation" flow.
4. Nested pipelines need to stay drillable through breadcrumbs, but they should not force users into a full graph mental model.

This spec replaces the visible free-DAG canvas with a fixed linear port map. The underlying state can remain DAG-shaped, but the UI constrains the authored topology to a single image spine plus side-loaded operation and pipeline parameters.

---

## 2. Goals

- Restore the simple fixed-node mental model from the previous linear builder.
- Preserve visible ports as first-class controls.
- Make every port a button that can select a target and open a compact menu.
- Add a floating continuation port after the last image-output port.
- Let palette clicks add to the currently selected target:
  - main continuation target means "insert into the image spine"
  - side parameter target means "fill this operation-valued or pipeline-valued parameter"
- Make the selected target green.
- Keep embedded pipelines drillable through breadcrumbs.
- Keep current DAG validation and serialization where practical.
- Follow `DESIGN.md` v1.2 for colors, typography, logo use, spacing, elevation, and geometry.
- Treat production alignment as a hard requirement, even if the standalone mockup is approximate.

---

## 3. Non-goals

- No runtime changes to `ImagePipeline`, `Image`, operation ABCs, or operation execution.
- No support for multi-image-input or multi-image-output operations.
- No freeform node placement in the default builder.
- No arbitrary graph branching in the default builder.
- No shared aux operation instance between multiple consumers in the default builder.
- No cross-scope wires. A nested pipeline scope is isolated and edited by drilling in.
- No permanent return to hidden popover-only aux state. Popovers are quick menus; the side loader is the durable editor.
- No drag-to-wire, palette drag/drop, or freeform graph manipulation in the default builder.
- No guided state-repair workflow in the first implementation. DAG authoring flows exist in development ledgers/tutorials, but they have not been released to production, so malformed DAG states only need a defensive unsupported-state path.

---

## 4. High-level decision

Use the current DAG model as the internal contract, but render a constrained authoring surface:

```text
[Input Image] -> [Op] -> [Op] -> [Terminal Op] -> (floating add port)
                                  |
                                  + side parameter ports
```

The visible canvas shows one ordered image spine per scope. Aux values are represented by side ports on consumer nodes and edited in the right side loader. When the value is itself a pipeline, the user can drill into that pipeline through breadcrumbs and see another fixed spine.

The old popover model is not restored as the primary editor. It returns only as a small action menu attached to a clicked port.

---

## 5. Design-system contract

The implementation must follow `DESIGN.md` v1.2.

### 5.1 Fonts

- Page title and content headings use IBM Plex Serif.
- UI titles, buttons, palette rows, and component headings use DM Sans.
- Badges, captions, parameter labels, breadcrumb text, numeric values, and code-like values use JetBrains Mono.

### 5.2 Colors

- UI chrome uses the primary palette only:
  - navy `#003660`
  - blue `#1b75bc`
  - gold `#febc11`
  - white `#ffffff`
  - canvas `#FBFEF8`
- Okabe-Ito colors are used only for semantic state indicators in this UI:
  - green for selected or valid
  - orange for warning or required-but-empty
  - vermilion for blocking error
- Do not use Okabe-Ito purple for aux chrome. Side parameter ports use brand gold because they are UI controls, not data series.

### 5.3 Radius and elevation

- Keep panel and card radii on the `--radius` / `--radius-md` ladder.
- Builder-local compact action buttons and port-menu buttons use `--radius-sm` for the sharper button shape requested in the mockup review.
- App-shell primary buttons keep the standard button treatment unless `DESIGN.md` is updated.
- Popovers use floating elevation (`--shadow-md`) and must not obscure adjacent port buttons.

### 5.4 Logo

- The builder topbar uses `light_logo_exfab.svg` in expanded desktop contexts.
- The icon mark is reserved for narrow or collapsed contexts.

---

## 6. User-facing layout

### 6.1 Three-column workspace

The production builder remains a work surface, not a landing page:

1. Left palette: searchable operations grouped by stage/type.
2. Center map: fixed linear image spine and visible ports.
3. Right side loader: active add/fill target, selected node or parameter editor, validation details, and drill actions.

The center map is the primary work area. The right side loader is not a passive inspector; it is the persistent editor for the current node or parameter context, with a persistent strip showing the active add/fill target.

### 6.2 Center map

Each active scope renders:

- a non-deletable `Input Image` source
- fixed-position operation nodes in image-flow order
- blue image input/output ports
- gold side parameter ports for operation-valued and pipeline-valued params
- a floating continuation port connected to the last output by a short line
- compact view-only zoom controls for zoom out, zoom in, reset to 100%, and fit/view all

Nodes are not draggable. Layout is computed from spine order and stable dimensions.
Zoom controls affect only the rendered HTML map viewport. They do not pan or
drag nodes, they do not write DAG state, and they do not serialize into
pipeline JSON.

Production alignment requirements:

- The main spine must fit the desktop builder viewport without accidental clipping at common desktop sizes.
- Port buttons must remain clickable when popovers are open and after map zoom changes.
- Menus must not cover the port that opened them or adjacent parameter ports.
- Long operation names and parameter names must wrap or ellipsize without resizing nodes.
- Node width, port size, menu width, and side-loader field height must be stable.
- Browser checks must cover at least 1280x720, 1440x900, and one tablet-width layout.

### 6.2.1 Map zoom controls

The linear map keeps compact viewport controls in the map header:

- minus icon button: zoom out
- plus icon button: zoom in
- reset zoom button: restore 100%
- fit/view all button: fit the full linear pipeline into the visible map viewport

Use lucide `Scan` for Fit when the icon set is available; otherwise use
`Maximize2` or an equivalent compact maximize glyph. The controls follow the
builder-local `--radius-sm` button treatment and the `DESIGN.md` chrome tokens.

Zoom is UI-only state:

- it may live in browser DOM/client asset state
- it must not mutate `_DagBuilderState`
- it must not write to any Dash store that serializes with the pipeline
- it must not change `ImagePipeline` JSON
- it must not re-enable Cytoscape pan, drag, or viewport behavior

### 6.3 Node anatomy

Each node includes:

- stage badge
- operation label
- short secondary metadata line
- left image input port when applicable
- right image output port when applicable
- one row per operation-valued or pipeline-valued parameter

Parameter rows place the port button to the left of the parameter label:

```text
(gold port) inoculum_detector
```

This left-first treatment makes the side-input affordance visible before the parameter name and matches the requested sidecar direction.

### 6.4 Floating continuation port

The terminal output port extends a short line to a floating circular port.

Default state:

- the floating port is selected green when no other target is selected
- the side loader shows "Continuation port"
- palette clicks insert after the current terminal node

Clicking the floating port:

- selects it green
- opens the continuation port menu
- updates the side loader to main-spine insertion mode

The continuation popover is not open by default. Selection and editing state live in the side loader.

### 6.5 Side loader

The side loader header places the state badge before the title:

```text
[Selected] Aux parameter port
```

The body includes:

- active add/fill target
- selected node or parameter context
- accepted value type
- slots or current value
- actions relevant to the target
- validation state
- drill-in button when the selected value is a pipeline or nested operation/pipeline scope
- class docstring help for the selected class or value

The side loader is the durable editing surface. If a popover is dismissed, the side loader still shows the selected node or parameter context and the active add/fill target.

---

## 7. Port model

### 7.1 Port categories

| Port | Color role | Placement | Meaning |
| ---- | ---------- | --------- | ------- |
| Image input | blue UI interactive | node left | receives image flow from previous node |
| Image output | blue UI interactive | node right | emits image flow to next node or terminal add target |
| Parameter input | gold UI accent | parameter row | fills operation-valued or pipeline-valued parameter |
| Floating continuation | blue plus selected green state | after last output | add target for next image-spine operation |

### 7.2 Target selection

Exactly one add/fill target is selected per scope:

```text
selected_target = {
  scope_path: breadcrumb path,
  kind: "continuation" | "image_port" | "parameter_port" | "parameter_slot",
  block_id: optional block id,
  param: optional parameter name,
  slot: optional integer
}
```

Selected target renders green. If the selected target becomes invalid after deletion or breadcrumb change, selection falls back to the current scope's floating continuation port.

The selected target is the only destination for palette clicks. A green port means "the next compatible palette click will insert or fill here."

### 7.3 Node selection

Node selection is separate from target selection.

Node selection is used for:

- side-loader scalar field editing
- node delete, replace, and reorder actions
- showing node-level validation details
- showing node class docstring help

Node selection should use a quieter outline or surface state. It must not use the green selected-target state. A node can be selected while the active palette target remains the floating continuation port.

After adding an operation to the main spine:

- the new operation becomes the selected node
- the floating continuation port becomes the selected add target
- the side loader shows the selected node editor plus the active continuation target strip

### 7.4 Port menus

Port menus are compact, local action surfaces.

Continuation port menu:

- Add operation
- Add pipeline
- Preview here
- Close menu

Image output port menu:

- Insert after this node
- Add pipeline after this node
- Preview to this point
- Select as continuation target

Image input port menu:

- Insert before this node
- Replace this node
- Preview upstream

Parameter port menu:

- Create operation
- Create pipeline
- Replace value
- Clear value
- Append list item, when list-valued
- Drill into value, when the value is drillable
- Show class docs, when a class is selected

Menu constraints:

- A menu must never be the only representation of a value.
- Dismissal must not clear selection.
- Popovers must be positioned to avoid covering clickable ports.
- Keyboard escape closes the menu without mutating state.

---

## 8. Palette behavior

Palette clicks are routed by the selected target.

### 8.1 Selected continuation target

Clicking an `ImageOperation` adds it to the image spine after the selected target.

Clicking `ImagePipeline` adds a nested pipeline node to the image spine.

After insertion:

- the new node is selected for editing
- the floating continuation port moves after it
- the floating continuation port becomes the next selected add target
- green selection appears on the floating continuation port, not on the node

### 8.2 Selected parameter target

Clicking a compatible `ImageOperation` creates or replaces the value for that parameter or slot.

Clicking `ImagePipeline` creates a nested aux pipeline value when the parameter accepts `ImagePipeline` or compatible operation pipelines.

After insertion:

- for an `ImageOperation` value, the consumer node remains selected for editing and the selected add target returns to the current scope's floating continuation port
- for an `ImagePipeline` value, the builder drills into the new embedded pipeline scope and selects that scope's floating continuation port
- the filled parameter row shows the new value and exposes replace, clear, drill, and help actions as appropriate
- the help icon for that slot opens the filled class docstring

### 8.3 Incompatible palette click

An incompatible operation does not mutate state. The UI should:

- keep the selected target
- flash the target warning state briefly
- show a short message in the side loader or toast

---

## 9. Embedded pipelines and breadcrumbs

Embedded pipelines remain click-into scopes.

Breadcrumb examples:

```text
Pipeline
Pipeline > Segmenter
Pipeline > FilamentousFungiDetector.inoculum_detector
Pipeline > MeasureFeatures.preprocess[1]
```

Drill-in entry points:

- selected pipeline node
- selected parameter value that is an `ImagePipeline`
- side-loader drill button
- port menu drill action

Drill behavior:

- breadcrumb pushes a typed segment
- canvas swaps to the nested scope
- nested scope renders its own fixed spine and floating continuation port
- the parent target is restored when navigating back, when possible

Each scope stores its own selected target. Entering or returning to a scope restores that scope's previous target when it still exists; otherwise it selects the floating continuation port for that scope.

---

## 10. Pipeline configurations

The redesign must support these configurations.

### 10.1 Main spine

- Empty root pipeline with only `Input Image` and floating continuation port.
- Single operation after input.
- Multi-operation linear image chain.
- Chain containing detector, measurement, and post-measurement operations.
- Insert before an existing node.
- Insert after an existing node.
- Replace a node while preserving compatible params when possible.
- Delete a node and reconnect predecessor to successor.
- Reorder by explicit move controls or menu actions, not dragging.

### 10.2 Scalar parameter values

- Optional operation-valued param left as default.
- Required operation-valued param missing.
- Operation-valued param filled by one operation.
- Operation-valued param filled by one nested pipeline.
- Existing param replaced by another compatible operation.
- Existing param cleared back to default or missing.

### 10.3 List parameter values

- Empty list.
- One filled slot.
- Multiple filled slots.
- Append slot.
- Slot remove.
- Slot reorder.
- Slot replacement.
- List slot filled by nested pipeline.

List UX stays contiguous. Adding appends to the end, deleting compacts the list, and reordering moves filled slots without leaving empty gaps.

### 10.4 Nested and recursive cases

- Pipeline node on the main spine.
- Pipeline value in an aux parameter.
- Aux operation with its own operation-valued parameter.
- Aux pipeline containing operations with their own side parameters.
- Breadcrumb return after nested edits.

### 10.5 Loaded or migrated state

- Legacy linear builder state.
- Current DAG state that maps cleanly to one image spine plus side params.
- Unknown operation class after loading saved JSON.

Saved pipeline JSON and builder states that map cleanly to the constrained editor remain supported. Development-only DAG states, tutorial artifacts, or GUI ledger flows that depend on arbitrary graph wiring are treated as retired/replaced in the implementation PR unless they can be normalized into one image spine plus side-loaded parameter values without data loss.

States that cannot be represented by the constrained editor should show a defensive unsupported-state panel. They should not silently drop nodes. No guided remediation workflow is required for the first implementation because DAG authoring has not been released to production.

---

## 11. Validation and recovery

Validation rules remain strict, but their presentation changes from graph-first to target-first.

Blocking issues:

- missing required parameter value
- incompatible parameter value
- no terminal image spine
- more than one terminal image spine after import
- cycle after import
- cross-scope edge after import
- unknown class that cannot be safely edited
- unsupported non-linear development-only graph state detected before editing

Advisory issues:

- unusual stage order
- optional parameter left at default

Issue presentation:

- issue list in side loader
- issue badge on affected node or port
- clicking an issue selects the offending port or node
- if the issue is inside a nested pipeline, clicking the issue drills to the scope and selects the target

Save and preview:

- global save is disabled for blocking issues
- global preview is disabled for blocking issues that affect the whole current pipeline
- advisory issues do not block save or preview
- port-menu `Preview here` can run to a selected upstream output when the prefix ending at that output is valid, even if a downstream node or parameter has a blocking issue

---

## 12. Internal state and conversion

The preferred implementation keeps the current DAG state classes and conversion code.

Rendering derives:

- active scope from breadcrumb
- image spine from the unique path rooted at `InputImage`
- terminal node from the last image-spine block
- parameter ports from operation registry metadata
- selected target from a UI store
- selected node from a UI store

The constrained renderer does not persist node coordinates. It computes positions from spine order.

State mutations should use existing dispatch concepts where possible:

- create block
- delete block
- create or replace edge
- delete edge
- drill into container
- drill to scope
- update param value
- reorder list aux
- append list aux slot

New or adapted dispatch concepts:

- select target
- select node
- insert after selected continuation
- insert before image input
- fill selected parameter target
- replace selected parameter slot
- reject unsupported graph state defensively

The public serialized pipeline format should remain unchanged.

---

## 13. Interaction details

### 13.1 Initial empty builder

User sees:

- `Input Image`
- its output port
- short line to floating continuation port
- continuation port selected green
- side loader in continuation mode

Palette click adds the first operation.

### 13.2 Adding to the main spine

1. User selects floating continuation port or image output port.
2. Target turns green.
3. Side loader shows accepted insertion type.
4. User clicks operation in palette.
5. Operation appears in the fixed spine.
6. Previous terminal connects to new node.
7. Floating continuation port moves after new node.
8. New node is selected for editing, while the floating continuation port is selected green as the next add target.

### 13.3 Filling a parameter

1. User clicks a gold side port.
2. Port turns green.
3. Side loader shows parameter name, accepted type, current slot state, and actions.
4. User clicks compatible operation or pipeline in palette.
5. Value is created and assigned.
6. If the value is an operation, the consumer node remains selected and the active add target returns to the current scope's floating continuation port.
7. If the value is a pipeline, the builder drills into the embedded pipeline and selects its floating continuation port.
8. The parent parameter row shows the value plus drill, replace, clear, and help actions.

### 13.4 Editing a nested pipeline

1. User selects a pipeline node or pipeline-valued parameter.
2. Side loader shows `Drill in`.
3. User drills in.
4. Breadcrumb pushes the nested scope.
5. Canvas shows the nested fixed spine.
6. User edits nested operations with the same target/port model.
7. Breadcrumb back returns to parent scope.

### 13.5 Deleting

Node deletion:

- deleting a main-spine node reconnects its predecessor to its successor
- deleting the first non-input node reconnects `Input Image` to the next node
- deleting a node with side parameter values requires confirmation if values would be lost

Parameter deletion:

- clearing scalar param removes the aux value and returns to default/missing state
- clearing list slot removes that value and compacts the list
- deleting nested pipeline from a parameter removes the nested scope after confirmation

### 13.6 Reorder

No free dragging. Use explicit controls:

- move left
- move right
- insert before
- insert after

For list params:

- move slot up
- move slot down
- reorder buttons only in the first implementation; no drag handle in the side loader

---

## 14. Retired default-builder affordances

The following affordances are retired from the default builder surface:

- Cytoscape node dragging and persisted node coordinates
- drag-to-wire connection authoring
- palette drag/drop onto the canvas
- visible aux wires between side parameter values and consumer nodes
- arbitrary container collapse, reparent, or cross-scope graph editing

The implementation PR should update `FEATURES.md` and `WORKFLOWS.md` to mark these development-only DAG flows as replaced by the fixed linear port-map workflow. If a loaded state uses one of these affordances but can still be represented as a linear image spine plus side-loaded parameters, it may render normally. If not, it enters the unsupported-state panel.

---

## 15. Defensive unsupported state

Some development or corrupted saved states may not fit the constrained model.

The first implementation does not build a guided remediation flow. If state cannot be represented as one image spine plus side-loaded parameter values, the builder shows a blocking unsupported-state panel.

Unsupported-state panel:

- explains that this saved builder state is not representable in the linear editor
- lists the offending condition, such as fork, orphan, cycle, shared aux source, or cross-scope edge
- blocks editing, preview, and save from that state
- offers only safe exits: load another pipeline, start a new pipeline, or export the raw JSON for debugging
- at mobile widths, the destructive start-new action is disabled; raw JSON export and read-only inspection remain available

---

## 16. Class docstring help

Inline parameter summaries are intentionally not part of the node card. They consume too much space and make compact nodes harder to align.

Instead, class help is available through a small question-mark icon:

- node header help opens the selected node class docstring
- parameter slot help opens the filled value's class docstring
- empty required parameter help opens the parameter description and accepted type
- help popovers are read-only
- help popovers close on Escape or outside click
- help content is sourced from registry metadata/docstrings, not handwritten UI text

This keeps the canvas dense while preserving discoverability.

---

## 17. Accessibility and keyboard behavior

- Every port is a real button with an accessible label.
- Every help icon is a real button with an accessible label that names the class or parameter.
- Selected target is conveyed by color and side-loader text, not color alone.
- Keyboard navigation can reach palette, canvas ports, port menus, and side loader.
- Enter or Space activates a focused port.
- Escape closes an open port menu without clearing selected target.
- Delete clears selected parameter value or deletes selected node after confirmation when destructive.
- Focus should move predictably:
  - port click keeps focus on port or moves to menu
  - palette add moves focus to side loader status or new node
  - drill-in moves focus to breadcrumb/current scope title

---

## 18. Responsive behavior

Desktop is the primary builder experience.

At desktop widths:

- three columns are visible
- center map does not horizontally clip the main spine for common pipelines
- long pipelines scroll horizontally inside the map, not the page

At tablet widths:

- palette may collapse to a drawer or top rail
- side loader may move below the map
- port controls remain at least 24x24 px with spacing

At mobile widths:

- mobile is read-only or limited inspection
- side loader stacks below map
- breadcrumbs, port selection, docstring help, validation issue inspection, and raw JSON/export affordances may remain available
- zoom, reset zoom, and fit/view all controls may remain available
- selected ports may still turn green for inspection, but palette-driven mutation remains disabled
- palette insertion, parameter editing, delete, reorder, replace, and save are disabled
- no text overlaps or clipped buttons are allowed

---

## 19. Testing and verification

Unit tests:

- selected target fallback after delete
- node selection remains independent from selected target
- palette click routes to continuation target
- palette click routes to parameter target
- operation-valued parameter fill returns selected target to current continuation port
- pipeline-valued parameter fill drills into embedded scope and selects nested continuation port
- incompatible palette click does not mutate state
- scalar param replace
- list param add/remove/reorder
- breadcrumb drill into aux pipeline
- unsupported graph state classification
- retired DAG affordances do not mutate default-builder state

Callback tests:

- port click updates target store
- node click updates node-selection store without overwriting target store
- side loader reflects selected target
- side loader can show selected node editing while selected target remains continuation
- popover open/close does not clear selection
- issue click selects offending target

Rendering tests:

- empty builder has `Input Image` plus floating port
- terminal output line exists
- gold param ports render before parameter labels
- class help button renders for selected class/parameter rows
- side-loader badge renders before title
- buttons use the approved compact radius in builder-local controls
- map zoom controls render for zoom out, zoom in, reset to 100%, and fit/view all
- zoom state is UI-only and absent from DAG/pipeline serialization

Browser verification:

- 1280x720 desktop
- 1440x900 desktop
- tablet-width layout
- no incoherent overlap
- no port covered by an open popover
- long labels wrap or ellipsize inside fixed boxes
- aux port click switches side loader to parameter mode
- continuation port click switches side loader to insertion mode
- class help icon opens the expected docstring popover
- zoom in/out/reset/fit does not mutate the pipeline and ports remain clickable
- nested pipeline breadcrumb drill-in and back
- mobile disables editing affordances while preserving inspection and zoom controls

Screenshot workflow:

- update `src/phenotypic/gui/FEATURES.md` for user-visible affordances
- update `src/phenotypic/gui/WORKFLOWS.md` only if tutorial-worthy flows change
- refresh GUI tutorial screenshots if production GUI chrome changes

---

## 20. Implementation sequence

1. Add selected-target state and tests.
2. Build constrained spine derivation from current DAG scope.
3. Render fixed nodes, image ports, side parameter ports, and floating continuation port.
4. Route palette clicks through selected target.
5. Add side-loader target modes.
6. Add port menus as secondary action surfaces.
7. Restore breadcrumb drill-in for parameter pipeline values.
8. Add class docstring help buttons and popovers.
9. Add defensive unsupported-state handling for non-linear development states.
10. Retire free-drag, drag-to-wire, and palette drag/drop affordances in default builder.
11. Update GUI ledgers, tests, and screenshots.

---

## 21. Acceptance criteria

The redesign is accepted when:

- a user can build a normal linear image pipeline without moving nodes
- the last output always exposes a floating add port
- selected main or side targets are green
- palette clicks add to the selected target
- side parameter ports are visible, clickable buttons
- side-loader controls are left-aligned as specified
- embedded pipelines can be drilled into and out of with breadcrumbs
- class docstrings are available from question-mark help icons, not inline summaries
- production layout has no port/menu/label overlap in verified browser sizes
- existing pipeline serialization remains compatible
- blocking validation prevents global save and global preview
- port-menu preview can still run valid upstream prefixes
- unsupported non-linear development states are blocked explicitly, not silently simplified

---

## 22. Notes from the mockup

The standalone mockup is a visual aid, not the final layout contract. It established the direction:

- fixed linear spine
- blue image ports
- gold parameter ports
- green selected target
- port menus
- right side loader
- badge before side-loader title
- parameter port before parameter label
- compact builder-local button radius
- question-mark help icon for class docstrings

The production implementation must be stricter than the mockup about alignment. Any alignment issue tolerated in the mockup should become a Playwright/browser verification case before shipping.
