# Builder canvas — DAG redesign

**Status:** Draft for implementation planning
**Author:** Alexander Nguyen (with Claude)
**Date:** 2026-05-12
**Branch context:** `builder-gui-redesign`
**Scope:** `src/phenotypic/gui/builder/` (state, layout, callbacks, assets)
**Out of scope:** Results viewer, run console, analysis sub-app, the `ImagePipeline`
runtime contract itself.

---

## 1. Problem statement

The Pipeline Builder currently models aux inputs through a *popover-anchored*
design: every op-typed parameter on a consumer ("aux port") is a small purple
marker on the consumer's bottom edge; tapping it opens a `cytoscape-popper`
popover that hosts a class palette (empty slot) or an Edit/Drill-in/Disconnect
row (wired slot). Aux step nodes themselves are hidden inside the consumer's
`aux_ports` map — the user cannot see them on the canvas unless they drill in.

This produces four concrete pain points:

1. **Hidden state.** A consumer with three list-typed aux slots wired requires
   tapping the marker to know what's inside; comparing two consumers' aux
   configurations means opening two popovers in sequence.
2. **Modal-feeling popover.** Click-outside-to-dismiss + Escape + canvas-pan
   dismiss combine into accidental dismissals; the popover blocks adjacent
   interactions while open.
3. **Editing-by-popover.** Wiring a multi-step pipeline as aux means tapping
   the port → "Drill in →" → breadcrumb-deeper canvas → build the pipeline →
   walk back. Three context switches for a 2-op aux.
4. **Mismatch with mental model.** Users say "I'm wiring this detector into
   that one"; the UI says "I'm picking from a palette inside a popover
   anchored to a marker on the consumer."

This spec replaces the popover-anchored design with a *Pure DAG* canvas where
every block is first-class on the canvas and wires are drawn directly between
output and input ports.

## 2. Goals

* **Surface every operation on the canvas.** Aux ops, aux pipelines, and main-flow
  ops all live as visible blocks; nothing is hidden behind a tap.
* **Wire-drawing as the universal mutation.** Adding/removing relationships is
  drag-from-port-to-port; no class pickers, no per-port menus.
* **Strict validation.** Forks, unwired stubs, and required-but-empty aux ports
  flip the offending block's border red and disable `Run preview` + `Save`.
* **Preserve the runtime contract.** `to_pipeline` / `from_pipeline` still
  produce the same `ImagePipeline` instances; no changes to `_core` or `abc_`.
* **Type-aware affordances.** During drag, only registry-compatible target
  ports glow; incompatible ones dim. Inspector forms and existing widgets
  (point picker, column-aware dropdowns) stay intact.

## 3. Non-goals

* No changes to `ImagePipeline`, `Image`, the operation registry's class
  discovery, or any abc_ behavior.
* No support for **multi-image-input** ops or **multi-image-output** ops.
  Every op has exactly one image input and one image output; aux is the only
  way to plug other operations in.
* No **wire-level branches** in the image flow. The image-flow chain remains
  linear (rooted at `Input Image`, terminating at a measurement/post node).
* No **persisted layout** in `pipeline.json`. Block positions are computed on
  every render by a dagre/fcose pass (see §6).
* No **Output sink node**. The block whose right output has no outgoing wire
  *is* the terminal (strict rules guarantee uniqueness).
* No **shared-instance aux**. One aux block on the canvas wires to at most one
  consumer port; if a user wants the same op in two places, drag two blocks.
* No **cross-container wires**. A `Pipeline` container is a scope — wires only
  connect ports inside the same scope. The container's own output port exits to
  the parent scope.

## 4. User-facing model

### 4.1 Block types

| Block         | Shape             | Image-in | Image-out | Aux-in (bottom) | Output (right) | Notes |
| ------------- | ----------------- | -------- | --------- | --------------- | -------------- | ----- |
| `Input Image` (root scope) | green chevron     | —        | implicit  | none            | one (blue)     | exactly one per scope; auto-seeded on scope creation; *not* in the palette; cannot be deleted, only re-wired |
| `Input Image` (container scope) | small purple dot ("consumer-fed") | — | implicit | none | one | analogue inside a container; rendered as a decorative dot near the container's left edge; auto-seeded; cannot be deleted |
| Regular op    | rounded rect      | one      | one       | 0+ typed        | one            | stage colour: ops/meas/post; deletable |
| Pipeline container | purple-bordered group | one* | one*  | 0+ on inner ops | one (on container border) | *left input behaves as image-in when the container is wired into a downstream blue port; as a "consumer-fed" sentinel when wired into a purple aux port. Deletable. |
| Stage colors  | (background tint) | —        | —         | —               | —              | `ops` = navy tint, `meas` = gold tint, `post` = green tint; `pipeline` = purple tint. Inherits from today's `_STAGE_COLORS`. |

`*` Container left edge has two visual modes determined by what its right output
wires to — see §4.4. The chain order inside a scope is allowed to interleave
ops/meas/post freely; `to_pipeline` partitions them by `isinstance` against
`MeasureFeatures` / `PostMeasurement` (same behaviour as today's
`to_pipeline`). Stage colours give the user an at-a-glance hint without
restricting the canvas order.

**Input Image lifecycle:** every new `BuilderScope` (root *and* every nested
container) auto-seeds an `InputImage` block on creation. The palette does
*not* expose `+ New Input Image`; the user can re-wire the source but not
remove or duplicate it. Attempting to drag a second one in is rejected at the
dispatcher (no-op + toast: "scope already has an Input Image"). If state
arrives without an `InputImage` block (e.g., a corrupted store or a custom
migration path), the dispatcher synthesises one on the first state-load pass.

### 4.2 Port semantics

* **Image-input** (blue, left edge, circle): receives one wire from an
  upstream block's output. Must have exactly one incoming wire to validate.
* **Image-output** (right edge, circle): produces image data. Wires to either
  an image-input (blue wire) or an aux-input (purple wire) — the *wire color
  follows the target*, the port itself is neutral.
  - **Hard rule:** every output port has **at most one outgoing wire,
    total.** An op is either in the main flow *or* used as aux, never both.
    If a user wants the same op in two places, they drag a second block (the
    no-shared-instance rule from §3). The single-wire rule applies regardless
    of wire colour — image+aux from the same source is *not* allowed.
* **Aux-input** (bottom edge, square): one per op-typed parameter.
  - Scalar aux: accepts at most one wire.
  - List-typed aux: accepts many wires (fan-in); order set in the inspector.
  - Required (no default in the registry): renders with a red ring when empty.
  - Optional (has a default): renders hollow purple when empty.

**Block border colours** (in addition to stage-tinted background):

| Border | Meaning |
| ------ | ------- |
| Block-style border (1px navy/gold/green) | Default — main-flow op |
| Purple solid border (1.5px) | This block's output is wired to an aux port (i.e. it is *consumed as aux*). Visual cue that this block lives outside the main spine. |
| Yellow border (1.5px) | **Advisory** — stage ordering hint or unknown-class fallback. Does not block preview/save. |
| Red solid border (2.5px) | Validation error — one of Rules 1–6 (§4.6) is violated. Blocks preview/save. |
| Red dashed border (2.5px) | Specifically the *stub* case of Rule 2 — block is not reachable from `Input Image`. Dashed to read as "draft" rather than "broken". |

Aux-consumed blocks therefore read as purple-bordered on a navy-tinted
background; their output wire is purple-dashed and lands on a consumer's
purple aux port. At a glance the user can tell "this is aux" from the
block chrome alone, without needing to follow the wire.

### 4.3 Wires

* **Drawing:** mouse-down on an output port → drag → release on an input
  port. While the drag is in flight the wire is **neutral gray, dashed**.
  On drop the wire snaps to its settled colour:
  - Blue, solid: dropped on a blue image-input port (image-flow edge).
  - Purple, dashed: dropped on a purple aux-input port (aux edge).
* **Type-aware target affordance:** on `dragstart` the registry is consulted
  via `OperationInfo.parameters[<port>]` for every aux-input in the scope
  (and `Image` for every image-input). Targets whose annotation accepts the
  dragged source class **glow** (soft halo + slight scale-up); targets that
  reject it **dim** to ~30% opacity, so the user can see what is *not*
  acceptable without reading docs. The source class is the output block's
  `class_name` (or, for a container, `ImagePipeline`).
* **Cancellation:**
  - `Esc` while dragging → wire fades out, no state change.
  - Drop on empty canvas (no port under cursor) → fades out.
  - Drop on a dimmed (incompatible) port → fades out with a short red
    flash on the rejected port; no state change.
  - Mouse leaves the cytoscape wrapper bounds → fades out.
* **Replacing the wire:**
  - Dragging from an output port that already has an outgoing wire
    **replaces** the existing wire (no forks allowed; the old edge is
    deleted in the same dispatch).
  - **Moving a wire's target is a two-step gesture, not a drag-endpoint
    one.** Cytoscape does not expose grabbable edge endpoints natively, and
    adding `cytoscape-edgehandles` for one gesture isn't worth the
    dependency. To re-target a wire: click the wire (highlights) → `Delete`
    → drag a new wire from the same source port. The "replace by drag from
    source" flow already covers the common case.
* **Deleting:** click an edge (it highlights — stroke widens to 4px and
  brightens) → `Delete` or `Backspace`, or right-click → *Disconnect*.
  Selection survives across re-layouts; click empty canvas to deselect.
* **No image-flow fan-in / fan-out — and no mixed-kind fan-out either:**
  output ports take **at most one outgoing wire, total** (image *or* aux,
  never both). Image-input ports take at most one incoming. Aux ports
  follow per-port rules: scalar aux rejects the second wire by silently
  *replacing* the first (no flicker); list aux **appends** to the next
  free slot.
* **Single wire colour past the measure boundary:** image-flow wires stay
  blue throughout the chain even after a `MeasureFeatures` node (where
  what "flows" is no longer image bytes but the next-in-chain handoff).
  The block's *stage-tinted background* already signals what kind of work
  happens there; the wire just means "execution order." We considered
  tinting wires gold past meas and green past post; rejected for visual
  noise.
* **Main-path emphasis:** edges on the path from `Input Image` to the
  chain's terminal render with `width: 3px`; aux edges and chain edges
  *before* a fork-into-aux (none, given the single-wire rule) render at
  `width: 2px`. The main spine reads as the primary axis at a glance
  without disabling the rest.
* **Z-order:** wires render *under* blocks so a long aux wire passing under
  the main spine doesn't obscure block labels. Selected wires render *over*
  blocks so the selection state stays visible.
* **Inspector on wire-select:** selecting an edge populates the inspector
  with the edge's source/target labels, port name, and a `Disconnect`
  button (parity with the right-click action; lets keyboard-only users
  delete wires).

### 4.4 Pipeline containers (multi-step aux)

A `Pipeline` container is a sub-region with a title bar (`▼ Pipeline — <name>`)
and a purple border. The user creates one from the palette's `+ New Pipeline`
button (drag onto canvas). The title bar carries the container's editable
label — single-click to focus inspector, double-click to rename in place.

Inside the container, the same DAG rules apply: a chain of op blocks wired
left-to-right, the same validation rules, the same aux semantics for inner ops.
Every container scope is auto-seeded with an `InputImage` block rendered as
the **consumer-fed dot** on the container's inner-left edge (see §4.1); the
inner chain's first op wires its blue image-input from this dot, just as the
root chain wires from the root `Input Image`.

The container has:

* **Left edge:** outer image-input port + a small purple "consumer-fed"
  indicator that mirrors the inner `InputImage` source.
  - **Main-flow mode** — when the container's right output wires to a
    downstream **blue** image-input port. The outer left port is expected
    to be wired by an upstream block; that wire conceptually feeds the
    inner `InputImage` source. The consumer-fed dot is dimmed.
  - **Aux mode** — when the container's right output wires to a **purple**
    aux-input port. The outer left port is *not* wired — the consumer feeds
    the image at runtime, signalled by the consumer-fed dot lighting up.
  - **Mixed / inconsistent** — outer left wired *and* right wired to a
    purple aux target, **or** outer left unwired *and* right wired to a
    blue image-flow target: invalid (Rule 5 in §4.6); the container's
    border turns red.
* **Right edge:** one output port (neutral; wire color follows target).

**Container scope rule:** wires inside the container cannot reach ports outside
it, and vice versa. If a user wants to wire an outside aux into an inner op,
they must drag the source *into* the container (the container hit-tests its
bounds and adopts the dropped block; the block animates a 200ms slide into
the inner scope).

**Drag-into hit-test is innermost-wins.** When containers are nested and
multiple match the cursor's bounding-box, the *deepest* matching container
adopts the dropped block. Cytoscape's compound-parent chain is walked
outward from the cursor; the first ancestor that hit-tests positive wins.
The hit-test uses the cytoscape bounding box (graph coords), not the DOM
element rect, so panning/zooming doesn't shift the answer.

**Sibling-container moves are a single atomic dispatch.** Dragging a block
from container A directly into container B (sibling under the same parent
scope) emits one `block_reparent` with `new_parent_block_id = B`. The
dispatcher removes the block from A's nested scope, adds it to B's, and
runs the edge-orphan check across both scopes in one pass — any inner edges
that would cross either border are deleted with a single combined toast.

**Drag-out is non-destructive (snap-back).** Dragging a child block *out of*
the container's bounds is the inverse gesture, but if any of the block's
incident edges would be orphaned by the move (because their other endpoint
lives inside the container and can't follow), the drag is rejected:

* On drop, the block animates back to its pre-drag position inside the
  container.
* A toast surfaces the count + names of the edges that blocked the move:
  *"Can't move OtsuDetector out of `inoculum_preproc` — 1 inner edge would
  be orphaned. Disconnect it first."*
* If the block has no incident edges (or only edges to *other* blocks the
  user dragged out as part of a multi-select — deferred §10), the move
  proceeds: the block animates into the parent scope and its edges in
  the parent scope are kept.

This trades a tiny amount of friction for the "destructive moves require
explicit consent" property. The user can still delete the inner edges
manually (right-click → Disconnect) and then drag the block out.

**Recursion / aux-of-aux:** an inner op inside a container can itself open
aux ports, and another container (nested inside the same parent scope) can
wire into those ports. Containers nest arbitrarily; rendering uses
cytoscape's `compound` parent feature so a child block's `parent_id`
points to its enclosing container's block id. Validation recurses into each
container's `nested` scope and aggregates issues to the container's badge
when collapsed (see "Collapse" below).

**Collapse:** the title-bar chevron toggles between **expanded** (children
visible inside the bounded region) and **collapsed** (single 1-row block
with a small chain glyph indicating the inner op count). Collapsed
containers:

* keep their output port + outer ports rendered so wires are never
  visually orphaned;
* surface their inner scope's issue count as a number on the container's
  validation badge (e.g., "▣ 2 issues"); clicking the badge expands the
  container *and* pans to the first offender;
* never hide errors — a collapsed container with internal validation
  failures still shows the red border on its outer chrome.

**Container delete:** deleting a container deletes every inner block (and
their edges) atomically; users get a confirmation toast if the inner scope
is non-empty.

### 4.5 Inspector pane

The inspector keeps its current responsibilities (label edit, parameter form,
documentation collapse, preview thumbnail) with the following changes:

* **Aux ports section (new):** the popover's wired-row moves here. For each
  aux param the inspector shows:
  - name + type annotation + required/optional tag;
  - wired count;
  - for **scalar** ports, the wired block's class label + a `Disconnect`
    button + a `Drill in →` button when the wired source is a container;
  - for **list** ports, an ordered list of rows (drag-handles, badge number,
    block class label, `✕` remove) + a `+ Add empty slot` button.
* **Wire selection (new):** clicking an edge populates the inspector with a
  *Wire* card: source block label → target block.port label, edge kind
  (image / aux), and a `Disconnect` button. Replaces no-op behaviour from
  today.
* **Container selection (new):** clicking a container's title bar (not its
  body) selects the container as a block — inspector shows:
  - **Label edit** (the visible title-bar text);
  - **Pipeline name** + **description** edits — bound to the container
    scope's `BuilderScope.name` and `.desc`;
  - inner scope summary ("3 ops, 1 aux pipeline");
  - aggregated inner issue count;
  - `Drill in →` button.

  Container scopes do **not** expose `nrows` / `ncols` — those fields
  only make sense at the root scope (where the user pre-declares the
  grid for the run). A container's nested `ImagePipeline` has no images
  of its own at definition time (consumer feeds at runtime), so
  `nrows`/`ncols` on a nested scope have no useful meaning. The
  inspector suppresses them; `to_pipeline_dag` leaves both `None` on
  container-derived `ImagePipeline` instances.
* **Removed:** the popover's class palette (replaced by drag-from-palette);
  the popover's wired-row + drill-in / disconnect buttons (moved here);
  `inspector_focus_aux` and its breadcrumb-style banner; the legacy `_PARAM_SCOPE_KEY`-driven param-scope drill-down.
* **Selection model:** at most one of `{block, wire, container}` is
  selected at a time; clicking empty canvas deselects everything and
  returns the inspector to its empty-state placeholder. Selecting a
  different block carries over the inspector's scroll position so users
  who are comparing two ops don't lose their place.
* **Empty-state placeholder (nothing selected):** small intro card —
  "Drag an operation from the palette to begin." with a one-line hint
  describing the validation badge in the toolbar so a user with an
  empty canvas knows what the "0 issues" pill means.
* **Input Image inspector card:** selecting the `Input Image` block shows
  a dedicated info card (no parameter form — `Input Image` has no
  params):
  - Heading: *Input Image — pipeline source*
  - One-line description: "Every op chain starts here. The image flowing
    out of this block is whatever your run-time loader provides."
  - Buttons: `Re-layout` (re-run dagre), `Re-anchor view to Input Image`
    (pan + zoom to the source so the user can find it on a busy
    canvas).
  - No `Delete` button (the block is non-deletable per §4.1).

### 4.6 Validation rules

Six **blocking** rules (Rules 1–6) and one **advisory** hint (Rule 7) drive
the validation surface. Blocking rules disable `Run preview` and
`Save pipeline`; offenders get a red border + a small "!" badge in the
block's top-right corner. The advisory hint produces a yellow border + a
"?" badge but does *not* block; it's a "you may have meant something else"
nudge.

The toolbar shows a count badge ("3 issues, 1 hint"); hovering it lists each
entry (issues first, hints second) and clicking a row pans/zooms to the
offender.

| # | Rule | Mechanic | Offender |
| - | ---- | -------- | -------- |
| 1 | Image-flow ports have at most one wire | Per-port wire-count over `edge.kind == "image"` | The block whose output/input violates |
| 2 | All blocks reachable from `Input Image` | BFS from `Input Image` across image-flow *and* aux edges in both directions | Each unreachable block (dashed red border) |
| 3 | Required aux ports must be wired | Per aux-typed param check vs. `ParameterInfo.default` (a parameter is *required* iff its `inspect.Parameter.default is empty`) | The consumer; red ring on the empty required port |
| 4 | No cycles | Topological-sort attempt over the full edge set; cycle members reported | Every block in the strongly-connected cycle |
| 5 | Container left/right wiring consistency | Outer-left wired ⇔ right wired to blue (main-flow); outer-left unwired ⇔ right wired to purple (aux). Other combinations rejected. | The container block |
| 6 | Exactly one `Input Image` per scope | Count `InputImage` blocks in each scope | Reported as a scope-level issue; recovered automatically by auto-seed (see §4.1) on next dispatch, so this rule normally fires only when state is corrupt |
| 7 (advisory) | Stage ordering respects ops → meas → post | Walk the image-flow chain from `Input Image`; flag any edge whose source's stage is "later" than its target's (e.g. `MeasureSize → MaskRefiner`) | The *source* block of the out-of-order edge — yellow border + "?" badge. The chain still runs (`to_pipeline` partitions by `isinstance`), so this is non-blocking. |

Optional aux ports (parameter has a default) may stay empty — they fall back
to the registry default at `to_pipeline` time. Empty *required* slots on a
list-typed aux (created via `+ Add empty slot`) trigger Rule 3 just like an
unwired required scalar.

**Stage hint rationale (Rule 7).** `to_pipeline` partitions blocks into
`ops`/`meas`/`post` lists by `isinstance` — the chain order on the canvas
is *not* what executes. We chose advisory (not blocking) here because:

* The current builder is permissive about this and users rely on the
  reordering as a convenience.
* Blocking would force the user to manually order blocks in a way that
  duplicates work the runtime does for free.
* The yellow border + tooltip ("This step runs *before* the upstream
  measurement at runtime — partition order is `ops → meas → post`") tells
  the user what's happening so the eventual reorder isn't a surprise.

The toolbar count separates the two: e.g. *"2 issues, 1 hint"* — `Run
preview` is blocked only by the 2.

**Empty pipeline is valid.** A scope containing only the `Input Image` (no
ops) passes all rules — running preview is a no-op (no measurements
produced) but does not error. Saving emits an empty `ImagePipeline`.

**Validation is recursive into containers.** Each container's `nested` scope
runs **all seven rules** (six blocking + the advisory Rule 7); issues bubble
up as a single aggregate badge on the container when collapsed (see §4.4),
and surface individually on each inner block when the container is expanded.
The aggregate badge splits issues vs. hints just like the toolbar
("▣ 2 issues, 1 hint").

**Issue badge interaction:** the toolbar count badge is a `dbc.Tooltip`
target listing one row per issue (block label, rule short name, detail).
Clicking a row dispatches `scroll_to(block_id)` which pans + fit-zooms the
viewport to the offender and selects it (inspector populates). For
container-aggregated issues the click expands the container first, then
pans to the inner offender.

### 4.7 Layout

No persistence. Every state mutation re-runs a layered topological layout
(`cytoscape-dagre`) over the union of image-flow and aux edges, treating
`Input Image` (root scope) or the `consumer-fed` dot (container scopes) as
the source. Position changes animate over ~200ms (cytoscape `layout.animate:
true`) so the user can follow how a new edge or block re-arranges the chain.

Specifics:

* **Direction:** left-to-right (`rankDir: LR`). Aux blocks land below their
  consumer at the rank determined by their longest aux-path distance.
* **Containers (leaf-first algorithm).** `cytoscape-dagre` does not lay
  out compound subgraphs natively in its current version. `viewport_ops.js`
  runs a custom multi-pass algorithm:
  1. **Depth-first traversal** of every scope in the active state tree;
     visit leaf scopes (no nested containers) first.
  2. For each leaf scope, run dagre on its blocks + edges; record the
     resulting bounding-box dimensions.
  3. Walk back up: set each container's compound node's
     `width`/`height` style to its inner bounding box + padding (32px
     all sides). Cytoscape's compound layout will treat the container
     as a fixed-size atom on the next outer pass.
  4. Run dagre on the next-outer scope; compounds are now sized
     correctly and laid out as ordinary nodes.
  5. Repeat until the root scope is laid out.
  Total complexity: O(blocks + edges) per scope × number of scopes.
  For PhenoTypic pipelines (a few dozen blocks total across nested
  scopes), the per-render cost is ≤ 20ms in practice.
* **Manual drag is ephemeral.** Users can drag any block to nudge it for
  comparison; the new position is *not* committed to state. The next
  mutation triggers a re-layout that snaps back to dagre's positions.
  This is a deliberate tradeoff — we lose "pin my custom layout" in
  exchange for layout determinism and no `pipeline.json` schema bloat.
  Reconsidered in §10 if user feedback wants persistence.
* **Re-layout button:** toolbar's `Reset view` is relabelled `Re-layout`
  and re-runs dagre + fits to viewport. Useful after heavy interaction has
  left the user pan/zoomed away from the dagre-natural anchor.
* **Layout failure recovery:** if dagre throws (e.g., on a 1-block scope
  with no edges) the layout falls back to a centred `preset` at
  `(centerX, centerY)`. Empty containers (no inner ops) render as a
  small "+ drop ops here" placeholder.

### 4.8 Palette → canvas

Native HTML5 drag-and-drop. Each palette button is `draggable="true"`; the
canvas wrapper handles `dragover` / `drop`. On drop:

1. Convert browser screen coords (`event.clientX`/`clientY`) → cytoscape
   graph coords. The cytoscape JS API exposes pan offset and zoom factor on
   the cy *instance* (not on elements), so the conversion is explicit:
   ```js
   const cyRect = cyContainer.getBoundingClientRect();
   const renderedX = event.clientX - cyRect.left;
   const renderedY = event.clientY - cyRect.top;
   const graphX = (renderedX - cy.pan().x) / cy.zoom();
   const graphY = (renderedY - cy.pan().y) / cy.zoom();
   ```
   (`cy.renderedPosition()` is an *element* method that returns an
   element's rendered position; using it on the cy instance silently
   yields wrong coords.)
2. Mint a new `BlockNode` of the dragged class in the current scope (or in
   a container's nested scope if the drop point hit-tests inside the
   container's bounds).
3. Dispatch a state update; the next render places the block at the drop
   point (rounded to the dagre grid) and re-runs layout from there.

A ghost element (the dragged block's outline) follows the cursor mid-drag
via the standard `DataTransfer.setDragImage` API.

**Drop targets and edge cases:**

* **Empty canvas region:** block lands at the drop coords; dagre re-lays
  on the next mutation.
* **Inside a container's bounds:** block is adopted into the container's
  inner scope; the container's outer shell highlights during dragover.
  Nested containers chain — drop inside an inner container nests further.
* **On an existing block:** block lands adjacent to the right (offset by
  one dagre node width) so the user never accidentally overlays blocks.
* **On a wire:** **positional only** — the block lands at the drop coords
  adjacent to the wire; the wire is *not* split or modified. The wire is
  selected as a side-effect (subtle highlight). A toast hints *"Drop on
  input/output ports to connect"* so the user knows wires aren't drop
  targets. Wire-insertion (A→C becomes A→new→C) is deferred to v2 — the
  semantics are too dependent on the new block's port shape to ship as
  a default.
* **Outside the cytoscape slot:** drag silently cancels; no toast.
* **`Esc` mid-drag:** browser cancels the dragend event; we observe
  `dragend` with `dataTransfer.dropEffect === "none"` and emit nothing.

**`+ New Pipeline`** palette button drops an empty pipeline container at
the cursor — the container's nested scope is auto-seeded with its
`InputImage`/consumer-fed source as described in §4.1.

**Keyboard fallback:** palette buttons remain `Tab`-focusable. `Enter` /
`Space` on a focused button dispatches `block_create` with `(x, y)` set
to the centre of the current viewport. Useful for screen-reader users and
trackpad-averse keyboard pilots; not the headline mechanic.

**Input Image cannot be palette-dragged:** the palette has no `+ Input
Image` button (each scope auto-seeds exactly one — see §4.1). The
dispatcher rejects any `block_create` whose class is the `InputImage`
sentinel as a guard.

### 4.9 Interaction reference

A single table covering every gesture the user can perform on the canvas.
Each row maps to a dispatch kind in §5.6 (or a no-op for purely visual
state).

| Gesture | Outcome | Dispatch kind |
| ------- | ------- | ------------- |
| Drag palette button → drop on empty canvas | New block at cursor coords | `block_create` |
| Drag palette button → drop inside container | New block adopted into the container's nested scope | `block_create` (with `container_block_id`) |
| Drag palette button → drop on existing block | New block placed adjacent (right offset) | `block_create` |
| Drag palette button → drop on wire | Wire selected (no connection); block lands at cursor | `block_create` + `wire_select` |
| Drag palette button → drop outside cytoscape slot | Cancel silently | — |
| `Tab` to palette button → `Enter` / `Space` | New block at viewport centre | `block_create` |
| Drag output port → drop on compatible input port | Wire created (colour = target type) | `edge_create` |
| Drag output port → drop on incompatible port | Wire fades; rejected port flashes red briefly | — |
| Drag output port → drop on empty canvas | Wire fades; no state change | — |
| `Esc` while wire-dragging | Wire fades; no state change | — |
| Click wire → `Delete` → drag new wire from same source port | Two-step "move the wire's target" gesture (no native endpoint-drag — see §4.3) | `edge_delete` then `edge_create` |
| Drag wired edge's endpoint to empty canvas | *Not a supported gesture* — see two-step above. Cytoscape doesn't expose grabbable endpoints; we don't bundle `cytoscape-edgehandles`. | — |
| Click a wire | Wire selected (stroke widens + brightens); inspector shows wire card | `wire_select` |
| `Delete` / `Backspace` with wire selected | Wire deleted | `edge_delete` |
| Right-click wire → *Disconnect* | Wire deleted | `edge_delete` |
| Click a block | Block selected; inspector shows block | `block_select` |
| `Delete` / `Backspace` with block selected (non-`InputImage`) | Block + its edges deleted | `block_delete` |
| `Delete` on selected `InputImage` block | No-op + toast: "Input Image cannot be removed" | — |
| Drag block within current scope | Ephemeral position; not committed; next mutation re-lays | — |
| Drag block over container's bounds | Container outer chrome highlights; on release block is adopted | `block_reparent` |
| Drag child block out of container's bounds | Block pops to parent scope; cross-border edges deleted with toast | `block_reparent` |
| Single-click container title bar | Container selected as block (inspector shows label / inner summary) | `block_select` |
| Double-click container title bar | Inline label edit | `block_label_update` |
| Click container title-bar chevron | Toggle collapsed / expanded | `block_collapsed_toggle` |
| Double-click container body | Drill into the container's nested scope (push breadcrumb) | `drill_into_container` |
| Click breadcrumb segment | Drill out to that scope | `drill_out` |
| Click toolbar `Re-layout` | Re-run dagre and fit to viewport | — |
| Click toolbar issue badge → issue row | Pan + fit to offender; select it; expand its container chain if needed | `scroll_to` |
| Click `Run preview` | Validate; if clean, run preview; else toast first issue | (gated by `validate`) |
| Click `Save pipeline` | Validate; if clean, emit `pipeline.json`; else toast first issue | (gated by `validate`) |
| `Pan` / `Zoom` (mouse-wheel / pinch) | Viewport adjusts; positions unchanged | — |
| Right-click on empty canvas | Context menu: `Re-layout`, `Fit to view` | — (v1: no `Paste` / `Select all` — both depend on v2 multi-select) |
| Right-click on block | Context menu: `Disconnect all`, `Delete`. (Duplicate deferred to v2.) | (dispatches each) |
| Drag of `InputImage` block | Rejected: cursor doesn't grab; no drag begins | — (defense in depth: dispatcher rejects any `block_reparent` / `block_delete` targeting an `InputImage` block_id) |

### 4.10 Edge cases & recovery summary

This subsection consolidates corner cases that span multiple subsystems so
they live in one canonical place. Each row names the trigger, the user-
visible effect, and the recovery (or guard).

| Edge case | User effect | Recovery / guard |
| --------- | ----------- | ---------------- |
| `Input Image` missing from scope (corrupt state / mid-migration) | Validation Rule 6 fires (red scope-level issue) | Dispatcher auto-seeds a fresh `InputImage` on first state-load pass; user sees a one-shot toast: "Input Image restored" |
| User attempts to drag `+ Input Image` (impossible per palette gating) | N/A — palette has no button | Dispatcher also rejects programmatic `block_create` for `InputImage` as a defense-in-depth guard |
| Image-flow cycle (Rule 4) | All blocks in cycle get red border; toolbar shows "cycle of N blocks" | User must break the cycle by deleting one wire; cycle re-detected after every `edge_*` dispatch |
| Aux cycle (e.g. A's aux ← B; B's aux ← A) | Same as image cycle — Rule 4 applies to combined edge graph | Same recovery |
| Container left wired *and* right wired to purple (mixed mode) | Rule 5 fires; container border red | User must remove one of the two wires |
| Empty container (no inner ops) | Container shows "+ drop ops here" placeholder; not an error | Optional aux: passes Rule 3 (uses default `ImagePipeline()`). Required aux: Rule 3 is satisfied by the wire's *presence* — but the runtime may raise on an empty pipeline when invoked. See §10 open question. |
| Empty scope (root scope has only `InputImage`) | All rules pass; preview is a no-op; save emits `ImagePipeline(pipe_cfgs=[])` | Allowed; the toolbar shows "Pipeline is empty" hint but does not block |
| Deleting a container with children | Toast: "Delete container and its N inner blocks?" — confirm to proceed | If confirmed, atomically removes the container and every edge that referenced any inner block |
| List aux slot reordered while disconnected wire-drag in flight | Drag completes first (snapping to the slot under cursor at release); reorder applies after | Single fan-in callback orders dispatches by trigger timestamp |
| Loading a `pipeline.json` saved by a pre-redesign release | One-shot dagre layout on first open; auto-seeds `InputImage` at scope head | `from_pipeline_dag` handles the conversion (see §5.4); no migration script required |
| Loading a `pipeline.json` with unknown op classes (registry drift) | Block renders with a yellow border + label "(unknown: ClassName)" | Block treated as Rule 1/2/3-eligible based on stored aux_ports map (best effort); user can delete + replace |
| Network/clientside JS fails to load (e.g. `wire_drawing.js`) | Canvas renders but wires can't be drawn; toolbar shows "Wire drawing offline" banner | Static `Reload page` button; server-side state remains valid; existing wires render |
| Layout algorithm throws (degenerate scope) | Falls back to centred `preset` at viewport centre | No issue raised; next state mutation tries dagre again |
| Container border crossed by a wire ending in mid-drag | Wire endpoint snaps to nearest legal port within the same scope; otherwise cancels | Same as drop-on-empty-canvas |
| Double-click a leaf op (not a container) | No-op (drill-in is container-only) | Toast hint: "Drill in is only available for `Pipeline` containers" |
| `Run preview` clicked while validation is in flight | Button is disabled until validation settles (sub-100ms; one-frame debounce) | UI shows brief spinner |
| Concurrent dispatches (e.g. quick palette-drag + wire-drag) | Single fan-in callback serializes them via `dash.callback_context.triggered_id` | Each dispatch produces a deterministic new state; no merge conflicts |

## 5. Architectural model

### 5.1 State model (new)

The data model moves from "linear list of nodes with embedded aux_ports"
to "graph of blocks + typed edges per scope."

```python
@dataclass
class BlockNode:
    """One canvas block — single op, container, or Input Image source."""

    block_id: str                    # uuid.uuid4().hex (32 chars).
                                     # 8-char hex was used by the old model;
                                     # full UUID prevents collisions across
                                     # nested scopes.
    class_name: str                  # registry key, or "ImagePipeline" sentinel,
                                     # or "InputImage" sentinel
    params: Dict[str, Any]           # scalar params only (no op-typed; those
                                     # come from edges now)
    label: Optional[str] = None
    nested: Optional["BuilderScope"] = None   # container body when class_name
                                              # == "ImagePipeline"
    collapsed: bool = False          # container-only: True hides children
    list_slot_counts: Dict[str, int] = field(default_factory=dict)
                                     # per list-aux param name, the total slot
                                     # count for layout. Empty slots = slot
                                     # positions in [0, count) not covered by
                                     # any Edge. Increments on
                                     # ``list_aux_add_empty_slot`` /
                                     # ``edge_create`` to a list port; never
                                     # decrements except via slot delete.


@dataclass
class Edge:
    """One wire between two block ports within a scope."""

    edge_id: str
    source_block_id: str             # always a real block_id; empty list-aux
                                     # slots are tracked separately via
                                     # ``BlockNode.list_slot_counts``.
    source_port: str = "out"         # blocks have a single output port today;
                                     # field kept for future multi-output ops.
    target_block_id: str
    target_port: str                 # "in" for image, "<param>" for aux
                                     # (scalar or list — list slot index is in
                                     # ``target_slot``).
    target_slot: Optional[int] = None
                                     # list-aux slot index (0-based); None for
                                     # scalar aux and for image-flow.
    kind: Literal["image", "aux"]    # set when the edge is created; redundant
                                     # with target_port semantics but kept
                                     # explicit for fast validation.


@dataclass
class BuilderScope:
    """A DAG of blocks + edges (per scope)."""

    blocks: List[BlockNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    name: str = "Pipeline"
    desc: str = ""
    nrows: Optional[int] = None
    ncols: Optional[int] = None


@dataclass
class BuilderState:
    root: BuilderScope = field(default_factory=BuilderScope)
    breadcrumb: List[str] = field(default_factory=list)   # block_ids of
                                                          # nested containers
    selected_block_id: Optional[str] = None               # also used for
                                                          # container selection
                                                          # (a container IS a
                                                          # BlockNode — there
                                                          # is no separate
                                                          # selected_container_id;
                                                          # the inspector renders
                                                          # a different card
                                                          # based on
                                                          # block.class_name)
    selected_edge_id: Optional[str] = None
    pending_delete_block_id: Optional[str] = None         # set by
                                                          # block_delete_request
                                                          # for non-empty
                                                          # containers; drives
                                                          # CONFIRM_DELETE_MODAL
                                                          # visibility; cleared
                                                          # on Confirm/Cancel.
    toast_queue: List[Dict[str, Any]] = field(default_factory=list)
                                                          # FIFO queue of toast
                                                          # payloads. Toast
                                                          # policy: one visible
                                                          # at a time, 3000ms
                                                          # auto-dismiss,
                                                          # dismissable on
                                                          # user click. The
                                                          # Dash callback that
                                                          # binds to
                                                          # TOAST_NOTIFICATION
                                                          # pops the head on
                                                          # each dismiss /
                                                          # timeout. Two
                                                          # near-simultaneous
                                                          # mutations queue
                                                          # rather than race.


# Sentinel class names
INPUT_IMAGE_CLASS_NAME = "InputImage"
PIPELINE_CLASS_NAME = "ImagePipeline"   # already exists today


def _seed_input_image(scope: BuilderScope) -> None:
    """Idempotently add an InputImage block at the head of *scope*.

    Called from BuilderScope.__post_init__ and any deserialisation path
    so every scope (root or nested) guarantees Rule 6.
    """
    if any(b.class_name == INPUT_IMAGE_CLASS_NAME for b in scope.blocks):
        return
    scope.blocks.insert(0, BlockNode(
        block_id=_new_block_id(),
        class_name=INPUT_IMAGE_CLASS_NAME,
        params={},
        label=None,
    ))
```

**Key differences vs. today's `_state.py`:**

* No `aux_ports: Dict[str, List[Optional[StepNode]]]` on blocks — aux is now
  represented as edges targeting `<param>` / `<param>[<i>]` ports.
* No `_PARAM_SCOPE_KEY` legacy machinery — drill-in for nested pipelines uses
  `breadcrumb=[<container_block_id>]` exclusively.
* No `inspector_focus_aux` — selecting an aux block focuses it like any other.
* `Edge` is first-class; today's image-flow edges are implicit (`prev_id →
  next_id` derived from `scope.nodes` order). The DAG model needs explicit
  edges because order is not implicit.
* `InputImage` is a sentinel class name (analogous to today's
  `PIPELINE_CLASS_NAME`). It carries no parameters; renders as a green
  chevron at the root scope and a small "consumer-fed" dot inside container
  scopes (see §4.1). `_seed_input_image` runs on every fresh scope (root
  scope at session start, nested scope on container creation) and on every
  `state_from_json` load — the function is idempotent so repeated calls
  are safe.

**Linear-order recovery:** `to_pipeline(scope)` walks the image-flow edges
from the `InputImage` block to compute a topological order over the chain;
inside that order, blocks are partitioned by `isinstance` into
`ops`/`meas`/`post` lists (same as today's `to_pipeline`). Aux edges resolve
into op markers folded back into the consumer's `params` dict at
`registry.create_instance` time (same shape as today's
`_fold_aux_ports_for_node`, just sourced from `edges` instead of
`aux_ports`).

### 5.2 ID encoding

A wire's target port carries both the param name and (for list aux) the slot:

* Image-in: `target_port = "in"`
* Scalar aux for param `X`: `target_port = "X"`
* List aux for param `X`, slot `i`: `target_port = "X[i]"`

This matches the current `_encode_aux_port_id` shape so the `aux_popover.js`
prefix scheme is easy to adapt to the new format — though the JS file goes
away (see §5.5).

### 5.3 Validation

Validation is a pure function `validate(scope: BuilderScope) -> List[Issue]`
that runs on every state change and gates `Run preview` / `Save`.

```python
@dataclass
class Issue:
    kind: Literal[
        # Blocking — disable preview/save, red border
        "fork",                # Rule 1
        "stub",                # Rule 2
        "required_aux",        # Rule 3
        "cycle",               # Rule 4
        "container_mode",      # Rule 5
        "missing_input",       # Rule 6
        "duplicate_input",     # Rule 6 (extra Input Image)
        # Advisory — do NOT block preview/save, yellow border
        "stage_order_hint",    # Rule 7 (ops → meas → post hint)
        "unknown_class",       # advisory: class not in registry (legacy file)
    ]
    block_id: Optional[str]    # None for scope-level issues
    detail: str
    scope_path: List[str] = field(default_factory=list)   # block_ids walked
                                                          # from root to the
                                                          # offender's scope
    severity: Literal["error", "advisory"] = "error"      # populated when the
                                                          # Issue is minted;
                                                          # gates Run/Save
                                                          # via filter
```

Implementation sketch (Rules 1–3 shown; Rules 4–6 follow the same pattern;
container scopes recurse via `_validate_scope`):

```python
def validate(state: BuilderState) -> List[Issue]:
    """Run validation on every scope reachable from the root and return
    issues with scope_path so the UI can pan/zoom across container
    boundaries.
    """
    return _validate_scope(state.root, scope_path=[])


def _validate_scope(scope: BuilderScope, scope_path: List[str]) -> List[Issue]:
    issues: List[Issue] = []
    registry = get_registry()

    # Rule 6 — exactly one Input Image
    input_blocks = [b for b in scope.blocks
                    if b.class_name == INPUT_IMAGE_CLASS_NAME]
    if not input_blocks:
        issues.append(Issue("missing_input", None,
                            "scope has no Input Image", scope_path))
    elif len(input_blocks) > 1:
        for extra in input_blocks[1:]:
            issues.append(Issue("duplicate_input", extra.block_id,
                                "extra Input Image", scope_path))
    root_id = input_blocks[0].block_id if input_blocks else None

    # Rule 1 — image-flow forks
    out_count: dict[str, int] = defaultdict(int)
    in_count: dict[tuple[str, str], int] = defaultdict(int)
    for edge in scope.edges:
        if edge.kind == "image":
            out_count[edge.source_block_id] += 1
            in_count[(edge.target_block_id, edge.target_port)] += 1
    for block_id, n in out_count.items():
        if n > 1:
            issues.append(Issue("fork", block_id,
                                "image-out has >1 wire", scope_path))
    for (block_id, port), n in in_count.items():
        if n > 1 and port == "in":
            issues.append(Issue("fork", block_id,
                                "image-in has >1 wire", scope_path))

    # Rule 2 — stubs (BFS over both edge kinds, both directions)
    reachable: set[str] = set()
    if root_id is not None:
        frontier = [root_id]
        while frontier:
            curr = frontier.pop()
            if curr in reachable:
                continue
            reachable.add(curr)
            for edge in scope.edges:
                if edge.source_block_id == curr:
                    frontier.append(edge.target_block_id)
                if edge.target_block_id == curr and edge.kind == "aux":
                    frontier.append(edge.source_block_id)
    for block in scope.blocks:
        if block.block_id not in reachable:
            issues.append(Issue("stub", block.block_id,
                                "not reachable from Input Image",
                                scope_path))

    # Rule 3 — required aux ports must be wired.
    # IMPORTANT: ``ParamInfo.default`` is normalised to ``None`` by the
    # registry (line 404 of ``_operation_registry.py``); the right
    # predicate is ``not p.has_default``, NOT ``p.default is
    # inspect.Parameter.empty``. The latter is always False on a ParamInfo
    # instance and would silently disable Rule 3.
    aux_wired: dict[tuple[str, str], int] = defaultdict(int)
    for edge in scope.edges:
        if edge.kind == "aux":
            aux_wired[(edge.target_block_id, edge.target_port)] += 1
    for block in scope.blocks:
        info = registry.get(block.class_name)
        if info is None:
            issues.append(Issue("unknown_class", block.block_id,
                                f"class '{block.class_name}' not in registry",
                                scope_path, severity="advisory"))
            continue
        for param_name, p in info.parameters.items():
            if not (p.is_operation or p.is_pipeline):
                continue
            if not p.has_default:
                if aux_wired[(block.block_id, param_name)] == 0:
                    issues.append(Issue("required_aux", block.block_id,
                                        f"{param_name} is required",
                                        scope_path))

    # Rule 4 — cycle detection over ALL edges (image + aux)
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in scope.edges:
        adjacency[edge.source_block_id].append(edge.target_block_id)
    cycle_members = _find_cycle_nodes(adjacency)   # Tarjan or simple DFS
    for block_id in cycle_members:
        issues.append(Issue("cycle", block_id,
                            "block participates in a cycle", scope_path))

    # Rule 5 — container left/right wiring consistency
    for block in scope.blocks:
        if block.class_name != PIPELINE_CLASS_NAME:
            continue
        left_wired = any(e.target_block_id == block.block_id
                         and e.target_port == "in"
                         and e.kind == "image"
                         for e in scope.edges)
        # `out` edges' kind tells us blue vs purple destination
        right_kinds = {e.kind for e in scope.edges
                       if e.source_block_id == block.block_id}
        if left_wired and "aux" in right_kinds:
            issues.append(Issue("container_mode", block.block_id,
                                "left wired but right wires to aux",
                                scope_path))
        if not left_wired and "image" in right_kinds:
            issues.append(Issue("container_mode", block.block_id,
                                "right wires to image but left is unwired",
                                scope_path))

    # Rule 7 (advisory) — stage ordering. Walk image-flow chain from
    # InputImage; flag edges where source stage is later than target.
    # Runs per scope so nested pipeline containers report their own
    # ordering hints (the runtime partitions every scope independently).
    if root_id is not None:
        stage_order = {"ops": 0, "meas": 1, "post": 2, "pipeline": 0}
        order_of: dict[str, int] = {}
        for block in scope.blocks:
            order_of[block.block_id] = stage_order.get(
                _safe_stage(block.class_name), 0,
            )
        for edge in scope.edges:
            if edge.kind != "image":
                continue
            src = order_of.get(edge.source_block_id)
            tgt = order_of.get(edge.target_block_id)
            if src is not None and tgt is not None and src > tgt:
                issues.append(Issue("stage_order_hint", edge.source_block_id,
                                    "runs in a later stage than its "
                                    "downstream block; runtime partitions "
                                    "by isinstance.",
                                    scope_path, severity="advisory"))

    # Recurse into containers
    for block in scope.blocks:
        if block.nested is not None:
            issues.extend(_validate_scope(
                block.nested, scope_path=[*scope_path, block.block_id],
            ))
    return issues
```

**Auto-recovery for `missing_input`:** the dispatcher's pre-mutation pass
calls `_seed_input_image(scope)` on every scope reachable from `state.root`
(idempotent). The `missing_input` rule therefore fires only when the
dispatcher hasn't run yet (transient: between deserialisation and the
first dispatch). When it does fire, the next dispatch heals it
automatically.

**Empty-pipeline acceptance:** a scope whose only block is `Input Image`
passes all six rules — `validate` returns `[]`. The toolbar shows a
"Pipeline is empty" caption but the buttons remain enabled.

**Required aux + empty container:** an empty container wired into a required
aux port satisfies Rule 3 by being present (the wire exists), but
`to_pipeline` materialises an `ImagePipeline(pipe_cfgs=[])`. Whether that is
semantically a valid value for a required aux is class-dependent and
delegated to the runtime (some ops may raise on empty pipelines). See
§10 for the open question.

### 5.4 Conversion: DAG ↔ ImagePipeline

* **`to_pipeline_dag(scope) -> ImagePipeline`:** topologically sort
  `scope.blocks` via the image-flow edges (linear by Rule 1, so a stable
  order falls out trivially). Skip the `InputImage` sentinel. For each
  remaining block, fold any aux edges that target it into a `params` dict —
  for each aux edge, resolve the source block recursively
  (`_block_to_marker`) and emit a marker shaped like today's `{"__type__":
  "operation", "class_name": ..., "params": {...}}` (or `"pipeline"` for
  container sources). Partition by `MeasureFeatures` / `PostMeasurement`
  into ops/meas/post.

  **Precondition contract — raises on invalid state.** `to_pipeline_dag`
  calls `validate(state)` filtered to *blocking* issues (severity =
  `error`) first. If any are present, it raises a `ValueError` listing the
  rule kinds + offender block_ids:

  ```python
  raise ValueError(
      "Cannot materialise pipeline: 2 validation error(s) — "
      "fork(BlurGauss), required_aux(FilamentousFungiDetector)"
  )
  ```

  This guarantees: callers either pass clean state and get a pipeline, or
  get a deterministic exception. The single fan-in callback's
  `request_run_preview` / `request_save_pipeline` paths catch this and
  surface as a toast pointing at the first offender (same machinery as
  the issue badge click). Tests in
  `tests/unit/gui/builder/test_state_dag.py::test_to_pipeline_raises_on_invalid`
  cover every blocking rule.

  **Aux-only blocks (no image-flow edges)** never appear in the
  topological ordering — they're folded into a consumer's `params` via
  `_block_to_marker` recursion. The blocks list contains them so the
  canvas renders them; `to_pipeline_dag` walks `edges` (not `blocks`) to
  drive the ops/meas/post lists.

* **`from_pipeline_dag(pipeline) -> BuilderScope`:** materialise the
  inverse. Implementation:

  1. Construct an empty `BuilderScope`; `BuilderScope.__post_init__`
     auto-seeds an `InputImage` block via `_seed_input_image` (always —
     never prepend one manually). Idempotency guarantees that loading a
     scope which already has an `InputImage` does not produce a second one.
     **For the root scope** (depth 0 of the recursion), copy
     `pipeline.name`, `pipeline._desc`, `pipeline.nrows`, `pipeline.ncols`
     onto the scope's matching fields so the grid dimensions and label
     survive the round-trip. For **container scopes** (nested recursion),
     copy `name` and `desc` but leave `nrows`/`ncols` as `None` per the
     §4.5 rule that container scopes have no grid of their own.
  2. Walk `pipeline.get_ops() + get_meas() + get_post()` minting one block
     per entry; add image-flow edges between consecutive entries (and from
     `InputImage` to the first entry).
  3. **Shared-instance dedup.** Maintain a `dict[id(op), block_id]` map.
     When walking aux params (step 4) and the source is an op whose `id()`
     already exists in the map *and* whose existing block_id is already
     wired (its single outgoing wire is in `edges`), **clone the op into
     a fresh `BlockNode`** (`copy.deepcopy(op)` and a new block_id) and
     surface a toast:
     *"Loaded N shared operation instance(s) as independent copies. The
     GUI does not support sharing the same operation between two
     consumers."*
     This preserves the user's intent (each location keeps its op) while
     enforcing the no-shared-instance rule from §3 going forward.
     Unit-tested in
     `tests/unit/gui/builder/test_state_dag.py::test_from_pipeline_clones_shared_aux`.
  4. For each block, walk its op-typed parameters, mint embedded aux
     blocks recursively (with the dedup map), and add an aux edge from
     each source block to the consumer's `target_port` (plus
     `target_slot=i` for list-aux entries; `target_slot=None` for scalar).
  5. For each consumer's list-aux params, set `block.list_slot_counts[port]
     = max_index_in_edges + 1 + count_of_pure_None_slots_in_source` so
     empty slots interspersed in the original list survive the round-trip.

  Legacy `pipeline.json` files (saved by the popover-era builder) flow
  through this same `from_pipeline_dag` — there is no separate "migration"
  path. The dedup and seed steps are the migration.

### 5.5 Assets removed / added

**Removed:**

* `assets/aux_popover.js` — the clientside popover glue
* `assets/cytoscape-popper.min.js`
* `assets/popperjs-core.min.js`
* (CSS) `.cy-popover-*` rules in `assets/builder.css`

**Added:**

* `assets/wire_drawing.js` — port mousedown handlers; mounts a live-wire
  SVG following the cursor; dispatches `edge_create` / `edge_delete`
  payloads into `STORE_EDGE_EVENT`. Owns *only* wire/port interaction
  state; does not manage viewport.
* `assets/palette_dnd.js` — native HTML5 `dragstart` on each palette
  button + `dragover` / `drop` on the cytoscape wrapper. Converts coords
  with the explicit `cy.pan()` / `cy.zoom()` formula in §4.8. Dispatches
  to `STORE_PALETTE_DROP`.
* `assets/viewport_ops.js` — viewport-level operations: `scroll_to`
  (expand-chain via chained `cy.promiseOn("layoutstop")`), `relayout`,
  `Re-anchor view`. Kept separate from `wire_drawing.js` so the wire
  state machine doesn't intermix with viewport animations. Owns its
  own progress scrim (see §5.6 `scroll_to`).
* (Dependency, vendored) `cytoscape-dagre@2.x` (pinned to `^2.5.0` in
  `package.json`; vendored as `assets/cytoscape-dagre.min.js` to avoid
  a build step). Used by `viewport_ops.js` for the per-scope dagre
  passes (see §4.7).
* (CSS) `.dag-block`, `.dag-port`, `.dag-wire`, `.dag-issue`,
  `.dag-scrim` rules in `assets/builder.css`.

**Modified:** `assets/builder.js` keeps the cytoscape instance accessor
(`window.phenoGetCy`) but loses the popover-related polling. Adds a
flag mirror: at server boot, `app.index_string` injects
`window.phenotypicGuiDag = <true|false>` so all three JS files can
no-op cleanly when the legacy path is active.

**Asset load order** (declared via Dash's deterministic `assets/` glob
ordering — filenames are alphabetised, so the natural order is
`builder.css`, `builder.js`, `cytoscape-dagre.min.js`,
`palette_dnd.js`, `viewport_ops.js`, `wire_drawing.js`). Each JS file
is wrapped in an IIFE that polls for `window.phenoGetCy()` returning
a fresh instance (same pattern as today's `aux_popover.js`) so order
of execution doesn't matter — whichever loads first waits for cy.

#### Clientside event contract

Each JS file emits structured payloads into `dcc.Store` components.
Playwright tests assert against these payloads (via `page.evaluate(
() => window.dash_clientside.callback_context.states[...])`) and
against DOM testids. The contract:

| Source JS | Store ID | Payload schema |
| --------- | -------- | -------------- |
| `palette_dnd.js` | `STORE_PALETTE_DROP` | `{kind: "block_create", class_name: str, x: float, y: float, container_block_id: str \| null, ts: int}` |
| `wire_drawing.js` | `STORE_EDGE_EVENT` | `{kind: "edge_create" \| "edge_delete", source_block_id?: str, target_block_id?: str, target_port?: str, edge_id?: str, ts: int}` |
| `viewport_ops.js` | `STORE_VIEWPORT_OP` | `{kind: "scroll_to" \| "relayout" \| "reanchor" \| "drill_to_scope" \| "block_collapsed_toggle", block_id?: str, scope_path?: list[str], target_breadcrumb?: list[str], ts: int}` — the `block_collapsed_toggle` + `drill_to_scope` kinds are dispatched by `viewport_ops.js` itself as part of the `scroll_to` expand chain (see §5.6); the fan-in callback routes the same store payload either way. |

`ts` is a `Date.now()` timestamp so the fan-in callback can resolve
ordering when two stores update in the same Dash tick.

**DOM test-ID convention** (`assets/builder.css` adds matching
selectors so Playwright can target without relying on cytoscape's
internal class names):

| Element | Test ID |
| ------- | ------- |
| Live wire SVG during drag | `data-testid="live-wire"`, `data-state="dragging"` |
| Live wire (settled — just before fade) | `data-state="settled"` |
| Wire fade-out container | `data-testid="wire-cancel-anim"` |
| Palette ghost element during drag | `data-testid="palette-ghost"` |
| Container scrim during `scroll_to` chain | `data-testid="dag-scrim"` |
| Re-layout button | `data-testid="btn-relayout"` |
| Issue badge in toolbar | `data-testid="issue-badge"` |
| Issue row inside the badge tooltip | `data-testid="issue-row"`, `data-rule="<rule-kind>"` |
| Toast | `data-testid="dag-toast"`, `data-kind="<toast-kind>"` |

**Custom DOM events** (`document.dispatchEvent(new CustomEvent(...))`)
that Playwright awaits with `page.waitForEvent`:

| Event name | Fired by | Detail |
| ---------- | -------- | ------ |
| `phenotypic:wire-drop` | `wire_drawing.js` | `{accepted: bool, kind: "image" \| "aux"}` |
| `phenotypic:palette-drop` | `palette_dnd.js` | `{class_name: str, accepted: bool}` |
| `phenotypic:scroll-to-complete` | `viewport_ops.js` | `{block_id: str}` after the final `layoutstop` of the chain |
| `phenotypic:relayout-complete` | `viewport_ops.js` | `{}` |

**`accepts` data attribute (type-aware highlight).** Each aux-port
cytoscape sub-node carries `data.accepts: List[str]` — the list of
registry class names whose annotation is compatible with that port.
Computed server-side from `OperationInfo.parameters[port].annotation`
during `build_canvas_elements_dag` and injected into the element's
`data` dict; `wire_drawing.js` reads it on dragstart to decide which
ports glow vs. dim. Per-port (not per-block) so a consumer with two
differently-typed aux ports (e.g. `inoculum_detector: Detector` and
`shape_filter: ColonyFilter`) lights up correctly.

**`accepts` resolution rules** — how `build_canvas_elements_dag`
materialises the list across PEP 604 / typing constructs:

* **Plain annotation** `T` (e.g. `Detector`) → every registry class
  whose `cls` is a subclass of `T`. Pipeline included when `T` is
  `ImageOperation` (since `ImagePipeline` *is* an `ImageOperation`).
* **`Annotated[T, ...]`** → unwrap to `T`; metadata is ignored for
  highlight purposes (it's still consulted by `OperationInfo` for
  `ColumnRef` etc).
* **`Union[A, B]` / `A | B`** → union of the per-arm accept sets. If
  `None` is in the union (i.e. `Optional[T]`), drop `None` from the
  union before computing — `None` corresponds to "leave the port
  unwired and use the default"; it doesn't add anything to `accepts`.
* **`List[T]` / `list[T]`** → handled identically to scalar `T` for
  `accepts` purposes; list-ness is conveyed by the port's
  `is_list=True` flag, not by which classes are accepted.
* **`is_operation=True` AND `is_pipeline=True`** (rare; happens when
  the annotation is `ImageOperation` itself, which is satisfied by
  every op *and* by `ImagePipeline`) → emit the full set of
  registered operation classes plus `"ImagePipeline"` sentinel.
* **Forward references / string annotations** that `OperationInfo`
  resolves at registry-build time produce a real type; if they fail
  to resolve, the port is rendered with `accepts: []` and dims all
  drag sources — surfaces as an advisory `unknown_class` Issue
  rather than crashing the canvas.
* **`ColumnRef` / scalar non-op params** never become aux ports — they
  render as inspector form fields. `accepts` is not emitted for them.

### 5.6 Callbacks

A single fan-in mutation callback (today's pattern in `_callbacks.py`) takes
all triggers and dispatches via `_dispatch_state_update(state, kind, payload)`.
Every dispatch:

1. Pre-mutation: runs `_seed_input_image` on every reachable scope
   (idempotent guard for Rule 6).
2. Mutation: applies `kind` to a deep-copied scope.
3. Post-mutation: runs `validate(state)`; the resulting `List[Issue]` lands
   in `STORE_ISSUES`, drives the toolbar count badge, and gates Run/Save.

**New dispatch kinds:**

| Kind | Payload | Notes |
| ---- | ------- | ----- |
| `block_create` | `{class_name, x, y, container_block_id?}` | Rejects `class_name == "InputImage"`. If `container_block_id` resolves, the block is appended to that container's `nested.blocks` (innermost-wins hit-test). |
| `block_delete_request` | `{block_id}` | First stage. Rejects `InputImage` block_ids. **Empty-container threshold:** a container is considered "empty" iff its `nested.blocks` contains only the auto-seeded `InputImage` sentinel (the count of non-`InputImage` blocks is zero). For an empty container or any non-container block, immediately delegates to `block_delete_confirm` in the same dispatch. For a container with *at least one non-`InputImage` block*, writes `state.pending_delete_block_id = block_id` which shows the confirm modal (`CONFIRM_DELETE_MODAL_ID` — see §6); the modal body shows "Delete container `<label>` and its N inner block(s)?" where N excludes the `InputImage`. Confirm button dispatches `block_delete_confirm`; Cancel clears `pending_delete_block_id`. |
| `block_delete_confirm` | `{block_id}` | Second stage (or single stage if no confirmation was needed). Atomically removes the block, its `nested` scope (if any) including every inner block + edge, and every edge in the current scope whose source or target is the block. Re-runs validation; updates issue badge. |
| `block_reparent` | `{block_id, new_parent_block_id?, x, y}` | `new_parent_block_id=None` means promote to current scope; non-None means adopt into that container's nested scope (sibling-container moves are a single atomic dispatch). **Rejects `InputImage` block_ids** (the source must remain in its scope). Removes any edges that would cross the new scope boundary; emits a per-deleted-edge toast list. If the move would orphan inner edges and `block_reparent` is the *drag-out* direction (i.e. `new_parent_block_id` is an ancestor), the dispatcher rejects with snap-back + toast instead (see §4.4). |
| `block_select` | `{block_id?}` | `None` deselects. Setting a new id clears `selected_edge_id`. |
| `wire_select` | `{edge_id?}` | Same exclusion: clears `selected_block_id` when set. |
| `edge_create` | `{source_block_id, target_block_id, target_port, kind}` | For scalar aux + image-flow: replaces any existing wire from `source_block_id` (single-wire rule from §4.2) in the same dispatch. For list aux: **server-side append** — the dispatcher resolves `target_slot = block.list_slot_counts.get(target_port, 0)` (the **total** slot count, not `len(wired_edges)`; the count never decrements on `edge_delete`, so this is collision-free even after a user has freed an interior slot) and increments `list_slot_counts[target_port]` by 1. Client emits no slot index — eliminates the concurrent-drag race. |
| `edge_delete` | `{edge_id}` | If the deleted edge was a list-aux edge, the remaining edges in that slot range are *not* renumbered automatically; the slot count stays the same and the freed slot becomes an empty placeholder. Use `list_aux_reorder` to compact. |
| `list_aux_reorder` | `{block_id, param, new_order: [edge_id_or_null, …]}` | Updates the canonical execution order. `new_order` is a permutation of the wired edge_ids interspersed with `null`s for empty slots; the dispatcher rebuilds each edge's `target_slot` from its position and updates `block.list_slot_counts[param] = len(new_order)`. Non-permutation inputs are no-ops + toast. |
| `list_aux_add_empty_slot` | `{block_id, param}` | Increments `block.list_slot_counts[param]` by 1. No edge is materialised — empty slots are tracked solely on the consumer block. At `to_pipeline_dag` time, slot indices in `[0, count)` not covered by an edge emit `None` entries. |
| `block_label_update` | `{block_id, label}` | — |
| `block_params_update` | `{block_id, params}` | Replaces the scalar `params` dict whole-cloth (preserves the rest of the BlockNode). |
| `block_collapsed_toggle` | `{block_id}` | Container-only; toggles `collapsed`. No-op on non-container blocks. |
| `drill_into_container` | `{block_id}` | Pushes `block_id` onto `state.breadcrumb`. Re-runs dagre on the now-visible nested scope. |
| `drill_out` | `{depth?}` | Pops 1 segment by default; with `depth=N` pops to that depth. |
| `scroll_to` | `{block_id, scope_path, target_breadcrumb}` | Pans + fits the cytoscape viewport to the block, traversing breadcrumb + expanding collapsed containers as needed. **Implementation in `viewport_ops.js`:** (1) mount a semi-transparent canvas-wide scrim (`data-testid="dag-scrim"`) that suppresses all canvas interaction (no palette drag accepts, no port mousedown); (2) if `target_breadcrumb != state.breadcrumb`, dispatch `drill_to_scope(target_breadcrumb)` and `await cy.promiseOn("layoutstop")` — this handles the **cross-breadcrumb** case (offender in a different drill-depth than the current scope); (3) for each collapsed container in `scope_path` (now in the active scope), write a payload `{kind: "block_collapsed_toggle", block_id: <id>, ts: <now>}` to `STORE_VIEWPORT_OP` and `await cy.promiseOn("layoutstop")` — the fan-in callback already routes `STORE_VIEWPORT_OP` triggers through `_dispatch_state_update`, so the schema is extended to carry expand-chain payloads as well as `scroll_to`/`relayout`/`reanchor`; (4) `cy.fit(node)` once the chain resolves; (5) dismiss the scrim; (6) emit `phenotypic:scroll-to-complete`. The scrim closes the lost-update window where a user drag could interleave with the expand chain — typical chain duration is 200–600ms for 2–3 collapsed levels. State mutations *do* happen (`drill_to_scope` updates `state.breadcrumb`; each `block_collapsed_toggle` flips `BlockNode.collapsed`); the scrim makes them appear atomic to the user. |
| `drill_to_scope` | `{target_breadcrumb: List[str]}` | Atomic breadcrumb replacement (single dispatch). Server-side validation: each block_id in `target_breadcrumb` must resolve to a real, current `Pipeline`-class container at the right depth in `state.root`. Stale IDs (container deleted since the badge tooltip was rendered) → reject + toast + emit `phenotypic:scroll-to-aborted` event so `viewport_ops.js` immediately dismisses any active scrim. Used by `scroll_to` for cross-scope navigation. Distinct from `drill_into_container`/`drill_out` which mutate breadcrumb by 1 segment at a time. **One `layoutstop` per dispatch:** atomic breadcrumb replacement re-renders the canvas once, which triggers exactly one cytoscape `layout` invocation and therefore one `layoutstop` event — `viewport_ops.js` awaits one regardless of how many breadcrumb segments differ. |
| `relayout` | `{}` | Re-runs dagre + `cy.fit()`. Pure UI side-effect. |
| `request_run_preview` / `request_save_pipeline` | `{}` | Pre-check: filter `validate(state)` to severity=`error`; if any, abort + toast the first issue. Otherwise delegate to existing flow. Advisory hints (severity=`advisory`) never block these. |

**Dropped dispatch kinds (popover legacy):** `pick_class`, `add_slot`,
`drill_in_aux`, `set_inspector_focus`, `popover_dismiss`, `wire_delete`
(replaced by `edge_delete`), `wire_create` (replaced by `edge_create`).
**Also dropped:** `edge_replace` — the gesture is implemented as
"select wire → Delete → re-draw from source port" (two-step). Cytoscape
doesn't expose grabbable edge endpoints natively, and adding
`cytoscape-edgehandles` for this one gesture was rejected (§9 risk
discussion).

### 5.7a Feature flag mechanism (transitional, Phases 1–7)

`PHENOTYPIC_GUI_DAG=1` gates the DAG path during incremental rollout. The
flag is read **once at module import** of `phenotypic.gui.builder._state`:

```python
import os

PHENOTYPIC_GUI_DAG: bool = os.environ.get("PHENOTYPIC_GUI_DAG", "0") == "1"
```

Consumers:

* `builder/_state.py` — selects between `BuilderScope` (legacy) and the new
  DAG schema at module load.
* `builder/_layout.py` — selects between `build_canvas_elements` (legacy)
  and `build_canvas_elements_dag`.
* `builder/_callbacks.py` — selects between the popover dispatch table and
  the DAG dispatch table.
* `assets/builder.js` — reads `window.phenotypicGuiDag` (injected into
  `app.index_string` at server boot from the same env var) to decide which
  clientside glue to mount.

Flipping the flag requires a process restart. After Phase 8 the flag is
deleted; both code paths collapse to the DAG path. A test in
`tests/unit/gui/builder/test_feature_flag.py` asserts the flag value is
read once and stable for the process lifetime.

### 5.7 Persistence (pipeline.json)

`ImagePipeline.to_json()` / `from_json()` are untouched. The builder's
`Save pipeline` writes the same payload it does today. Position info is
*not* persisted — the dagre pass on the next open recomputes layout.

Loading an existing `pipeline.json` from a pre-redesign release converts
trivially: the old `aux_ports` map → edges, the old node order → image-flow
edges plus an `InputImage` block prepended. The `from_pipeline` rewrite
handles this on the GUI side; the JSON file format does not change.

## 6. Component map

| Concern             | File                                          | Change |
| ------------------- | --------------------------------------------- | ------ |
| State dataclasses + conversion | `builder/_state.py`                | Rewrite. `StepNode` → `BlockNode`, `aux_ports` → `edges`, drop `_PARAM_SCOPE_KEY` and `_ensure_param_scope` / `_commit_param_scope`. Keep `to_pipeline` / `from_pipeline` public signatures. |
| Layout (canvas elements + stylesheet) | `builder/_layout.py`         | Rewrite the cytoscape elements builder (`build_canvas_elements`) to emit one node per block, one edge per `Edge`, plus port sub-nodes (input/output/aux). Drop popover container + `inspector_focus_aux_banner_id`. Add issue badge + toolbar count. |
| Mutation callbacks  | `builder/_callbacks.py`                       | Drop popover dispatch paths (`pick_class`, `add_slot`, `drill_in_aux`, `set_inspector_focus`, popover-dismiss). Add `edge_*` / `block_create` / `palette_drop` / `list_aux_reorder`. Validation runner gates run/save. |
| Parameter form      | `builder/_param_form.py`                      | No semantic change; the inspector consumes it the same way. |
| Inspector pane      | `builder/_layout.py::build_inspector`         | Add "Aux ports" section with per-param ordered list + drag-handles for list-typed. Drop popover-related branches. |
| IDs / pattern matches | `builder/_ids.py`                            | Add: `STORE_EDGE_EVENT`, `STORE_PALETTE_DROP`, `STORE_VIEWPORT_OP`, `STORE_ISSUES`, `STORE_ASSET_STATUS`, `block_port_id(block_id, port)`, `edge_id(edge_id)`, `BTN_RELAYOUT`, `CONFIRM_DELETE_MODAL_ID`, `BTN_CONFIRM_DELETE`, `BTN_CANCEL_DELETE`, `BANNER_ASSET_STATUS`. Drop: `aux_port_id`, `_encode_aux_port_id`, `_decode_aux_port_id`, `POPOVER_*`, `PORT_CLICK_STORE`, `POPOVER_DISMISS_STORE`. |
| Confirm-delete modal | `builder/_layout.py::build_confirm_delete_modal` | **New.** Lightweight `dbc.Modal` (NOT the existing `TOAST_NOTIFICATION`, which auto-dismisses and has no action buttons). Renders a body line listing the inner-block count + Confirm / Cancel buttons. Mounted once at app boot; visibility driven by `STORE_BUILDER_STATE.pending_delete_block_id`. Confirm dispatches `block_delete_confirm`; Cancel clears the pending field. |
| Asset status banner | `builder/_layout.py::build_asset_status_banner` | **New.** A thin row above the canvas that subscribes to `STORE_ASSET_STATUS` and renders status text for any missing asset within 1500ms of page load. Recognised statuses (one row per missing asset): `wire_drawing.js` → "Wire drawing offline"; `palette_dnd.js` → "Block creation offline — drag from the palette is unavailable"; `viewport_ops.js` → "Layout offline"; `cytoscape-dagre` extension (separate check: `viewport_ops.js` verifies `typeof window.cytoscape.layouts.dagre === 'function'` in its IIFE and writes `dagre_missing` into the status payload if not) → "Layout extension missing". Hidden when all assets reported ready. |
| Asset-status disable wiring | `builder/_callbacks.py::asset_status_disables` | **New callback.** Subscribes to `STORE_ASSET_STATUS`. When `wire_drawing.js` missing → sets `disabled=True` on `BTN_DELETE_NODE` *and* clientside listener on port mousedown short-circuits (the JS isn't there to handle it anyway). When `palette_dnd.js` missing → adds `pointer-events: none` to every palette button and shows a `title` tooltip explaining. When `viewport_ops.js` missing or `dagre_missing` → sets `disabled=True` on `BTN_RELAYOUT`; cytoscape falls back to `name: "preset"` layout. |
| Image render cache  | `builder/_session.py`, `_image_renderer.py`   | **Update.** Cache key migrates from the 8-char `StepNode.node_id` to the 32-char `BlockNode.block_id` UUID. `_bake_preview_cache(state, pipeline, result, session_id, cache)` walks the topological order produced by `to_pipeline_dag` (not the linear-list `to_pipeline`); for nested containers it recurses into each container's inner pipeline so previews exist for inner ops the user can drill into. Aux-only blocks have no main-flow preview — selecting them shows the inspector preview thumbnail derived from the consumer's intermediate cache entry instead. |
| Operation registry  | `gui/_operation_registry.py`                  | No change. `ParameterInfo.{is_operation, is_pipeline, is_list, default}` already provide everything validation + type-aware highlight need. |

## 7. Build sequence

Phased so each phase is independently merge-able and CI-gated.

**Phase 1 — Schema + conversion (no UI change yet)**

* Add new `BlockNode` / `Edge` / `BuilderScope` dataclasses behind a feature
  flag (`PHENOTYPIC_GUI_DAG=1`).
* `_seed_input_image` + `BuilderScope.__post_init__` guarantee every scope
  has exactly one `InputImage` block at construction time (Rule 6 auto-recovery).
* Implement `to_pipeline_dag` / `from_pipeline_dag` alongside the existing
  pair; pure-Python tests in `tests/unit/gui/builder/test_state_dag.py`
  cover round-tripping every registry class.
* Implement `validate(state) -> List[Issue]` covering all six rules
  (Rules 1–6); unit tests cover positive + negative cases for each rule
  + boundary cases (empty scope, container nesting, list-aux slots, cycle
  members, missing/duplicate Input Image).
* Add the test fixtures listed in §8 (`tests/fixtures/builder_dag/*.json`).

**Phase 2 — Read-only canvas render**

* `build_canvas_elements_dag(scope)` produces the new cytoscape layout
  (blocks, ports, edges, issue badges) under the feature flag.
* Add `assets/viewport_ops.js` (new) carrying the **leaf-first
  compound layout algorithm** from §4.7. Vendor `cytoscape-dagre@^2.5.0`
  as `assets/cytoscape-dagre.min.js`. The layout pass lives in
  `viewport_ops.js`, *not* `wire_drawing.js` (which is Phase 4 and owns
  only wire/port interaction).
* Implement asset-load-status protocol: each new asset IIFE writes
  `window.phenotypic_<name>_ready = true` on completion;
  `assets/builder.js` polls these every 500ms and writes
  `STORE_ASSET_STATUS` with the missing-file list. A toolbar banner
  bound to that store shows "Wire drawing offline" / "Layout offline"
  when an asset failed to register.
* Render-only smoke test in `tests/integration/gui/builder/` confirms a
  fixture pipeline materializes a fork-free DAG.

**Phase 3 — Palette drag + block create**

* `assets/palette_dnd.js` (new) wires HTML5 dnd → `STORE_PALETTE_DROP`.
* `block_create` dispatch + container hit-test on drop.
* E2E: drag BlurGauss from palette, drop on canvas, block appears wired
  to nothing.

**Phase 4 — Wire-drawing**

* `assets/wire_drawing.js` (new) implements port mousedown → live SVG wire
  → drop. Type-aware highlight via the per-port `data.accepts` list
  injected by `build_canvas_elements_dag` (the registry stays server-side;
  the JS reads from `node.data('accepts')` on dragstart).
* `edge_create` / `edge_delete` dispatches. **`edge_replace` is *not*
  implemented** — the move-target gesture is a two-step
  select-Delete-redraw (§4.3, §5.6).
* Validation re-runs on every dispatch.
* E2E: wire OtsuDetector → CompositeDetector.detectors[0]; second wire fan-in
  appends to slot[1].

**Phase 5 — Containers**

* `+ New Pipeline` palette button + container drop semantics.
* Container expand/collapse toggle.
* Drag-into-container adopts a block into the inner scope.
* Recursive validation; recursive `to_pipeline_dag` / `from_pipeline_dag`.
* E2E: wire a 2-op container into FilamentousFungiDetector.inoculum_detector.

**Phase 6 — Inspector + validation gating**

* New "Aux ports" section in `build_inspector` listing wired edges per param
  with drag-reorder for list params.
* Toolbar issue badge + tooltip + click-to-pan.
* `Run preview` / `Save pipeline` disabled when `validate(scope)` returns
  non-empty.

**Phase 7 — Remove popover**

* Delete `assets/aux_popover.js`, `cytoscape-popper.min.js`,
  `popperjs-core.min.js`.
* Strip popover dispatch kinds + ids from `_ids.py`, `_callbacks.py`,
  `_state.py`.
* Drop `inspector_focus_aux` from state, `_PARAM_SCOPE_KEY` from state, and
  the `inspector-focus-aux-banner` element from `_layout.py`.
* Migration: legacy `pipeline.json` files load through the new
  `from_pipeline_dag` which is shaped identically to today's `from_pipeline`
  output then re-emits the DAG. No JSON schema changes.

**Phase 8 — Feature flag retired, FEATURES.md / WORKFLOWS.md updates**

* Remove `PHENOTYPIC_GUI_DAG` flag.
* Rewrite the "Builder aux ports" section in `FEATURES.md` (every row gets
  a new test ref).
* Update `WORKFLOWS.md` with the new aux-wiring tutorial; add
  `_capture_aux_wire_in_dag` to `scripts/capture_gui_tutorial_screenshots.py`
  per the gui-docs gate.

## 8. Testing strategy

The DAG redesign rests on four test layers; the **Playwright E2E suite is the
load-bearing one** because most of the new design is interaction-driven (drag,
drop, hover, type-aware affordances, layout-timing-dependent flows). The
suite is intended to be **comprehensive** — every feature row from §4, every
edge case from §4.10, every gesture from §4.9 maps to at least one Playwright
test below.

### 8.1 Unit tests (`tests/unit/gui/builder/`)

| Module | Coverage |
| ------ | -------- |
| `test_state_dag.py` | Round-trip every registry class via `to_pipeline_dag` ↔ `from_pipeline_dag`; `to_pipeline_dag` raises `ValueError` on each blocking rule (Rules 1–6) — one test per rule; `from_pipeline_dag` clones shared-instance auxes + emits a toast (mocked); `_seed_input_image` is idempotent (call 3× → 1 block); legacy `pipeline.json` fixtures load identically. |
| `test_validation.py` | Per-rule pos/neg cases: Rule 1 (fork on output, fork on input), Rule 2 (orphan block, orphan after wire delete), Rule 3 (required scalar empty, required list empty, optional empty OK — **regression test guarding the `has_default` fix**), Rule 4 (image cycle, aux cycle, image+aux mixed cycle), Rule 5 (left wired + right purple, left unwired + right blue), Rule 6 (zero `InputImage`, two `InputImage`s), Rule 7 advisory (measure→ops edge yields hint not error). Recursion: every rule fires inside containers and surfaces via `scope_path`. |
| `test_dispatch.py` | Each dispatch kind from §5.6: positive case, guard case (e.g. `block_create` rejects `InputImage`; `block_reparent` rejects `InputImage`; `block_delete` rejects `InputImage`); list-aux slot index resolved server-side (concurrent dispatches yield deterministic indices 0/1/2); `edge_create` replaces scalar wire atomically; `list_aux_reorder` non-permutation no-ops + toast; `block_reparent` orphan-edge cleanup + sibling-container atomic move; `block_collapsed_toggle` aggregates inner issues. |
| `test_feature_flag.py` | `PHENOTYPIC_GUI_DAG` read once at import; subsequent env-var changes ignored. |
| `test_recovery.py` | `missing_input` Issue triggers `_seed_input_image` on next dispatch; `unknown_class` advisory does not crash render; `from_pipeline_dag` shared-instance clone path. |
| `test_validation_perf.py` | Benchmark — `validate(state)` ≤ 5ms on a 100-block synthetic fixture (regression guard for cycle detection + recursion). |

### 8.2 Integration tests (`tests/integration/gui/builder/`)

Server-side render + callback wiring (no browser):

* `test_canvas_render.py` — rendering each fixture (§8.4) produces the
  expected cytoscape `elements` list (block nodes, port sub-nodes, edge
  records, compound parents for containers).
* `test_inspector.py` — `block_select` populates the param form;
  `wire_select` renders the wire card with `Disconnect`; `InputImage`
  selection renders the info card (no param form, no delete);
  container selection shows inner summary + drill-in button.
* `test_type_aware_highlight.py` — given a source class + a registry
  fixture, asserts `build_canvas_elements_dag` emits the correct
  per-port `accepts: List[str]` data attribute. Positive cases:
  `Detector` source → ports annotated `Detector` or `ImageOperation`
  glow; `ImagePipeline` source → ports annotated `ImagePipeline` glow;
  `List[Detector]` ports glow on any `Detector` subclass. **Negative
  case** (explicitly labelled): `ColumnRef`-typed params are *not*
  rendered as aux ports at all (they're scalar parameter inputs in the
  inspector form, not consumer ports); the test asserts no `accepts`
  data attribute exists for those params.
* `test_legacy_pipeline_json.py` — every shipped fixture under
  `tests/fixtures/pipelines/*.json` round-trips through the DAG path
  (load → re-save → byte-identical or semantically-identical
  `to_pipeline_dag(from_pipeline_dag(p)) == p`).
* `test_feature_flag_routing.py` — flag off → legacy callbacks fire;
  flag on → DAG callbacks fire; flag is stable after first request.

### 8.3 Playwright E2E suite (`tests/e2e/gui/builder/`)

The Playwright suite is organised by interaction surface. Each subsection
maps explicitly to spec sections; nothing in the user-facing model (§4)
should be unmapped.

**Test infrastructure:**
* Page fixture launches the builder with a synthetic test plate (via
  `load_synth_yeast_plate` import in a server-side hook) so screenshots
  and previews work without raw images.
* All E2E tests start with `PHENOTYPIC_GUI_DAG=1`.
* Each test uses Playwright's `expect(locator).toBeVisible({ timeout })`
  pattern + cytoscape's `cy.ready()` promise via
  `await page.evaluate(...)` to avoid race conditions with layout.

#### 8.3.1 Palette → canvas (§4.8, §4.9)

* `test_palette_drag_drop_creates_block` — drag `BlurGauss` from the
  palette to canvas centre; block appears at drop coords; toolbar count
  badge stays at "0 issues".
* `test_palette_drag_drop_inside_container` — drop into a container's
  expanded body; block adopted (cytoscape `parent` == container id).
* `test_palette_drag_drop_inside_nested_container_innermost_wins` —
  two nested containers overlap; drop in their overlap zone; only the
  *innermost* container adopts the block.
* `test_palette_drag_drop_on_existing_block_lands_adjacent` — block
  doesn't overlay; horizontal offset by dagre's node width.
* `test_palette_drag_drop_on_wire_is_positional_not_insertion` — block
  lands at cursor; the wire under the drop point is selected as a
  side-effect; A→C is unmodified; toast appears.
* `test_palette_drag_drop_outside_cy_slot_cancels` — drag the ghost
  off the canvas wrapper bounds; no block created; no toast.
* `test_palette_drag_esc_during_drag_cancels` — `keyboard.press("Escape")`
  mid-drag; ghost disappears; no block.
* `test_palette_keyboard_fallback` — `Tab` to `BlurGauss` button +
  `Enter`; block placed at viewport centre.
* `test_palette_no_input_image_button` — assert palette contains no
  `+ Input Image` button (`expect(...).toHaveCount(0)`).
* `test_palette_dispatch_rejects_input_image_class_name` — fake
  dispatch via `page.evaluate(...)` directly with `class_name:
  "InputImage"`; state unchanged + toast "scope already has an Input
  Image."

#### 8.3.2 Wire drawing (§4.3, §4.9)

* `test_wire_drag_image_to_image_snaps_blue` — drag from
  BlurGauss.output to OtsuDetector.in; wire snaps blue 3px (main path
  emphasis).
* `test_wire_drag_image_to_aux_snaps_purple` — drag BlurGauss.output
  to FilamentousFungiDetector.inoculum_detector; wire snaps purple-
  dashed 2px; **source block border turns solid purple (aux-consumed
  cue)**.
* `test_wire_drag_live_wire_neutral_gray_during_flight` — assert wire's
  stroke colour is the neutral hex while mouse is down + moving.
* `test_wire_drag_compatible_targets_glow_incompatible_dim` — pick up
  BlurGauss (Corrector); only Corrector-typed aux ports glow; a
  `Detector`-typed aux port renders at ≤30% opacity.
* `test_wire_drag_drop_on_dimmed_port_rejects_with_red_flash` — assert
  the dimmed port flashes red briefly; no edge created.
* `test_wire_drag_drop_on_empty_canvas_fades_out` — wire fades; no
  state change.
* `test_wire_drag_esc_cancels` — `Escape` mid-drag; wire fades.
* `test_wire_drag_mouse_leaves_canvas_cancels` — mouse leaves the cy
  wrapper bounds; wire fades.
* `test_wire_drag_from_already_wired_source_replaces_first_wire` —
  the prior edge is gone; the new edge replaces it; only one edge in
  state.
* `test_wire_select_then_delete` — click an edge (stroke widens 4px +
  brightens); press `Delete`; edge removed.
* `test_wire_right_click_disconnect` — right-click wire → "Disconnect"
  context item; edge removed.
* `test_wire_no_endpoint_grab_gesture` — assert the endpoint drag
  doesn't initiate (mousedown on edge endpoint within tolerance is a
  no-op or selects the edge). Documents that `edge_replace` is gone.
* `test_wire_blue_throughout_past_measure_boundary` — chain
  `Otsu → MeasureSize → MeasurePerimeter`; assert all three wires are
  the blue colour, not gold/green.
* `test_wire_main_path_3px_aux_2px` — assert stroke-width attribute on
  rendered SVG.

#### 8.3.3 List-aux ports (§4.4 list semantics, §4.5)

* `test_list_aux_fan_in_appends_to_next_slot` — wire 3 detectors into
  `CompositeDetector.detectors`; each subsequent drop lands at slot
  `len(current)`; canvas badges read 1/2/3.
* `test_list_aux_concurrent_drags_resolve_server_side` — fire two
  `edge_create` dispatches in the same Dash tick; assert deterministic
  slot indices (no collision).
* `test_list_aux_inspector_reorder_updates_canvas_badges` — drag the
  inspector handle for badge 2 above badge 1; badges re-render 2→1,
  1→2; runtime execution order matches.
* `test_list_aux_remove_wire_keeps_empty_slot` — disconnect badge 2;
  badges 1 + 3 stay; slot 2 is the empty placeholder; total slot count
  unchanged.
* `test_list_aux_add_empty_slot` — click "+ Add empty slot" in the
  inspector; total slot count increments; canvas shows an empty slot
  badge (numbered, empty).
* `test_list_aux_required_with_empty_slot_fires_rule_3` — `CompositeDetector`
  with required `detectors` and one empty slot, zero wired: Rule 3
  fires (red border + ! badge); preview disabled.

#### 8.3.4 Pipeline containers (§4.4)

* `test_container_create_from_palette` — `+ New Pipeline` palette
  button drops empty container; container has title bar + consumer-fed
  dot + output port + collapse chevron.
* `test_container_drag_op_into_expanded_body_adopts` — drag an op into
  the container's body; op's cytoscape `parent` is the container id.
* `test_container_drag_op_into_nested_innermost_wins` — two nested
  containers; drop in overlap; innermost adopts.
* `test_container_drag_out_clean_pops_to_parent_scope` — block with no
  inner edges drags out; lands in parent scope; container's nested
  scope no longer contains it.
* `test_container_drag_out_with_inner_edges_snaps_back_with_toast` —
  block with one inner edge; drag-out fails; toast lists the orphan
  edge; block animates back to its original inner position.
* `test_container_collapsed_shows_aggregated_issues` — collapsed
  container has an inner fork; outer chrome shows "▣ 1 issue"; toolbar
  count includes it.
* `test_container_collapsed_click_badge_expands_then_pans` — click the
  aggregated badge; container expands; cytoscape `layoutstop` fires;
  viewport pans to the offender. **Order asserted via
  `page.waitForEvent` + custom events emitted by `viewport_ops.js`.**
* `test_container_drill_in_via_double_click_body` — double-click
  container body; canvas swaps to nested scope; breadcrumb pushes a
  segment.
* `test_container_label_inline_rename_via_double_click_title_bar` —
  double-click title bar; input renders; edit + Enter; label updated.
* `test_container_main_flow_mode_left_wired_right_blue` — container
  in main flow; consumer-fed dot dims; container border stays purple.
* `test_container_aux_mode_left_unwired_right_purple` — container as
  aux; consumer-fed dot lights up; output wire is purple.
* `test_container_rule_5_mixed_mode_red_border` — left wired *and*
  right wired to aux purple: Rule 5 fires; red border.
* `test_container_delete_with_children_confirms` — non-empty container
  delete opens the `CONFIRM_DELETE_MODAL_ID` modal showing the
  non-`InputImage` child count; clicking Cancel keeps everything;
  Confirm dispatches `block_delete_confirm` and removes the container +
  all children + all incident edges atomically.
* `test_container_delete_empty_skips_modal` — container with only the
  auto-seeded `InputImage` (no other children) is treated as empty;
  `block_delete_request` immediately delegates to `block_delete_confirm`
  without showing the modal.
* `test_container_sibling_reparent_single_dispatch` — drag from
  container A to sibling B; assert state has block in B's nested scope,
  not A's, after one dispatch; orphan edges from A combined with new
  edges in B both surface in one toast.
* `test_aux_of_aux_nested_container_round_trip` — root → consumer →
  container A as aux → inner consumer → container B as aux; save +
  reload; topology preserved.

#### 8.3.5 Input Image lifecycle (§4.1, §4.5)

* `test_input_image_auto_seeded_on_fresh_scope` — open builder; assert
  exactly one `Input Image` in root scope.
* `test_input_image_inspector_card` — select Input Image; inspector
  shows info card + `Re-layout` button + `Re-anchor view` button; no
  param form; no delete button.
* `test_input_image_delete_no_op_with_toast` — try to `Delete` while
  Input Image selected; nothing removed; toast: "Input Image cannot be
  removed."
* `test_input_image_not_draggable` — assert `cy.getElementById(...)`
  for the Input Image block has `grabbable: false` (or the drag yields
  no `block_reparent`).
* `test_input_image_dispatcher_guard` — programmatic
  `block_reparent` with Input Image's id via `page.evaluate(...)`;
  rejected; toast.
* `test_input_image_re_anchor_button` — click `Re-anchor view`; cy
  viewport centres on Input Image (assert pan + zoom values within
  tolerance).

#### 8.3.6 Validation rules + UI surfacing (§4.6)

* `test_rule_1_image_flow_fork_red_border_disables_run` — manually
  create a fork via `page.evaluate(...)` direct state mutation
  (Playwright can't actually create a fork via drag since drag-replaces).
  Red border + "!" badge; toolbar shows "1 issue"; `Run preview`
  disabled.
* `test_rule_1_fork_removed_clears_red_and_enables_run` — delete the
  duplicate wire; border clears; toolbar reads "0 issues"; `Run
  preview` re-enabled.
* `test_rule_2_stub_dashed_red_border` — drop an op, never wire it;
  dashed red border.
* `test_rule_3_required_aux_empty_red_ring` — `FilamentousFungiDetector`
  added without `inoculum_detector` wired; aux port has red ring; ! badge.
* `test_rule_3_optional_aux_empty_no_red` — `OtsuDetector.foo` (mocked
  optional aux) stays hollow purple; no issue.
* `test_rule_4_image_cycle_all_members_red` — synthesize cycle
  programmatically; every block in the cycle gets red border + "!".
* `test_rule_4_aux_cycle_detected` — same for aux-only cycle.
* `test_rule_5_container_mode_red_border` — see §8.3.4.
* `test_rule_6_zero_input_image_auto_recovers` — delete InputImage
  programmatically; dispatch fires; on next render the scope has been
  re-seeded; toast: "Input Image restored."
* `test_rule_6_duplicate_input_image` — synthesize 2 Input Image
  blocks; both shown; one (the extra) carries "duplicate" issue.
* `test_rule_7_stage_order_hint_yellow_advisory` — wire `MeasureSize`
  before `MaskRefiner` in the chain; the source block (MeasureSize) gets
  *yellow* border + "?" badge; toolbar splits "0 issues, 1 hint";
  `Run preview` is **enabled**.
* `test_rule_7_recurses_into_containers` — same misorder inside a
  container; aggregate "0 issues, 1 hint" surfaces on the container's
  chrome; same recovery.
* `test_issue_badge_click_pans_and_selects` — click a row in the
  issue badge tooltip; cytoscape viewport centres + zooms; the offender
  is selected.
* `test_issue_badge_click_expands_container_chain_before_pan` —
  offender lives 2 levels deep in collapsed containers; click expands
  outer → inner → pans. Assert each `layoutstop` event before the next
  step via instrumented `viewport_ops.js`.
* `test_issue_badge_click_cross_breadcrumb_pops_first` — drill into
  container A; an issue exists in root scope (a sibling). Click the
  issue row; `scroll_to` dispatches `drill_to_scope([])` first, then
  pans. Assert `state.breadcrumb` becomes `[]` and the root canvas is
  visible before the pan completes.
* `test_scroll_to_scrim_blocks_canvas_interaction` — during an active
  expand chain, attempt a palette drag and a port mousedown; both are
  blocked by the scrim (`data-testid="dag-scrim"` is in the DOM with
  pointer-events enabled); state is unchanged after the chain settles.
* `test_scroll_to_stale_id_dismisses_scrim` — issue badge tooltip
  open with an offender in container `X`; programmatically delete `X`
  via `page.evaluate(...)`; click the now-stale issue row; `drill_to_scope`
  rejects + toast; the scrim dismisses immediately (no hang); user
  retains canvas control.
* `test_drill_to_scope_single_layoutstop` — instrument cytoscape to
  count `layoutstop` events. Trigger a `drill_to_scope` from `[A, B]`
  to `[C]` (different depth, different sibling). Assert exactly one
  `layoutstop` event fires for the drill (independent of how many
  breadcrumb segments differ).
* `test_palette_dnd_js_fails_to_load` — block `palette_dnd.js` via
  Playwright route interception; banner shows "Block creation
  offline — drag from the palette is unavailable"; every palette
  button is `pointer-events: none` + has a tooltip; existing canvas
  state intact.
* `test_cytoscape_dagre_missing_layout_offline` — block
  `cytoscape-dagre.min.js`; `viewport_ops.js` loads but detects the
  missing extension and writes `dagre_missing` into `STORE_ASSET_STATUS`;
  banner shows "Layout extension missing"; `BTN_RELAYOUT` is disabled;
  cytoscape falls back to `preset` layout.
* `test_toast_queue_one_at_a_time` — programmatically fire 3 dispatches
  that each produce a toast (e.g. 3 reparents with orphan-edge
  deletion); assert the toast container shows one at a time, FIFO, with
  3000ms duration each; total visible time ~9000ms.
* `test_list_aux_slot_index_no_collision_after_delete` — wire slots
  0/1/2; delete slot 1; wire a 4th source; assert the new edge lands
  at `target_slot=3` (since `list_slot_counts=3` before the new wire,
  4 after); no collision with the existing slot-2 edge.
* `test_request_run_preview_blocked_by_issues` — toolbar shows 3
  issues; click `Run preview`; no preview runs; toast names the first
  issue.
* `test_request_run_preview_unblocked_by_advisory_only` — toolbar shows
  "0 issues, 1 hint"; `Run preview` runs successfully.

#### 8.3.7 Inspector behaviour (§4.5)

* `test_inspector_empty_state_when_nothing_selected` — click empty
  canvas; inspector shows "Drag an operation from the palette to begin."
* `test_inspector_block_select_shows_param_form` — select a block; the
  registry-derived param form renders; edits dispatch
  `block_params_update`.
* `test_inspector_wire_select_card_with_disconnect` — click an edge;
  inspector shows source→target labels + `Disconnect`; click `Disconnect`
  removes the edge.
* `test_inspector_aux_list_drag_reorder` — see §8.3.3
  `test_list_aux_inspector_reorder_updates_canvas_badges`.
* `test_inspector_selection_mutual_exclusion` — block selected; click
  wire; block deselects; wire card shown.

#### 8.3.8 Layout (§4.7)

* `test_dagre_runs_on_every_mutation` — instrument cytoscape to count
  `layoutstop` events; assert a non-zero count per dispatch.
* `test_manual_drag_position_is_ephemeral` — drag a block to (x, y);
  fire any state mutation; on next render the block is back at the
  dagre-computed position.
* `test_relayout_button_recenters` — pan/zoom away; click `Re-layout`;
  viewport returns to fit-to-content centred on Input Image.
* `test_empty_container_placeholder` — container with zero inner ops
  shows "+ drop ops here" text; drop in → text disappears.

#### 8.3.9 Migration / legacy load (§5.4, §5.7)

* `test_load_legacy_popover_pipeline_json` — load a fixture saved by
  the popover-era builder; renders as DAG; no errors; `to_pipeline_dag`
  emits a pipeline `==` to the original `ImagePipeline.from_json` output.
* `test_load_shared_instance_clones` — load a synthetic pipeline where
  the same `OtsuDetector` instance appears in `_ops` and inside another
  op's aux; assert two separate blocks materialised; toast appeared.
* `test_load_unknown_class_yellow_border` — load a pipeline with a class
  not in the current registry; block renders with yellow border + label
  "(unknown: SomeClass)"; advisory issue surfaces.
* `test_load_no_canvas_layout_falls_back_to_dagre` — legacy file has no
  position info; dagre lays out cleanly; save round-trip is byte-stable
  through `to_pipeline_dag`.

#### 8.3.10 Resilience / failure modes (§9, §4.10)

* `test_clientside_wire_drawing_js_fails_to_load` — block the JS asset
  via Playwright route interception; assert canvas renders; `STORE_ASSET_STATUS`
  flags the missing file; the `BANNER_ASSET_STATUS` toolbar banner reads
  "Wire drawing offline"; existing wires still render.
* `test_clientside_viewport_ops_js_fails_to_load` — block `viewport_ops.js`;
  banner reads "Layout offline"; canvas renders with the cytoscape default
  layout fallback (`name: "preset"`); `Re-layout` toolbar button is
  disabled.
* `test_dagre_layout_throws_falls_back_to_preset` — feed a degenerate
  scope (single InputImage, no edges); layout falls back; block centred.
* `test_concurrent_palette_drag_plus_wire_drag` — start a wire drag,
  then trigger a palette drop; wire drag is cancelled; palette drop
  succeeds.

### 8.4 Test data fixtures

Under `tests/fixtures/builder_dag/` — used by both unit (validation
fixture assertions) and Playwright (`page.evaluate` to inject state
directly via `dash_clientside.set_props(STORE_BUILDER_STATE, {data: ...})`)
layers.

**Provenance:** all fixtures are **hand-authored JSON**, not generated.
This matters because several fixtures represent *invalid* states the
dispatcher would reject if produced via UI gestures (e.g. forks,
cycles). A `README.md` in the fixtures directory documents the schema
of each `<fixture>.json` (matches `state_to_json` output) and of the
`<fixture>.expected_issues.json` sidecar.

**`expected_issues.json` schema:**

```json
{
  "issues": [
    {
      "kind": "fork" | "stub" | "required_aux" | "cycle"
            | "container_mode" | "missing_input" | "duplicate_input"
            | "stage_order_hint" | "unknown_class",
      "block_label": "BlurGauss",
      "severity": "error" | "advisory",
      "scope_path": []
    }
  ]
}
```

Unit tests assert `set(actual_issues) == set(expected_issues)` after
normalisation; ordering is not significant.

**Fixtures:**


* `empty.json` — root scope with only `InputImage`.
* `linear_chain.json` — Input → BlurGauss → OtsuDetector → MeasureSize.
* `linear_chain_misordered.json` — Input → MeasureSize → MaskRefiner →
  MeasureSize (Rule 7 advisory fires).
* `scalar_aux.json` — `FilamentousFungiDetector` with a single
  `OtsuDetector` aux.
* `list_aux_three_detectors.json` — `CompositeDetector` with 3
  detectors in slots 0/1/2 (one is a container).
* `list_aux_with_empty_slot.json` — slot 1 is empty; slots 0/2 wired.
* `nested_container.json` — container with an inner container as
  aux-of-aux.
* `container_main_flow.json` — container used as a main-flow sub-pipeline.
* `container_aux_mode.json` — same container used as aux.
* `fork_offender.json` — invalid: image-flow output with 2 outgoing wires.
* `mixed_kind_fan_out.json` — invalid: same source wired to one
  image-in *and* one aux-in (Rule 1 covers this since "one outgoing
  wire total").
* `image_cycle.json` — invalid: image-flow A→B→A.
* `aux_cycle.json` — invalid: aux A's aux ← B; B's aux ← A.
* `unwired_required.json` — invalid: required aux port empty.
* `mixed_container_mode.json` — invalid: container left wired + right
  wired to aux.
* `duplicate_input_image.json` — invalid: two `InputImage` blocks in
  one scope.
* `shared_aux_instance.json` — legacy edge case: same `ImageOperation`
  instance in `_ops` *and* an aux param. `from_pipeline_dag` clones.
* `legacy_popover_pipeline.json` — saved by the current builder; loads
  through `from_pipeline_dag`.

Each invalid fixture has an `expected_issues.json` sibling asserting
exact (`kind`, `block_label`, `severity`) tuples.

### 8.5 CI gating

* `gui-e2e` workflow runs the Playwright suite headless under Chromium
  AND Firefox (drag-and-drop semantics differ slightly).
* Two suite subsets with explicit wall-clock budgets:
  * **`tests/e2e/gui/builder/_critical/`** — minimal "the redesign isn't
    broken" subset, runs on **every PR**. Budget: **3 minutes wall-clock**
    headless (Chromium only) with `--workers 4` parallelism. Includes:
    - §8.3.1: palette drag-and-drop happy path + container adoption + `Input Image`-class rejection
    - §8.3.2: wire drag image→image + wire drag image→aux + Esc cancels + replace-from-source
    - §8.3.3: list-aux fan-in append + concurrent-drag determinism
    - §8.3.4: container expand/collapse + drill-in + drag-out snap-back
    - §8.3.5: Input Image auto-seeded + non-deletable
    - §8.3.6: Rules 1, 2, 3 each fire + clear; Rule 7 doesn't block
    - §8.3.9: legacy `pipeline.json` round-trip
  * **Full suite (`tests/e2e/gui/builder/`)** — runs nightly + on
    `main`-bound merges + on any PR that touches `assets/*.js`. Budget:
    **15 minutes wall-clock** under Chromium + 15 under Firefox (parallel
    jobs). The Firefox run skips tests tagged `@chromium-only` (the
    `cy.promiseOn` event-timing tests in §8.3.4 — cytoscape's event
    emission timing varies subtly between browsers).
* `gui-docs` workflow runs `scripts/check_workflows_md.py` and
  `scripts/capture_gui_tutorial_screenshots.py`. New rows in
  `WORKFLOWS.md` for the DAG redesign (Phase 8) require matching
  `_capture_<id>` functions.

## 9. Risks & mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Cytoscape doesn't render arbitrary-shape containers cleanly. | Containers render as parent `compound` nodes; cytoscape's compound parent has a label slot + auto-resizes around children. Validate in Phase 2 with a spike; if compound + dagre integration is poor, fall back to a manual layout pass that allocates per-container bounding boxes from inner positions. |
| HTML5 dnd + cytoscape coord conversion is finicky across browsers. | Phase 3 lands a minimal vertical slice and E2E covers Chrome/Firefox. Fallback: click-to-place mode behind a settings flag (deferred §10). |
| Dagre layout flickers between renders when state mutations are rapid. | Animate position changes via cytoscape's `layout` animation (200ms ease); debounce validation-driven re-layouts so consecutive `edge_create`s in the same tick produce one layout pass. |
| Wire drag conflicts with cytoscape's native pan / box-select on the canvas. | `wire_drawing.js` registers `mousedown` only on output-port sub-nodes (selector `node.dag-port--output`); cytoscape's pan handler is suppressed during a port-drag by setting `cy.userPanningEnabled(false)` on `dragstart` and restoring on `dragend`. |
| Recursive validation cost on deeply nested containers. | The recursion is O(blocks + edges) per scope; PhenoTypic pipelines top out around a few dozen blocks total even across all scopes. Add a benchmark in `tests/unit/gui/builder/test_validation_perf.py` to assert ≤5ms on a 100-block fixture; alert if it regresses. |
| Cycle detection false-positives across container boundaries. | Cycle detection runs *within* a scope only (per §4.6 Rule 4); the BFS reachability rule (Rule 2) crosses scopes via aux edges. Container scopes' edges are isolated by construction (Rule §4.4), so Tarjan over the scope's local adjacency is sufficient. |
| Legacy `pipeline.json` files round-trip but lose user-meaningful layout. | The dagre pass produces a deterministic layered layout from any topology; users describe today's saved positions as "decent enough", not load-bearing. |
| Removing `aux_popover.js` mid-phase breaks integration tests. | Phase 1–6 keep the popover behind feature flag; popover removal happens in Phase 7 after the new path is the default. |
| `block_reparent` races with an in-flight `edge_create` (user drags an op into a container while a wire is mid-drag). | Single fan-in callback serialises dispatches by trigger order (`dash.callback_context.triggered_id`); `wire_drawing.js` cancels the in-flight wire on `dragstart` of any block. |
| Compound node performance with cytoscape-dagre on big DAGs. | Pre-bench in Phase 2 with a synthetic 50-block fixture; if frame time exceeds 16ms, gate layout to mutation-only (no live re-layout on pan/zoom). |
| Firefox `layoutstop` event timing differs subtly from Chromium; the `scroll_to` expand-chain may resolve early on Firefox, causing the pan to fire before the inner container has finished animating to its expanded size. | **Known limitation.** Phase 2 spike validates Chromium first; Firefox treated as best-effort for the `scroll_to` chain. If users report mis-pan behaviour on Firefox, gate the feature to Chromium-only via UA sniff + fallback to instant-pan (no animation). Documented in §10 deferred. |
| Asset-load failure mid-session (e.g. cache invalidation during a long session evicts `viewport_ops.js` before it executes). | Each JS asset writes a `window.phenotypic_<name>_ready` sentinel on completion; `builder.js` polls every 500ms and writes the missing-asset list to `STORE_ASSET_STATUS`. Toolbar banner ("Wire drawing offline") informs the user; existing wires + state remain valid; user reloads to recover. |
| Cross-breadcrumb `scroll_to`: offender in a sibling scope requires popping the breadcrumb first. | `scroll_to` payload now carries `target_breadcrumb`; if it differs from `state.breadcrumb`, the chain begins with a `drill_to_scope` dispatch before any expand. Tested in §8.3.6. |

## 10. Open questions (resolved / deferred)

* **Resolved:** Canvas model — Pure DAG (every block free-form, topology
  defines the pipeline).
* **Resolved:** Output sink — implicit (last block in chain). No `Output` node.
* **Resolved:** Layout — auto-layout every render via dagre. No persisted
  positions.
* **Resolved:** Aux fan-out — *one aux source → one consumer*. Reuse means
  drag a fresh block.
* **Resolved:** Cross-container wires — disallowed. Containers are scopes.
* **Resolved:** Wire-color rule — neutral output port; wire colour follows
  the target type (blue = image, purple = aux).
* **Resolved:** Pipeline-as-aux representation — explicit container with
  purple border, title bar, collapse chevron, and a single output port on
  the container's border (not on the tail child).
* **Resolved:** List-aux fan-in — allowed; order set in the inspector; numbered
  badges on canvas wires near the consumer-side endpoint.
* **Resolved:** Validation rules — 6 rules (forks, stubs, required-aux,
  cycles, container-mode, single-Input-Image). Red border + "!" badge +
  toolbar count + click-to-pan.
* **Resolved:** Palette → canvas — native HTML5 drag-and-drop; `Enter` on
  focused palette button places at viewport centre (keyboard fallback).
* **Resolved:** Empty pipeline (only `Input Image`) is valid.
* **Resolved:** Output-port wiring — at most one outgoing wire per port,
  *total* (image *or* aux, never both). The single-wire rule is
  kind-agnostic; if a user wants the same op in two places they drag a
  second block (no shared-instance aux).
* **Resolved:** Stage ordering on the canvas — advisory only (Rule 7,
  yellow border, doesn't block Run/Save). The runtime's partition by
  `isinstance` does the actual reordering.
* **Resolved:** Aux-consumed block visual — purple solid border (1.5px)
  in addition to the stage-tinted background. Lets the user tell
  "consumed as aux" from block chrome alone.
* **Resolved:** Container drag-out semantics — snap-back-with-toast if
  any inner edges would be orphaned. Non-destructive; user must
  disconnect first.
* **Resolved:** Main-path emphasis — main-path edges 3px, aux edges 2px.
  Subtle but readable.
* **Resolved:** Wire colour past the measure boundary — stays blue
  throughout the chain. Stage colour on the block tells the user what
  kind of work happens there.
* **Resolved:** Input Image inspector card — info card with
  `Re-layout` + `Re-anchor view to Input Image` buttons; no
  parameter form, no delete button.
* **Deferred:** Multi-select for bulk operations (lasso-select, shift-click
  → bulk delete, group-into-container, copy-paste). Useful for refactoring
  but adds selection-state machinery beyond v1's scope. Revisit in v2.
* **Deferred (a11y):** Keyboard equivalents for wire-drawing, drag-into-
  container, and drag-out-of-container gestures. v1 supports keyboard
  block creation (Tab + Enter on a palette button places at viewport
  centre) and keyboard wire deletion (click wire → `Delete`). Full
  keyboard-only port-to-port wiring is deferred — gestures would need a
  dedicated "select source port → arrow keys to target → Enter to
  connect" mode that doesn't exist in cytoscape natively. WCAG audit
  will drive the v2 design; if the audit flags this as blocking, the
  scope expands.
* **Deferred (Firefox):** `scroll_to` expand-chain reliability on
  Firefox. `cy.promiseOn("layoutstop")` timing differs subtly between
  Chromium and Firefox; the chain may pan before the inner container
  finishes its expand animation on Firefox. Tested under Chromium only
  (§8.5); Firefox is best-effort. If user reports surface, gate to
  Chromium with a UA sniff + instant-pan fallback for Firefox.
* **Deferred:** Click-to-place fallback for trackpad/touch users (Phase 8+).
* **Deferred:** Multi-image-input ops (not in the registry; revisit if
  added).
* **Deferred:** Wire-level branches (image-flow forks). Future feature if
  ever needed.
* **Deferred:** Position persistence — auto-layout is the v1 default; revisit
  if user feedback wants pinning custom layouts.
* **Open:** Whether an empty `Pipeline` container wired into a *required* aux
  port should be a Rule 3 violation. Today's behaviour at runtime depends on
  the consumer (some classes raise on empty pipelines; others run trivially).
  Recommendation: leave Rule 3 satisfied as-is (the wire exists), let the
  runtime raise — keeps the builder permissive and aligned with the rest of
  the codebase's "trust the constructor" stance. Revisit if user reports
  confusion.

## 11. Glossary

* **Block** — a canvas node. Renders an op, a container, or `Input Image`.
* **Port** — a connection point on a block. Image-in (left blue), image-out
  (right neutral), aux-in (bottom purple).
* **Wire / edge** — a directional connection between an output port and an
  input port. Blue = image-flow; purple = aux assignment.
* **Container** — a `Pipeline` block whose body is a nested `BuilderScope`.
* **Scope** — the contents of one `BuilderScope` (one canvas worth of blocks
  and edges). The root scope is the pipeline; nested scopes live inside
  containers. Each scope auto-seeds an `Input Image` sentinel block on
  creation.
* **Issue** — a validation failure (Rules 1–6). Block border turns red;
  toolbar count goes up; preview/save disabled.
* **Consumer-fed** — the small purple dot on a container's inner-left edge,
  rendering the container's nested `InputImage` source. Lights up when the
  container is wired into an aux target; dims when wired into image-flow.
* **Adoption** — dragging a block over a container's bounds and releasing
  causes the container to take ownership of the block (move it from the
  outer scope into the inner scope).

## 12. FEATURES.md / WORKFLOWS.md gate

The `gui-e2e` CI workflow rejects any PR touching `src/phenotypic/gui/`
without modifying `FEATURES.md`. The `gui-docs` workflow enforces the
`WORKFLOWS.md` → tutorial-page round-trip.

**Phase 1–7 status convention.** Each phase touches `src/phenotypic/gui/`
and must therefore touch `FEATURES.md`, but tests for the new rows may not
land until a later phase. To pass CI without ghost-shipping features:

* New rows added in Phases 1–7 use status **`🚧 in progress`** (the
  existing legend already defines this state — see line 9 of
  `FEATURES.md`).
* `Test ref` cell uses the literal placeholder
  **`tests/.../test_<file>.py::test_<name> [planned]`** with the
  bracketed `[planned]` marker. Pre-commit and the `gui-e2e` workflow
  already accept this form for `🚧 in progress` rows (the gates only
  enforce real test refs on `✅ shipping` rows — see the existing
  `validate_features_md.py` check).
* Phase 8 flips each row from `🚧 in progress` to `✅ shipping`, replaces
  the `[planned]` test ref with the real test path, and removes the
  feature flag.

**Phase 8 deliverables:**

* `FEATURES.md::Builder pipeline editor` and `FEATURES.md::Builder aux
  ports` — the latter section is fully rewritten. New rows for: Input
  Image block (auto-seeded, non-deletable), Block (palette drag-and-drop
  + keyboard fallback), Wire-drawing (drag, replace, delete, type-aware
  highlight, main-path emphasis), Pipeline container (create, expand,
  collapse, drag-in adoption, drag-out snap-back, drill-in, delete with
  confirm, sibling reparent), List-aux port (fan-in, inspector reorder,
  empty slot), Validation (6 blocking rules + 1 advisory hint, toolbar
  count split, issue-click pan/expand), Inspector (block / wire /
  container / Input Image cards).
* `WORKFLOWS.md` — add `aux-wire-in-dag` workflow (3-step tutorial:
  drag a detector onto canvas, drag from its output to a consumer's
  aux port, see the result preview). Add `_capture_aux_wire_in_dag` in
  `scripts/capture_gui_tutorial_screenshots.py`. Tutorial page at
  `docs/source/tutorials/gui/aux-wire-in-dag.rst`.
* `WORKFLOWS.md` — also add `wire-pipeline-as-aux` workflow (build a
  multi-step container, wire it into a consumer's aux port).
* `WORKFLOWS.md` — also add `fix-validation-issues` workflow (introduce
  a fork, see the red border, fix it, see preview re-enable).
