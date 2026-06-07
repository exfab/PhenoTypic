/* PhenoTypic Pipeline Builder — wire drawing clientside glue
 * (spec §4.2, §4.3, §4.9, §5.5).
 *
 * Auto-loaded by Dash via the ``assets/`` convention. Owns port-level
 * wire interactions on the new DAG canvas:
 *
 *   * ``mousedown`` on an image-output port (``dag-port--output``)
 *     spawns a live SVG wire (``<svg><path/>``) overlaid on the
 *     cytoscape canvas tracking the cursor; type-aware highlight
 *     decorates every candidate target port.
 *   * ``mouseup`` on a compatible input port emits an
 *     ``edge_create`` payload to ``STORE_EDGE_EVENT``.  The server
 *     dispatcher handles the single-wire rule (replace any existing
 *     outgoing wire from the same source) and list-aux server-side
 *     slot resolution (spec §5.6).
 *   * Click an existing cytoscape edge → ``wire_select`` partial via
 *     ``STORE_BUILDER_STATE``; ``Delete`` / ``Backspace`` with a wire
 *     selected emits ``edge_delete``.  Right-click on an edge opens a
 *     small DOM context menu with a ``Disconnect`` action.
 *
 * Cancellation paths (spec §4.3):
 *   * ``Esc`` while wire-dragging → wire fades; no state change.
 *   * Drop on empty canvas / outside cytoscape wrapper bounds → fades.
 *   * Drop on incompatible (dimmed) port → red flash on the rejected
 *     port; wire fades.
 *
 * Custom DOM events emitted (spec §5.5):
 *   * ``phenotypic:wire-drop`` — ``{detail: {accepted: bool,
 *     kind: "image" | "aux" | null}}``.
 *
 * Asset-readiness sentinel:
 *   ``window.phenotypic_wire_drawing_ready = true`` once the IIFE has
 *   bound its handlers; ``builder.js``'s poller writes the value into
 *   ``STORE_ASSET_STATUS.wire_drawing`` (spec §5.5 / §6).
 *
 * Conventions:
 *   * Vanilla JS only (no jQuery / ES modules — assets/ load as
 *     ``<script>`` tags via Dash's ``assets/`` glob).
 *   * Polls ``window.phenoGetCy()`` per ``viewport_ops.js`` precedent.
 *   * Idempotent binding via a ``cy.__phenoWireDrawingBound`` sentinel.
 *   * ``mousemove`` is rAF-coalesced — pointer events fire 60-100Hz and
 *     each one would otherwise re-walk every candidate port.
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------
    // Constants — keep in sync with builder/_ids.py + spec §5.5.
    // -----------------------------------------------------------------
    /** Mirror of ``builder/_ids.CANVAS_CYTOSCAPE``. */
    const CY_CANVAS_ID = "canvas-cytoscape";

    /** Wrapper div that hosts the cytoscape slot.  Mouse events outside
     *  the wrapper bounds are treated as drop-on-empty-canvas (spec
     *  §4.3 "Mouse leaves the cytoscape wrapper bounds"). */
    const CY_WRAPPER_ID = "canvas-cytoscape-wrapper";

    /** Mirror of ``builder/_ids.STORE_EDGE_EVENT``; keep the literal
     *  in sync with the server-side ``_ids.py`` value.  Contract:
     *  ``{kind: "edge_create" | "edge_delete",
     *    source_block_id?, target_block_id?, target_port?,
     *    edge_kind?: "image" | "aux", edge_id?, ts}``. */
    const STORE_EDGE_EVENT_ID = "store-edge-event";

    /** Mirror of ``builder/_ids.STORE_BUILDER_STATE``.  Used for the
     *  ``wire_select`` partial when the user clicks an existing edge. */
    const STORE_BUILDER_STATE_ID = "store-builder-state";

    /** ``data.port_kind`` value on output ports (image-output).  Spec
     *  §4.2: every block has exactly one image output. */
    const PORT_KIND_OUT = "image-out";

    /** ``data.port_kind`` value on image-input ports.  Image inputs
     *  accept any output (universal compatibility — spec §4.2). */
    const PORT_KIND_IN = "image-in";

    /** ``data.port_kind`` value on aux-input ports (purple, bottom
     *  edge).  Aux ports carry ``data.accepts: List[str]`` for the
     *  type-aware highlight algorithm. */
    const PORT_KIND_AUX = "aux";

    /** CSS class flipped on candidate input ports during a wire drag
     *  to signal "this is a compatible target" — soft halo via the
     *  cytoscape inline style we apply alongside the class flag. */
    const PORT_GLOW_CLASS = "dag-port--glow";

    /** CSS class flipped on candidate input ports during a wire drag
     *  to signal "this is incompatible" — drops to ~30% opacity. */
    const PORT_DIM_CLASS = "dag-port--dim";

    /** Transient class applied to a rejected port for ~300ms so the
     *  user sees a visible red pulse before the wire fades. */
    const PORT_RED_FLASH_CLASS = "dag-port--red-flash";

    /** Cytoscape edge class flipped on click-selected wires.  CSS rule
     *  in ``builder.css`` widens the stroke and brightens the colour. */
    const WIRE_SELECTED_CLASS = "dag-wire--selected";

    /** Halo + opacity timings.  300ms matches the red-flash duration
     *  used elsewhere in the canvas chrome. */
    const RED_FLASH_MS = 300;
    /** Live-wire fade-out duration (transition).  Long enough for the
     *  user to register the cancellation, short enough not to feel
     *  draggy. */
    const FADE_OUT_MS = 200;

    /** Glow / dim cytoscape inline-style values.  The Python-side
     *  stylesheet (``_canvas_stylesheet``) doesn't carry ``glow`` /
     *  ``dim`` selectors, so we use cytoscape's ``.style()`` API to
     *  apply per-instance overrides during the drag and reset them on
     *  drag end.  Class flags survive too (for test assertions via
     *  ``cy.$('.dag-port--glow')``). */
    const GLOW_STYLE = {
        "border-width": 3,
        "border-color": "#cc79a7", // OI_PURPLE — matches the aux accent.
    };
    const DIM_STYLE = {
        opacity: 0.3,
    };

    /** SVG namespace for the live-wire element. */
    const SVG_NS = "http://www.w3.org/2000/svg";

    /** SVG colour tokens.  Mirrors ``_design.py`` palette literals via
     *  the existing cytoscape canvas (where rem / var are unsupported). */
    const WIRE_COLOR_NEUTRAL = "#6c757d";  // muted gray — drag-in-flight.
    const WIRE_COLOR_IMAGE = "#1b75bc";    // COLOR_BLUE — settled image.
    const WIRE_COLOR_AUX = "#cc79a7";      // OI_PURPLE — settled aux.

    // -----------------------------------------------------------------
    // Asset-readiness sentinel (eager).  builder.js's poller surfaces
    // the value via ``STORE_ASSET_STATUS.wire_drawing``.
    // -----------------------------------------------------------------
    window.phenotypic_wire_drawing_ready = false;

    // -----------------------------------------------------------------
    // Drag state — single in-flight wire drag at a time. ``activeDrag``
    // captures the source port + the live SVG element + the cached
    // candidate target ports decorated for the drag.
    // -----------------------------------------------------------------
    /** {sourceNode, sourceBlockId, sourcePort, sourceClassName,
     *    sourceClientPoint, svgRoot, pathEl, candidates: Map<id, kind>}
     *  ``candidates`` maps cytoscape node id → "compatible" | "incompatible"
     *  so we can revert each port's class + inline style on drag end
     *  without re-scanning every port. */
    let activeDrag = null;

    /** Open DOM context-menu element (Disconnect dropdown).  Tracked
     *  so a subsequent mousedown / scroll can dismiss it. */
    let activeContextMenu = null;

    /** rAF coalescing for the mousemove handler — port-set scanning is
     *  O(V) per frame in the worst case. */
    let mousemoveRafId = null;
    let mousemovePending = null;

    /** Track which cytoscape edge is currently click-selected so the
     *  keydown handler can dispatch ``edge_delete`` for it. */
    let selectedEdgeId = null;

    // -----------------------------------------------------------------
    // Helpers — generic.
    // -----------------------------------------------------------------
    /** Resolve the live cytoscape instance via the shared accessor
     *  ``window.phenoWhenCyReady`` exposed by ``builder.js`` (with a
     *  defensive inline fallback for the cold-load case where this asset
     *  evaluates before ``builder.js``). */
    function whenCyReady(cb) {
        if (typeof window.phenoWhenCyReady === "function") {
            window.phenoWhenCyReady(cb);
            return;
        }
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (cy) {
            cb(cy);
            return;
        }
        if (window.phenoLinearMapMounted && window.phenoLinearMapMounted()) {
            return;
        }
        setTimeout(function () {
            whenCyReady(cb);
        }, 100);
    }

    /** Convert ``event.clientX/Y`` (or ``{clientX, clientY}``) to the
     *  rendered (screen) coordinates inside the cytoscape canvas
     *  container.  Mirrors the formula in ``palette_dnd.js`` but skips
     *  the graph-coord transform — the live wire is drawn in DOM-screen
     *  space relative to the wrapper. */
    function clientToWrapperPoint(wrapper, clientX, clientY) {
        const rect = wrapper.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top,
        };
    }

    /** True when ``(clientX, clientY)`` lies inside ``wrapper``'s
     *  bounding rect.  Used to treat off-wrapper drops as cancellations
     *  (spec §4.3). */
    function isPointInsideWrapper(wrapper, clientX, clientY) {
        const rect = wrapper.getBoundingClientRect();
        return (
            clientX >= rect.left &&
            clientX <= rect.right &&
            clientY >= rect.top &&
            clientY <= rect.bottom
        );
    }

    /** Return the cytoscape node whose ``renderedBoundingBox`` contains
     *  ``(clientX, clientY)`` (translated to renderer coords inside the
     *  cy container).  Used by the drop hit-test — cytoscape's
     *  ``elementsAt`` would also work but is private API on older
     *  builds. */
    function findPortAt(cy, clientX, clientY) {
        const container = document.getElementById(CY_CANVAS_ID);
        if (!container) return null;
        const rect = container.getBoundingClientRect();
        const renderedX = clientX - rect.left;
        const renderedY = clientY - rect.top;
        let hit = null;
        cy.nodes().forEach(function (node) {
            if (hit) return;
            if (!node.data("is_port")) return;
            const bb = node.renderedBoundingBox({ includeOverlays: false });
            if (
                bb.x1 <= renderedX &&
                renderedX <= bb.x2 &&
                bb.y1 <= renderedY &&
                renderedY <= bb.y2
            ) {
                hit = node;
            }
        });
        return hit;
    }

    /** Emit ``phenotypic:wire-drop`` (spec §5.5).  ``kind`` is the
     *  source's intended target kind:
     *    * ``"image"`` — wire drop landed (or was attempted) on an
     *      image-input port (or aborted while one was the most recent
     *      candidate).
     *    * ``"aux"`` — similar for an aux-input port.
     *    * ``null`` — no port hovered; fade-out on empty canvas / Esc. */
    function emitWireDrop(accepted, kind) {
        try {
            document.dispatchEvent(
                new CustomEvent("phenotypic:wire-drop", {
                    detail: { accepted: !!accepted, kind: kind || null },
                })
            );
        } catch (err) {
            // Older browsers without CustomEvent constructor — ignore.
        }
    }

    /** Publish ``payload`` to ``STORE_EDGE_EVENT`` via
     *  ``dash_clientside.set_props`` (Dash 2.18+).  When the API or the
     *  store hasn't mounted yet, the call is a silent no-op — the
     *  dispatcher will simply not run.  Mirrors the publish pattern in
     *  ``palette_dnd.js``. */
    function publishEdgeEvent(payload) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            return;
        }
        try {
            window.dash_clientside.set_props(STORE_EDGE_EVENT_ID, {
                data: payload,
            });
        } catch (err) {
            // Store not yet mounted; silently drop the dispatch — the
            // user can retry.
        }
    }

    /** Hint partial: write a ``wire_select`` action to
     *  ``STORE_BUILDER_STATE``.  The server-side dispatcher is the
     *  canonical authority; this clientside hint produces an immediate
     *  visual cue ahead of the round trip.  Mirrors the same pattern in
     *  ``palette_dnd.js`` (drop-on-wire side-effect). */
    function publishWireSelect(edgeId) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            return;
        }
        try {
            window.dash_clientside.set_props(STORE_BUILDER_STATE_ID, {
                data: {
                    kind: "wire_select",
                    edge_id: edgeId,
                    ts: Date.now(),
                },
            });
        } catch (err) {
            // Store not yet mounted; ignore.
        }
    }

    // -----------------------------------------------------------------
    // Type-aware highlight (spec §4.3 / §5.5 ``accepts``).
    // -----------------------------------------------------------------
    /** Compatibility check for one candidate input port against the
     *  source's ``class_name``.
     *
     *  * Image-input ports are universally compatible (image-flow → blue
     *    wire — spec §4.2 "image-in receives one wire from an upstream
     *    block's output").
     *  * Aux-input ports carry ``data.accepts: List[str]`` (server-side
     *    resolution — spec §5.5).  Compatible iff the source's class
     *    name is in that list. */
    function isPortCompatible(targetNode, sourceClassName) {
        if (!targetNode || !sourceClassName) return false;
        const portKind = targetNode.data("port_kind");
        if (portKind === PORT_KIND_IN) return true;
        if (portKind === PORT_KIND_AUX) {
            const accepts = targetNode.data("accepts");
            if (Array.isArray(accepts)) {
                return accepts.indexOf(sourceClassName) !== -1;
            }
            return false;
        }
        return false;
    }

    /** Decorate ``targetNode`` per the drag-in-flight state.  Stores
     *  the assignment in ``activeDrag.candidates`` so the cleanup loop
     *  can revert both the class flag and the inline cytoscape style.
     *
     *  ``status`` is one of:
     *    * ``"compatible"`` — adds ``dag-port--glow`` + halo style.
     *    * ``"incompatible"`` — adds ``dag-port--dim`` + opacity 0.3.
     *
     *  The source port itself is exempt (skipped by the caller) so the
     *  user can see where the wire originated. */
    function decoratePort(targetNode, status) {
        if (!targetNode || !activeDrag) return;
        const id = targetNode.id();
        // Skip the source port itself.
        if (activeDrag.sourceNode && id === activeDrag.sourceNode.id()) {
            return;
        }
        // Already decorated — keep the existing tag.
        if (activeDrag.candidates.has(id)) return;
        activeDrag.candidates.set(id, status);
        try {
            if (status === "compatible") {
                targetNode.addClass(PORT_GLOW_CLASS);
                targetNode.style(GLOW_STYLE);
            } else {
                targetNode.addClass(PORT_DIM_CLASS);
                targetNode.style(DIM_STYLE);
            }
        } catch (err) {
            // Cytoscape may have torn down the node mid-drag — ignore.
        }
    }

    /** Decorate every candidate input port (image-input or aux-input)
     *  in the current scope based on its compatibility with the dragged
     *  source.  Runs once on ``mousedown`` so the user sees the target
     *  set immediately. */
    function decorateAllCandidatePorts(cy) {
        if (!cy || !activeDrag) return;
        cy.nodes().forEach(function (node) {
            if (!node.data("is_port")) return;
            const portKind = node.data("port_kind");
            if (portKind !== PORT_KIND_IN && portKind !== PORT_KIND_AUX) {
                return;
            }
            const compatible = isPortCompatible(
                node, activeDrag.sourceClassName
            );
            decoratePort(node, compatible ? "compatible" : "incompatible");
        });
    }

    /** Revert every decorated port back to its baseline appearance.
     *  Cleared on drag end (mouseup, Esc, off-wrapper drop). */
    function clearPortDecorations(cy) {
        if (!cy || !activeDrag) return;
        activeDrag.candidates.forEach(function (_status, id) {
            try {
                const node = cy.getElementById(id);
                if (!node || !node.length) return;
                node.removeClass(PORT_GLOW_CLASS);
                node.removeClass(PORT_DIM_CLASS);
                // Reset inline-style overrides applied by ``decoratePort``.
                node.removeStyle("border-width");
                node.removeStyle("border-color");
                node.removeStyle("opacity");
            } catch (err) {
                // Node tear-down — ignore.
            }
        });
        activeDrag.candidates.clear();
    }

    /** Flash a port red for ``RED_FLASH_MS`` to indicate rejection
     *  (spec §4.3: "drop on a dimmed port → red flash on the rejected
     *  port"). */
    function flashRejectedPort(cy, portNode) {
        if (!portNode) return;
        try {
            portNode.addClass(PORT_RED_FLASH_CLASS);
            // Inline cytoscape style for the actual red border — the
            // cytoscape stylesheet doesn't carry a selector for this
            // transient class.
            portNode.style({
                "border-width": 3,
                "border-color": "#d55e00",  // OI_VERMILION
            });
        } catch (err) {
            // Tear-down — ignore.
        }
        setTimeout(function () {
            try {
                portNode.removeClass(PORT_RED_FLASH_CLASS);
                portNode.removeStyle("border-width");
                portNode.removeStyle("border-color");
            } catch (err) {
                // Tear-down — ignore.
            }
        }, RED_FLASH_MS);
    }

    // -----------------------------------------------------------------
    // Live-wire SVG (spec §4.3 + §5.5 DOM test IDs).
    // -----------------------------------------------------------------
    /** Build the SVG overlay that hosts the in-flight wire.  Returns
     *  ``{svg, path}``; the caller appends ``svg`` into the wrapper. */
    function buildLiveWireSvg(originPoint) {
        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("data-testid", "live-wire");
        svg.setAttribute("data-state", "dragging");
        svg.style.position = "absolute";
        svg.style.top = "0";
        svg.style.left = "0";
        svg.style.width = "100%";
        svg.style.height = "100%";
        svg.style.pointerEvents = "none";
        svg.style.zIndex = "100";
        svg.style.transition = "opacity " + FADE_OUT_MS + "ms ease-out";

        const path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("stroke", WIRE_COLOR_NEUTRAL);
        path.setAttribute("stroke-width", "2");
        path.setAttribute("stroke-dasharray", "6 4");
        path.setAttribute("fill", "none");
        path.setAttribute(
            "d",
            "M " + originPoint.x + " " + originPoint.y +
                " L " + originPoint.x + " " + originPoint.y
        );
        svg.appendChild(path);
        return { svg: svg, path: path };
    }

    /** Update the live wire's endpoint to track the cursor.  Uses a
     *  shallow bezier so the wire reads as draggable rather than a hard
     *  straight line. */
    function updateLiveWirePath(pathEl, originPoint, cursorPoint) {
        if (!pathEl) return;
        const dx = cursorPoint.x - originPoint.x;
        const c1x = originPoint.x + dx * 0.5;
        const c1y = originPoint.y;
        const c2x = originPoint.x + dx * 0.5;
        const c2y = cursorPoint.y;
        pathEl.setAttribute(
            "d",
            "M " + originPoint.x + " " + originPoint.y +
                " C " + c1x + " " + c1y +
                " " + c2x + " " + c2y +
                " " + cursorPoint.x + " " + cursorPoint.y
        );
    }

    /** Recolour the live wire on a settle (drop on compatible target)
     *  before its parent SVG fades out.  Blue solid for image-in,
     *  purple dashed for aux-in. */
    function settleLiveWirePath(svg, pathEl, kind) {
        if (!svg || !pathEl) return;
        try {
            svg.setAttribute("data-state", "settled");
            if (kind === "aux") {
                pathEl.setAttribute("stroke", WIRE_COLOR_AUX);
                pathEl.setAttribute("stroke-dasharray", "6 4");
            } else {
                pathEl.setAttribute("stroke", WIRE_COLOR_IMAGE);
                pathEl.removeAttribute("stroke-dasharray");
            }
        } catch (err) {
            // Tear-down — ignore.
        }
    }

    /** Fade out and remove the live wire's SVG from the DOM.  Adds a
     *  secondary testid (``wire-cancel-anim``) during the fade so
     *  Playwright tests can ``waitForSelector`` on the cancellation
     *  marker — the spec-defined testid pair (``live-wire`` +
     *  ``wire-cancel-anim``) lets tests assert "wire was canceled, not
     *  accepted". */
    function teardownLiveWire(svg, immediate) {
        if (!svg) return;
        const wrapper = svg.parentNode;
        // Add the fade-out testid as a sibling attribute so the
        // original ``data-testid="live-wire"`` stays queryable during
        // the fade window.  We use ``data-cancel-anim`` to avoid
        // overwriting the original ``data-testid`` attribute.
        try {
            svg.setAttribute("data-cancel-anim", "1");
        } catch (err) {
            // ignore
        }
        if (immediate) {
            if (wrapper) {
                try {
                    wrapper.removeChild(svg);
                } catch (err) {
                    // ignore
                }
            }
            return;
        }
        // Mount a ``wire-cancel-anim`` testid wrapper sibling so tests
        // can ``waitForSelector('[data-testid=wire-cancel-anim]')``
        // during the fade window.  The marker is removed once the
        // fade completes.
        let cancelMarker = null;
        if (wrapper) {
            try {
                cancelMarker = document.createElement("div");
                cancelMarker.setAttribute("data-testid", "wire-cancel-anim");
                cancelMarker.style.display = "none";
                wrapper.appendChild(cancelMarker);
            } catch (err) {
                cancelMarker = null;
            }
        }
        svg.style.opacity = "0";
        setTimeout(function () {
            try {
                if (svg && svg.parentNode) {
                    svg.parentNode.removeChild(svg);
                }
            } catch (err) {
                // ignore — DOM may have been torn down.
            }
            try {
                if (cancelMarker && cancelMarker.parentNode) {
                    cancelMarker.parentNode.removeChild(cancelMarker);
                }
            } catch (err) {
                // ignore
            }
        }, FADE_OUT_MS + 20);
    }

    // -----------------------------------------------------------------
    // Cytoscape user-pan suppression (spec §4.3 "Suppress cytoscape pan
    // during drag").
    // -----------------------------------------------------------------
    /** Toggle ``cy.userPanningEnabled``.  Wrapped so we can swallow
     *  errors if cytoscape was torn down between drag start and end. */
    function setCyPanning(cy, enabled) {
        if (!cy || typeof cy.userPanningEnabled !== "function") return;
        try {
            cy.userPanningEnabled(enabled);
        } catch (err) {
            // ignore
        }
    }

    // -----------------------------------------------------------------
    // Drag lifecycle.
    // -----------------------------------------------------------------
    /** Resolve the source port's ``class_name``.  Output ports
     *  themselves are compound children of their parent block; the
     *  ``class_name`` we need for type-aware highlight lives on the
     *  parent block's ``data``.  Walk one step up via cytoscape's
     *  ``parent()``. */
    function resolveSourceClassName(portNode) {
        if (!portNode) return null;
        // Prefer the explicit ``block_id`` data on the port itself.
        const blockId = portNode.data("block_id");
        const cy = portNode.cy && portNode.cy();
        if (cy && blockId) {
            const block = cy.getElementById(blockId);
            if (block && block.length) {
                return block.data("class_name") || null;
            }
        }
        const parent = portNode.parent && portNode.parent();
        if (parent && parent.length) {
            return parent.data("class_name") || null;
        }
        return null;
    }

    /** Begin a wire drag from ``portNode``.  Spawns the live SVG,
     *  decorates all candidate target ports, suppresses pan, and binds
     *  document-level ``mousemove`` / ``mouseup`` / ``keydown``
     *  listeners.  Idempotent — if a drag is already in flight, a
     *  fresh mousedown is treated as a fresh start (abort the
     *  in-flight drag first). */
    function startDrag(cy, portNode, event) {
        if (activeDrag) {
            // Defensive: shouldn't happen given the document-level
            // mouseup that ends the drag, but tear down anything left
            // over from a previous drag before starting fresh.
            endDrag(cy, "cancel");
        }
        if (!portNode) return;

        const wrapper = document.getElementById(CY_WRAPPER_ID);
        if (!wrapper) return;

        const blockId = portNode.data("block_id");
        const portName = portNode.data("port");
        const sourceClassName = resolveSourceClassName(portNode);

        // Use the port's rendered centre as the wire's origin so the
        // first frame doesn't snap to the cursor (jumpy).  ``rendered*``
        // coords are in container-pixel space; convert to wrapper-pixel
        // space via the bounding rect diff.
        const cyContainer = document.getElementById(CY_CANVAS_ID);
        const cyRect = cyContainer
            ? cyContainer.getBoundingClientRect()
            : null;
        const wrapRect = wrapper.getBoundingClientRect();
        let originPoint;
        if (cyRect && typeof portNode.renderedPosition === "function") {
            const rp = portNode.renderedPosition();
            originPoint = {
                x: rp.x + (cyRect.left - wrapRect.left),
                y: rp.y + (cyRect.top - wrapRect.top),
            };
        } else {
            originPoint = clientToWrapperPoint(
                wrapper, event.clientX, event.clientY
            );
        }
        const liveWire = buildLiveWireSvg(originPoint);
        wrapper.appendChild(liveWire.svg);

        activeDrag = {
            sourceNode: portNode,
            sourceBlockId: blockId,
            sourcePort: portName,
            sourceClassName: sourceClassName,
            sourcePoint: originPoint,
            wrapper: wrapper,
            svgRoot: liveWire.svg,
            pathEl: liveWire.path,
            candidates: new Map(),
        };

        setCyPanning(cy, false);
        decorateAllCandidatePorts(cy);

        // Document-level binding so the drag completes even if the
        // cursor wanders off the wrapper / over a different DOM layer.
        document.addEventListener("mousemove", onDocMouseMove, true);
        document.addEventListener("mouseup", onDocMouseUp, true);
        document.addEventListener("keydown", onDragKeyDown, true);

        // Cytoscape would otherwise drag the block (or pan).  The
        // ``box`` selection action is suppressed by setting
        // userPanningEnabled = false above; preventing default on the
        // browser event suppresses the native text-selection drag too.
        event.preventDefault();
        event.stopPropagation();
    }

    /** rAF-coalesced mousemove handler.  Updates the live wire's
     *  endpoint to track the cursor.  Pointer events fire 60-100Hz; the
     *  cy-scope-wide port set is decorated once on dragstart, so the
     *  per-frame work is just the SVG path geometry (a single DOM
     *  attribute write).  Hover hit-testing is deferred to ``mouseup``
     *  so we don't pay O(P) ``renderedBoundingBox`` calls per animation
     *  frame just to keep ``lastHoverKind`` warm — that metadata is
     *  only consulted on cancellation paths that pass an explicit
     *  ``cancelKind`` or where ``null`` is a perfectly fine fallback
     *  (Esc / off-wrapper / empty-canvas — all UI-cosmetic). */
    function onDocMouseMove(event) {
        if (!activeDrag) return;
        mousemovePending = { clientX: event.clientX, clientY: event.clientY };
        if (mousemoveRafId !== null) return;
        mousemoveRafId = requestAnimationFrame(function () {
            mousemoveRafId = null;
            const pending = mousemovePending;
            mousemovePending = null;
            if (!activeDrag || pending === null) return;
            const wrapper = activeDrag.wrapper;
            if (!wrapper) return;
            const cursorPoint = clientToWrapperPoint(
                wrapper, pending.clientX, pending.clientY
            );
            updateLiveWirePath(
                activeDrag.pathEl,
                activeDrag.sourcePoint,
                cursorPoint
            );
        });
    }

    /** Mouseup handler — runs the drop hit-test, dispatches the
     *  appropriate ``edge_create`` (or no-op on cancellation), and
     *  tears down the drag. */
    function onDocMouseUp(event) {
        if (!activeDrag) return;
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (!cy) {
            endDrag(cy, "cancel");
            return;
        }
        const wrapper = activeDrag.wrapper;
        // Off-wrapper drop → treat as empty canvas (spec §4.3).
        if (
            wrapper &&
            !isPointInsideWrapper(wrapper, event.clientX, event.clientY)
        ) {
            endDrag(cy, "cancel");
            return;
        }
        const target = findPortAt(cy, event.clientX, event.clientY);
        if (!target || !target.data("is_port")) {
            endDrag(cy, "cancel");
            return;
        }
        const portKind = target.data("port_kind");
        // Only image-in / aux input ports are valid drop targets.
        if (portKind !== PORT_KIND_IN && portKind !== PORT_KIND_AUX) {
            endDrag(cy, "cancel");
            return;
        }
        // Reject drops on the source's own ports (no self-wiring).
        const targetBlockId = target.data("block_id");
        if (targetBlockId && targetBlockId === activeDrag.sourceBlockId) {
            // Treat as incompatible-flash so the user sees a clear
            // rejection.
            flashRejectedPort(cy, target);
            endDrag(cy, "cancel", portKind === PORT_KIND_AUX ? "aux" : "image");
            return;
        }
        const compatible = isPortCompatible(target, activeDrag.sourceClassName);
        const kind = portKind === PORT_KIND_AUX ? "aux" : "image";
        if (!compatible) {
            flashRejectedPort(cy, target);
            endDrag(cy, "cancel", kind);
            return;
        }
        // Compatible drop — settle the wire, emit ``edge_create``.
        // Spec §5.6 line for ``edge_create``: the server-side
        // dispatcher handles the single-wire rule (replace any
        // existing outgoing wire from ``source_block_id`` in the same
        // dispatch tick).  ``edge_replace`` was *dropped* in the spec
        // (§5.6 "Dropped dispatch kinds (popover legacy)") — the
        // gesture lives entirely on the dispatcher side; the client
        // simply emits ``edge_create`` and the dispatcher performs the
        // atomic replace.
        settleLiveWirePath(activeDrag.svgRoot, activeDrag.pathEl, kind);
        const payload = {
            kind: "edge_create",
            source_block_id: activeDrag.sourceBlockId,
            target_block_id: target.data("block_id"),
            target_port: target.data("port"),
            edge_kind: kind,
            ts: Date.now(),
        };
        publishEdgeEvent(payload);
        emitWireDrop(true, kind);
        // Server dispatcher will re-render the canvas with the new
        // edge; tear down the transient live wire on the next frame.
        endDrag(cy, "accept", kind);
    }

    /** Keyboard cancel — ``Esc`` while wire-dragging fades the wire and
     *  emits ``phenotypic:wire-drop`` ``{accepted: false}``.  Also
     *  catches stray ``Delete`` / ``Backspace`` keys during a drag to
     *  prevent accidental edge deletion mid-gesture. */
    function onDragKeyDown(event) {
        if (!activeDrag) return;
        if (event.key === "Escape") {
            event.preventDefault();
            const cy = window.phenoGetCy && window.phenoGetCy();
            endDrag(cy, "cancel");
        }
    }

    /** Tear down the in-flight drag.  Mode is one of:
     *    * ``"accept"`` — compatible drop landed; settle + short fade.
     *    * ``"cancel"`` — Esc, off-wrapper, empty-canvas, incompatible.
     *
     *  ``cancelKind`` carries the source's intended wire kind so the
     *  emitted ``phenotypic:wire-drop`` event can report it on
     *  cancellation paths. */
    function endDrag(cy, mode, cancelKind) {
        if (!activeDrag) return;
        // Unbind document-level listeners.
        document.removeEventListener("mousemove", onDocMouseMove, true);
        document.removeEventListener("mouseup", onDocMouseUp, true);
        document.removeEventListener("keydown", onDragKeyDown, true);

        if (mousemoveRafId !== null) {
            cancelAnimationFrame(mousemoveRafId);
            mousemoveRafId = null;
        }
        mousemovePending = null;

        clearPortDecorations(cy);
        setCyPanning(cy, true);

        if (mode === "accept") {
            // Live wire briefly stays in its settled state before the
            // canvas re-renders with the real cytoscape edge.  Short
            // fade so the swap reads as continuous.
            teardownLiveWire(activeDrag.svgRoot, false);
        } else {
            // Cancellation path — fade out.  ``cancelKind`` is supplied
            // explicitly by callers that already hit-tested the drop
            // target (incompatible drop, self-wire); Esc / off-wrapper /
            // empty-canvas cancellations pass ``undefined`` and emit
            // ``kind: null``, matching the documented contract.
            teardownLiveWire(activeDrag.svgRoot, false);
            emitWireDrop(false, cancelKind || null);
        }

        activeDrag = null;
    }

    // -----------------------------------------------------------------
    // Output-port mousedown handler (spec §4.3 entry point).
    //
    // Cytoscape doesn't bubble raw ``mousedown`` to the document for
    // canvas-rendered nodes, so we use ``cy.on('mousedown', 'node',
    // ...)`` to intercept the gesture at the cytoscape layer.  The
    // handler discriminates by ``data.port_kind === "image-out"`` and
    // hands off to :func:`startDrag`.
    // -----------------------------------------------------------------
    function onCyOutputPortMouseDown(evt) {
        const node = evt.target || evt.cyTarget;
        if (!node || typeof node.data !== "function") return;
        if (!node.data("is_port")) return;
        if (node.data("port_kind") !== PORT_KIND_OUT) return;
        // Skip when the user is mid-pan with the spacebar (rare; defense
        // in depth — we already suppress pan on dragstart).  The
        // browser-level event is on ``evt.originalEvent``.
        const cy = node.cy && node.cy();
        const raw = evt.originalEvent;
        if (!raw) return;
        // Right-clicks on output ports are not drag triggers (cytoscape
        // routes button 2 through ``cxttap``).  Only left button starts
        // a wire drag.
        if (typeof raw.button === "number" && raw.button !== 0) return;
        startDrag(cy, node, raw);
    }

    // -----------------------------------------------------------------
    // Edge selection + delete (spec §4.3 / §4.9).
    // -----------------------------------------------------------------
    /** Cytoscape ``tap`` on an edge → mark it selected, emit a
     *  ``wire_select`` partial to ``STORE_BUILDER_STATE``, and remember
     *  the id so a subsequent ``Delete`` / ``Backspace`` keypress can
     *  emit ``edge_delete``. */
    function onCyEdgeTap(evt) {
        const edge = evt.target || evt.cyTarget;
        if (!edge || typeof edge.data !== "function") return;
        if (!edge.isEdge || !edge.isEdge()) return;
        selectEdge(edge);
    }

    /** Click on empty canvas clears any edge selection. */
    function onCyBackgroundTap(evt) {
        const target = evt.target;
        const cy = evt.cy;
        // ``evt.target`` is the cy instance itself when the click landed
        // on the canvas background.
        if (target !== cy) return;
        clearEdgeSelection(cy);
    }

    /** Apply selection styling + bookkeeping for a clicked edge. */
    function selectEdge(edge) {
        const cy = edge.cy && edge.cy();
        if (!cy) return;
        clearEdgeSelection(cy);
        try {
            edge.addClass(WIRE_SELECTED_CLASS);
            // Apply inline cytoscape style for the selected wire
            // affordance — stylesheet doesn't carry this selector.
            edge.style({
                width: 4,
                "line-color": "#003660",  // COLOR_NAVY
                "target-arrow-color": "#003660",
            });
        } catch (err) {
            // ignore
        }
        selectedEdgeId = edge.data("edge_id") || edge.id();
        publishWireSelect(selectedEdgeId);
    }

    /** Revert every previously-selected edge's styling. */
    function clearEdgeSelection(cy) {
        if (!cy) return;
        try {
            cy.edges("." + WIRE_SELECTED_CLASS).forEach(function (edge) {
                edge.removeClass(WIRE_SELECTED_CLASS);
                edge.removeStyle("width");
                edge.removeStyle("line-color");
                edge.removeStyle("target-arrow-color");
            });
        } catch (err) {
            // ignore
        }
        selectedEdgeId = null;
    }

    /** Document-level ``keydown`` for the wire-selected ``Delete`` /
     *  ``Backspace`` shortcut.  Intentionally NOT bound during a drag —
     *  ``onDragKeyDown`` swallows Esc; this handler runs only when
     *  there's no active drag. */
    function onGlobalKeyDown(event) {
        if (activeDrag) return;
        if (event.key !== "Delete" && event.key !== "Backspace") return;
        if (!selectedEdgeId) return;
        // Skip when focus is inside an editable field — don't intercept
        // text-deletion gestures.
        const active = document.activeElement;
        if (active) {
            const tag = active.tagName;
            if (
                tag === "INPUT" ||
                tag === "TEXTAREA" ||
                tag === "SELECT" ||
                active.isContentEditable
            ) {
                return;
            }
        }
        event.preventDefault();
        publishEdgeEvent({
            kind: "edge_delete",
            edge_id: selectedEdgeId,
            ts: Date.now(),
        });
        // The server re-renders without the edge; reset our local
        // selection tracking.
        selectedEdgeId = null;
    }

    // -----------------------------------------------------------------
    // Right-click → small DOM context menu (spec §4.3 "Right-click →
    // Disconnect").
    // -----------------------------------------------------------------
    /** Dismiss any open context menu.  Bound on ``mousedown`` /
     *  ``scroll`` once a menu is showing so it behaves like a native
     *  popover. */
    function dismissContextMenu() {
        if (!activeContextMenu) return;
        try {
            if (activeContextMenu.parentNode) {
                activeContextMenu.parentNode.removeChild(activeContextMenu);
            }
        } catch (err) {
            // ignore
        }
        activeContextMenu = null;
        document.removeEventListener("mousedown", onDocMenuDismiss, true);
        document.removeEventListener("scroll", dismissContextMenu, true);
    }

    function onDocMenuDismiss(event) {
        if (!activeContextMenu) return;
        if (activeContextMenu.contains(event.target)) return;
        dismissContextMenu();
    }

    /** Open a small DOM context menu at ``(clientX, clientY)`` with a
     *  single ``Disconnect`` action that emits ``edge_delete`` for the
     *  given ``edgeId``. */
    function showEdgeContextMenu(edgeId, clientX, clientY) {
        dismissContextMenu();
        const menu = document.createElement("div");
        menu.setAttribute("data-testid", "wire-context-menu");
        menu.className = "dag-wire-context-menu";
        menu.style.position = "fixed";
        menu.style.top = clientY + "px";
        menu.style.left = clientX + "px";
        menu.style.zIndex = "1000";
        menu.style.background = "var(--color-surface, #fff)";
        menu.style.border = "1px solid var(--color-border, #cdd5e0)";
        menu.style.borderRadius = "var(--radius, 6px)";
        menu.style.boxShadow = "var(--shadow-md, 0 6px 16px rgba(0,0,0,0.15))";
        menu.style.padding = "4px";
        menu.style.fontFamily = "var(--font-body, sans-serif)";
        menu.style.fontSize = "var(--font-size-body, 14px)";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = "Disconnect";
        btn.setAttribute("data-testid", "wire-context-menu-disconnect");
        btn.style.display = "block";
        btn.style.width = "100%";
        btn.style.padding = "4px 12px";
        btn.style.background = "transparent";
        btn.style.border = "none";
        btn.style.color = "var(--oi-vermilion, #d55e00)";
        btn.style.cursor = "pointer";
        btn.style.textAlign = "left";
        btn.style.font = "inherit";

        btn.addEventListener("click", function () {
            publishEdgeEvent({
                kind: "edge_delete",
                edge_id: edgeId,
                ts: Date.now(),
            });
            selectedEdgeId = null;
            dismissContextMenu();
        });
        menu.appendChild(btn);
        document.body.appendChild(menu);
        activeContextMenu = menu;

        // Defer the dismiss-binding by one tick so the very ``mousedown``
        // that opened the menu doesn't immediately close it.
        setTimeout(function () {
            document.addEventListener("mousedown", onDocMenuDismiss, true);
            document.addEventListener("scroll", dismissContextMenu, true);
        }, 0);
    }

    /** Cytoscape ``cxttap`` (right-click) on an edge.  Show the
     *  Disconnect context menu at the original browser cursor coords. */
    function onCyEdgeCxtTap(evt) {
        const edge = evt.target || evt.cyTarget;
        if (!edge || typeof edge.data !== "function") return;
        if (!edge.isEdge || !edge.isEdge()) return;
        const raw = evt.originalEvent;
        if (!raw) return;
        const edgeId = edge.data("edge_id") || edge.id();
        // Select the edge too so the inspector card matches the menu's
        // target — parity with native right-click semantics.
        selectEdge(edge);
        showEdgeContextMenu(edgeId, raw.clientX, raw.clientY);
        // Prevent the browser's native context menu from appearing.
        if (typeof raw.preventDefault === "function") {
            raw.preventDefault();
        }
    }

    // -----------------------------------------------------------------
    // Cytoscape binding lifecycle.  Cytoscape swaps its DOM container
    // on every state mutation; the wire-drawing handlers re-bind via
    // a per-cy sentinel so we don't double-bind on a single instance.
    // -----------------------------------------------------------------
    function bindCytoscape(cy) {
        if (!cy) return;
        if (cy.__phenoWireDrawingBound) return;
        cy.__phenoWireDrawingBound = true;

        // Output-port mousedown → start wire drag.
        cy.on("mousedown", "node", onCyOutputPortMouseDown);
        // Edge click → select.
        cy.on("tap", "edge", onCyEdgeTap);
        // Background click → clear selection.
        cy.on("tap", onCyBackgroundTap);
        // Edge right-click → context menu.
        cy.on("cxttap", "edge", onCyEdgeCxtTap);

        // Suppress the native browser context menu over the cytoscape
        // canvas — our cytoscape-side ``cxttap`` handler owns the right-
        // click gesture, but the browser would otherwise open the OS
        // context menu over the canvas pixel and mask the DOM menu we
        // mount.  Bound once per cy container; safe to attach because
        // ``cxttap`` already does its own routing internally.
        const cyContainer = cy.container && cy.container();
        if (cyContainer && !cyContainer.__phenoWireCtxBound) {
            cyContainer.__phenoWireCtxBound = true;
            cyContainer.addEventListener("contextmenu", function (event) {
                // Only suppress when an edge was hit — otherwise let
                // the browser show its default menu (e.g. for empty
                // canvas right-clicks the spec leaves open).
                const target = findPortAt(cy, event.clientX, event.clientY);
                if (target) {
                    // Suppress only over input/output/aux ports so the
                    // user doesn't see two menus stacked.
                    event.preventDefault();
                    return;
                }
                // No port hit; check edges via cytoscape's own
                // hit-test by hovering at the cursor coords.
                if (selectedEdgeId) {
                    event.preventDefault();
                }
            });
        }
    }

    /** Bind the global keydown handler once per document.  Survives
     *  cytoscape re-renders because it lives at the document level. */
    function bindGlobalKeyHandler() {
        if (document.__phenoWireKeyBound) return;
        document.__phenoWireKeyBound = true;
        document.addEventListener("keydown", onGlobalKeyDown);
    }

    /** Watch for the cytoscape instance being replaced (Dash may rebuild
     *  the canvas wrapper subtree on certain state mutations).  Rebinds
     *  the cytoscape handlers each time a fresh instance appears. */
    function watchCytoscape() {
        if (typeof MutationObserver === "undefined") return;
        const observer = new MutationObserver(function () {
            const cy = window.phenoGetCy && window.phenoGetCy();
            if (cy && !cy.__phenoWireDrawingBound) {
                bindCytoscape(cy);
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // -----------------------------------------------------------------
    // Module init.
    // -----------------------------------------------------------------
    whenCyReady(function (cy) {
        bindCytoscape(cy);
        bindGlobalKeyHandler();
        watchCytoscape();
        window.phenotypic_wire_drawing_ready = true;
    });
})();
