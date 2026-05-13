/* PhenoTypic Pipeline Builder — palette drag-and-drop clientside glue
 * (spec §4.8 + §5.5).
 *
 * Auto-loaded by Dash via the ``assets/`` convention. Wires native HTML5
 * drag-and-drop from the palette buttons (left rail, server-rendered with
 * ``draggable="true"`` + ``data-palette-class="<ClassName>"``) onto the
 * cytoscape canvas (id ``canvas-cytoscape``).
 *
 * Drag → drop flow:
 *   1. ``dragstart`` on a palette button caches the dragged class name and
 *      installs a ghost element (``data-testid="palette-ghost"``) as the
 *      drag image.
 *   2. ``dragover`` on the cytoscape wrapper hit-tests the cursor against
 *      the innermost ``.dag-block--container`` (cytoscape compound),
 *      toggling ``.dag-block--container-hover`` on the matched node.
 *   3. ``drop`` translates the client coords to cytoscape graph coords via
 *      the explicit ``(x - cy.pan().x) / cy.zoom()`` formula (spec §4.8),
 *      detects an existing-block / wire / container target, and writes a
 *      ``{kind: "block_create", class_name, x, y, container_block_id,
 *      ts}`` payload to ``STORE_PALETTE_DROP`` via
 *      ``dash_clientside.set_props("store-palette-drop", {data: payload})``.
 *      Emits the ``phenotypic:palette-drop`` custom DOM event so Playwright
 *      can ``page.waitForEvent`` on it.
 *
 * Cancellation paths (spec §4.8):
 *   * Off-canvas drop / ``Esc``: ``dragend`` observes
 *     ``dataTransfer.dropEffect === "none"`` and emits nothing.
 *   * Drop outside the cytoscape slot: ``drop`` never fires on the canvas
 *     wrapper, so the IIFE silently no-ops.
 *
 * Keyboard fallback (spec §4.8):
 *   ``Enter`` / ``Space`` on a focused palette button dispatches
 *   ``block_create`` at the centre of the current cytoscape viewport
 *   (``cy.extent()`` midpoint).
 *
 * Custom DOM events emitted:
 *   * ``phenotypic:palette-drop`` — fires on every drop attempt (accepted
 *     or rejected) with ``{detail: {class_name, accepted}}``.
 *
 * Asset-readiness sentinel:
 *   ``window.phenotypic_palette_dnd_ready = true`` once the IIFE has bound
 *   its palette + canvas handlers; ``builder.js``'s poller writes the
 *   value into ``STORE_ASSET_STATUS.palette_dnd`` (spec §5.5 / §6).
 *
 * Conventions:
 *   * Vanilla JS only (no jQuery / ES modules — assets/ are loaded via
 *     ``<script>`` tags by Dash).
 *   * Polls ``window.phenoGetCy()`` per ``viewport_ops.js`` precedent.
 *   * Co-located with ``viewport_ops.js`` + ``wire_drawing.js`` (Phase 4);
 *     each IIFE writes its own readiness sentinel.
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------
    // Constants — keep in sync with builder/_ids.py.
    // -----------------------------------------------------------------
    /** Mirror of ``builder/_ids.CANVAS_CYTOSCAPE``. */
    const CY_CANVAS_ID = "canvas-cytoscape";

    /** Wrapper div that hosts the cytoscape slot (see ``_layout.py``).
     *  The wrapper survives Dash subtree swaps; the inner cytoscape
     *  container may be replaced when ``elements`` re-render. */
    const CY_WRAPPER_ID = "canvas-cytoscape-wrapper";

    /** Mirror of ``builder/_ids.STORE_PALETTE_DROP`` (Agent 3B). The
     *  literal is hard-coded because ``_ids.py`` may not have shipped
     *  the constant when this asset first loads. */
    const STORE_PALETTE_DROP_ID = "store-palette-drop";

    /** Mirror of ``builder/_ids.STORE_BUILDER_STATE``. Used for the
     *  drop-on-wire side-effect of selecting the wire. */
    const STORE_BUILDER_STATE_ID = "store-builder-state";

    /** Server-rendered palette buttons carry this attribute (Agent 3B
     *  ensures it lands on the dbc.Button DOM). */
    const PALETTE_CLASS_ATTR = "data-palette-class";

    /** Cytoscape class on container compound nodes. Matches
     *  ``_layout._block_classes`` and ``viewport_ops.js`` so the
     *  hit-test stays consistent. */
    const CONTAINER_CSS_CLASS = "dag-block--container";

    /** DOM class toggled on a hovered container during dragover; styled
     *  by ``builder.css`` (subtle outline + tint). */
    const CONTAINER_HOVER_CLASS = "dag-block--container-hover";

    /** MIME type used on ``dataTransfer``. Browsers require a non-empty
     *  set-data call for the drag to register on some platforms. */
    const DT_MIME = "application/x-phenotypic-class";

    /** Horizontal offset (px in graph coords) applied when the user
     *  drops on an existing non-container block. Mirrors
     *  ``viewport_ops.DAGRE_NODE_SEP`` + a small buffer so the new
     *  block clears the existing block's bounding box. */
    const ADJACENT_OFFSET_X = 150;

    // -----------------------------------------------------------------
    // Asset-readiness sentinel (eager). The IIFE flips this to ``true``
    // after binding; ``builder.js``'s poller surfaces the value via
    // ``STORE_ASSET_STATUS``.
    // -----------------------------------------------------------------
    window.phenotypic_palette_dnd_ready = false;

    // -----------------------------------------------------------------
    // Drag state — single in-flight drag at a time. ``activeDrag`` is
    // ``null`` between drags.
    // -----------------------------------------------------------------
    let activeDrag = null; // {className, startX, startY, ts}
    let hoverContainerId = null; // cytoscape node id of currently hovered container

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------
    /** Resolve the live cytoscape instance via ``window.phenoGetCy``.
     *  Mirrors ``viewport_ops.js`` so this asset works regardless of
     *  load order. */
    function whenCyReady(cb) {
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (cy) {
            cb(cy);
            return;
        }
        setTimeout(function () {
            whenCyReady(cb);
        }, 100);
    }

    /** Walk every cytoscape container node that contains ``(graphX,
     *  graphY)`` in its bounding box and return the deepest one
     *  (innermost-wins). Returns ``null`` when no container is under the
     *  cursor.
     *
     *  Depth is measured via ``node.parents().length`` so a doubly-
     *  nested container ranks above its outer wrapper. */
    function findInnermostContainer(cy, graphX, graphY) {
        const containers = cy.nodes("." + CONTAINER_CSS_CLASS);
        let best = null;
        let bestDepth = -1;
        containers.forEach(function (node) {
            const bb = node.boundingBox({
                includeOverlays: false,
                includeLabels: false,
            });
            if (
                bb.x1 <= graphX &&
                graphX <= bb.x2 &&
                bb.y1 <= graphY &&
                graphY <= bb.y2
            ) {
                const depth = node.parents().length;
                if (depth > bestDepth) {
                    bestDepth = depth;
                    best = node;
                }
            }
        });
        return best;
    }

    /** Return the topmost non-container cytoscape node under ``(graphX,
     *  graphY)`` whose class list is ``.dag-block`` but NOT
     *  ``.dag-block--container``. Used to detect drop-on-block and emit
     *  the adjacent-offset coords. Returns ``null`` when no block is
     *  under the cursor. */
    function findBlockAt(cy, graphX, graphY) {
        const blocks = cy.nodes(".dag-block");
        let best = null;
        let bestDepth = -1;
        blocks.forEach(function (node) {
            if (node.hasClass(CONTAINER_CSS_CLASS)) return;
            if (node.data("is_port")) return;
            const bb = node.boundingBox({
                includeOverlays: false,
                includeLabels: false,
            });
            if (
                bb.x1 <= graphX &&
                graphX <= bb.x2 &&
                bb.y1 <= graphY &&
                graphY <= bb.y2
            ) {
                const depth = node.parents().length;
                // Prefer the deepest (innermost) hit so nested blocks
                // win over their container parent's body region.
                if (depth > bestDepth) {
                    bestDepth = depth;
                    best = node;
                }
            }
        });
        return best;
    }

    /** Return the cytoscape edge nearest ``(graphX, graphY)`` if the
     *  cursor is within a small graph-coord threshold of the edge's
     *  bounding box. Used for the drop-on-wire side-effect (select the
     *  wire; block lands at coords; no split). Returns ``null`` if no
     *  edge is hit.
     *
     *  Cytoscape doesn't expose a public "edge under point" hit-test,
     *  so we approximate via bounding-box containment with a 6px
     *  cushion. This matches the spec's "loose" wire-hit semantics. */
    function findEdgeAt(cy, graphX, graphY) {
        const TOL = 6;
        let hit = null;
        cy.edges().forEach(function (edge) {
            if (hit) return;
            const bb = edge.boundingBox({ includeOverlays: false });
            if (
                bb.x1 - TOL <= graphX &&
                graphX <= bb.x2 + TOL &&
                bb.y1 - TOL <= graphY &&
                graphY <= bb.y2 + TOL
            ) {
                hit = edge;
            }
        });
        return hit;
    }

    /** Translate ``event.clientX/Y`` into cytoscape graph coordinates,
     *  per the explicit formula in spec §4.8.
     *
     *  Returns ``null`` if the cy container cannot be located (i.e. the
     *  canvas has been torn down mid-drop). */
    function clientToGraph(cy, clientX, clientY) {
        const container = document.getElementById(CY_CANVAS_ID);
        if (!container) return null;
        const rect = container.getBoundingClientRect();
        const renderedX = clientX - rect.left;
        const renderedY = clientY - rect.top;
        const pan = cy.pan();
        const zoom = cy.zoom() || 1;
        return {
            x: (renderedX - pan.x) / zoom,
            y: (renderedY - pan.y) / zoom,
        };
    }

    /** Publish ``payload`` to ``STORE_PALETTE_DROP``. Uses
     *  ``dash_clientside.set_props`` (Dash 2.18+). When the API is
     *  unavailable (older Dash) or the store hasn't rendered yet, the
     *  call is a no-op — the dispatcher will simply not run. */
    function publishPaletteDrop(payload) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            return;
        }
        try {
            window.dash_clientside.set_props(STORE_PALETTE_DROP_ID, {
                data: payload,
            });
        } catch (err) {
            // Store not yet mounted; silently drop the dispatch — the
            // user can retry.
        }
    }

    /** Side-effect: select the wire when the user drops on top of it.
     *  Per spec §4.8 the block still lands at drop coords; we just push
     *  a partial ``{selected_edge_id}`` mutation into
     *  ``STORE_BUILDER_STATE``. The server-side ``wire_select``
     *  dispatch performs the canonical mutation; this clientside hint
     *  exists so the wire visually highlights immediately. */
    function selectWire(edgeId) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            return;
        }
        try {
            // Only the dispatch payload matters here — the fan-in
            // callback re-derives state from the canonical store.
            window.dash_clientside.set_props(STORE_BUILDER_STATE_ID, {
                data: { kind: "wire_select", edge_id: edgeId, ts: Date.now() },
            });
        } catch (err) {
            // Store not yet mounted; ignore.
        }
    }

    /** Emit ``phenotypic:palette-drop`` with the given detail. */
    function emitPaletteDrop(className, accepted) {
        try {
            document.dispatchEvent(
                new CustomEvent("phenotypic:palette-drop", {
                    detail: { class_name: className, accepted: !!accepted },
                })
            );
        } catch (err) {
            // Older browsers without CustomEvent — ignore.
        }
    }

    /** Build the small ghost element used as the drag image. We attach
     *  it briefly to the body so ``setDragImage`` can rasterise it,
     *  then schedule removal on the next tick. The ``data-testid``
     *  attribute matches the spec §5.5 DOM test ID for Playwright. */
    function buildGhostElement(className) {
        const ghost = document.createElement("div");
        ghost.setAttribute("data-testid", "palette-ghost");
        ghost.className = "palette-ghost";
        ghost.textContent = className;
        // Off-screen positioning so the rendered card doesn't flicker
        // visibly before the browser captures it.
        ghost.style.position = "absolute";
        ghost.style.top = "-9999px";
        ghost.style.left = "-9999px";
        document.body.appendChild(ghost);
        // Auto-remove on the next tick (after the browser snapshot).
        setTimeout(function () {
            if (ghost && ghost.parentNode) {
                ghost.parentNode.removeChild(ghost);
            }
        }, 0);
        return ghost;
    }

    /** Clear any container-hover decoration set by ``dragover``. */
    function clearHoverHighlight(cy) {
        if (!hoverContainerId) return;
        try {
            const node = cy.getElementById(hoverContainerId);
            if (node && node.length) {
                node.removeClass(CONTAINER_HOVER_CLASS);
            }
        } catch (err) {
            // Cytoscape may have been torn down — ignore.
        }
        hoverContainerId = null;
    }

    // -----------------------------------------------------------------
    // Palette button event handlers.
    // -----------------------------------------------------------------
    function onPaletteDragStart(event) {
        const target = event.target.closest("[" + PALETTE_CLASS_ATTR + "]");
        if (!target) return;
        const className = target.getAttribute(PALETTE_CLASS_ATTR);
        if (!className) return;

        activeDrag = {
            className: className,
            startX: event.clientX,
            startY: event.clientY,
            ts: Date.now(),
        };

        if (event.dataTransfer) {
            try {
                event.dataTransfer.setData(DT_MIME, className);
                // ``effectAllowed`` controls the cursor + drop-effect
                // negotiation; ``copy`` matches the create-new-block
                // semantics (we're not moving the palette button).
                event.dataTransfer.effectAllowed = "copy";
            } catch (err) {
                // Some browsers throw on cross-origin drag — ignore.
            }

            const ghost = buildGhostElement(className);
            try {
                event.dataTransfer.setDragImage(ghost, 8, 8);
            } catch (err) {
                // setDragImage unsupported (e.g. IE) — accept the
                // default cursor preview.
            }
        }
    }

    function onPaletteDragEnd(event) {
        // ``dropEffect === "none"`` means the browser cancelled the
        // drag (Esc, or drop landed somewhere with no drop handler).
        // Spec §4.8: emit nothing.
        const cancelled =
            !event.dataTransfer || event.dataTransfer.dropEffect === "none";

        if (cancelled && activeDrag) {
            emitPaletteDrop(activeDrag.className, false);
        }
        // Clear hover decoration even on cancellation.
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (cy) clearHoverHighlight(cy);
        activeDrag = null;
    }

    /** Keyboard fallback (spec §4.8): focused palette button + Enter /
     *  Space dispatches at viewport centre. */
    function onKeyboardFallback(event) {
        if (event.key !== "Enter" && event.key !== " ") return;
        const target = document.activeElement;
        if (!target) return;
        const className = target.getAttribute(PALETTE_CLASS_ATTR);
        if (!className) return;
        event.preventDefault();

        whenCyReady(function (cy) {
            const ext = cy.extent();
            const centerX = (ext.x1 + ext.x2) / 2;
            const centerY = (ext.y1 + ext.y2) / 2;
            const payload = {
                kind: "block_create",
                class_name: className,
                x: centerX,
                y: centerY,
                container_block_id: null,
                ts: Date.now(),
            };
            publishPaletteDrop(payload);
            emitPaletteDrop(className, true);
        });
    }

    // -----------------------------------------------------------------
    // Canvas event handlers.
    // -----------------------------------------------------------------
    function onCanvasDragOver(event) {
        if (!activeDrag) return;
        // ``preventDefault`` is required to mark the wrapper as a valid
        // drop target. Without it the browser silently rejects drops.
        event.preventDefault();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }

        const cy = window.phenoGetCy && window.phenoGetCy();
        if (!cy) return;
        const graph = clientToGraph(cy, event.clientX, event.clientY);
        if (!graph) return;

        const container = findInnermostContainer(cy, graph.x, graph.y);
        const newHoverId = container ? container.id() : null;
        if (newHoverId === hoverContainerId) return;

        // Swap the hover decoration.
        clearHoverHighlight(cy);
        if (container) {
            container.addClass(CONTAINER_HOVER_CLASS);
            hoverContainerId = container.id();
        }
    }

    function onCanvasDragLeave(event) {
        // ``dragleave`` fires whenever the cursor crosses a child
        // boundary too — only clear the hover when we've actually left
        // the wrapper. ``relatedTarget`` is the element entered next;
        // if it's still inside the wrapper, ignore.
        const wrapper = event.currentTarget;
        if (
            wrapper &&
            event.relatedTarget &&
            wrapper.contains(event.relatedTarget)
        ) {
            return;
        }
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (cy) clearHoverHighlight(cy);
    }

    function onCanvasDrop(event) {
        event.preventDefault();
        if (!activeDrag) return;

        const cy = window.phenoGetCy && window.phenoGetCy();
        if (!cy) {
            // Cytoscape unmounted between dragstart and drop — bail.
            emitPaletteDrop(activeDrag.className, false);
            activeDrag = null;
            return;
        }

        clearHoverHighlight(cy);

        const graph = clientToGraph(cy, event.clientX, event.clientY);
        if (!graph) {
            emitPaletteDrop(activeDrag.className, false);
            activeDrag = null;
            return;
        }

        // 1. Innermost-wins container hit-test.
        const container = findInnermostContainer(cy, graph.x, graph.y);
        const containerId = container ? container.data("block_id") : null;

        // 2. Existing-block hit-test (non-container). Offsets the
        //    payload coords to land adjacent.
        let dropX = graph.x;
        let dropY = graph.y;
        const block = findBlockAt(cy, graph.x, graph.y);
        if (block) {
            dropX = graph.x + ADJACENT_OFFSET_X;
        }

        // 3. Wire hit-test (drop-on-wire side-effect: select the wire,
        //    but still land at drop coords).
        const edge = findEdgeAt(cy, graph.x, graph.y);
        if (edge && !block && !container) {
            const edgeId = edge.data("edge_id") || edge.id();
            selectWire(edgeId);
            // Hint the user that wires aren't drop targets. The toast
            // queue may not exist yet; emit a console warning as a
            // forward-compatible fallback (spec §4.8).
            if (
                window.phenoToast &&
                typeof window.phenoToast.show === "function"
            ) {
                try {
                    window.phenoToast.show({
                        kind: "hint",
                        message: "Drop on input/output ports to connect",
                    });
                } catch (err) {
                    // ignore
                }
            } else if (typeof console !== "undefined" && console.warn) {
                console.warn(
                    "[palette_dnd] Wires are not drop targets; drop on " +
                        "input/output ports to connect."
                );
            }
        }

        // 4. Publish the dispatch payload.
        const payload = {
            kind: "block_create",
            class_name: activeDrag.className,
            x: dropX,
            y: dropY,
            container_block_id: containerId,
            ts: Date.now(),
        };
        publishPaletteDrop(payload);
        emitPaletteDrop(activeDrag.className, true);

        activeDrag = null;
    }

    // -----------------------------------------------------------------
    // Binding lifecycle. Palette buttons are server-rendered and may
    // be replaced on every Dash re-render; we use event delegation on
    // ``document`` so the handlers survive subtree swaps. Canvas
    // handlers attach to the stable wrapper id (also survives subtree
    // swaps because the wrapper itself is fixed in ``_layout.py``).
    // -----------------------------------------------------------------
    function attachPaletteHandlers() {
        // Event delegation: listen at the document level and filter by
        // ``data-palette-class`` on the closest ancestor. Idempotent
        // via a single sentinel so we don't double-bind on re-runs.
        if (document.__phenoPaletteBound) return;
        document.__phenoPaletteBound = true;
        document.addEventListener("dragstart", onPaletteDragStart, true);
        document.addEventListener("dragend", onPaletteDragEnd, true);
        document.addEventListener("keydown", onKeyboardFallback);
    }

    function attachCanvasHandlers() {
        const wrapper = document.getElementById(CY_WRAPPER_ID);
        if (!wrapper) return false;
        if (wrapper.__phenoDndBound) return true;
        wrapper.__phenoDndBound = true;
        wrapper.addEventListener("dragover", onCanvasDragOver);
        wrapper.addEventListener("dragleave", onCanvasDragLeave);
        wrapper.addEventListener("drop", onCanvasDrop);
        return true;
    }

    /** Watch for the cytoscape wrapper being re-mounted (Dash may swap
     *  its parent subtree). Rebinds canvas handlers each time a new
     *  wrapper appears. */
    function watchWrapper() {
        if (typeof MutationObserver === "undefined") return;
        const observer = new MutationObserver(function () {
            const wrapper = document.getElementById(CY_WRAPPER_ID);
            if (wrapper && !wrapper.__phenoDndBound) {
                attachCanvasHandlers();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    // -----------------------------------------------------------------
    // Module init.
    // -----------------------------------------------------------------
    whenCyReady(function (_cy) {
        attachPaletteHandlers();
        attachCanvasHandlers();
        watchWrapper();
        window.phenotypic_palette_dnd_ready = true;
    });
})();
