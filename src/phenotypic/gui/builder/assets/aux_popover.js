/* PhenoTypic Pipeline Builder — aux-port popover clientside glue.
 *
 * Auto-loaded by Dash via the ``assets/`` convention (alongside
 * ``builder.js``). Bridges cytoscape ``tap`` events on bottom-edge aux
 * port nodes (id-prefix ``aux-port__``) to the Python-side popover
 * machinery by writing to ``dcc.Store`` components, and mounts a
 * Popper.js instance that keeps the popover DOM container anchored to
 * the tapped port across cytoscape pan / zoom.
 *
 * Lifecycle:
 *   1. On page load, register ``cytoscape.use(cytoscapePopper)`` once
 *      (idempotent — both vendored libs must be in window scope).
 *   2. Poll for ``window.phenoGetCy()`` returning a fresh cy instance
 *      (the builder swaps it on every state mutation); when one
 *      appears, rebind tap handlers.
 *   3. Tap on an aux port -> write to ``store-port-click``, then mount
 *      a popper instance anchored at the tapped node tracking the
 *      ``cy-popover-container`` div.
 *   4. Click-outside or Escape -> hide container, destroy popper,
 *      write to ``store-popover-dismiss``.
 *
 * Conventions:
 *   - Vanilla JS only (no jQuery / ES modules — assets/ are loaded via
 *     ``<script>`` tags by Dash).
 *   - Writes to ``dcc.Store`` use ``window.dash_clientside.set_props``
 *     (Dash 2.18+ API).
 *   - Does not interfere with ``builder.js`` (no shared state apart
 *     from ``window.phenoGetCy``).
 */

(function () {
    "use strict";

    // ---------------------------------------------------------------
    // Element ids (mirror src/phenotypic/gui/builder/_ids.py)
    // ---------------------------------------------------------------
    const CY_CANVAS_ID = "canvas-cytoscape";
    const POPOVER_CONTAINER_ID = "cy-popover-container";
    const PORT_CLICK_STORE_ID = "store-port-click";
    const POPOVER_DISMISS_STORE_ID = "store-popover-dismiss";
    const AUX_PORT_PREFIX = "aux-port__";

    // ---------------------------------------------------------------
    // Extension registration (runs once when both libs are loaded).
    // ---------------------------------------------------------------
    let _extensionRegistered = false;

    function registerExtension() {
        if (_extensionRegistered) return true;
        if (window.cytoscape && window.cytoscapePopper) {
            try {
                window.cytoscape.use(window.cytoscapePopper);
                _extensionRegistered = true;
                return true;
            } catch (err) {
                // ``cytoscape.use`` throws if the same extension is
                // registered twice — treat that as success.
                _extensionRegistered = true;
                return true;
            }
        }
        return false;
    }

    // ---------------------------------------------------------------
    // Popper lifecycle.
    // ---------------------------------------------------------------
    function destroyPopper() {
        if (window._auxPopperInstance) {
            try {
                window._auxPopperInstance.destroy();
            } catch (err) {
                // Cytoscape may have already torn down the node.
            }
            window._auxPopperInstance = null;
        }
        if (window._auxPopperUpdate && window._auxPopperCy) {
            try {
                window._auxPopperCy.off(
                    "pan zoom resize",
                    window._auxPopperUpdate,
                );
            } catch (err) {
                // cy may already be disposed.
            }
        }
        window._auxPopperUpdate = null;
        window._auxPopperCy = null;
    }

    function mountPopover(cy, node) {
        const popoverDiv = document.getElementById(POPOVER_CONTAINER_ID);
        if (!popoverDiv) return;

        // Tear down any previous instance.
        destroyPopper();

        // Make container visible (server-side callback renders its
        // children based on ``store-port-click``).
        popoverDiv.style.display = "block";

        // ``cytoscape-popper`` (registered above) attaches ``.popper()``
        // to every cytoscape element. The returned Popper instance
        // auto-tracks the node's screen position.
        if (typeof node.popper !== "function") {
            // Extension not loaded — bail silently. The store write
            // above still happens so the server can fall back if it
            // wishes.
            return;
        }

        window._auxPopperInstance = node.popper({
            content: () => popoverDiv,
            popper: {
                placement: "bottom",
                modifiers: [
                    { name: "offset", options: { offset: [0, 8] } },
                ],
            },
        });

        // Keep the popover glued to the port during pan / zoom /
        // canvas resize.
        const updatePopper = () => {
            if (window._auxPopperInstance) {
                try {
                    window._auxPopperInstance.update();
                } catch (err) {
                    // Cytoscape may have re-rendered and orphaned the
                    // popper — let dismiss handle it.
                }
            }
        };
        cy.on("pan zoom resize", updatePopper);
        window._auxPopperUpdate = updatePopper;
        window._auxPopperCy = cy;
    }

    function dismissPopover() {
        const popoverDiv = document.getElementById(POPOVER_CONTAINER_ID);
        if (popoverDiv) {
            popoverDiv.style.display = "none";
        }
        destroyPopper();
        if (
            window.dash_clientside &&
            typeof window.dash_clientside.set_props === "function"
        ) {
            try {
                window.dash_clientside.set_props(POPOVER_DISMISS_STORE_ID, {
                    data: Date.now(),
                });
            } catch (err) {
                // Store may not be mounted yet; non-fatal.
            }
        }
    }

    // ---------------------------------------------------------------
    // Tap binding on the live cytoscape instance.
    // ---------------------------------------------------------------
    function bindAuxPortTap(cy) {
        // Cytoscape selector matches nodes whose id begins with the
        // aux-port prefix.
        cy.on("tap", 'node[id ^= "aux-port__"]', (event) => {
            const node = event.target;
            const id = node.id();
            // Encoded as ``aux-port__<target_node_id>__<param>`` (see
            // ``_encode_aux_port_id`` in builder/_ids.py). Param names
            // may contain underscores so rejoin everything past the
            // target_node_id segment.
            const parts = id.split("__");
            if (parts.length < 3) return;
            const targetNodeId = parts[1];
            const param = parts.slice(2).join("__");

            if (
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            ) {
                try {
                    window.dash_clientside.set_props(PORT_CLICK_STORE_ID, {
                        data: {
                            target_node_id: targetNodeId,
                            param: param,
                            ts: Date.now(),
                        },
                    });
                } catch (err) {
                    // Store may not be mounted yet — non-fatal.
                }
            }

            mountPopover(cy, node);
        });
    }

    // ---------------------------------------------------------------
    // Global dismiss handlers: click-outside + Escape.
    // ---------------------------------------------------------------
    document.addEventListener("click", (e) => {
        const popoverDiv = document.getElementById(POPOVER_CONTAINER_ID);
        if (!popoverDiv) return;
        if (popoverDiv.style.display === "none") return;
        if (popoverDiv.contains(e.target)) return;
        // Clicks on the cytoscape canvas itself trip the cy.on('tap')
        // path; if the user clicked an aux port the handler re-opens
        // the popover, so dismissing here is safe.
        dismissPopover();
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") dismissPopover();
    });

    // ---------------------------------------------------------------
    // Cytoscape-instance lifecycle.
    //
    // The builder swaps the cytoscape instance every time dash-cytoscape
    // re-renders (which happens on most state mutations). Poll the
    // ``phenoGetCy`` getter exposed by builder.js and rebind whenever
    // the instance identity changes.
    // ---------------------------------------------------------------
    let _lastCy = null;
    function pollForCy() {
        registerExtension();
        if (typeof window.phenoGetCy !== "function") return;
        const cy = window.phenoGetCy();
        if (!cy) return;
        if (cy === _lastCy) return;
        _lastCy = cy;
        // The previous cy's listeners go away with it; bind to the
        // new instance.
        bindAuxPortTap(cy);
    }

    // Poll at a modest rate — dash-cytoscape doesn't expose a "new
    // instance mounted" hook, and ``MutationObserver`` on the wrapper
    // fires too eagerly (every child re-render). A 200ms tick is well
    // below the latency required for the popover handshake to feel
    // immediate.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            setInterval(pollForCy, 200);
        });
    } else {
        setInterval(pollForCy, 200);
    }
})();
