/* PhenoTypic Pipeline Builder — viewport operations clientside glue.
 *
 * Auto-loaded by Dash via the ``assets/`` convention. Owns viewport-level
 * operations (layout, anchoring, scrolling) on the new DAG canvas:
 *
 *   * ``window.phenotypicRelayout()`` — re-run the leaf-first dagre
 *     compound layout algorithm (see spec §4.7) and call ``cy.fit()``.
 *   * ``window.phenotypicReanchor()`` — pan + zoom so the root ``InputImage``
 *     block sits centered on the canvas.
 *   * ``window.phenotypicBlockCollapsedToggle(blockId)`` — flip the
 *     ``dag-block--collapsed`` CSS class on the container block; the body's
 *     ``display: none`` is handled by ``builder.css``.
 *   * ``window.phenotypicScrollTo(...)`` and ``window.phenotypicDrillToScope(...)``
 *     are Phase-2 stubs; Phase 6 fills in the full expand-chain + scrim flow.
 *
 * On completion the IIFE writes the asset-readiness sentinels expected by
 * ``builder.js``'s polling routine (spec §5.5 / §5.6):
 *
 *   * ``window.phenotypic_viewport_ops_ready = true`` once the module has
 *     bound its viewport handlers.
 *   * ``window.phenotypic_viewport_ops_dagre_missing = true`` if the
 *     vendored ``cytoscape-dagre.min.js`` failed to register (in which
 *     case ``relayout`` falls back to ``cy.layout({name: "preset"})``).
 *
 * Conventions:
 *   * Vanilla JS only (no jQuery / ES modules — assets/ are <script> tags).
 *   * Polls for ``window.phenoGetCy()`` returning a fresh cy instance,
 *     mirroring ``builder.js`` / ``aux_popover.js``.
 *   * No DOM mutations outside the cytoscape container; CSS-only chrome
 *     (e.g. ``.dag-block--collapsed``) is owned by ``builder.css``.
 */

(function () {
    "use strict";

    // -----------------------------------------------------------------
    // Constants — keep in sync with builder/_ids.py + spec §4.6.
    // -----------------------------------------------------------------
    /** Padding (px) added to each container's inner bounding box before
     *  promoting it to a fixed-size compound on the outer layout pass. */
    const COMPOUND_PADDING = 32;

    /** Dagre direction + animation knobs (spec §4.7). */
    const DAGRE_DIRECTION = "LR";
    const ANIMATION_DURATION = 200;
    const ANIMATION_EASING = "ease-out";

    /** CSS class toggled on the compound parent during collapse.
     *  ``builder.css`` owns the visibility rule:
     *      .dag-block--collapsed > .dag-block__body { display: none; } */
    const COLLAPSED_CLASS = "dag-block--collapsed";

    /** ``data.class_name`` value identifying the root-scope source block. */
    const INPUT_IMAGE_CLASS = "InputImage";

    // -----------------------------------------------------------------
    // Helpers.
    // -----------------------------------------------------------------
    /** Resolve the live cytoscape instance via the accessor builder.js
     *  publishes on ``window``. ``cb(cy)`` runs whenever an instance
     *  becomes available; if cytoscape hasn't mounted yet we retry every
     *  100ms. Mirrors the polling pattern used by ``aux_popover.js`` so
     *  asset load order doesn't matter. */
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

    /** Best-effort check that the ``cytoscape-dagre`` extension registered
     *  with the global cytoscape lib. Tries the documented spec check
     *  (``window.cytoscape.layouts.dagre``) first; if cytoscape's API
     *  doesn't expose ``layouts`` directly (older builds keep extension
     *  metadata on a private map), falls back to attempting a no-op
     *  ``cy.layout({name: "dagre"})`` on an empty cy clone, catching the
     *  "No such layout" error class. Returns ``true`` when dagre appears
     *  registered. */
    function isDagreRegistered(cy) {
        const csc = window.cytoscape;
        if (!csc) return false;
        // Spec-prescribed check — works on the vendored cytoscape build
        // (it exposes ``cytoscape.layouts`` once an extension has been
        // registered via ``cytoscape("layout", "dagre", ...)``).
        if (csc.layouts && typeof csc.layouts.dagre === "function") {
            return true;
        }
        // Fallback: probe ``cy.layout()`` and detect cytoscape's
        // "No such layout `dagre`" exception. ``cy.layout()`` itself
        // is cheap — it just constructs the layout object, no run.
        if (cy && typeof cy.layout === "function") {
            try {
                const probe = cy.layout({ name: "dagre", animate: false });
                // If layout() returns a real object with a run() method,
                // dagre is registered. Don't actually call run().
                return !!(probe && typeof probe.run === "function");
            } catch (err) {
                return false;
            }
        }
        return false;
    }

    /** Pre-set the sentinel up-front; the IIFE replaces this with the
     *  real ``true`` after binding viewport ops to cytoscape.
     *  Initial value is ``false`` so builder.js's poller sees the asset
     *  as "not ready yet" rather than "missing entirely". */
    window.phenotypic_viewport_ops_ready = false;
    // The ``_dagre_missing`` flag is intentionally NOT initialised here —
    // a missing flag means "dagre status unknown" (asset still loading);
    // an explicit ``true`` means "we checked, dagre is gone".

    // -----------------------------------------------------------------
    // Leaf-first dagre compound layout (spec §4.7).
    // -----------------------------------------------------------------
    /** Run dagre on the union of image-flow + aux edges in the active
     *  state tree, treating compound containers as opaque atoms on each
     *  outer pass.
     *
     *  Algorithm (spec §4.7):
     *    1. Depth-first walk every scope. The "root" scope is cytoscape's
     *       top-level (nodes with ``parent() === null``); each compound
     *       parent (``data.is_container``) is a nested scope.
     *    2. Visit leaf scopes first (those with no nested containers).
     *    3. For each leaf scope, run dagre on its blocks + edges with
     *       ``boundingBox`` constrained to the parent's coords; record
     *       the resulting bounding-box dimensions.
     *    4. Walk back up: set each container compound's ``width``/
     *       ``height`` data so cytoscape's compound layout treats it as
     *       a fixed-size atom on the next outer pass.
     *    5. Run dagre on the next-outer scope; compounds are now sized
     *       correctly and laid out as ordinary nodes.
     *    6. Repeat until the root scope is laid out.
     *
     *  Direction: ``rankDir: "LR"``. Animations are batched at the root
     *  level only — per-scope sub-layouts run synchronously so the final
     *  positions are known before the outer animation kicks in.
     *
     *  Fallback: if ``cytoscape-dagre`` did not register, runs a single
     *  ``preset`` layout (use existing positions / cytoscape defaults).
     *  Per spec §5.5: ``BTN_RELAYOUT`` is disabled in that case, but a
     *  manual call here still degrades gracefully. */
    function leafFirstDagre(cy) {
        if (!cy) return;

        // Fallback: dagre extension absent. ``preset`` keeps any existing
        // positions; the spec says we lean on ``cy.fit()`` to at least
        // recentre the view. Surface the degraded state via the sentinel
        // flag so the asset-status banner can warn the user.
        if (!isDagreRegistered(cy)) {
            window.phenotypic_viewport_ops_dagre_missing = true;
            try {
                cy.layout({ name: "preset", animate: false }).run();
            } catch (err) {
                // Preset can't fail on a sane cy — but we swallow because
                // the relayout path is purely cosmetic.
            }
            cy.fit(undefined, 24);
            return;
        }

        // Step 1 — Enumerate scopes. A scope is identified by its
        // compound parent (``null`` = root scope). We bucket every node
        // by its parent's id; "leaf" scopes have no inner compound
        // children.
        const scopes = new Map(); // parentId | null -> {parent, children, hasContainerChild}
        cy.nodes().forEach(function (node) {
            // Skip the port/aux sub-nodes — they're rendered as part of
            // the parent block's chrome and shouldn't participate in
            // the dagre rank assignment. Ports are emitted as compound
            // children of their parent block; we filter via
            // ``data.is_port`` (undefined treated as "not a port").
            if (node.data("is_port")) return;
            // Only laid-out elements are the block compounds + their
            // direct atoms. Compounds themselves *are* laid out by the
            // outer pass; we don't filter them here.
            const parent = node.parent();
            const parentId = parent.length ? parent.id() : null;
            if (!scopes.has(parentId)) {
                scopes.set(parentId, {
                    parent: parent.length ? parent : null,
                    children: cy.collection(),
                    hasContainerChild: false,
                });
            }
            const scope = scopes.get(parentId);
            scope.children = scope.children.add(node);
            if (node.isParent && node.isParent()) {
                scope.hasContainerChild = true;
            }
        });

        // Step 2 — Order scopes leaf-first. A scope is a leaf iff none
        // of its children are themselves compound parents. We sort by
        // depth descending (deepest first) so children settle before
        // their containers consult their bounding box.
        const scopeEntries = Array.from(scopes.entries()).map(function (entry) {
            const parentId = entry[0];
            const meta = entry[1];
            // Depth = number of ancestor compounds. Root scope has
            // depth 0; first-level container scope has depth 1; etc.
            let depth = 0;
            if (meta.parent) {
                depth = meta.parent.parents().length + 1;
            }
            return [parentId, meta, depth];
        });
        scopeEntries.sort(function (a, b) {
            return b[2] - a[2]; // deepest first
        });

        // Step 3 — Per-scope dagre passes. Each pass runs synchronously
        // (animate: false) because the outer pass needs the inner
        // bounding box NOW. We record bounding boxes by parent id so
        // step 4 can write width/height back onto the compound atom.
        const innerBBoxes = new Map(); // parentId -> {w, h}
        scopeEntries.forEach(function (entry) {
            const meta = entry[1];
            const children = meta.children;
            if (!children || children.length === 0) return;

            // Only edges whose endpoints both live in this scope.
            const childIds = new Set(children.map(function (n) { return n.id(); }));
            const edges = cy.edges().filter(function (e) {
                return (
                    childIds.has(e.source().id()) &&
                    childIds.has(e.target().id())
                );
            });
            const eles = children.add(edges);

            // Per-scope dagre. ``ranker: "longest-path"`` gives the
            // determinism the spec wants; ``nodeSep`` / ``rankSep``
            // keep blocks readable without forcing the user to scroll
            // for short chains.
            try {
                eles
                    .layout({
                        name: "dagre",
                        rankDir: DAGRE_DIRECTION,
                        animate: false,
                        ranker: "longest-path",
                        nodeSep: 40,
                        rankSep: 80,
                        edgeSep: 16,
                        // ``fit: false`` keeps the *outer* scope's pan
                        // intact while we lay out an inner one — only
                        // the final root-level pass calls cy.fit().
                        fit: false,
                    })
                    .run();
            } catch (err) {
                // Per-scope dagre can throw on a 1-block scope with no
                // edges (cytoscape-dagre 2.5.0 issue). Spec §4.7 says
                // fall back to a centred preset and continue.
                eles.layout({ name: "preset", animate: false }).run();
            }

            // Record this scope's inner bounding box for the outer pass.
            // ``children.boundingBox()`` returns ``{x1,y1,x2,y2,w,h}``
            // in graph coords.
            if (meta.parent) {
                const bb = children.boundingBox();
                innerBBoxes.set(meta.parent.id(), {
                    w: bb.w + COMPOUND_PADDING * 2,
                    h: bb.h + COMPOUND_PADDING * 2,
                });
            }
        });

        // Step 4 — Propagate inner sizes to each compound. cytoscape's
        // stylesheet inspects ``data.compound_width`` /
        // ``data.compound_height`` to set ``width`` / ``height`` once
        // the stylesheet ships; meanwhile the next outer pass reads
        // these data values via cy's positioning engine.
        innerBBoxes.forEach(function (size, parentId) {
            const compound = cy.getElementById(parentId);
            if (!compound || !compound.length) return;
            compound.data("compound_width", size.w);
            compound.data("compound_height", size.h);
        });

        // Step 5/6 — Root scope animated fit. Per spec, only the final
        // pan / zoom animates; per-scope passes ran synchronously above.
        cy.animate(
            { fit: { eles: cy.elements(), padding: 24 } },
            { duration: ANIMATION_DURATION, easing: ANIMATION_EASING }
        );

        // Emit the completion event so tests can ``page.waitForEvent``
        // on it (spec §5.5 custom DOM events).
        try {
            document.dispatchEvent(
                new CustomEvent("phenotypic:relayout-complete", { detail: {} })
            );
        } catch (err) {
            // Older browsers without CustomEvent constructor — ignore.
        }
    }

    // -----------------------------------------------------------------
    // Public viewport ops attached to ``window`` for fan-in callbacks.
    // -----------------------------------------------------------------
    /** Re-run the leaf-first dagre layout + fit. Server-side callbacks
     *  invoke this via ``STORE_VIEWPORT_OP`` ``{kind: "relayout"}``. */
    function phenotypicRelayout() {
        whenCyReady(function (cy) {
            leafFirstDagre(cy);
        });
    }

    /** Pan + zoom so the root ``InputImage`` block sits centered. Falls
     *  back to a plain ``cy.fit()`` if no InputImage is present (e.g. on
     *  a corrupted state — validation surfaces a ``missing_input`` Issue
     *  in that case but we still want a graceful view). */
    function phenotypicReanchor() {
        whenCyReady(function (cy) {
            const input = cy
                .nodes()
                .filter(function (n) {
                    return n.data("class_name") === INPUT_IMAGE_CLASS;
                })
                .first();
            if (input && input.length) {
                cy.animate(
                    {
                        center: { eles: input },
                        zoom: Math.min(cy.zoom(), 1.0),
                    },
                    {
                        duration: ANIMATION_DURATION,
                        easing: ANIMATION_EASING,
                    }
                );
            } else {
                cy.fit(undefined, 24);
            }
        });
    }

    /** Toggle the ``dag-block--collapsed`` CSS class on the block's
     *  cytoscape node. ``builder.css``'s rule
     *  (``.dag-block--collapsed > .dag-block__body { display: none; }``)
     *  hides the inner ports + body. Cytoscape doesn't expose a node's
     *  DOM directly, so we apply a ``classes`` toggle and rely on the
     *  canvas stylesheet selectors. */
    function phenotypicBlockCollapsedToggle(blockId) {
        if (!blockId) return;
        whenCyReady(function (cy) {
            const node = cy.getElementById(blockId);
            if (!node || !node.length) return;
            node.toggleClass(COLLAPSED_CLASS);
            // After a collapse/expand the bounding box of any ancestor
            // compound changes — trigger a relayout so neighbours don't
            // overlap.
            leafFirstDagre(cy);
        });
    }

    /** Pan/zoom-only stub. Phase 6 expands the full chain:
     *    - mounts a canvas-wide scrim,
     *    - dispatches ``drill_to_scope`` if the offender lives in a
     *      different breadcrumb,
     *    - chains ``block_collapsed_toggle`` for each collapsed
     *      container in ``scope_path``,
     *    - emits ``phenotypic:scroll-to-complete`` on settle.
     *  PHASE 6: full expand-chain + scrim + drill_to_scope. */
    function phenotypicScrollTo(blockId, _scopePath, _targetBreadcrumb) {
        if (!blockId) return;
        whenCyReady(function (cy) {
            const node = cy.getElementById(blockId);
            if (!node || !node.length) return;
            cy.animate(
                { center: { eles: node } },
                {
                    duration: ANIMATION_DURATION,
                    easing: ANIMATION_EASING,
                    complete: function () {
                        try {
                            document.dispatchEvent(
                                new CustomEvent("phenotypic:scroll-to-complete", {
                                    detail: { block_id: blockId },
                                })
                            );
                        } catch (err) {
                            // Older browsers — ignore.
                        }
                    },
                }
            );
        });
    }

    /** Phase-2 stub. The full implementation dispatches a
     *  ``drill_to_scope`` mutation through ``STORE_BUILDER_STATE`` and
     *  awaits ``layoutstop``; for now we no-op so the asset is forward-
     *  compatible.
     *  PHASE 6: atomic breadcrumb replacement + validation. */
    function phenotypicDrillToScope(_targetBreadcrumb) {
        // Intentional no-op — STORE_BUILDER_STATE writes belong on the
        // server side; the dispatcher already handles drill_to_scope
        // payloads. Phase 6 wires the clientside trigger.
    }

    // -----------------------------------------------------------------
    // Module init.
    // -----------------------------------------------------------------
    // Eagerly probe the dagre extension presence — sets the sentinel
    // even if no cytoscape instance exists yet (e.g. asset gate check on
    // the empty-canvas placeholder route).
    if (!(window.cytoscape && window.cytoscape.layouts && window.cytoscape.layouts.dagre)) {
        // cytoscape-dagre.min.js failed to load OR did not auto-register
        // with the cytoscape global. Don't abort the IIFE — we still
        // want the other viewport ops (reanchor, block toggle) to work.
        // The relayout path falls back to ``preset`` (see leafFirstDagre).
        // builder.js's asset-status poller reads this flag and surfaces
        // the "Layout extension missing" banner.
        window.phenotypic_viewport_ops_dagre_missing = true;
    }

    whenCyReady(function (_cy) {
        // Publish viewport-op handlers under the documented namespace.
        // The server-side clientside callback dispatches these from a
        // ``STORE_VIEWPORT_OP`` payload (kind switch).
        window.phenotypicRelayout = phenotypicRelayout;
        window.phenotypicReanchor = phenotypicReanchor;
        window.phenotypicBlockCollapsedToggle = phenotypicBlockCollapsedToggle;
        window.phenotypicScrollTo = phenotypicScrollTo;
        window.phenotypicDrillToScope = phenotypicDrillToScope;

        // Signal readiness — builder.js poller writes the missing-asset
        // list to STORE_ASSET_STATUS based on these sentinels.
        window.phenotypic_viewport_ops_ready = true;
    });
})();
