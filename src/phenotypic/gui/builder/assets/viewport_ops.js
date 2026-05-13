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
 *   * ``window.phenotypicScrollTo(blockId, scopePath, targetBreadcrumb)`` —
 *     pans + fits the canvas to the offender, traversing the breadcrumb
 *     and expanding collapsed containers as needed (spec §5.6).  Mounts
 *     a canvas-wide scrim (``data-testid="dag-scrim"``) so a user drag
 *     cannot interleave with the expand chain; dismisses the scrim and
 *     emits ``phenotypic:scroll-to-complete`` on settle.
 *   * ``window.phenotypicDrillToScope(targetBreadcrumb)`` — dispatch an
 *     atomic breadcrumb replacement through ``STORE_VIEWPORT_OP``.  The
 *     server-side dispatcher validates each id; stale ids → reject +
 *     toast + ``STORE_VIEWPORT_OP`` sentinel ``scroll_to_aborted`` that
 *     this module relays as a ``phenotypic:scroll-to-aborted`` DOM event.
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

    /** Dagre node + rank separation (px). Tuned per spec §4.7 so short
     *  chains stay above the fold without crowding longer ones. */
    const DAGRE_NODE_SEP = 40;
    const DAGRE_RANK_SEP = 80;
    const DAGRE_EDGE_SEP = 16;

    /** Padding (px) ``cy.fit()`` reserves around the final bounding box. */
    const FIT_PADDING = 24;

    /** Padding (px) the ``scroll_to`` chain reserves around the offender
     *  block once the expand chain resolves and ``cy.fit()`` zooms in. */
    const SCROLL_TO_FIT_PADDING = 60;

    /** Duration (ms) of the final ``cy.animate({fit})`` after the expand
     *  chain settles.  Matches the canvas-toolbar fit/zoom buttons so the
     *  visual rhythm reads as one consistent motion language. */
    const SCROLL_TO_FIT_DURATION = 300;

    /** Max time (ms) ``waitForLayoutstopOrAbort`` will block on a single
     *  ``layoutstop`` event.  Sized so a 2–3 level expand chain
     *  (~200–600ms per the spec) plus a generous network round-trip
     *  fits within the budget; the timeout fires only when the server
     *  silently dropped the dispatch. */
    const LAYOUTSTOP_TIMEOUT_MS = 5000;

    /** CSS class toggled on the compound parent during collapse.
     *  ``builder.css`` owns the visibility rule:
     *      .dag-block--collapsed > .dag-block__body { display: none; } */
    const COLLAPSED_CLASS = "dag-block--collapsed";

    /** ``data.class_name`` value identifying the root-scope source block. */
    const INPUT_IMAGE_CLASS = "InputImage";

    /** ``data-testid`` attribute on the scroll-to scrim.  Playwright +
     *  the spec §6 row both spell this exact string; tests resolve the
     *  scrim element by ``[data-testid="dag-scrim"]``. */
    const SCRIM_TEST_ID = "dag-scrim";

    /** Custom-event name emitted by the abort path.  ``phenotypicScrollTo``
     *  listens for this so the scrim dismisses immediately on a stale-id
     *  rejection rather than waiting for the ``layoutstop`` timeout. */
    const SCROLL_TO_ABORTED_EVENT = "phenotypic:scroll-to-aborted";

    /** Custom-event name emitted on a successful chain settle. */
    const SCROLL_TO_COMPLETE_EVENT = "phenotypic:scroll-to-complete";

    // -----------------------------------------------------------------
    // Helpers.
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
            cy.fit(undefined, FIT_PADDING);
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
                        nodeSep: DAGRE_NODE_SEP,
                        rankSep: DAGRE_RANK_SEP,
                        edgeSep: DAGRE_EDGE_SEP,
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
            { fit: { eles: cy.elements(), padding: FIT_PADDING } },
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
    // ``scroll_to`` chain helpers (spec §5.6).
    // -----------------------------------------------------------------

    /** Compare two breadcrumb arrays element-wise. ``null`` / ``undefined``
     *  coerce to an empty array so both default cases (root scope,
     *  uninitialised state) compare equal. */
    function arraysEqual(a, b) {
        const aa = Array.isArray(a) ? a : [];
        const bb = Array.isArray(b) ? b : [];
        if (aa.length !== bb.length) return false;
        for (let i = 0; i < aa.length; i++) {
            if (aa[i] !== bb[i]) return false;
        }
        return true;
    }

    /** Read the active breadcrumb out of ``STORE_BUILDER_STATE``.
     *
     *  The store is a Dash ``dcc.Store`` whose data field hangs off the
     *  hidden ``<div id="store-builder-state">`` element's
     *  ``__dashprivate_initial_props`` attribute *during the initial
     *  render*; for ongoing state we read the live React props via
     *  ``window.dash_clientside`` — but Dash doesn't expose those.
     *  Instead, ``builder.js`` mirrors the breadcrumb onto
     *  ``window.__phenoBreadcrumb`` after every fan-in render so we have
     *  a stable hook here.
     *
     *  If the mirror isn't present (older builder.js / pre-render race),
     *  fall back to reading the breadcrumb pill DOM via the
     *  ``.pheno-breadcrumb [data-breadcrumb-segment]`` attribute
     *  rendered by ``_layout.build_breadcrumb``. */
    function getCurrentBreadcrumb() {
        if (Array.isArray(window.__phenoBreadcrumb)) {
            return window.__phenoBreadcrumb.slice();
        }
        // DOM fallback — read every breadcrumb segment id attribute.
        const segments = document.querySelectorAll(
            ".pheno-breadcrumb [data-breadcrumb-segment]"
        );
        const ids = [];
        segments.forEach(function (el) {
            const segId = el.getAttribute("data-breadcrumb-segment");
            if (segId) ids.push(segId);
        });
        return ids;
    }

    /** Mount a ``data-testid="dag-scrim"`` div on the canvas wrapper.
     *
     *  The scrim:
     *    * Covers the entire canvas wrapper (CSS rules in ``builder.css``
     *      set ``position: absolute; top/left/right/bottom: 0``).
     *    * Captures pointer events so palette drag-over and port-mousedown
     *      gestures cannot interleave with the expand chain.
     *    * Carries a ``data-testid`` attribute so Playwright can assert
     *      its lifecycle (mount on chain start, unmount on completion).
     *
     *  Returns the scrim element so the caller can ``.remove()`` it on
     *  chain completion. */
    function mountScrim() {
        const cyContainer = document.getElementById("canvas-cytoscape");
        // Mount on the cytoscape container's parent — that's where the
        // canvas chrome (toolbar, asset banner) shares a positioning
        // context.  Falling back to the cy container itself if the
        // parent isn't `position: relative` keeps the scrim covering
        // the right surface.
        const host = (cyContainer && cyContainer.parentElement) || cyContainer || document.body;
        // Ensure the host can position the scrim absolutely.
        const computed = window.getComputedStyle(host);
        if (computed.position === "static") {
            host.style.position = "relative";
        }
        const scrim = document.createElement("div");
        scrim.className = "dag-scrim";
        scrim.setAttribute("data-testid", SCRIM_TEST_ID);
        scrim.setAttribute("aria-busy", "true");
        host.appendChild(scrim);
        return scrim;
    }

    /** Resolve once ``cy`` fires its next ``layoutstop`` OR once a
     *  ``phenotypic:scroll-to-aborted`` event is dispatched on document.
     *
     *  This race lets ``phenotypicScrollTo`` short-circuit when the
     *  server-side ``drill_to_scope`` rejects a stale breadcrumb id —
     *  without the race, the scrim would block for the full timeout
     *  before dismissing.
     *
     *  Times out after ``LAYOUTSTOP_TIMEOUT_MS`` so a silently-dropped
     *  dispatch can't pin the scrim indefinitely. */
    function waitForLayoutstopOrAbort(cy, timeoutMs) {
        const budget = typeof timeoutMs === "number" ? timeoutMs : LAYOUTSTOP_TIMEOUT_MS;
        return new Promise(function (resolve, reject) {
            let timer = null;
            function cleanup() {
                if (timer !== null) {
                    clearTimeout(timer);
                    timer = null;
                }
                document.removeEventListener(SCROLL_TO_ABORTED_EVENT, onAbort);
                if (cy && typeof cy.removeListener === "function") {
                    cy.removeListener("layoutstop", onStop);
                }
            }
            function onAbort() {
                cleanup();
                reject(new Error("scroll-to-aborted"));
            }
            function onStop() {
                cleanup();
                resolve();
            }
            document.addEventListener(SCROLL_TO_ABORTED_EVENT, onAbort, { once: true });
            // cy.one binds a one-shot handler that auto-detaches on fire.
            if (cy && typeof cy.one === "function") {
                cy.one("layoutstop", onStop);
            }
            timer = setTimeout(function () {
                cleanup();
                reject(new Error("layoutstop timeout"));
            }, budget);
        });
    }

    /** Helper: write a payload to ``STORE_VIEWPORT_OP`` via
     *  ``dash_clientside.set_props``.  The fan-in callback subscribed to
     *  the store picks up the change and routes through the dispatcher. */
    function publishViewportOp(payload) {
        if (
            window.dash_clientside &&
            typeof window.dash_clientside.set_props === "function"
        ) {
            window.dash_clientside.set_props("store-viewport-op", {
                data: payload,
            });
            return true;
        }
        return false;
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
                cy.fit(undefined, FIT_PADDING);
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

    /** Full scroll-to chain (spec §5.6 ``scroll_to`` row).
     *
     *  Pans + fits the cytoscape viewport to the block, traversing
     *  the breadcrumb and expanding collapsed containers as needed.
     *  Mounts a canvas-wide scrim so a user drag cannot interleave
     *  with the expand chain — the scrim dismisses on settle or abort.
     *
     *  Steps:
     *    (1) Mount the ``data-testid="dag-scrim"`` overlay.
     *    (2) If ``targetBreadcrumb`` differs from the current breadcrumb,
     *        publish a ``drill_to_scope`` payload to ``STORE_VIEWPORT_OP``
     *        and await ``layoutstop`` (with a race against
     *        ``phenotypic:scroll-to-aborted`` for the stale-id case).
     *    (3) For each collapsed container in ``scopePath`` (now in the
     *        active scope), publish a ``block_collapsed_toggle`` payload
     *        and await ``layoutstop``.
     *    (4) ``cy.animate({fit})`` to the offender block.
     *    (5) Remove the scrim.
     *    (6) Dispatch ``phenotypic:scroll-to-complete``.
     *
     *  Args:
     *    blockId: ``BlockNode.block_id`` of the offender to pan to.
     *    scopePath: List of container block_ids on the path from
     *      the active scope down to the offender's enclosing scope.
     *      Each entry is an *intermediate* container — the offender
     *      itself is NOT included.
     *    targetBreadcrumb: Breadcrumb the offender lives under.
     *      When this differs from the current breadcrumb, the chain
     *      begins with a single ``drill_to_scope`` dispatch.
     */
    async function phenotypicScrollTo(blockId, scopePath, targetBreadcrumb) {
        if (!blockId) return;
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (!cy) return;

        const scopes = Array.isArray(scopePath) ? scopePath : [];
        const breadcrumb = Array.isArray(targetBreadcrumb) ? targetBreadcrumb : [];

        // (1) Mount the scrim.
        const scrim = mountScrim();

        try {
            // (2) Cross-breadcrumb navigation if needed.
            const currentBreadcrumb = getCurrentBreadcrumb();
            if (!arraysEqual(breadcrumb, currentBreadcrumb)) {
                const published = publishViewportOp({
                    kind: "drill_to_scope",
                    target_breadcrumb: breadcrumb,
                    ts: Date.now(),
                });
                if (!published) {
                    // dash_clientside not ready — abort the chain.
                    return;
                }
                await waitForLayoutstopOrAbort(cy);
            }

            // (3) Expand collapsed containers in scope_path.  Each
            // collapsed-toggle dispatch may re-render the canvas; we
            // re-resolve the node each iteration in case the prior
            // dispatch dropped & re-created the cytoscape elements.
            for (let i = 0; i < scopes.length; i++) {
                const containerBlockId = scopes[i];
                if (!containerBlockId) continue;
                const node = cy.getElementById(containerBlockId);
                if (!node || !node.length) continue;
                // ``data.collapsed`` is the canonical truth set by the
                // server-side reducer; the DOM class lags behind a tick
                // after the cytoscape re-render.
                if (!node.data("collapsed")) continue;
                const published = publishViewportOp({
                    kind: "block_collapsed_toggle",
                    block_id: containerBlockId,
                    ts: Date.now(),
                });
                if (!published) return;
                await waitForLayoutstopOrAbort(cy);
            }

            // (4) Pan + fit to the target block.
            const targetNode = cy.getElementById(blockId);
            if (targetNode && targetNode.length) {
                await new Promise(function (resolve) {
                    cy.animate(
                        {
                            fit: {
                                eles: targetNode,
                                padding: SCROLL_TO_FIT_PADDING,
                            },
                        },
                        {
                            duration: SCROLL_TO_FIT_DURATION,
                            easing: ANIMATION_EASING,
                            complete: resolve,
                        }
                    );
                });
            }
        } catch (err) {
            // Aborted or timed out — fall through to scrim removal +
            // emit no completion event.  The toast queue already
            // surfaced the user-facing message; nothing else to do.
            try {
                scrim.remove();
            } catch (cleanupErr) {
                // Scrim already detached — ignore.
            }
            return;
        }

        // (5) Dismiss the scrim.
        try {
            scrim.remove();
        } catch (err) {
            // Scrim already detached — ignore.
        }

        // (6) Emit the completion event.
        try {
            document.dispatchEvent(
                new CustomEvent(SCROLL_TO_COMPLETE_EVENT, {
                    detail: { block_id: blockId },
                })
            );
        } catch (err) {
            // Older browsers — ignore.
        }
    }

    /** Atomic breadcrumb replacement (spec §5.6 ``drill_to_scope`` row).
     *
     *  Publishes a ``drill_to_scope`` payload to ``STORE_VIEWPORT_OP``.
     *  The server-side fan-in callback validates each segment in
     *  ``targetBreadcrumb`` against the current state tree:
     *
     *    * Every id must resolve to a ``Pipeline``-class container at
     *      the right depth.
     *    * Stale (deleted) ids → reject + queue toast + emit the
     *      ``scroll_to_aborted`` sentinel back to ``STORE_VIEWPORT_OP``
     *      (which the bottom-of-this-file relay turns into a
     *      ``phenotypic:scroll-to-aborted`` DOM event).
     *
     *  No client-side custom event from the success path; the
     *  ``layoutstop`` after dispatch is the signal the caller awaits. */
    function phenotypicDrillToScope(targetBreadcrumb) {
        const breadcrumb = Array.isArray(targetBreadcrumb) ? targetBreadcrumb : [];
        publishViewportOp({
            kind: "drill_to_scope",
            target_breadcrumb: breadcrumb,
            ts: Date.now(),
        });
        // Server-side rejects stale ids and queues a toast.  No client-
        // side custom event from the success path; the layout-stop
        // after dispatch is the signal callers await.
    }

    // -----------------------------------------------------------------
    // ``scroll_to_aborted`` relay.
    // -----------------------------------------------------------------
    //
    // When the server-side ``drill_to_scope`` dispatch rejects a stale
    // breadcrumb id, the ``viewport_op_fan_in`` callback writes back to
    // ``STORE_VIEWPORT_OP`` with ``{kind: "scroll_to_aborted", ts}``.
    // We can't subscribe to the store directly from this asset
    // (Dash mediates store changes), so the equivalent clientside
    // callback registered in ``_callbacks.py`` (Phase 6) calls
    // ``window.phenotypicScrollToAbortedRelay`` whenever it sees the
    // sentinel.  That function dispatches the DOM event our
    // ``waitForLayoutstopOrAbort`` helper races against.
    function phenotypicScrollToAbortedRelay() {
        try {
            document.dispatchEvent(
                new CustomEvent(SCROLL_TO_ABORTED_EVENT, { detail: {} })
            );
        } catch (err) {
            // Older browsers without CustomEvent constructor — ignore.
        }
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
        window.phenotypicScrollToAbortedRelay = phenotypicScrollToAbortedRelay;

        // Signal readiness — builder.js poller writes the missing-asset
        // list to STORE_ASSET_STATUS based on these sentinels.
        window.phenotypic_viewport_ops_ready = true;
    });
})();
