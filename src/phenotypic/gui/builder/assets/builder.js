/* PhenoTypic Pipeline Builder — clientside helpers (auto-loaded by Dash).
 *
 * Exposes ``window.phenoGetCy()`` so clientside callbacks (registered in
 * ``_callbacks.py``) can drive the cytoscape canvas directly via its native
 * API — ``cy.fit()`` / ``cy.zoom()`` — instead of fighting dash-cytoscape's
 * prop change detection.
 */

(function () {
    "use strict";

    /** Locate the live cytoscape.js instance backing the dash-cytoscape
     *  component with id ``canvas-cytoscape``.
     *
     *  Tries (in order):
     *    1. The cytoscape.js registry attached to the container element.
     *    2. The React fiber tree, walking down looking for a node whose
     *       ``stateNode`` exposes ``_cy`` or ``cy``.
     *
     *  Returns the cy instance, or ``null`` if it can't be located (e.g.
     *  before the canvas has finished mounting).
     */
    window.phenoGetCy = function phenoGetCy() {
        const container = document.getElementById("canvas-cytoscape");
        if (!container) return null;

        // 1. cytoscape.js attaches the instance via ``_cyreg`` on the container.
        if (container._cyreg && container._cyreg.cy) {
            return container._cyreg.cy;
        }

        // 2. Walk the React fiber tree from the wrapper.
        const wrap =
            document.getElementById("canvas-cytoscape-wrapper") || container;
        const fiberKey = Object.keys(wrap).find(
            (k) => k.startsWith("__reactFiber") || k.startsWith("__reactInternalInstance"),
        );
        if (!fiberKey) return null;

        const stack = [wrap[fiberKey]];
        const seen = new Set();
        while (stack.length) {
            const node = stack.pop();
            if (!node || seen.has(node)) continue;
            seen.add(node);
            const inst = node.stateNode;
            if (inst) {
                if (inst._cy) return inst._cy;
                if (inst.cy) return inst.cy;
            }
            if (node.child) stack.push(node.child);
            if (node.sibling) stack.push(node.sibling);
        }
        return null;
    };

    /* Watch the cytoscape container for size changes (the layout settles
     * after fonts load, accordions expand, etc.) and call ``cy.resize()`` +
     * ``cy.fit()`` so the chain is always centered. ``cy.resize()`` is a
     * no-op when the size genuinely hasn't changed; ``cy.fit()`` re-centers
     * without re-running the layout algorithm so user-dragged positions
     * are preserved. */
    function watchCanvasResize() {
        const container = document.getElementById("canvas-cytoscape");
        if (!container || typeof ResizeObserver === "undefined") return;
        let lastW = 0,
            lastH = 0;
        const ro = new ResizeObserver(() => {
            const cy = window.phenoGetCy();
            if (!cy) return;
            const w = cy.width();
            const h = cy.height();
            if (w === lastW && h === lastH) return;
            lastW = w;
            lastH = h;
            if (w > 0 && h > 0) {
                cy.resize();
                cy.fit(undefined, 24);
            }
        });
        ro.observe(container);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", watchCanvasResize);
    } else {
        // ``setTimeout`` defers past Dash's initial render so the cytoscape
        // component has had time to mount its container.
        setTimeout(watchCanvasResize, 100);
    }
    // Re-bind whenever Dash swaps the canvas wrapper subtree.
    new MutationObserver(() => {
        const c = document.getElementById("canvas-cytoscape");
        if (c && !c.__phenoResizeBound) {
            c.__phenoResizeBound = true;
            watchCanvasResize();
        }
    }).observe(document.body, { childList: true, subtree: true });
})();


/* PhenoTypic Pipeline Builder — scroll affordance helper.
 *
 * Dash auto-loads any *.js under the ``assets/`` folder colocated with the
 * module that calls ``dash.Dash(__name__)``. This script wires the
 * ``data-more-up`` / ``data-more-down`` attributes on each
 * ``.pheno-scroll-wrap`` so its CSS chevrons (top / bottom) appear strictly
 * when there is content above / below the visible viewport.
 *
 * Designed to survive Dash callbacks that swap subtrees (e.g. the inspector):
 * a single MutationObserver re-attaches scroll listeners whenever a new
 * ``.pheno-scroll-wrap`` is inserted into the DOM.
 */

(function () {
    "use strict";

    const TOP_THRESHOLD = 4;
    const BOTTOM_THRESHOLD = 4;

    /** Update wrapper data attributes for the current inner-scroll metrics. */
    function refreshIndicators(wrap) {
        const inner = wrap.querySelector(".pheno-scroll");
        if (!inner) {
            wrap.removeAttribute("data-more-up");
            wrap.removeAttribute("data-more-down");
            return;
        }
        const scrollTop = inner.scrollTop;
        const scrollHeight = inner.scrollHeight;
        const clientHeight = inner.clientHeight;
        const overflows = scrollHeight - clientHeight > 1;

        const moreUp = overflows && scrollTop > TOP_THRESHOLD;
        const moreDown =
            overflows && scrollTop + clientHeight < scrollHeight - BOTTOM_THRESHOLD;

        if (moreUp) {
            wrap.setAttribute("data-more-up", "1");
        } else {
            wrap.removeAttribute("data-more-up");
        }
        if (moreDown) {
            wrap.setAttribute("data-more-down", "1");
        } else {
            wrap.removeAttribute("data-more-down");
        }
    }

    /** Bind scroll + resize listeners to a single wrapper. Idempotent. */
    function bindWrapper(wrap) {
        if (wrap.__phenoScrollBound) return;
        wrap.__phenoScrollBound = true;

        const inner = wrap.querySelector(".pheno-scroll");
        if (!inner) return;

        // Scroll listener (passive — read-only metrics).
        inner.addEventListener("scroll", () => refreshIndicators(wrap), {
            passive: true,
        });

        // ResizeObserver covers content additions/removals (palette grows,
        // inspector form swaps in, viewport resizes).
        if (typeof ResizeObserver !== "undefined") {
            const ro = new ResizeObserver(() => refreshIndicators(wrap));
            ro.observe(inner);
            // Observe a few likely children so adding a tall accordion section
            // updates the indicators without a manual scroll.
            for (const child of inner.children) {
                ro.observe(child);
            }
        }

        // Initial paint.
        // Defer one frame so layout has settled (Bootstrap accordions need a
        // tick to compute their own heights).
        requestAnimationFrame(() => refreshIndicators(wrap));
    }

    /** Discover and bind every ``.pheno-scroll-wrap`` currently in the DOM. */
    function bindAll(root) {
        const scope = root || document;
        const wraps = scope.querySelectorAll(".pheno-scroll-wrap");
        wraps.forEach(bindWrapper);
        // Bootstrap accordions emit transitionend events when expanding —
        // refresh the indicators after each so the post-expand height is
        // reflected immediately.
        scope
            .querySelectorAll(".pheno-scroll-wrap .accordion-collapse")
            .forEach((collapse) => {
                if (collapse.__phenoTransitionBound) return;
                collapse.__phenoTransitionBound = true;
                collapse.addEventListener("transitionend", () => {
                    const wrap = collapse.closest(".pheno-scroll-wrap");
                    if (wrap) refreshIndicators(wrap);
                });
            });
    }

    function watchForNewWrappers() {
        const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
                for (const node of m.addedNodes) {
                    if (!(node instanceof Element)) continue;
                    if (node.classList && node.classList.contains("pheno-scroll-wrap")) {
                        bindWrapper(node);
                    }
                    bindAll(node);
                }
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    function start() {
        bindAll();
        watchForNewWrappers();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();


/* PhenoTypic Pipeline Builder — asset-readiness poller (spec §5.5).
 *
 * Each new DAG-canvas JS asset (currently ``viewport_ops.js``; later
 * ``palette_dnd.js``, ``wire_drawing.js``) writes a sentinel onto
 * ``window`` when its IIFE finishes binding:
 *
 *   * ``window.phenotypic_viewport_ops_ready = true``
 *   * ``window.phenotypic_palette_dnd_ready  = true``  (Phase 3)
 *   * ``window.phenotypic_wire_drawing_ready = true``  (Phase 4)
 *
 *  ``viewport_ops.js`` may additionally set
 *  ``window.phenotypic_viewport_ops_dagre_missing = true`` if the
 *  vendored ``cytoscape-dagre.min.js`` failed to register; the asset is
 *  still "ready" in that case but degraded, and the toolbar banner
 *  surfaces a separate "Layout extension missing" row.
 *
 *  This routine polls every 500ms for the first 1500ms after page load,
 *  then writes the final missing-asset list (an empty list = all green)
 *  into ``STORE_ASSET_STATUS`` via ``dash_clientside.set_props``. Agent
 *  2A's server-side wiring (``asset_status_disables`` callback, asset-
 *  status banner in ``_layout.py``) reads the store and disables the
 *  relayout button + shows banner rows as needed.
 *
 *  The poll interval is short (500ms) and only runs three iterations
 *  total, so the cost is negligible. The window flags themselves are
 *  always available, so any future code (e.g. graceful-degradation
 *  shortcircuits in ``wire_drawing.js``) can read them directly without
 *  going through the store. */
(function () {
    "use strict";

    /** Mirror of ``builder/_ids.py`` — ``STORE_ASSET_STATUS`` is a new
     *  store id added in Phase 2 (Agent 2A's responsibility). Kept as a
     *  constant here so a future rename only touches one place. */
    const STORE_ASSET_STATUS_ID = "store-asset-status";

    /** Map of asset name -> {ready_flag, dagre_flag?}. The ``dagre_flag``
     *  is asset-specific: only ``viewport_ops`` consults it. When the
     *  dagre extension is missing we still report ``viewport_ops`` as
     *  ready (degraded), but emit a separate ``cytoscape-dagre`` entry
     *  in the missing-list. */
    const ASSETS = [
        {
            name: "viewport_ops",
            ready_flag: "phenotypic_viewport_ops_ready",
            extension_flag: "phenotypic_viewport_ops_dagre_missing",
            extension_label: "cytoscape-dagre",
        },
        // Phase 3 / Phase 4 will append palette_dnd + wire_drawing.
    ];

    /** Total poll budget. Matches spec §5.5 (1500ms ceiling). */
    const TOTAL_BUDGET_MS = 1500;
    const POLL_INTERVAL_MS = 500;

    /** Construct the missing-asset list from the current ``window``
     *  sentinels. Returns ``{missing: string[], degraded: string[]}``
     *  where ``missing`` are assets whose ready flag is still false and
     *  ``degraded`` are auxiliary extensions (e.g. cytoscape-dagre) that
     *  reported themselves missing. */
    function collectAssetStatus() {
        const missing = [];
        const degraded = [];
        for (const asset of ASSETS) {
            if (!window[asset.ready_flag]) {
                missing.push(asset.name);
            }
            if (asset.extension_flag && window[asset.extension_flag]) {
                degraded.push(asset.extension_label);
            }
        }
        return { missing: missing, degraded: degraded };
    }

    /** Write the final status into ``STORE_ASSET_STATUS`` via Dash's
     *  ``set_props`` clientside hook (Dash 2.18+). The payload shape:
     *
     *    {missing: string[], degraded: string[], ts: number}
     *
     *  - ``missing``: names of asset files whose IIFE never ran to
     *    completion (e.g. CDN cache eviction blocked the load).
     *  - ``degraded``: auxiliary extension names that reported their
     *    own absence (currently ``cytoscape-dagre`` only).
     *  - ``ts``: monotonic timestamp so the server-side callback can
     *    distinguish "first write" vs. "no change". */
    function publishAssetStatus(status) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            // Dash older than 2.18 — fall back to a plain ``dcc.Store``
            // mutation via a CustomEvent that the asset-status callback
            // can listen for. We do not attempt to install the event
            // listener here; Phase 2's Agent 2A wires the store at
            // server-render time. The fallback path is documented for
            // future graceful-degradation work.
            return;
        }
        try {
            window.dash_clientside.set_props(STORE_ASSET_STATUS_ID, {
                data: {
                    missing: status.missing,
                    degraded: status.degraded,
                    ts: Date.now(),
                },
            });
        } catch (err) {
            // STORE_ASSET_STATUS hasn't been rendered yet (e.g. legacy
            // popover route active, or initial page render still in
            // flight). The poll routine will retry on the next tick
            // until either the store exists or the budget is exhausted.
        }
    }

    /** Drive the poll loop. Runs every ``POLL_INTERVAL_MS`` for at
     *  most ``TOTAL_BUDGET_MS`` total; whichever comes first:
     *   - all assets report ready (early exit + final publish), or
     *   - the budget is exhausted (final publish with whatever's still
     *     missing).
     *  The final publish is unconditional so the asset-status banner
     *  always reflects an authoritative state. */
    function pollAssetStatus() {
        const start = Date.now();
        let last;
        const handle = setInterval(function () {
            const status = collectAssetStatus();
            const allReady = status.missing.length === 0;
            const elapsed = Date.now() - start;
            // Publish on every iteration so the banner clears as soon as
            // each asset reports ready (not just at the budget cap).
            if (
                !last ||
                last.missing.length !== status.missing.length ||
                last.degraded.length !== status.degraded.length
            ) {
                publishAssetStatus(status);
                last = status;
            }
            if (allReady || elapsed >= TOTAL_BUDGET_MS) {
                clearInterval(handle);
                // Final write — guarantees the store reflects the
                // post-budget state even if no transitions happened.
                publishAssetStatus(status);
            }
        }, POLL_INTERVAL_MS);
    }

    // Defer the first poll one tick so cytoscape + Dash have had a
    // chance to materialise the store DOM. ``setTimeout(_, 0)`` is
    // enough; subsequent iterations rely on ``setInterval``.
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () {
            setTimeout(pollAssetStatus, 0);
        });
    } else {
        setTimeout(pollAssetStatus, 0);
    }
})();
