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
