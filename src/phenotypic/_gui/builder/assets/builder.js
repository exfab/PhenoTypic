/* PhenoTypic Pipeline Builder — clientside helpers (auto-loaded by Dash).
 *
 * Exposes ``window.phenoGetCy()`` so clientside callbacks (registered in
 * ``_callbacks.py``) can drive the cytoscape canvas directly via its native
 * API — ``cy.fit()`` / ``cy.zoom()`` — instead of fighting dash-cytoscape's
 * prop change detection.
 *
 * Also exposes ``window.phenoWhenCyReady(cb)`` — a shared accessor that
 * polls ``phenoGetCy()`` every 100ms and invokes ``cb(cy)`` once the
 * cytoscape instance has mounted.  All three IIFE assets
 * (``viewport_ops.js``, ``palette_dnd.js``, ``wire_drawing.js``) used to
 * carry an identical copy of this pattern; centralising it here keeps
 * load-order tolerance in one place and means new assets only need to
 * call ``window.phenoWhenCyReady`` rather than re-implement the loop.
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

    window.phenoLinearMapMounted = function phenoLinearMapMounted() {
        return Boolean(
            document.getElementById("linear-map-container") &&
                !document.getElementById("canvas-cytoscape"),
        );
    };

    /** Poll ``window.phenoGetCy`` every 100ms and invoke ``cb(cy)`` once
     *  the cytoscape instance has mounted.  Shared by every asset that
     *  binds clientside handlers but doesn't know whether the canvas
     *  has rendered yet (load-order tolerance).
     *
     *  Idempotency: each asset's IIFE is expected to call this exactly
     *  once at bind time; the recursive ``setTimeout`` walks at most
     *  one outstanding chain at a time per caller. */
    window.phenoWhenCyReady = function phenoWhenCyReady(cb) {
        const cy = window.phenoGetCy && window.phenoGetCy();
        if (cy) {
            cb(cy);
            return;
        }
        if (window.phenoLinearMapMounted && window.phenoLinearMapMounted()) {
            return;
        }
        setTimeout(function () {
            window.phenoWhenCyReady(cb);
        }, 100);
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


/* PhenoTypic Pipeline Builder — linear-map zoom controls.
 *
 * Fixed-map zoom is intentionally UI-only. It stores no Dash state and never
 * writes pipeline JSON; it only scales the HTML map content inside the current
 * viewport. Button ports remain ordinary clickable DOM buttons after scaling.
 */

(function () {
    "use strict";

    const VIEWPORT_ID = "linear-map-container";
    const VIEWPORT_SELECTOR = ".linear-map-zoom-viewport";
    const CONTENT_SELECTOR = ".linear-map-zoom-content";
    const ZOOM_MIN = 0.35;
    const ZOOM_MAX = 2.0;
    const ZOOM_STEP = 1.2;

    function clampZoom(value) {
        return Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, value));
    }

    function parts() {
        const viewport =
            document.getElementById(VIEWPORT_ID) ||
            document.querySelector(VIEWPORT_SELECTOR);
        if (!viewport) return null;
        const content = viewport.querySelector(CONTENT_SELECTOR);
        if (!content) return null;
        return { viewport, content };
    }

    function currentZoom(viewport) {
        const raw =
            viewport.__phenotypicLinearZoom ||
            Number.parseFloat(viewport.dataset.linearZoom || "1");
        return Number.isFinite(raw) ? raw : 1;
    }

    function applyZoom(zoom) {
        const found = parts();
        if (!found) return;
        const next = clampZoom(zoom);
        found.viewport.__phenotypicLinearZoom = next;
        found.viewport.dataset.linearZoom = String(Number(next.toFixed(2)));
        // Scale from the top-left so the map grows down/right into the
        // scrollable directions (a centered origin would push the upper-left
        // overflow into negative space the viewport can never reach).
        found.content.style.transformOrigin = "left top";
        found.content.style.transform = `scale(${next})`;
        // CSS transforms paint-scale but don't resize the layout box, so the
        // viewport's scroll extent would stay frozen at the un-zoomed size and
        // panning would "lock" once zoomed in. Reserve the extra scaled space
        // with margins (which transforms ignore) so the scroll bounds grow to
        // match what the user actually sees. `scrollWidth`/`scrollHeight` read
        // the natural, transform-independent content extent.
        const naturalWidth = found.content.scrollWidth || found.content.offsetWidth || 0;
        const naturalHeight = found.content.scrollHeight || found.content.offsetHeight || 0;
        const extraWidth = naturalWidth * (next - 1);
        const extraHeight = naturalHeight * (next - 1);
        found.content.style.marginRight = extraWidth > 0 ? `${extraWidth}px` : "0px";
        found.content.style.marginBottom = extraHeight > 0 ? `${extraHeight}px` : "0px";
    }

    function zoomBy(factor) {
        const found = parts();
        if (!found) return;
        applyZoom(currentZoom(found.viewport) * factor);
    }

    function resetZoom() {
        applyZoom(1);
        const found = parts();
        if (!found) return;
        found.viewport.scrollLeft = 0;
        found.viewport.scrollTop = 0;
    }

    function fitZoom() {
        const found = parts();
        if (!found) return;
        const width = found.content.scrollWidth || found.content.offsetWidth || 1;
        const height = found.content.scrollHeight || found.content.offsetHeight || 1;
        const availableWidth = Math.max(1, found.viewport.clientWidth - 16);
        const availableHeight = Math.max(1, found.viewport.clientHeight - 16);
        const fit = Math.min(1, availableWidth / width, availableHeight / height);
        applyZoom(fit);
        found.viewport.scrollLeft = 0;
        found.viewport.scrollTop = 0;
    }

    function bindButton(id, handler) {
        const button = document.getElementById(id);
        if (!button || button.__phenotypicLinearZoomBound) return;
        button.__phenotypicLinearZoomBound = true;
        button.addEventListener("click", handler);
    }

    function bindZoomControls() {
        bindButton("linear-zoom-out", () => zoomBy(1 / ZOOM_STEP));
        bindButton("linear-zoom-in", () => zoomBy(ZOOM_STEP));
        bindButton("linear-zoom-reset", resetZoom);
        bindButton("linear-zoom-fit", fitZoom);

        const found = parts();
        if (found && !found.content.style.transform) {
            applyZoom(currentZoom(found.viewport));
        }
    }

    function scheduleBind() {
        requestAnimationFrame(bindZoomControls);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", scheduleBind);
    } else {
        scheduleBind();
    }

    new MutationObserver(scheduleBind).observe(document.body, {
        childList: true,
        subtree: true,
    });
})();


/* PhenoTypic Pipeline Builder — inspector slide-over toggle (UI-only).
 *
 * The inspector docks as a right-edge overlay; its tab handle flips the
 * ``.is-closed`` class on the stable ``#inspector-slideover`` wrapper. The
 * wrapper survives the inspector's inner re-renders, so the open/closed
 * choice (persisted to localStorage) sticks across node selections. A
 * MutationObserver re-binds after Dash swaps subtrees, mirroring the zoom
 * controls above; a per-element flag keeps the listener single-bound.
 */
(function () {
    "use strict";

    const SLIDEOVER_ID = "inspector-slideover";
    const STORAGE_KEY = "phenotypicBuilderInspectorClosed";

    function bindSlideover() {
        const slideover = document.getElementById(SLIDEOVER_ID);
        if (!slideover || slideover.__phenotypicSlideoverBound) return;
        const tab = slideover.querySelector(".builder-slideover-tab");
        if (!tab) return;
        slideover.__phenotypicSlideoverBound = true;

        // Restore the persisted open/closed choice on first bind (both
        // directions, so a previously-opened inspector re-opens even though
        // the layout renders closed by default).
        try {
            const stored = window.localStorage.getItem(STORAGE_KEY);
            if (stored === "1") {
                slideover.classList.add("is-closed");
            } else if (stored === "0") {
                slideover.classList.remove("is-closed");
            }
        } catch (err) {
            /* localStorage unavailable (private mode) — keep default. */
        }

        tab.addEventListener("click", function () {
            const closed = slideover.classList.toggle("is-closed");
            try {
                window.localStorage.setItem(STORAGE_KEY, closed ? "1" : "0");
            } catch (err) {
                /* ignore persistence failures */
            }
        });
    }

    function scheduleSlideoverBind() {
        requestAnimationFrame(bindSlideover);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", scheduleSlideoverBind);
    } else {
        scheduleSlideoverBind();
    }

    new MutationObserver(scheduleSlideoverBind).observe(document.body, {
        childList: true,
        subtree: true,
    });
})();


/* PhenoTypic Pipeline Builder — operations palette: collapse + resize.
 *
 * The palette pane's width lives in the ``--builder-palette-width`` custom
 * property on the stable ``#builder-columns`` wrapper. The divider between the
 * palette and the canvas is a ``col-resize`` drag handle; the small button on
 * it toggles a ``palette-collapsed`` class. Both the width and the collapsed
 * choice persist to localStorage and restore on first bind. A MutationObserver
 * re-binds after Dash subtree swaps; per-element flags keep the listeners
 * single-bound. UI-only — no Dash state is written.
 */
(function () {
    "use strict";

    const COLUMNS_ID = "builder-columns";
    const DIVIDER_ID = "builder-palette-divider";
    const PANE_ID = "builder-palette-pane";
    const COLLAPSE_ID = "btn-palette-collapse";
    const WIDTH_KEY = "phenotypicBuilderPaletteWidth";
    const COLLAPSED_KEY = "phenotypicBuilderPaletteCollapsed";
    const MIN_WIDTH = 180;
    const MAX_WIDTH = 560;

    function clampWidth(px) {
        return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, px));
    }

    function setWidth(columns, px) {
        columns.style.setProperty("--builder-palette-width", px + "px");
    }

    function bindPalette() {
        const columns = document.getElementById(COLUMNS_ID);
        if (!columns || columns.__phenotypicPaletteBound) return;
        const divider = document.getElementById(DIVIDER_ID);
        const collapseBtn = document.getElementById(COLLAPSE_ID);
        if (!divider || !collapseBtn) return;
        columns.__phenotypicPaletteBound = true;

        // Restore persisted width + collapsed state on first bind.
        try {
            const storedW = parseFloat(
                window.localStorage.getItem(WIDTH_KEY) || ""
            );
            if (Number.isFinite(storedW)) {
                setWidth(columns, clampWidth(storedW));
            }
            if (window.localStorage.getItem(COLLAPSED_KEY) === "1") {
                columns.classList.add("palette-collapsed");
            }
        } catch (err) {
            /* localStorage unavailable (private mode) — keep defaults. */
        }

        // Collapse toggle.
        collapseBtn.addEventListener("click", function (event) {
            event.stopPropagation();
            const collapsed = columns.classList.toggle("palette-collapsed");
            try {
                window.localStorage.setItem(
                    COLLAPSED_KEY,
                    collapsed ? "1" : "0"
                );
            } catch (err) {
                /* ignore persistence failures */
            }
        });

        // Drag the divider to resize — unless the pointer-down landed on the
        // collapse button, or the palette is currently collapsed.
        let resizing = false;
        let startX = 0;
        let startWidth = 0;
        let currentWidth = 0;

        divider.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) return;
            if (event.target.closest(".builder-palette-collapse")) return;
            if (columns.classList.contains("palette-collapsed")) return;
            resizing = true;
            startX = event.clientX;
            const pane = document.getElementById(PANE_ID);
            startWidth = pane ? pane.getBoundingClientRect().width : 320;
            currentWidth = startWidth;
            columns.classList.add("is-resizing");
            try {
                divider.setPointerCapture(event.pointerId);
            } catch (err) {}
            event.preventDefault();
        });

        divider.addEventListener("pointermove", function (event) {
            if (!resizing) return;
            currentWidth = clampWidth(startWidth + (event.clientX - startX));
            setWidth(columns, currentWidth);
            event.preventDefault();
        });

        function endResize(event) {
            if (!resizing) return;
            resizing = false;
            columns.classList.remove("is-resizing");
            try {
                divider.releasePointerCapture(event.pointerId);
            } catch (err) {}
            try {
                window.localStorage.setItem(
                    WIDTH_KEY,
                    String(Math.round(currentWidth))
                );
            } catch (err) {
                /* ignore persistence failures */
            }
        }

        divider.addEventListener("pointerup", endResize);
        divider.addEventListener("pointercancel", endResize);
    }

    function schedulePaletteBind() {
        requestAnimationFrame(bindPalette);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", schedulePaletteBind);
    } else {
        schedulePaletteBind();
    }

    new MutationObserver(schedulePaletteBind).observe(document.body, {
        childList: true,
        subtree: true,
    });
})();


/* PhenoTypic Pipeline Builder — drag-to-pan the fixed linear map.
 *
 * Grab the empty canvas and drag to scroll the map (in addition to the
 * wheel / scrollbars). Pointer-downs that land on an interactive control
 * (node title button, port, help "?") are left alone so clicking still
 * selects nodes; a drag past a small threshold suppresses the trailing
 * click so a pan never accidentally selects a node. The viewport
 * (#linear-map-container) is stable across re-renders, so a per-element
 * flag keeps the listeners single-bound.
 */
(function () {
    "use strict";

    const VIEWPORT_ID = "linear-map-container";
    const INTERACTIVE =
        "button, a, input, select, textarea, [role='button']";
    const THRESHOLD = 4;

    function bindPan() {
        const vp = document.getElementById(VIEWPORT_ID);
        if (!vp || vp.__phenotypicPanBound) return;
        vp.__phenotypicPanBound = true;

        let panning = false;
        let moved = false;
        let startX = 0;
        let startY = 0;
        let startLeft = 0;
        let startTop = 0;

        vp.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) return; // left button only
            if (event.target.closest(INTERACTIVE)) return; // let controls work
            panning = true;
            moved = false;
            startX = event.clientX;
            startY = event.clientY;
            startLeft = vp.scrollLeft;
            startTop = vp.scrollTop;
            vp.classList.add("is-grabbing");
            // Capture the pointer so move/up reliably reach the viewport even
            // if the cursor leaves it or crosses other elements mid-drag.
            try {
                vp.setPointerCapture(event.pointerId);
            } catch (err) {
                /* pointer capture unsupported — fall back to bubbling */
            }
        });

        vp.addEventListener("pointermove", function (event) {
            if (!panning) return;
            const dx = event.clientX - startX;
            const dy = event.clientY - startY;
            if (!moved && Math.abs(dx) + Math.abs(dy) < THRESHOLD) return;
            moved = true;
            vp.scrollLeft = startLeft - dx;
            vp.scrollTop = startTop - dy;
            event.preventDefault();
        });

        function endPan(event) {
            if (!panning) return;
            panning = false;
            vp.classList.remove("is-grabbing");
            try {
                vp.releasePointerCapture(event.pointerId);
            } catch (err) {
                /* ignore */
            }
        }

        vp.addEventListener("pointerup", endPan);
        vp.addEventListener("pointercancel", endPan);

        // Suppress the click that trails a real pan so it doesn't select a
        // node. Capture phase so it beats the node's own click handler.
        vp.addEventListener(
            "click",
            function (event) {
                if (moved) {
                    event.preventDefault();
                    event.stopPropagation();
                    moved = false;
                }
            },
            true,
        );
    }

    function schedulePanBind() {
        requestAnimationFrame(bindPan);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", schedulePanBind);
    } else {
        schedulePanBind();
    }

    new MutationObserver(schedulePanBind).observe(document.body, {
        childList: true,
        subtree: true,
    });
})();


/* PhenoTypic Pipeline Builder — mobile limited-mode enforcement.
 *
 * CSS handles the visual treatment for narrow viewports, but edit affordances
 * also need real DOM disabling so keyboard and assistive-tech activation cannot
 * mutate the linear builder. Dash swaps large subtrees during normal rendering,
 * so this observer reapplies the attributes whenever controls are replaced.
 */

(function () {
    "use strict";

    const MOBILE_LIMITED_QUERY = "(max-width: 768px)";

    const DISABLED_SELECTORS = [
        ".palette-button",
        ".linear-port-menu-action:not(.linear-port-menu-close)",
        ".linear-side-action:not(.linear-side-drill-action)",
        "#btn-save",
        "#btn-save-confirm",
        "#btn-new-pipeline-node",
    ];

    const FIELD_SELECTORS = [
        ".linear-side-param-form input",
        ".linear-side-param-form textarea",
        ".linear-side-param-form select",
        "#input-node-label",
    ];

    const ENABLED_SELECTORS = [
        ".linear-port-button",
        ".linear-unsupported-export-action",
        ".linear-help-button",
        ".linear-map-zoom-control",
        ".linear-port-menu-close",
        ".linear-side-drill-action",
        ".breadcrumb button",
        "#issue-badge",
    ];

    function setDisabled(el, limited) {
        if ("disabled" in el) {
            el.disabled = limited;
        }
        el.setAttribute("aria-disabled", limited ? "true" : "false");
    }

    function setFieldLimited(el, limited) {
        const tag = el.tagName ? el.tagName.toLowerCase() : "";
        const type = (el.getAttribute("type") || "").toLowerCase();
        const mustDisable =
            tag === "select" ||
            type === "checkbox" ||
            type === "radio" ||
            type === "file";

        if (mustDisable && "disabled" in el) {
            el.disabled = limited;
        } else if ("readOnly" in el) {
            el.readOnly = limited;
        } else if ("disabled" in el) {
            el.disabled = limited;
        }
        el.setAttribute("aria-disabled", limited ? "true" : "false");
    }

    function enableAlwaysAvailable(el) {
        if ("disabled" in el) {
            el.disabled = false;
        }
        if ("readOnly" in el) {
            el.readOnly = false;
        }
        el.removeAttribute("aria-disabled");
    }

    function applyMobileLimitedMode() {
        const limited =
            window.matchMedia &&
            window.matchMedia(MOBILE_LIMITED_QUERY).matches;

        DISABLED_SELECTORS.forEach((selector) => {
            document.querySelectorAll(selector).forEach((el) => {
                setDisabled(el, limited);
            });
        });

        FIELD_SELECTORS.forEach((selector) => {
            document.querySelectorAll(selector).forEach((el) => {
                setFieldLimited(el, limited);
            });
        });

        if (limited) {
            ENABLED_SELECTORS.forEach((selector) => {
                document.querySelectorAll(selector).forEach(enableAlwaysAvailable);
            });
        }
    }

    function scheduleApply() {
        requestAnimationFrame(applyMobileLimitedMode);
    }

    function startMobileLimiter() {
        scheduleApply();

        if (window.matchMedia) {
            const mq = window.matchMedia(MOBILE_LIMITED_QUERY);
            if (typeof mq.addEventListener === "function") {
                mq.addEventListener("change", scheduleApply);
            } else if (typeof mq.addListener === "function") {
                mq.addListener(scheduleApply);
            }
        }

        new MutationObserver(scheduleApply).observe(document.body, {
            childList: true,
            subtree: true,
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startMobileLimiter);
    } else {
        startMobileLimiter();
    }
})();


/* PhenoTypic Pipeline Builder — asset-readiness poller (spec §5.5).
 *
 * Each new DAG-canvas JS asset writes a sentinel onto ``window`` when
 * its IIFE finishes binding (``phenotypic_<name>_ready``); ``viewport_ops``
 * additionally exposes a ``phenotypic_viewport_ops_dagre_missing``
 * flag when the vendored ``cytoscape-dagre.min.js`` failed to register.
 *
 * The poll publishes ``STORE_ASSET_STATUS`` as a boolean-flag dict
 * matching the server callback's contract (see
 * ``_callbacks.asset_status_disables``):
 *
 *    {wire_drawing: bool, palette_dnd: bool, viewport_ops: bool,
 *     dagre_missing: bool}
 *
 * ``dash_clientside.set_props`` is the publish channel (Dash 2.18+).
 * The poll runs every 500ms for at most 1500ms total; an asset whose
 * IIFE has not shipped (e.g. palette_dnd before Phase 3) defaults to
 * ``true`` so it doesn't trip the banner prematurely. */
(function () {
    "use strict";

    /** Mirror of ``builder/_ids.STORE_ASSET_STATUS``. Kept as a
     *  constant here so a future rename only touches one place. */
    const STORE_ASSET_STATUS_ID = "store-asset-status";

    /** Asset name → ``window`` sentinel that the corresponding IIFE
     *  raises once it has bound its handlers. Each key matches a
     *  field in the ``STORE_ASSET_STATUS`` payload the Python callback
     *  in ``_callbacks.asset_status_disables`` consumes. */
    const ASSET_READY_FLAGS = {
        viewport_ops: "phenotypic_viewport_ops_ready",
        palette_dnd: "phenotypic_palette_dnd_ready",
        wire_drawing: "phenotypic_wire_drawing_ready",
    };

    /** Auxiliary "extension missing" flags. Asset-specific: an asset
     *  may still report ``ready`` while an optional dependency (e.g.
     *  cytoscape-dagre) reported itself absent. */
    const ASSET_DEGRADED_FLAGS = {
        dagre_missing: "phenotypic_viewport_ops_dagre_missing",
    };

    const TOTAL_BUDGET_MS = 1500;
    const POLL_INTERVAL_MS = 500;

    /** Read each sentinel from ``window`` and emit the boolean-flag
     *  payload the server-side callback expects. The shape mirrors the
     *  initial ``STORE_ASSET_STATUS.data`` value mounted in
     *  ``_layout.build_app_layout``. */
    function collectAssetStatus() {
        const status = {};
        for (const name in ASSET_READY_FLAGS) {
            status[name] = Boolean(window[ASSET_READY_FLAGS[name]]);
        }
        for (const name in ASSET_DEGRADED_FLAGS) {
            status[name] = Boolean(window[ASSET_DEGRADED_FLAGS[name]]);
        }
        return status;
    }

    /** Push ``status`` into ``STORE_ASSET_STATUS`` via the
     *  ``set_props`` clientside hook (Dash 2.18+). Older Dash builds
     *  silently skip publishing — the callback then keeps the default
     *  "everything ready" state. */
    function publishAssetStatus(status) {
        if (
            !(
                window.dash_clientside &&
                typeof window.dash_clientside.set_props === "function"
            )
        ) {
            return;
        }
        try {
            window.dash_clientside.set_props(STORE_ASSET_STATUS_ID, {
                data: status,
            });
        } catch (err) {
            // STORE_ASSET_STATUS not rendered yet — the poll keeps
            // retrying until either it exists or the budget runs out.
        }
    }

    /** Cheap shallow-equality probe used to skip redundant publishes. */
    function statusEqual(a, b) {
        if (!a || !b) return false;
        for (const key in a) {
            if (a[key] !== b[key]) return false;
        }
        for (const key in b) {
            if (a[key] !== b[key]) return false;
        }
        return true;
    }

    /** Drive the poll loop. Runs every ``POLL_INTERVAL_MS`` for at
     *  most ``TOTAL_BUDGET_MS`` total; whichever comes first:
     *   - all assets report ready (early exit + final publish), or
     *   - the budget is exhausted (final publish with whatever's still
     *     missing). */
    function pollAssetStatus() {
        const start = Date.now();
        let last;
        const handle = setInterval(function () {
            const status = collectAssetStatus();
            const allReady =
                status.viewport_ops &&
                status.palette_dnd &&
                status.wire_drawing &&
                !status.dagre_missing;
            const elapsed = Date.now() - start;
            if (!statusEqual(last, status)) {
                publishAssetStatus(status);
                last = status;
            }
            if (allReady || elapsed >= TOTAL_BUDGET_MS) {
                clearInterval(handle);
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
