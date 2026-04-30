/*
 * results_viewer.js
 * -------------------------------------------------------------------------
 * Client-side lifecycle layer for the PhenoTypic Results Viewer.
 *
 * Responsibilities:
 *   1. Bootstrap OpenSeadragon (CDN first, vendored fallback).
 *   2. Maintain a registry of OSD viewer instances keyed by their host divId.
 *   3. Provide mount / dispose helpers for use by Dash clientside callbacks.
 *   4. Implement an opt-in "lock views" mode that synchronizes pan/zoom
 *      across all mounted viewers.
 *   5. Watch the cards container for DOM removals and dispose orphaned
 *      viewers automatically (cards come and go via Dash callbacks).
 *
 * The Dash app loads this file because `assets_folder="_assets"` is set in
 * `_app.py`, so it is served at `/assets/results_viewer.js`.
 *
 * Public surface (under `window.__phenotypicResultsViewer`):
 *   - osdReady              : Promise that resolves when OSD is loaded.
 *   - viewers               : Map<divId, OpenSeadragon.Viewer>.
 *   - mountViewer(id, dzi)  : (re)create a viewer in the given div.
 *   - disposeViewer(id)     : tear down and forget a viewer.
 *   - setLockViews(active)  : toggle synchronized pan/zoom.
 *   - applyImageSelection(states) : helper invoked from clientside callbacks.
 * -------------------------------------------------------------------------
 */

/* ============================================================
 * (A) Bootstrap: load OpenSeadragon, CDN-first with fallback.
 * ============================================================ */
(function () {
    "use strict";

    function loadOpenSeadragon() {
        return new Promise(function (resolve, reject) {
            // If something already loaded OSD before us, just resolve.
            if (window.OpenSeadragon) {
                resolve("preloaded");
                return;
            }
            const cdn = "https://cdn.jsdelivr.net/npm/openseadragon@5/build/openseadragon/openseadragon.min.js";
            const local = "/assets/openseadragon/openseadragon.min.js";
            const tag = document.createElement("script");
            tag.src = cdn;
            tag.async = true;
            tag.onload = function () {
                console.info("[results_viewer] OSD loaded from CDN");
                resolve("cdn");
            };
            tag.onerror = function () {
                console.warn("[results_viewer] OSD CDN failed, falling back to vendored copy");
                const fallback = document.createElement("script");
                fallback.src = local;
                fallback.async = true;
                fallback.onload = function () {
                    console.info("[results_viewer] OSD loaded from vendored copy");
                    resolve("vendored");
                };
                fallback.onerror = function () {
                    console.error("[results_viewer] OSD failed to load from both CDN and vendored copy");
                    reject(new Error("OSD load failure"));
                };
                document.head.appendChild(fallback);
            };
            document.head.appendChild(tag);
        });
    }

    const osdReady = loadOpenSeadragon();
    window.__phenotypicResultsViewer = window.__phenotypicResultsViewer || {};
    window.__phenotypicResultsViewer.osdReady = osdReady;
})();

/* ============================================================
 * (B) Viewer registry and lifecycle helpers.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer = window.__phenotypicResultsViewer || {};
    ns.viewers = ns.viewers || new Map();   // divId -> OpenSeadragon.Viewer

    /**
     * Create (or recreate) an OpenSeadragon viewer in the element with the
     * given DOM id and load the supplied DZI tile source.
     *
     * @param {string} divId  - the host element id (matches a card-osd-div).
     * @param {string} dziUrl - URL to a .dzi descriptor served by the app.
     * @returns {Promise<OpenSeadragon.Viewer | null>}
     */
    ns.mountViewer = async function (divId, dziUrl) {
        await ns.osdReady;
        const el = document.getElementById(divId);
        if (!el) {
            console.warn("[results_viewer] mountViewer: no element", divId);
            return null;
        }
        // Skip if the same DZI is already mounted on this div: the
        // pattern-matching ALL clientside callback fires for *every*
        // card-state change, not just the one that changed, so an
        // unconditional teardown would needlessly re-fetch tiles for
        // every other card on every selection.
        const existing = ns.viewers.get(divId);
        if (existing && existing._phenotypicDziUrl === dziUrl) {
            return existing;
        }
        if (existing) {
            try { existing.destroy(); }
            catch (e) { console.error(e); }
            ns.viewers.delete(divId);
        }
        const viewer = window.OpenSeadragon({
            element: el,
            prefixUrl: "/assets/openseadragon/images/",
            tileSources: dziUrl,
            showNavigator: false,
            showRotationControl: false,
            animationTime: 0.5,
            blendTime: 0.1,
            constrainDuringPan: true,
            visibilityRatio: 0.5,
            minZoomLevel: 0.5,
            maxZoomPixelRatio: 2,
            // Defer rendering until tiles are ready; produces a softer crossfade.
            immediateRender: false,
        });
        viewer._phenotypicDziUrl = dziUrl;
        ns.viewers.set(divId, viewer);
        if (ns.lockViewsActive) ns._attachLockHandlers(viewer);
        return viewer;
    };

    /**
     * Dispose a viewer (if any) registered under the given div id.
     */
    ns.disposeViewer = function (divId) {
        const v = ns.viewers.get(divId);
        if (!v) return;
        try { v.destroy(); }
        catch (e) { console.error(e); }
        ns.viewers.delete(divId);
    };
})();

/* ============================================================
 * (C) Lock-views: synchronized pan/zoom across all viewers.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;
    ns.lockViewsActive = false;
    ns._lockHandlers = new Map();   // viewer -> handler fn
    let _broadcasting = false;       // re-entrancy guard for cross-broadcast

    function broadcastViewport(srcViewer) {
        if (_broadcasting) return;
        _broadcasting = true;
        try {
            const center = srcViewer.viewport.getCenter(true);
            const zoom = srcViewer.viewport.getZoom(true);
            ns.viewers.forEach(function (v) {
                if (v === srcViewer) return;
                v.viewport.zoomTo(zoom, null, true);
                v.viewport.panTo(center, true);
            });
        } finally {
            _broadcasting = false;
        }
    }

    /**
     * Attach an "animation" handler that mirrors this viewer's viewport
     * onto all peers. Idempotent: re-attaching is a no-op.
     */
    ns._attachLockHandlers = function (viewer) {
        if (ns._lockHandlers.has(viewer)) return;
        const handler = function () { broadcastViewport(viewer); };
        viewer.addHandler("animation", handler);
        ns._lockHandlers.set(viewer, handler);
    };

    /**
     * Detach the lock handler that was attached by _attachLockHandlers.
     */
    ns._detachLockHandlers = function (viewer) {
        const h = ns._lockHandlers.get(viewer);
        if (!h) return;
        viewer.removeHandler("animation", h);
        ns._lockHandlers.delete(viewer);
    };

    /**
     * Toggle lock-views mode. The Python clientside callback in
     * `_callbacks.py` invokes this on STORE_LOCK_VIEWS changes.
     */
    ns.setLockViews = function (active) {
        ns.lockViewsActive = !!active;
        ns.viewers.forEach(function (v) {
            if (ns.lockViewsActive) ns._attachLockHandlers(v);
            else ns._detachLockHandlers(v);
        });
        return ns.lockViewsActive;
    };
})();

/* ============================================================
 * (D) MutationObserver: dispose viewers when their card is removed.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;

    function findOsdCanvases(node) {
        if (!node) return [];
        const out = [];
        const isCanvas = function (el) {
            return el.classList && el.classList.contains("osd-canvas");
        };
        if (node.nodeType === 1 && isCanvas(node)) out.push(node);
        if (node.querySelectorAll) {
            node.querySelectorAll(".osd-canvas").forEach(function (el) {
                out.push(el);
            });
        }
        return out;
    }

    function startObserver() {
        const container = document.getElementById("cards-container");
        if (!container) return false;
        const obs = new MutationObserver(function (mutations) {
            mutations.forEach(function (m) {
                m.removedNodes.forEach(function (n) {
                    findOsdCanvases(n).forEach(function (canvas) {
                        if (canvas.id) ns.disposeViewer(canvas.id);
                    });
                });
            });
        });
        obs.observe(container, { childList: true, subtree: true });
        ns._cardsObserver = obs;
        console.info("[results_viewer] cards-container observer attached");
        return true;
    }

    // Dash mounts the layout asynchronously; poll until the container
    // exists, then attach the observer.
    if (!startObserver()) {
        const interval = setInterval(function () {
            if (startObserver()) clearInterval(interval);
        }, 100);
    }
})();

/* ============================================================
 * (E) Helper invoked by the Python clientside callbacks.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer;

    /**
     * Apply a batch of image selections to mounted viewer cards.
     *
     * Each entry in `cardStates` describes the desired state of one card:
     *   { id: <divId>, dataset: <dataset>, stem: <imageStem> }
     * If `dataset` or `stem` is missing/falsy the card's viewer is
     * disposed (used when a card is "cleared").
     *
     * The DZI URL is `/tiles/<dataset>/<stem>.dzi`, served by the
     * Flask blueprint in `_tile_routes.py`.
     */
    ns.applyImageSelection = function (cardStates) {
        if (!Array.isArray(cardStates)) return null;
        cardStates.forEach(function (s) {
            if (!s || !s.id) return;
            if (!s.dataset || !s.stem) {
                ns.disposeViewer(s.id);
                return;
            }
            const dziUrl = "/tiles/" + encodeURIComponent(s.dataset) +
                           "/" + encodeURIComponent(s.stem) + ".dzi";
            ns.mountViewer(s.id, dziUrl);
        });
        return null;
    };
})();
