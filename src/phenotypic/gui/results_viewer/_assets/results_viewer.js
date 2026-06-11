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

    // ``window.__phenotypicAppPrefix`` is injected by the Dash factory
    // (see results_viewer/_app.py::_index_string_with_prefix). It carries
    // the mount-point prefix when the app is hosted under the unified
    // GUI hub (``/results/``); falls back to ``/`` for standalone.
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix
        : "/";

    function loadOpenSeadragon() {
        return new Promise(function (resolve, reject) {
            // If something already loaded OSD before us, just resolve.
            if (window.OpenSeadragon) {
                resolve("preloaded");
                return;
            }
            const cdn = "https://cdn.jsdelivr.net/npm/openseadragon@5/build/openseadragon/openseadragon.min.js";
            const local = appPrefix + "assets/openseadragon/openseadragon.min.js";
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
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix
        : "/";

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
            prefixUrl: appPrefix + "assets/openseadragon/images/",
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
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix
        : "/";

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
            const dziUrl = appPrefix + "tiles/" + encodeURIComponent(s.dataset) +
                           "/" + encodeURIComponent(s.stem) + ".dzi";
            ns.mountViewer(s.id, dziUrl);
        });
        return null;
    };
})();

/* ============================================================
 * (F) Tile multi-select shift-click bridge (colony grid + QC gallery).
 *
 * Bridges native checkbox click events on a tile container into a
 * Dash dcc.Store so the Python side can apply selection logic (single
 * toggle, shift-range, etc.) without round-tripping every image's
 * <input> through the server.
 *
 * The SAME bridge serves two surfaces (M1: QC-review selection parity):
 *   - #colony-grid-container -> store-colony-selection-delta
 *   - #qc-review-gallery     -> store-qc-gallery-selection-delta
 * QC tiles carry the same `.colony-cell-checkbox` + `data-key` chrome as
 * colony tiles, so one parameterized listener covers both; each surface
 * emits to its own delta store, and a per-surface Python consumer folds
 * the delta into the SHARED store-colony-selection.
 *
 * Each click emits a payload of the form:
 *   { key: [image_file, label], shift: <bool>, ts: <ms> }
 * via window.dash_clientside.set_props.
 * ============================================================ */
(function () {
    "use strict";

    // (containerId, deltaStoreId, datasetFlag) per surface. The dataset
    // flag is the single source of truth for "this container already has
    // our listener", so the polling path and the body MutationObserver
    // agree even if they race on the same fresh container.
    const BRIDGES = [
        {
            containerId: "colony-grid-container",
            deltaStoreId: "store-colony-selection-delta",
            flag: "_colonyShiftBridge",
        },
        {
            containerId: "qc-review-gallery",
            deltaStoreId: "store-qc-gallery-selection-delta",
            flag: "_qcShiftBridge",
        },
    ];

    function attachListener(container, deltaStoreId, flag) {
        if (container.dataset[flag] === "1") return;
        container.dataset[flag] = "1";
        container.addEventListener("click", function (event) {
            const tgt = event.target;
            if (!tgt || !tgt.classList ||
                !tgt.classList.contains("colony-cell-checkbox")) {
                return;
            }
            const raw = tgt.dataset ? tgt.dataset.key : null;
            if (!raw || typeof raw !== "string") {
                console.warn("[results_viewer] tile checkbox missing data-key");
                return;
            }
            const parts = raw.split("::");
            if (parts.length !== 2) {
                console.warn("[results_viewer] tile data-key malformed:", raw);
                return;
            }
            const imageFile = parts[0];
            const label = parseInt(parts[1], 10);
            if (Number.isNaN(label)) {
                console.warn("[results_viewer] tile label not an int:", raw);
                return;
            }
            // Suppress the native toggle: Python owns the checked state.
            event.preventDefault();
            const payload = {
                key: [imageFile, label],
                shift: !!event.shiftKey,
                ts: Date.now(),
            };
            const dc = window.dash_clientside;
            if (!dc || typeof dc.set_props !== "function") {
                console.warn("[results_viewer] dash_clientside.set_props unavailable");
                return;
            }
            dc.set_props(deltaStoreId, { data: payload });
        });
        console.info(
            "[results_viewer] shift-click bridge attached:", container.id
        );
    }

    function tryAttach() {
        // Returns true only once EVERY surface's container is present, so
        // the initial poll keeps running until both have mounted (QC mounts
        // later than the colony grid).
        let allPresent = true;
        BRIDGES.forEach(function (b) {
            const container = document.getElementById(b.containerId);
            if (container) {
                attachListener(container, b.deltaStoreId, b.flag);
            } else {
                allPresent = false;
            }
        });
        return allPresent;
    }

    // Dash mounts the tabs lazily; poll until every container exists, then
    // attach the delegated listeners and stop polling.
    if (!tryAttach()) {
        const interval = setInterval(function () {
            if (tryAttach()) clearInterval(interval);
        }, 100);
    }

    // If Dash later replaces a container (tab switch / re-render), the
    // dataset flag goes away with the old node. Watch <body> for re-mounts
    // and re-attach to whichever fresh container appears.
    function startReattachObserver() {
        if (!document.body) return false;
        const obs = new MutationObserver(function () {
            BRIDGES.forEach(function (b) {
                const container = document.getElementById(b.containerId);
                if (!container) return;
                // attachListener is idempotent — its dataset-flag guard
                // makes a no-op cheap, so we can fire on every mutation.
                attachListener(container, b.deltaStoreId, b.flag);
            });
        });
        obs.observe(document.body, { childList: true, subtree: true });
        return true;
    }

    if (!startReattachObserver()) {
        const bodyInterval = setInterval(function () {
            if (startReattachObserver()) clearInterval(bodyInterval);
        }, 100);
    }
})();

/* ============================================================
 * (G) QC Review worklist drag-splitter.
 *
 * A thin handle (#qc-review-splitter) between the worklist sidebar and
 * the detail/gallery pane. Dragging it widens/narrows the worklist
 * (#qc-review-worklist) live, clamped to [MIN, MAX] px; on mouse-up the
 * final width is persisted to the Dash store `store-qc-sidebar-width`
 * via window.dash_clientside.set_props, so a Python callback can re-apply
 * it across re-renders + collapse. Mirrors the colony shift-click bridge:
 * poll-to-attach + a <body> MutationObserver re-attach, both idempotent
 * via a dataset flag.
 *
 * `clampSidebarWidth` is exposed on the namespace so a test can drive the
 * exact clamp the drag uses without a real pointer drag.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicResultsViewer =
        window.__phenotypicResultsViewer || {};

    const MIN_W = 140;
    const MAX_W = 380;

    ns.clampSidebarWidth = function (px) {
        const n = Math.round(Number(px));
        if (!Number.isFinite(n)) return 180;  // default
        return Math.max(MIN_W, Math.min(MAX_W, n));
    };

    function persistWidth(px) {
        const dc = window.dash_clientside;
        if (!dc || typeof dc.set_props !== "function") {
            console.warn("[results_viewer] dash_clientside.set_props unavailable");
            return;
        }
        dc.set_props("store-qc-sidebar-width", { data: px });
    }

    function attachSplitter(handle) {
        if (handle.dataset._qcSplitter === "1") return;
        handle.dataset._qcSplitter = "1";
        handle.addEventListener("mousedown", function (downEvt) {
            const worklist = document.getElementById("qc-review-worklist");
            if (!worklist) return;
            downEvt.preventDefault();  // don't text-select while dragging
            const startX = downEvt.clientX;
            const startW = worklist.getBoundingClientRect().width;
            // Visual feedback during the drag.
            document.body.style.userSelect = "none";
            document.body.style.cursor = "col-resize";

            function onMove(moveEvt) {
                const next = ns.clampSidebarWidth(startW + (moveEvt.clientX - startX));
                worklist.style.width = next + "px";
            }
            function onUp() {
                document.removeEventListener("mousemove", onMove);
                document.removeEventListener("mouseup", onUp);
                document.body.style.userSelect = "";
                document.body.style.cursor = "";
                const finalW = ns.clampSidebarWidth(
                    worklist.getBoundingClientRect().width
                );
                worklist.style.width = finalW + "px";
                persistWidth(finalW);  // survives re-renders + collapse
            }
            document.addEventListener("mousemove", onMove);
            document.addEventListener("mouseup", onUp);
        });
        console.info("[results_viewer] QC review splitter attached");
    }

    function tryAttach() {
        const handle = document.getElementById("qc-review-splitter");
        if (!handle) return false;
        attachSplitter(handle);
        return true;
    }

    if (!tryAttach()) {
        const interval = setInterval(function () {
            if (tryAttach()) clearInterval(interval);
        }, 100);
    }

    // Re-attach if Dash re-mounts the Review subtree (tab/sub-view switch).
    function startReattachObserver() {
        if (!document.body) return false;
        const obs = new MutationObserver(function () {
            const handle = document.getElementById("qc-review-splitter");
            if (handle) attachSplitter(handle);  // idempotent
        });
        obs.observe(document.body, { childList: true, subtree: true });
        return true;
    }
    if (!startReattachObserver()) {
        const bodyInterval = setInterval(function () {
            if (startReattachObserver()) clearInterval(bodyInterval);
        }, 100);
    }
})();
