/*
 * point_picker.js
 * -------------------------------------------------------------------------
 * Client-side lifecycle layer for the PhenoTypic Builder Point Picker modal.
 *
 * Responsibilities:
 *   1. Bootstrap OpenSeadragon (CDN first, vendored fallback) — mirrors
 *      results_viewer.js so OSD is available as soon as the modal opens.
 *   2. Maintain a single OSD viewer instance (this modal hosts at most one
 *      viewer at a time, unlike the results viewer which keeps a Map).
 *   3. Provide mountViewer / redrawOverlay / disposeViewer helpers consumed
 *      by Dash clientside callbacks in `_callbacks.py`.
 *   4. Capture `canvas-click` events, convert pixel → image coordinates,
 *      and push the resulting [y, x] pair into PICKER_STAGED_STORE via
 *      `dash_clientside.set_props`.
 *   5. Render a small red SVG-style marker overlay for every staged point,
 *      using OSD's overlay-by-viewport-rect API so markers track zoom.
 *
 * Public surface (under `window.__phenotypicBuilderPointPicker`):
 *   - osdReady              : Promise resolving when OSD is loaded.
 *   - mountViewer(divId, dziUrl, stagedPoints) : (re)create the viewer.
 *   - redrawOverlay(points) : refresh the marker overlay only.
 *   - disposeViewer()       : tear down the cached viewer + clear overlay.
 * -------------------------------------------------------------------------
 */

/* ============================================================
 * (A) Bootstrap: load OpenSeadragon, CDN-first with fallback.
 * ============================================================ */
(function () {
    "use strict";

    // ``window.__phenotypicAppPrefix`` is injected by the Dash factory
    // (see builder/_app.py). It carries the mount-point prefix when the
    // app is hosted under the unified GUI hub (``/builder/``); falls back
    // to ``/`` for the standalone ``python -m phenotypic.gui.builder``.
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix
        : "/";
    function siblingPrefix(prefix, mountName) {
        let base = prefix.endsWith("/") ? prefix : prefix + "/";
        if (base.endsWith("/builder/")) {
            base = base.slice(0, -"builder/".length);
        }
        return base + mountName + "/";
    }
    const resultsPrefix = siblingPrefix(appPrefix, "results");

    function loadOpenSeadragon() {
        return new Promise(function (resolve, reject) {
            // If something already loaded OSD before us (e.g. results
            // viewer was visited in this tab), just resolve.
            if (window.OpenSeadragon) {
                resolve("preloaded");
                return;
            }
            const cdn = "https://cdn.jsdelivr.net/npm/openseadragon@5/build/openseadragon/openseadragon.min.js";
            // The vendored OSD copy lives under the results-viewer assets
            // tree (its own pip extra ships those tiles + button images).
            // Under the unified hub composer the results viewer's static
            // assets live at the sibling results prefix. Standalone
            // ``python -m phenotypic.gui.builder`` does not mount the results
            // viewer, so the fallback is CDN-only there.
            const local = resultsPrefix + "assets/openseadragon/openseadragon.min.js";
            const tag = document.createElement("script");
            tag.src = cdn;
            tag.async = true;
            tag.onload = function () {
                console.info("[point_picker] OSD loaded from CDN");
                resolve("cdn");
            };
            tag.onerror = function () {
                console.warn("[point_picker] OSD CDN failed, falling back to vendored copy");
                const fallback = document.createElement("script");
                fallback.src = local;
                fallback.async = true;
                fallback.onload = function () {
                    console.info("[point_picker] OSD loaded from vendored copy");
                    resolve("vendored");
                };
                fallback.onerror = function () {
                    console.error("[point_picker] OSD failed to load from both CDN and vendored copy");
                    reject(new Error("OSD load failure"));
                };
                document.head.appendChild(fallback);
            };
            document.head.appendChild(tag);
        });
    }

    const osdReady = loadOpenSeadragon();
    window.__phenotypicBuilderPointPicker =
        window.__phenotypicBuilderPointPicker || {};
    window.__phenotypicBuilderPointPicker.osdReady = osdReady;
})();

/* ============================================================
 * (B) Single-viewer lifecycle + click capture + overlay markers.
 * ============================================================ */
(function () {
    "use strict";
    const ns = window.__phenotypicBuilderPointPicker =
        window.__phenotypicBuilderPointPicker || {};

    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix
        : "/";
    function siblingPrefix(prefix, mountName) {
        let base = prefix.endsWith("/") ? prefix : prefix + "/";
        if (base.endsWith("/builder/")) {
            base = base.slice(0, -"builder/".length);
        }
        return base + mountName + "/";
    }
    const resultsPrefix = siblingPrefix(appPrefix, "results");

    // Single cached viewer — this modal hosts exactly one OSD instance.
    let viewer = null;
    // Element id of the host div (cached so disposeViewer can reach it
    // even when the caller doesn't pass it back in).
    let viewerDivId = null;
    // The staged point list as last seen by mountViewer / redrawOverlay.
    // The click handler reads from this closure to build the next list to
    // push back to Dash via set_props.
    let currentStagedPoints = [];
    // Track every overlay element we've added so we can tear them down
    // before each redraw (OSD owns the DOM; we keep refs to call
    // viewer.removeOverlay).
    let overlayElements = [];
    // Track whether OSD has fired the "open" event (i.e. viewport is
    // ready). Before this, viewport.imageToViewportCoordinates returns
    // garbage and overlays land in the wrong place.
    let viewerOpen = false;

    function _normalisePoints(points) {
        if (!Array.isArray(points)) return [];
        const out = [];
        for (let i = 0; i < points.length; i += 1) {
            const p = points[i];
            if (Array.isArray(p) && p.length >= 2) {
                const y = Number(p[0]);
                const x = Number(p[1]);
                if (Number.isFinite(y) && Number.isFinite(x)) {
                    out.push([y, x]);
                }
            }
        }
        return out;
    }

    function _clearOverlay() {
        if (!viewer) {
            overlayElements = [];
            return;
        }
        for (let i = 0; i < overlayElements.length; i += 1) {
            try {
                viewer.removeOverlay(overlayElements[i]);
            } catch (e) {
                // OSD logs its own warnings; nothing actionable here.
            }
        }
        overlayElements = [];
    }

    function _addMarker(yx) {
        if (!viewer || !window.OpenSeadragon || !viewerOpen) return null;
        const Point = window.OpenSeadragon.Point;
        // Image coords are row-major: yx = [y, x]; OSD expects (x, y).
        const imagePoint = new Point(yx[1], yx[0]);
        let viewportPoint;
        try {
            viewportPoint = viewer.viewport.imageToViewportCoordinates(imagePoint);
        } catch (e) {
            console.warn("[point_picker] could not project marker", yx, e);
            return null;
        }
        const el = document.createElement("div");
        el.className = "point-picker-marker";
        try {
            viewer.addOverlay({
                element: el,
                location: viewportPoint,
                placement: window.OpenSeadragon.Placement.CENTER,
            });
        } catch (e) {
            console.warn("[point_picker] addOverlay failed", e);
            return null;
        }
        overlayElements.push(el);
        return el;
    }

    function _drawMarkers(points) {
        for (let i = 0; i < points.length; i += 1) {
            _addMarker(points[i]);
        }
    }

    function _pointsEqual(a, b) {
        if (a === b) return true;
        if (!a || !b || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i += 1) {
            if (a[i][0] !== b[i][0] || a[i][1] !== b[i][1]) return false;
        }
        return true;
    }

    /**
     * Refresh just the overlay markers (no remount of the viewer).
     *
     * Diffs against the previously-rendered list and only adds/removes the
     * suffix that changed — clicks always append, undo always pops, so the
     * common case is O(1) DOM work instead of full teardown.
     *
     * @param {Array<[number, number]>} points - list of [y, x] image-coord
     *   pairs.
     */
    ns.redrawOverlay = function (points) {
        const next = _normalisePoints(points);
        if (_pointsEqual(currentStagedPoints, next)) return;
        const prev = currentStagedPoints;
        currentStagedPoints = next;
        if (!viewer || !viewerOpen) return;

        // Find the longest common prefix; only mutate from there.
        let common = 0;
        const cap = Math.min(prev.length, next.length);
        while (
            common < cap
            && prev[common][0] === next[common][0]
            && prev[common][1] === next[common][1]
        ) {
            common += 1;
        }
        // Pop any trailing markers that no longer match.
        for (let i = overlayElements.length - 1; i >= common; i -= 1) {
            try { viewer.removeOverlay(overlayElements[i]); }
            catch (e) { /* OSD warns; nothing actionable here. */ }
        }
        overlayElements.length = common;
        // Add new markers for the appended suffix.
        for (let i = common; i < next.length; i += 1) {
            _addMarker(next[i]);
        }
    };

    /**
     * Create (or recreate) the single OSD viewer in the host div, load the
     * supplied DZI source, attach a canvas-click handler, and seed the
     * overlay with `stagedPoints`.
     *
     * Idempotent: if a viewer with the same DZI URL is already mounted on
     * the same host div, this just re-syncs the overlay and returns.
     *
     * @param {string} divId  - DOM id of the host element.
     * @param {string} dziUrl - URL to a .dzi descriptor.
     * @param {Array<[number, number]>} stagedPoints - existing picks.
     */
    ns.mountViewer = async function (divId, dziUrl, stagedPoints) {
        await ns.osdReady;
        const el = document.getElementById(divId);
        if (!el) {
            console.warn("[point_picker] mountViewer: no element", divId);
            return null;
        }
        currentStagedPoints = _normalisePoints(stagedPoints);

        // Same DZI on the same host div -> skip remount, just sync overlay.
        if (
            viewer
            && viewerDivId === divId
            && viewer._phenotypicDziUrl === dziUrl
        ) {
            _clearOverlay();
            if (viewerOpen) {
                _drawMarkers(currentStagedPoints);
            }
            return viewer;
        }

        // Tear down the previous viewer (if any) before creating a new one.
        if (viewer) {
            _clearOverlay();
            try { viewer.destroy(); } catch (e) { console.error(e); }
            viewer = null;
            viewerDivId = null;
            viewerOpen = false;
        }

        el.setAttribute("aria-busy", "true");

        // OSD's `prefixUrl` resolves the icon images for the zoom/home/
        // fullpage buttons. The vendored icon set lives under the
        // results-viewer's assets tree (the OSD-JS bundle does too; see
        // `loadOpenSeadragon` in section (A)), so use the sibling results
        // prefix instead of `appPrefix + "assets/..."` — the builder assets
        // folder does not vendor the icons.
        viewer = window.OpenSeadragon({
            element: el,
            prefixUrl: resultsPrefix + "assets/openseadragon/images/",
            tileSources: dziUrl,
            showNavigator: false,
            showRotationControl: false,
            animationTime: 0.5,
            blendTime: 0.1,
            constrainDuringPan: true,
            visibilityRatio: 0.5,
            minZoomLevel: 0.5,
            maxZoomPixelRatio: 4,
            immediateRender: false,
            // Suppress the default click-to-zoom so single-tap picks don't
            // also zoom the canvas. We still preserve drag-to-pan and
            // wheel-to-zoom (controlled separately by gestureSettings).
            gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
            gestureSettingsTouch: { clickToZoom: false, dblClickToZoom: true },
        });
        viewer._phenotypicDziUrl = dziUrl;
        viewerDivId = divId;
        viewerOpen = false;

        viewer.addHandler("open", function () {
            viewerOpen = true;
            el.setAttribute("aria-busy", "false");
            // Now the viewport is initialised; safe to project overlays.
            _clearOverlay();
            _drawMarkers(currentStagedPoints);
        });

        viewer.addHandler("close", function () {
            viewerOpen = false;
        });

        // The click handler is the heart of the feature. Filter to "quick"
        // single-tap events so drags / pinch-zooms don't drop a point.
        viewer.addHandler("canvas-click", function (event) {
            if (!event.quick) return;
            // Block OSD's default click action (zoom-to-point).
            event.preventDefaultAction = true;
            if (!viewer || !viewerOpen) return;

            let viewportPoint;
            let imagePoint;
            try {
                viewportPoint = viewer.viewport.pointFromPixel(event.position);
                imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
            } catch (e) {
                console.warn("[point_picker] click projection failed", e);
                return;
            }
            // image coords are row-major in this codebase: push [y, x].
            const nextPoints = currentStagedPoints.slice();
            nextPoints.push([imagePoint.y, imagePoint.x]);

            const dc = window.dash_clientside;
            if (!dc || typeof dc.set_props !== "function") {
                console.warn("[point_picker] dash_clientside.set_props unavailable");
                return;
            }
            dc.set_props("picker-staged-store", { data: nextPoints });
        });

        return viewer;
    };

    /**
     * Tear down the single cached viewer and clear all overlay markers.
     * Called by the Cancel / Confirm clientside callbacks when the modal
     * closes.
     */
    ns.disposeViewer = function () {
        if (!viewer) {
            // Even with no viewer cached, drop any tracked elements so a
            // future mount starts from a clean slate.
            overlayElements = [];
            currentStagedPoints = [];
            viewerOpen = false;
            return;
        }
        _clearOverlay();
        try {
            viewer.destroy();
        } catch (e) {
            console.error(e);
        }
        // Clear the aria-busy flag in case the modal closes mid-load.
        if (viewerDivId) {
            const el = document.getElementById(viewerDivId);
            if (el) el.setAttribute("aria-busy", "false");
        }
        viewer = null;
        viewerDivId = null;
        viewerOpen = false;
        overlayElements = [];
        currentStagedPoints = [];
    };
})();
