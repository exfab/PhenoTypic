/*
 * browse.js — single-viewport OpenSeadragon lifecycle for the Browse tab.
 * Loads OSD from the vendored copy (no CDN; offline-safe over a tunnel) and
 * exposes window.__phenotypicBrowse.applyImage({token,label}), invoked by a
 * Dash clientside callback when the current-image store changes.
 */
(function () {
    "use strict";
    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";
    const OSD_DIV_ID = "browse-osd-div";
    const LOADING_ID = "browse-osd-loading";
    const LOADING_TEXT_ID = "browse-loading-text";

    const ns = window.__phenotypicBrowse = window.__phenotypicBrowse || {};
    ns.viewer = ns.viewer || null;
    // Tokens whose DZI conversion has already been warmed this session, so a
    // rapid prev/next sweep never re-fetches the same neighbour. The browse
    // cache is never evicted mid-run, so "warmed once" stays warm.
    ns.prefetched = ns.prefetched || new Set();

    // Background-fetch each neighbour's .dzi to pre-warm the server-side
    // normalize + DZI-tile cache. The body is discarded — the point is the
    // disk cache the next ‹/› step will hit. A failed warm is dropped from the
    // set so a real navigation can retry it. Fired after the active image
    // opens so it never competes with the visible image's first paint.
    function prefetchNeighbors(tokens) {
        if (!Array.isArray(tokens)) { return; }
        tokens.forEach(function (tok) {
            if (!tok || ns.prefetched.has(tok)) { return; }
            ns.prefetched.add(tok);
            const url = appPrefix + "tiles/" + encodeURIComponent(tok) + ".dzi";
            fetch(url, { credentials: "same-origin" }).then(function (resp) {
                if (!resp.ok) { ns.prefetched.delete(tok); }
            }).catch(function () { ns.prefetched.delete(tok); });
        });
    }

    function basename(p) {
        if (!p) { return ""; }
        const parts = String(p).split("/");
        return parts[parts.length - 1] || String(p);
    }

    // Toggle the spinner overlay. state: "loading" | "error" | "hidden".
    // First view of a large RAW pays a multi-second server-side
    // normalize + DZI-tile inside the .dzi request, so OSD's open event
    // is the precise "image is now displayed" signal to hide on.
    function setLoading(state, message) {
        const overlay = document.getElementById(LOADING_ID);
        if (!overlay) { return; }
        if (state === "hidden") {
            overlay.classList.remove("is-visible", "browse-loading-overlay--error");
            return;
        }
        const textEl = document.getElementById(LOADING_TEXT_ID);
        if (textEl && message) { textEl.textContent = message; }
        overlay.classList.toggle("browse-loading-overlay--error", state === "error");
        overlay.classList.add("is-visible");
    }

    function loadOSD() {
        return new Promise(function (resolve, reject) {
            if (window.OpenSeadragon) { resolve(); return; }
            const tag = document.createElement("script");
            tag.src = appPrefix + "assets/openseadragon/openseadragon.min.js";
            tag.async = true;
            tag.onload = function () { resolve(); };
            tag.onerror = function () { reject(new Error("OSD vendored load failed")); };
            document.head.appendChild(tag);
        });
    }
    ns.osdReady = ns.osdReady || loadOSD();

    ns.applyImage = async function (payload) {
        await ns.osdReady;
        const el = document.getElementById(OSD_DIV_ID);
        if (!el) { return; }
        if (!payload || !payload.token) {
            if (ns.viewer) { try { ns.viewer.destroy(); } catch (e) {} ns.viewer = null; }
            setLoading("hidden");
            return;
        }
        const dziUrl = appPrefix + "tiles/" + encodeURIComponent(payload.token) + ".dzi";
        if (ns.viewer && ns.viewer._phenotypicDziUrl === dziUrl) { return; }
        if (ns.viewer) { try { ns.viewer.destroy(); } catch (e) {} ns.viewer = null; }
        const name = basename(payload.label || payload.token);
        setLoading("loading", name ? ("Loading " + name + "…") : "Loading image…");
        const viewer = window.OpenSeadragon({
            element: el,
            prefixUrl: appPrefix + "assets/openseadragon/images/",
            tileSources: dziUrl,
            showNavigator: false,
            showRotationControl: false,
            animationTime: 0.4,
            constrainDuringPan: true,
            visibilityRatio: 0.5,
            minZoomLevel: 0.4,
            maxZoomPixelRatio: 4,
            immediateRender: false,
        });
        viewer._phenotypicDziUrl = dziUrl;
        viewer.addHandler("open", function () {
            setLoading("hidden");
            prefetchNeighbors(payload.prefetch);
        });
        viewer.addHandler("open-failed", function () {
            setLoading("error", name ? ("Could not load " + name) : "Could not load image");
        });
        ns.viewer = viewer;
    };
})();
