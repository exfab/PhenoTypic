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
    // Single shared OSD handle: the single-pane viewer (applyImage) and the
    // Timeline deep-zoom pop-out (applyPopoutImage) both mount through _mountOSD
    // and reuse ns.viewer. They are mutually exclusive — the view-mode toggle
    // shows only one body at a time — so one handle is sufficient.
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

    // Mount (or replace) an OSD viewer for `payload` into the div `divId`,
    // reusing the same /tiles/<token>.dzi source + loading/open handlers.
    // Pure extraction of the original applyImage body, parameterized only by
    // the target div id, so the single-viewport Browse pane (OSD_DIV_ID) and
    // the Timeline pop-out (browse-tl-popout-osd) share one lifecycle path.
    async function _mountOSD(divId, payload) {
        await ns.osdReady;
        const el = document.getElementById(divId);
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
    }

    ns.applyImage = function (payload) {
        return _mountOSD(OSD_DIV_ID, payload);
    };

    // Timeline deep-zoom pop-out: mount the same DZI viewer into the modal's
    // dedicated OSD div. Reuses _mountOSD so the loading/open lifecycle stays
    // identical to the single Browse viewer.
    ns.applyPopoutImage = function (payload) {
        return _mountOSD("browse-tl-popout-osd", payload);
    };

    // Browse-only Timeline event adapter. The shared timeline controller is
    // also used by Results, but Browse's source can change in place. Delegate
    // on document so a Dash remount cannot retire the listener, and publish
    // through Dash's supported set_props API rather than mutating a
    // React-controlled hidden input.
    const TL_GRID_ID = "browse-tl-grid";
    const TL_POPOUT_EVENT_ID = "browse-tl-popout-event";
    let timelineEventSequence = 0;

    function browseGrid() {
        return document.getElementById(TL_GRID_ID);
    }

    function publishPopout(cell) {
        const grid = browseGrid();
        const token = cell && cell.getAttribute("data-ref");
        const revision = grid && grid.getAttribute("data-grid-revision");
        const dc = window.dash_clientside;
        if (!grid || !grid.contains(cell) || !token || !revision
            || !dc || typeof dc.set_props !== "function") {
            return false;
        }
        timelineEventSequence += 1;
        dc.set_props(TL_POPOUT_EVENT_ID, {
            data: {
                token: token,
                revision: revision,
                sequence: timelineEventSequence,
            },
        });
        return true;
    }

    function decodeBrowseRef(ref) {
        try {
            let encoded = String(ref).replace(/-/g, "+").replace(/_/g, "/");
            encoded += "=".repeat((4 - encoded.length % 4) % 4);
            const binary = window.atob(encoded);
            const bytes = Uint8Array.from(binary, function (ch) {
                return ch.charCodeAt(0);
            });
            if (window.TextDecoder) {
                return new window.TextDecoder("utf-8", { fatal: true }).decode(bytes);
            }
            let escaped = "";
            bytes.forEach(function (byte) {
                escaped += "%" + byte.toString(16).padStart(2, "0");
            });
            return decodeURIComponent(escaped);
        } catch (e) {
            return String(ref);
        }
    }

    function browseTimelineDziUrl(ref) {
        return appPrefix + "tiles/" + encodeURIComponent(String(ref)) + ".dzi";
    }

    function compareCap(grid) {
        const parsed = parseInt(grid.getAttribute("data-compare-cap"), 10);
        return Number.isFinite(parsed) ? parsed : 12;
    }

    function selectedRefs(grid) {
        return Array.from(
            grid.querySelectorAll(".timeline-cell.timeline-cell--selected[data-ref]")
        ).sort(function (left, right) {
            const leftRow = parseInt(left.getAttribute("data-row-index"), 10) || 0;
            const rightRow = parseInt(right.getAttribute("data-row-index"), 10) || 0;
            if (leftRow !== rightRow) { return leftRow - rightRow; }
            return (parseInt(left.getAttribute("data-col-index"), 10) || 0)
                - (parseInt(right.getAttribute("data-col-index"), 10) || 0);
        }).map(function (cell) { return cell.getAttribute("data-ref"); });
    }

    function rowRefs(grid, rowValue) {
        return Array.from(
            grid.querySelectorAll(".timeline-cell[data-src][data-row][data-ref]")
        ).filter(function (cell) {
            return cell.getAttribute("data-row") === rowValue;
        }).map(function (cell) { return cell.getAttribute("data-ref"); });
    }

    function openBrowseCompare(grid, refs) {
        const timeline = window.__phenotypicTimeline;
        if (!timeline || !timeline.openCompareStrip || !refs.length) { return; }
        timeline.openCompareStrip(refs, {
            dziUrlBuilder: browseTimelineDziUrl,
            titleFor: decodeBrowseRef,
            cap: compareCap(grid),
        });
    }

    // Capture phase wins before the shared controller's per-node bubble
    // listeners. That keeps encoded transport refs out of visible titles while
    // leaving the shared Results controller byte-identical and untouched.
    document.addEventListener("click", function (ev) {
        const target = ev.target;
        if (!target || !target.closest) { return; }
        const grid = browseGrid();
        if (!grid) { return; }

        const popout = target.closest(".timeline-cell-popout");
        if (popout) {
            const cell = popout.closest(".timeline-cell[data-ref]");
            if (cell && grid.contains(cell) && publishPopout(cell)) {
                ev.preventDefault();
                ev.stopImmediatePropagation();
            }
            return;
        }

        const compareButton = target.closest("#browse-tl-compare-btn");
        if (compareButton) {
            openBrowseCompare(grid, selectedRefs(grid));
            ev.preventDefault();
            ev.stopImmediatePropagation();
            return;
        }

        const rowHeader = target.closest(".timeline-axis-label--y[data-row]");
        if (rowHeader && grid.contains(rowHeader)) {
            openBrowseCompare(grid, rowRefs(grid, rowHeader.getAttribute("data-row")));
            ev.preventDefault();
            ev.stopImmediatePropagation();
        }
    }, true);

    document.addEventListener("keydown", function (ev) {
        if (ev.key !== "Enter" && ev.key !== " ") { return; }
        const grid = browseGrid();
        const viewport = grid && grid.closest(".browse-tl-viewport");
        if (!grid || !viewport || !viewport.contains(document.activeElement)) { return; }
        const focused = grid.querySelector(".timeline-cell--focused[data-ref]");
        if (focused && publishPopout(focused)) {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        }
    }, true);

    ns.resetTimelineRevision = function (containerId) {
        const grid = document.getElementById(containerId || TL_GRID_ID);
        if (grid) {
            grid.querySelectorAll(".timeline-cell--selected").forEach(function (cell) {
                cell.classList.remove("timeline-cell--selected");
            });
        }
        const timeline = window.__phenotypicTimeline;
        if (timeline && timeline.closeCompareStrip) {
            timeline.closeCompareStrip();
        }
    };
})();
