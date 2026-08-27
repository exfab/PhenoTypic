/*
 * browse.js — Browse interaction and OpenSeadragon lifecycle.
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
    const PREVIEW_ID = "browse-preview-img";
    const LOADING_ID = "browse-osd-loading";
    const LOADING_TEXT_ID = "browse-loading-text";
    const SINGLE_BODY_ID = "browse-single-body";
    const KEEP_POSITION_ID = "browse-keep-position";
    const FILMSTRIP_ID = "browse-filmstrip";
    const POSITION_ID = "browse-position";
    const NAV_EVENT_ID = "browse-nav-event-store";
    const PREPARATION_STATUS_ID = "browse-preparation-status";
    const VIEW_MODE_TOGGLE_ID = "browse-view-mode-toggle";
    const PREPARATION_PROGRESS_ID = "browse-preparation-progress";
    const PREPARE_BUTTON_ID = "browse-prepare-btn";
    const STOP_BUTTON_ID = "browse-stop-prepare-btn";
    const CLEAR_BUTTON_ID = "browse-clear-cache-btn";
    const CACHE_USAGE_ID = "browse-cache-usage";
    const BACKEND_DETAILS_ID = "browse-backend-details";
    const FILMSTRIP_RADIUS = 4;
    const KEY_REPEAT_INTERVAL_MS = 80;

    const ns = window.__phenotypicBrowse = window.__phenotypicBrowse || {};
    ns.singleViewer = ns.singleViewer || null;
    ns.singleGeneration = ns.singleGeneration || 0;
    ns.navigationSequence = ns.navigationSequence || 0;
    ns.lastRepeatAt = ns.lastRepeatAt || 0;
    ns.singleState = ns.singleState || { dimensions: null };
    ns.singleStartedAt = ns.singleStartedAt || 0;
    ns.filmstripPollSequence = ns.filmstripPollSequence || 0;

    function basename(p) {
        if (!p) { return ""; }
        const parts = String(p).split("/");
        return parts[parts.length - 1] || String(p);
    }

    function clientId() {
        const key = "phenotypic.browse.client-id.v1";
        try {
            let value = window.sessionStorage.getItem(key);
            if (!value) {
                value = (window.crypto && window.crypto.randomUUID)
                    ? window.crypto.randomUUID()
                    : (Date.now().toString(36) + "-" + Math.random().toString(36).slice(2));
                window.sessionStorage.setItem(key, value);
            }
            return value;
        } catch (e) {
            return "session-unavailable";
        }
    }

    function publishNavigation(event) {
        const dc = window.dash_clientside;
        if (!dc || typeof dc.set_props !== "function") { return false; }
        ns.navigationSequence += 1;
        dc.set_props(NAV_EVENT_ID, {
            data: Object.assign({}, event, {
                sequence: ns.navigationSequence,
                session_id: clientId(),
            }),
        });
        return true;
    }

    function publishSessionState(forceEnabled) {
        const dc = window.dash_clientside;
        if (!dc || typeof dc.set_props !== "function") { return false; }
        const checkedMode = document.querySelector(
            "#" + VIEW_MODE_TOGGLE_ID + " input:checked"
        );
        const singleMode = !checkedMode || checkedMode.value === "single";
        const enabled = (typeof forceEnabled === "boolean")
            ? forceEnabled
            : (!document.hidden && navigator.onLine !== false && singleMode);
        dc.set_props(NAV_EVENT_ID, {
            data: {
                kind: "session",
                sequence: ns.navigationSequence,
                session_id: clientId(),
                speculation_enabled: enabled,
            },
        });
        return true;
    }

    function editingTarget(target) {
        return Boolean(target && target.closest && target.closest(
            "input, textarea, select, [contenteditable='true'], "
            + "[role='textbox'], [role='combobox'], .Select, .dash-dropdown"
        ));
    }

    function visibleModal() {
        return Array.from(document.querySelectorAll(
            ".modal.show, [role='dialog'][aria-modal='true']"
        )).some(function (modal) {
            return modal.getAttribute("aria-hidden") !== "true"
                && window.getComputedStyle(modal).display !== "none";
        });
    }

    function singleViewVisible() {
        const body = document.getElementById(SINGLE_BODY_ID);
        return Boolean(body && window.getComputedStyle(body).display !== "none");
    }

    if (!ns.sessionStateInstalled) {
        ns.sessionStateInstalled = true;
        document.addEventListener("visibilitychange", function () {
            publishSessionState();
        });
        window.addEventListener("online", function () { publishSessionState(); });
        window.addEventListener("offline", function () { publishSessionState(false); });
        document.addEventListener("change", function (event) {
            const target = event.target;
            if (target && target.closest && target.closest("#" + VIEW_MODE_TOGGLE_ID)) {
                window.setTimeout(function () { publishSessionState(); }, 0);
            }
        });
        window.setTimeout(function () { publishSessionState(); }, 0);
    }

    if (!ns.keyboardInstalled) {
        document.addEventListener("keydown", function (ev) {
            const key = String(ev.key || "").toLowerCase();
            if ((key !== "j" && key !== "k") || ev.defaultPrevented
                || ev.ctrlKey || ev.metaKey || ev.altKey || editingTarget(ev.target)
                || visibleModal() || !singleViewVisible()) {
                return;
            }
            const now = window.performance ? window.performance.now() : Date.now();
            if (ev.repeat && now - ns.lastRepeatAt < KEY_REPEAT_INTERVAL_MS) { return; }
            ns.lastRepeatAt = now;
            const magnitude = ev.shiftKey ? 10 : 1;
            const delta = key === "j" ? -magnitude : magnitude;
            if (publishNavigation({ kind: "offset", delta: delta, source: "keyboard" })) {
                ev.preventDefault();
            }
        }, true);
        ns.keyboardInstalled = true;
    }

    function keepPositionEnabled() {
        const control = document.getElementById(KEEP_POSITION_ID);
        return Boolean(control && control.checked);
    }

    function payloadDimensions(payload) {
        if (!payload) { return null; }
        const dimensions = payload.dimensions;
        const width = Number(payload.width || (Array.isArray(dimensions) && dimensions[0])
            || (dimensions && dimensions.width));
        const height = Number(payload.height || (Array.isArray(dimensions) && dimensions[1])
            || (dimensions && dimensions.height));
        return width > 0 && height > 0 ? { width: width, height: height } : null;
    }

    function equalDimensions(left, right) {
        return Boolean(left && right && left.width === right.width && left.height === right.height);
    }

    function captureViewport(viewer) {
        if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() < 1) {
            return null;
        }
        return {
            center: viewer.viewport.getCenter(),
            zoom: viewer.viewport.getZoom(),
        };
    }

    function scopedAssetUrl(rawUrl, generation) {
        const parsed = new URL(rawUrl, window.location.href);
        parsed.searchParams.set("client_id", clientId());
        parsed.searchParams.set("generation", String(generation));
        if (/^https?:\/\//i.test(rawUrl)) { return parsed.toString(); }
        return parsed.pathname + parsed.search + parsed.hash;
    }

    function dziUrl(payload, generation) {
        const rawUrl = payload.dzi_url
            || (appPrefix + "tiles/" + encodeURIComponent(payload.token) + ".dzi");
        return scopedAssetUrl(rawUrl, generation);
    }

    function previewUrl(payload, generation) {
        const rawUrl = payload && (payload.preview_url || payload.previewUrl) || "";
        return rawUrl ? scopedAssetUrl(rawUrl, generation) : "";
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

    function showPreview(payload, generation) {
        const preview = document.getElementById(PREVIEW_ID);
        const stage = preview && preview.closest(".browse-osd-stage");
        if (!preview || !stage) { return; }
        const url = previewUrl(payload, generation);
        preview.dataset.generation = String(generation);
        preview.classList.remove("is-visible");
        stage.classList.remove("has-preview");
        if (!url) {
            preview.removeAttribute("src");
            return;
        }
        preview.onload = function () {
            if (preview.dataset.generation !== String(generation)) { return; }
            stage.classList.add("has-preview");
            preview.classList.add("is-visible");
            if (ns.singleStartedAt && window.console && console.debug) {
                console.debug("Browse local timing", {
                    milestone: "preview",
                    elapsed_ms: (window.performance ? performance.now() : Date.now())
                        - ns.singleStartedAt,
                    revision: String(payload.revision || "").slice(0, 12),
                });
            }
        };
        preview.onerror = function () {
            if (preview.dataset.generation !== String(generation)) { return; }
            stage.classList.remove("has-preview");
            preview.classList.remove("is-visible");
        };
        preview.src = url;
    }

    function hidePreview(generation) {
        const preview = document.getElementById(PREVIEW_ID);
        const stage = preview && preview.closest(".browse-osd-stage");
        if (!preview || preview.dataset.generation !== String(generation)) { return; }
        preview.classList.remove("is-visible");
        if (!stage) { return; }
        const finishFade = function () {
            if (preview.dataset.generation === String(generation)
                && !preview.classList.contains("is-visible")) {
                stage.classList.remove("has-preview");
            }
        };
        if (window.matchMedia
            && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            finishFade();
            return;
        }
        preview.addEventListener("transitionend", finishFade, { once: true });
        window.setTimeout(finishFade, 300);
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

    function destroyViewer(kind) {
        const property = kind === "single" ? "singleViewer" : "popoutViewer";
        const viewer = ns[property];
        if (viewer) { try { viewer.destroy(); } catch (e) {} }
        ns[property] = null;
    }

    function ensureViewer(kind, divId) {
        const property = kind === "single" ? "singleViewer" : "popoutViewer";
        const el = document.getElementById(divId);
        if (!el) { return null; }
        let viewer = ns[property];
        if (viewer && viewer.element === el && el.isConnected) { return viewer; }
        destroyViewer(kind);
        viewer = window.OpenSeadragon({
            element: el,
            prefixUrl: appPrefix + "assets/openseadragon/images/",
            showNavigator: false,
            showRotationControl: false,
            animationTime: 0.4,
            constrainDuringPan: true,
            visibilityRatio: 0.5,
            minZoomLevel: 0.4,
            maxZoomPixelRatio: 4,
            immediateRender: false,
        });
        ns[property] = viewer;
        return viewer;
    }

    async function mountOSD(kind, divId, payload) {
        await ns.osdReady;
        const el = document.getElementById(divId);
        if (!el) { return; }
        if (!payload || !payload.token) {
            destroyViewer(kind);
            if (kind === "single") { setLoading("hidden"); }
            return;
        }
        const generationProperty = kind === "single" ? "singleGeneration" : "popoutGeneration";
        ns[generationProperty] += 1;
        const generation = ns[generationProperty];
        const url = dziUrl(payload, generation);
        const viewer = ensureViewer(kind, divId);
        if (!viewer) { return; }
        const name = basename(payload.label || payload.token);
        const dimensions = payloadDimensions(payload);
        let restore = null;
        if (kind === "single") {
            ns.singleStartedAt = window.performance ? performance.now() : Date.now();
            if (keepPositionEnabled() && equalDimensions(ns.singleState.dimensions, dimensions)) {
                restore = captureViewport(viewer);
            }
            ns.singleState.dimensions = dimensions;
            showPreview(payload, generation);
            setLoading("loading", name ? ("Loading " + name + "…") : "Loading image…");
        }
        if (viewer._phenotypicOpenHandler) {
            viewer.removeHandler("open", viewer._phenotypicOpenHandler);
        }
        if (viewer._phenotypicFailureHandler) {
            viewer.removeHandler("open-failed", viewer._phenotypicFailureHandler);
        }
        viewer._phenotypicOpenHandler = function () {
            if (generation !== ns[generationProperty] || viewer._phenotypicDziUrl !== url) {
                return;
            }
            if (restore) {
                viewer.viewport.panTo(restore.center, true);
                viewer.viewport.zoomTo(restore.zoom, restore.center, true);
                viewer.viewport.applyConstraints();
            } else {
                viewer.viewport.goHome(true);
            }
            if (kind === "single") {
                setLoading("hidden");
                hidePreview(generation);
                if (ns.singleStartedAt && window.console && console.debug) {
                    console.debug("Browse local timing", {
                        milestone: "osd-open",
                        elapsed_ms: (window.performance ? performance.now() : Date.now())
                            - ns.singleStartedAt,
                        revision: String(payload.revision || "").slice(0, 12),
                    });
                }
            }
        };
        viewer._phenotypicFailureHandler = function () {
            if (generation !== ns[generationProperty] || viewer._phenotypicDziUrl !== url) {
                return;
            }
            if (kind === "single") {
                setLoading("error", name ? ("Could not load " + name) : "Could not load image");
            }
        };
        viewer.addHandler("open", viewer._phenotypicOpenHandler);
        viewer.addHandler("open-failed", viewer._phenotypicFailureHandler);
        if (viewer._phenotypicDziUrl !== url) {
            viewer._phenotypicDziUrl = url;
            viewer.open(url);
        } else if (viewer.world && viewer.world.getItemCount() > 0) {
            viewer._phenotypicOpenHandler();
        }
    }

    ns.applyImage = function (payload) {
        renderPosition(payload && payload.position);
        renderFilmstrip(payload && payload.filmstrip, payload);
        return mountOSD("single", OSD_DIV_ID, payload);
    };

    function renderPosition(position) {
        const element = document.getElementById(POSITION_ID);
        if (!element || !position) { return; }
        const index = Number(position.index || position.position || 0);
        const total = Number(position.total || 0);
        element.textContent = String(index) + " of " + String(total);
    }

    function boundedFilmstrip(items, activeValue) {
        if (!Array.isArray(items)) { return []; }
        const active = items.findIndex(function (item) {
            return item && (item.current || item.selected || item.value === activeValue);
        });
        if (items.length <= FILMSTRIP_RADIUS * 2 + 1) { return items; }
        const centre = active >= 0 ? active : 0;
        const start = Math.max(0, Math.min(
            items.length - (FILMSTRIP_RADIUS * 2 + 1), centre - FILMSTRIP_RADIUS
        ));
        return items.slice(start, start + FILMSTRIP_RADIUS * 2 + 1);
    }

    function renderFilmstrip(items, payload) {
        const host = document.getElementById(FILMSTRIP_ID);
        if (!host) { return; }
        host.replaceChildren();
        const activeValue = payload && (payload.value || payload.filename);
        boundedFilmstrip(items, activeValue).forEach(function (item) {
            if (!item || !item.value) { return; }
            const wrapper = document.createElement("div");
            wrapper.setAttribute("role", "listitem");
            const button = document.createElement("button");
            const label = basename(item.label || item.value);
            const status = ["ready", "preparing", "queued", "failed"].includes(item.status)
                ? item.status : "queued";
            const current = Boolean(item.current || item.selected || item.value === activeValue);
            button.type = "button";
            button.className = "browse-filmstrip-item";
            button.title = label;
            button.setAttribute("aria-label", (current ? "Current image, " : "Open ") + label);
            button.setAttribute("aria-current", current ? "true" : "false");
            if (item.preview_url) {
                const image = document.createElement("img");
                image.className = "browse-filmstrip-thumb";
                image.alt = "";
                image.loading = "lazy";
                image.decoding = "async";
                image.dataset.previewUrl = item.preview_url;
                image.dataset.loaded = "false";
                image.onload = function () {
                    image.dataset.loaded = "true";
                    image.classList.remove("is-unavailable");
                    state.textContent = "Ready";
                    state.className = "browse-filmstrip-state browse-filmstrip-state--ready";
                };
                image.onerror = function () {
                    image.dataset.loaded = "false";
                    image.classList.add("is-unavailable");
                };
                image.src = item.preview_url;
                button.appendChild(image);
            } else {
                const placeholder = document.createElement("span");
                placeholder.className = "browse-filmstrip-placeholder";
                placeholder.setAttribute("aria-hidden", "true");
                placeholder.textContent = "···";
                button.appendChild(placeholder);
            }
            const name = document.createElement("span");
            name.className = "browse-filmstrip-name";
            name.textContent = label;
            button.appendChild(name);
            const state = document.createElement("span");
            state.className = "browse-filmstrip-state browse-filmstrip-state--" + status;
            state.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            button.appendChild(state);
            button.addEventListener("click", function () {
                publishNavigation({ kind: "select", value: item.value, source: "filmstrip" });
            });
            wrapper.appendChild(button);
            host.appendChild(wrapper);
        });
    }

    ns.applyPreparationStatus = function (payload) {
        payload = payload || {};
        const state = payload.state || "idle";
        const running = state === "running" || state === "stopping";
        const total = Math.max(0, Number(payload.total || 0));
        const ready = Math.max(0, Number(payload.ready || 0));
        const failed = Math.max(0, Number(payload.failed || 0));
        const status = document.getElementById(PREPARATION_STATUS_ID);
        const progress = document.getElementById(PREPARATION_PROGRESS_ID);
        const prepare = document.getElementById(PREPARE_BUTTON_ID);
        const stop = document.getElementById(STOP_BUTTON_ID);
        const clear = document.getElementById(CLEAR_BUTTON_ID);
        const cache = document.getElementById(CACHE_USAGE_ID);
        const backend = document.getElementById(BACKEND_DETAILS_ID);
        if (status) {
            status.textContent = payload.message || (running
                ? ("Prepared " + ready + " of " + total + (failed ? "; " + failed + " failed" : ""))
                : "Images prepare as you browse.");
        }
        if (progress) {
            progress.max = Math.max(1, total);
            progress.value = Math.min(progress.max, ready + failed);
        }
        if (prepare) { prepare.disabled = running; }
        if (stop) { stop.disabled = !running || state === "stopping"; }
        if (clear) { clear.disabled = running || Boolean(payload.clearing); }
        if (cache) { cache.textContent = payload.cache_usage || "—"; }
        if (backend) { backend.textContent = payload.backend || "—"; }
        ns.filmstripPollSequence += 1;
        if (ns.filmstripPollSequence % 2 === 0) {
            document.querySelectorAll(
                ".browse-filmstrip-thumb[data-loaded='false'][data-preview-url]"
            ).forEach(function (image) {
                const url = new URL(image.dataset.previewUrl, window.location.href);
                url.searchParams.set("filmstrip_probe", String(ns.filmstripPollSequence));
                image.src = url.pathname + url.search;
            });
        }
        return "";
    };

})();
