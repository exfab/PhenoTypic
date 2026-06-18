// Node-preview OSD glue. Exposes window.__phenotypicNodePreview.
(function () {
    "use strict";
    const ns = window.__phenotypicNodePreview =
        window.__phenotypicNodePreview || {};

    const appPrefix = (typeof window.__phenotypicAppPrefix === "string"
        && window.__phenotypicAppPrefix.length > 0)
        ? window.__phenotypicAppPrefix : "/";

    function siblingPrefix(prefix, mountName) {
        let base = prefix.endsWith("/") ? prefix : prefix + "/";
        if (base.endsWith("/builder/")) {
            base = base.slice(0, -"builder/".length);
        }
        return base + mountName + "/";
    }
    const resultsPrefix = siblingPrefix(appPrefix, "results");

    let viewer = null;

    function loadOSD(cb) {
        if (window.OpenSeadragon) { cb(); return; }
        const cdn = document.createElement("script");
        cdn.src = "https://cdn.jsdelivr.net/npm/openseadragon@5/build/openseadragon/openseadragon.min.js";
        cdn.onload = cb;
        cdn.onerror = function () {
            const v = document.createElement("script");
            v.src = resultsPrefix + "assets/openseadragon/openseadragon.min.js";
            v.onload = cb;
            document.head.appendChild(v);
        };
        document.head.appendChild(cdn);
    }

    ns.mountViewer = function (divId, dziUrl) {
        loadOSD(function () {
            const el = document.getElementById(divId);
            if (!el || !dziUrl) { return; }
            if (viewer && viewer._phenotypicDziUrl === dziUrl) { return; }
            if (viewer) { viewer.destroy(); viewer = null; }
            viewer = window.OpenSeadragon({
                element: el,
                prefixUrl: resultsPrefix + "assets/openseadragon/images/",
                tileSources: dziUrl,
                showNavigator: false,
                immediateRender: false,
            });
            viewer._phenotypicDziUrl = dziUrl;
        });
    };

    ns.disposeViewer = function () {
        if (viewer) { viewer.destroy(); viewer = null; }
    };
})();
