"""Executable browser contracts for Browse's OSD/Viv renderer switch."""

from __future__ import annotations

from pathlib import Path

import pytest


_CONTROLLER = (
    Path(__file__).parents[3]
    / "src/phenotypic/gui/browse/_assets/browse.js"
)
_FACADE = (
    Path(__file__).parents[3]
    / "src/phenotypic/gui/results_viewer/_assets/viv_viewer.js"
)


@pytest.fixture
def controller_page(page):
    page.set_content(
        """
        <base href="http://localhost/">
        <div class="browse-osd-stage">
          <div id="browse-osd-div"></div>
          <img id="browse-preview-img">
          <div id="browse-osd-loading">
            <span id="browse-loading-text"></span>
          </div>
        </div>
        <div id="browse-filmstrip"></div>
        <div id="browse-position"></div>
        <input id="browse-keep-position" type="checkbox">
        <div id="browse-single-body"></div>
        """
    )
    page.evaluate(
        """() => {
            window.__phenotypicAppPrefix = "/browse/";
            window.__rendererEvents = [];
            window.__vivPending = new Map();

            function Viewer(options) {
                this.element = options.element;
                this.handlers = {};
                this.world = {getItemCount: () => 1};
                this.viewport = {
                    getCenter: () => ({x: 0.5, y: 0.5}),
                    getZoom: () => 1,
                    panTo: () => {}, zoomTo: () => {},
                    applyConstraints: () => {}, goHome: () => {},
                };
                this.addHandler = (name, fn) => { this.handlers[name] = fn; };
                this.removeHandler = (name, fn) => {
                    if (this.handlers[name] === fn) delete this.handlers[name];
                };
                this.open = (url) => {
                    window.__rendererEvents.push(["osd-open", url]);
                    queueMicrotask(() => this.handlers.open && this.handlers.open());
                };
                this.destroy = () => {
                    window.__rendererEvents.push(["osd-destroy"]);
                    this.element.replaceChildren();
                };
            }
            window.OpenSeadragon = (options) => new Viewer(options);

            window.phenotypicViv = {
                ready: async () => {},
                mount: async (id) => {
                    window.__rendererEvents.push(["viv-mount", id]);
                    document.getElementById(id).appendChild(
                        Object.assign(document.createElement("canvas"), {
                            className: "viv-canvas",
                        })
                    );
                },
                setSource: async (id, spec) => {
                    window.__rendererEvents.push([
                        "viv-source", id, spec.storeUrl,
                    ]);
                    if (!spec.storeUrl.includes("deferred")) return {};
                    return new Promise((resolve, reject) => {
                        window.__vivPending.set(spec.storeUrl, {resolve, reject});
                    });
                },
                destroy: (id) => {
                    window.__rendererEvents.push(["viv-destroy", id]);
                    document.getElementById(id).replaceChildren();
                },
            };
        }"""
    )
    page.add_script_tag(content=_CONTROLLER.read_text(encoding="utf-8"))
    return page


@pytest.fixture
def integrated_controller_page(page):
    page.set_content(
        """
        <base href="http://localhost/">
        <div class="browse-osd-stage">
          <div id="browse-osd-div"></div>
          <img id="browse-preview-img">
          <div id="browse-osd-loading"><span id="browse-loading-text"></span></div>
        </div>
        <div id="browse-filmstrip"></div><div id="browse-position"></div>
        <input id="browse-keep-position" type="checkbox">
        <div id="browse-single-body"></div>
        """
    )
    page.evaluate(
        """() => {
            window.__phenotypicAppPrefix = "/browse/";
            window.__rendererEvents = [];
            window.__vivLoads = new Map();
            window.__vivPaints = [];
            window.__vivFinalizedUses = 0;
            function Viewer(options) {
                this.element = options.element;
                this.handlers = {};
                this.world = {getItemCount: () => 1};
                this.viewport = {
                    getCenter: () => ({x: 0.5, y: 0.5}), getZoom: () => 1,
                    panTo: () => {}, zoomTo: () => {}, applyConstraints: () => {},
                    goHome: () => {},
                };
                this.addHandler = (name, fn) => { this.handlers[name] = fn; };
                this.removeHandler = () => {};
                this.open = (url) => {
                    window.__rendererEvents.push(["osd-open", url]);
                    queueMicrotask(() => this.handlers.open && this.handlers.open());
                };
                this.destroy = () => this.element.replaceChildren();
            }
            window.OpenSeadragon = (options) => new Viewer(options);
            class Layer {
                constructor(props) { this.id = props.id; this.props = props; }
                clone(extra) { return new Layer({...this.props, ...extra}); }
            }
            function sourceFor(url) {
                return {sourceId: url, labels: ["y", "x"], shape: [8, 8],
                    dtype: "Uint8", tileSize: 4, getTile: () => null};
            }
            window.__vivBundle = {
                zarr: {registry: {set: () => {}}},
                numcodecs: {Zstd: function () {}},
                viv: {loadOmeZarrFromStore: (store) => new Promise((resolve) => {
                    window.__vivLoads.set(store.url, () => resolve({
                        data: [sourceFor(store.url)],
                    }));
                })},
                layers: {ImageLayer: Layer, MultiscaleImageLayer: Layer},
                extensions: {AdditiveColormapExtension: class {}},
                createViewer: (el) => {
                    let finalized = false;
                    el.appendChild(document.createElement("canvas"));
                    return {
                        deck: {setProps: () => {}},
                        setLayers: (layers) => {
                            if (finalized) window.__vivFinalizedUses += 1;
                            const image = layers.find((layer) => layer.id === "image");
                            window.__vivPaints.push(image.props.loader.sourceId);
                        },
                        setViewState: () => {}, setLayerVisibility: () => {},
                        finalize: () => { finalized = true; el.replaceChildren(); },
                    };
                },
            };
        }"""
    )
    page.add_script_tag(content=_FACADE.read_text(encoding="utf-8"))
    page.add_script_tag(content=_CONTROLLER.read_text(encoding="utf-8"))
    return page


def _store_payload(store_url: str, label: str = "plate.ome.zarr") -> dict:
    return {
        "render_kind": "ome-zarr",
        "token": "opaque-token",
        "label": label,
        "store_url": store_url,
        "source_spec": {
            "storeUrl": store_url,
            "seriesPath": "rgb",
            "labelPath": "rgb/labels/objmap",
            "token": "publication-token",
            "series": ["rgb"],
            "pyramid": {"levels": 2},
        },
    }


def test_switches_renderers_in_teardown_order_and_reuses_viv(controller_page):
    page = controller_page
    first = _store_payload("/browse/assets/a/rev/zarr/", "a.ome.zarr")
    second = _store_payload("/browse/assets/b/rev/zarr/", "b.ome.zarr")

    page.evaluate("payload => window.__phenotypicBrowse.applyImage(payload)", first)
    page.wait_for_function("window.__rendererEvents.length >= 2")
    page.evaluate("payload => window.__phenotypicBrowse.applyImage(payload)", second)
    page.wait_for_function("window.__rendererEvents.length >= 3")

    events = page.evaluate("window.__rendererEvents")
    assert [event[0] for event in events] == [
        "viv-mount",
        "viv-source",
        "viv-source",
    ]
    assert page.locator("#browse-osd-div canvas.viv-canvas").count() == 1

    flat = {
        "render_kind": "dzi",
        "token": "flat-token",
        "label": "flat.tif",
        "dzi_url": "http://localhost/browse/assets/flat/rev/image.dzi",
    }
    page.evaluate("payload => window.__phenotypicBrowse.applyImage(payload)", flat)
    page.wait_for_function(
        "window.__rendererEvents.some(e => e[0] === 'osd-open')"
    )
    events = page.evaluate("window.__rendererEvents")
    names = [event[0] for event in events]
    assert names.index("viv-destroy") < names.index("osd-open")

    page.evaluate("payload => window.__phenotypicBrowse.applyImage(payload)", first)
    page.wait_for_function(
        "window.__rendererEvents.filter(e => e[0] === 'viv-source').length === 3"
    )
    events = page.evaluate("window.__rendererEvents")
    names = [event[0] for event in events]
    assert names.index("osd-destroy") < len(names) - 2
    assert names[-2:] == ["viv-mount", "viv-source"]


def test_stale_viv_failure_cannot_replace_new_selection_loading(controller_page):
    page = controller_page
    old = _store_payload(
        "/browse/assets/deferred-old/rev/zarr/",
        "old.ome.zarr",
    )
    new = _store_payload(
        "/browse/assets/deferred-new/rev/zarr/",
        "new.ome.zarr",
    )

    page.evaluate(
        "payload => { window.__oldApply = "
        "window.__phenotypicBrowse.applyImage(payload); }",
        old,
    )
    page.wait_for_function("window.__vivPending.size === 1")
    page.evaluate(
        "payload => { window.__newApply = "
        "window.__phenotypicBrowse.applyImage(payload); }",
        new,
    )
    page.evaluate(
        "url => window.__vivPending.get(url).reject(new Error('old failed'))",
        old["store_url"],
    )
    page.wait_for_timeout(10)

    overlay = page.locator("#browse-osd-loading")
    assert "is-visible" in (overlay.get_attribute("class") or "")
    assert "browse-loading-overlay--error" not in (
        overlay.get_attribute("class") or ""
    )
    assert page.locator("#browse-loading-text").inner_text() == (
        "Loading new.ome.zarr…"
    )

    page.wait_for_function("window.__vivPending.size === 2")

    page.evaluate(
        "url => window.__vivPending.get(url).resolve({})",
        new["store_url"],
    )
    page.wait_for_function(
        "!document.getElementById('browse-osd-loading').classList.contains('is-visible')"
    )


def test_new_viv_source_starts_before_older_deferred_success_finishes(
    controller_page,
):
    page = controller_page
    old = _store_payload(
        "/browse/assets/deferred-old-success/rev/zarr/",
        "old.ome.zarr",
    )
    new = _store_payload(
        "/browse/assets/deferred-new-success/rev/zarr/",
        "new.ome.zarr",
    )

    page.evaluate(
        "payload => { window.__oldApply = "
        "window.__phenotypicBrowse.applyImage(payload); }",
        old,
    )
    page.wait_for_function("window.__vivPending.size === 1")
    page.evaluate(
        "payload => { window.__newApply = "
        "window.__phenotypicBrowse.applyImage(payload); }",
        new,
    )

    page.wait_for_function("window.__vivPending.size === 2")
    page.evaluate(
        "url => window.__vivPending.get(url).resolve({})",
        old["store_url"],
    )
    page.evaluate("() => window.__oldApply")

    overlay = page.locator("#browse-osd-loading")
    assert "is-visible" in (overlay.get_attribute("class") or "")
    assert page.locator("#browse-loading-text").inner_text() == (
        "Loading new.ome.zarr…"
    )

    page.evaluate(
        "url => window.__vivPending.get(url).resolve({})",
        new["store_url"],
    )
    page.evaluate("() => window.__newApply")
    assert "is-visible" not in (overlay.get_attribute("class") or "")


def test_pending_viv_source_cannot_paint_after_browse_switches_to_osd(
    integrated_controller_page,
):
    page = integrated_controller_page
    old = _store_payload(
        "http://localhost/browse/assets/pending/rev/zarr/",
        "pending.ome.zarr",
    )
    page.evaluate(
        "payload => { window.__oldApply = window.__phenotypicBrowse.applyImage(payload); }",
        old,
    )
    image_url = old["store_url"].rstrip("/") + "/rgb"
    page.wait_for_function(
        "url => window.__vivLoads.has(url)", arg=image_url
    )

    flat = {
        "render_kind": "dzi",
        "token": "flat-token",
        "label": "flat.tif",
        "dzi_url": "http://localhost/browse/assets/flat/rev/image.dzi",
    }
    page.evaluate(
        "payload => window.__phenotypicBrowse.applyImage(payload)", flat
    )
    page.wait_for_function(
        "window.__rendererEvents.some(e => e[0] === 'osd-open')"
    )
    page.evaluate("url => window.__vivLoads.get(url)()", image_url)
    page.evaluate("() => window.__oldApply")

    assert page.evaluate("window.__vivPaints") == []
    assert page.evaluate("window.__vivFinalizedUses") == 0
    assert page.evaluate(
        "window.__rendererEvents.filter(e => e[0] === 'osd-open').length"
    ) == 1
