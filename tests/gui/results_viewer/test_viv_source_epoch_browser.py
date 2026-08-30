"""Executable race contracts for the shared Viv facade's source lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest


_FACADE = (
    Path(__file__).parents[3]
    / "src/phenotypic/gui/results_viewer/_assets/viv_viewer.js"
)


@pytest.fixture
def facade_page(page):
    page.set_content('<div id="viv-probe"></div>')
    page.evaluate(
        """() => {
            window.__vivLoads = new Map();
            window.__vivPaints = [];
            window.__vivFinalizedUses = 0;

            class Layer {
                constructor(props) { this.id = props.id; this.props = props; }
                clone(extra) { return new Layer({...this.props, ...extra}); }
            }
            function sourceFor(url) {
                return {
                    sourceId: url,
                    labels: ["y", "x"],
                    shape: [8, 8],
                    dtype: "Uint8",
                    tileSize: 4,
                    getTile: () => null,
                };
            }
            window.__vivBundle = {
                zarr: {registry: {set: () => {}}},
                numcodecs: {Zstd: function () {}},
                viv: {
                    loadOmeZarrFromStore: (store) => new Promise((resolve) => {
                        window.__vivLoads.set(store.url, () => resolve({
                            data: [sourceFor(store.url)],
                        }));
                    }),
                },
                layers: {
                    ImageLayer: Layer,
                    MultiscaleImageLayer: Layer,
                },
                extensions: {AdditiveColormapExtension: class {}},
                createViewer: () => {
                    let finalized = false;
                    return {
                        deck: {setProps: () => {}},
                        setLayers: (layers) => {
                            if (finalized) window.__vivFinalizedUses += 1;
                            const image = layers.find((layer) => layer.id === "image");
                            window.__vivPaints.push(image.props.loader.sourceId);
                        },
                        setViewState: () => {},
                        setLayerVisibility: () => {},
                        finalize: () => { finalized = true; },
                    };
                },
            };
        }"""
    )
    page.add_script_tag(content=_FACADE.read_text(encoding="utf-8"))
    page.evaluate("() => window.phenotypicViv.mount('viv-probe', {})")
    return page


def _launch_source(page, name: str, *, label: bool) -> None:
    spec = {
        "storeUrl": f"http://store/{name}",
        "seriesPath": "rgb",
        "labelPath": "labels" if label else None,
    }
    page.evaluate(
        """([name, spec]) => {
            window[name] = window.phenotypicViv.setSource("viv-probe", spec)
                .then((value) => { window[name + "Done"] = value; });
        }""",
        [name, spec],
    )


def _resolve(page, url: str) -> None:
    page.evaluate("url => window.__vivLoads.get(url)()", url)


def test_new_source_claim_prevents_older_deferred_success_from_painting(
    facade_page,
):
    page = facade_page
    _launch_source(page, "sourceA", label=True)
    page.wait_for_function("window.__vivLoads.has('http://store/sourceA/rgb')")
    _resolve(page, "http://store/sourceA/rgb")
    page.wait_for_function("window.__vivLoads.has('http://store/sourceA/labels')")

    _launch_source(page, "sourceB", label=True)
    page.wait_for_function("window.__vivLoads.has('http://store/sourceB/rgb')")
    _resolve(page, "http://store/sourceA/labels")
    page.wait_for_function("'sourceADone' in window")

    assert page.evaluate("window.__vivPaints") == []

    _resolve(page, "http://store/sourceB/rgb")
    page.wait_for_function("window.__vivLoads.has('http://store/sourceB/labels')")
    _resolve(page, "http://store/sourceB/labels")
    page.wait_for_function("'sourceBDone' in window")

    assert page.evaluate("window.__vivPaints") == [
        "http://store/sourceB/rgb"
    ]


def test_destroyed_instance_cannot_commit_a_deferred_source(facade_page):
    page = facade_page
    _launch_source(page, "destroyed", label=False)
    page.wait_for_function("window.__vivLoads.has('http://store/destroyed/rgb')")

    page.evaluate("window.phenotypicViv.destroy('viv-probe')")
    _resolve(page, "http://store/destroyed/rgb")
    page.wait_for_function("'destroyedDone' in window")

    assert page.evaluate("window.__vivPaints") == []
    assert page.evaluate("window.__vivFinalizedUses") == 0
