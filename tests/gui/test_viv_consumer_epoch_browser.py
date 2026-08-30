"""Executable cancellation contracts for Results and Builder Viv consumers."""

from __future__ import annotations

from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[2]
_RESULTS = _ROOT / "src/phenotypic/gui/results_viewer/_assets/results_viewer.js"
_BUILDER = _ROOT / "src/phenotypic/gui/builder/assets/preview.js"


@pytest.fixture
def consumer_page(page):
    page.set_content(
        '<div id="stage"></div><div id="level"></div><div id="zoom"></div>'
    )
    page.evaluate(
        """() => {
            window.__sources = [];
            window.__sourcePending = [];
            window.__displayCalls = [];
            window.phenotypicViv = {
                mount: async () => ({}),
                setSource: (id, spec) => new Promise((resolve) => {
                    window.__sources.push(spec.storeUrl);
                    window.__sourcePending.push({id, spec, resolve});
                }),
                setLayerOpacity: (id, layer, value) => {
                    window.__displayCalls.push(["opacity", id, layer, value]);
                },
                setLayerVisibility: async (id, layer, value) => {
                    window.__displayCalls.push(["visible", id, layer, value]);
                },
                destroy: () => {},
            };
        }"""
    )
    return page


def _spec(name: str, *, image_visible: bool = True) -> dict:
    return {
        "storeUrl": f"http://store/{name}",
        "seriesPath": "rgb",
        "labelPath": "labels",
        "imageVisible": image_visible,
    }


def test_results_does_not_commit_epoch_cancelled_source(consumer_page):
    page = consumer_page
    page.add_script_tag(content=_RESULTS.read_text(encoding="utf-8"))
    old = {
        "id": "stage",
        "levelReadoutId": "level",
        "zoomReadoutId": "zoom",
        "spec": _spec("A"),
        "display": {"opacity": {"image": 0.2}, "labelVisible": False},
    }
    new = {
        **old,
        "spec": _spec("B"),
        "display": {"opacity": {"image": 0.8}, "labelVisible": True},
    }
    page.evaluate(
        "record => { window.__old = window.__phenotypicResultsViewer.mountStage(record); }",
        old,
    )
    page.wait_for_function("window.__sourcePending.length === 1")
    page.evaluate(
        "record => { window.__new = window.__phenotypicResultsViewer.mountStage(record); }",
        new,
    )
    page.wait_for_function("window.__sourcePending.length === 2")
    page.evaluate("() => window.__sourcePending[0].resolve(undefined)")
    page.evaluate("() => window.__old")

    assert page.evaluate("window.__displayCalls") == []
    assert page.evaluate(
        "window.__phenotypicResultsViewer.stages.get('stage').signature"
    ) is None

    page.evaluate("() => window.__sourcePending[1].resolve({})")
    page.evaluate("() => window.__new")
    assert page.evaluate("window.__displayCalls") == [
        ["opacity", "stage", "image", 0.8]
    ]


def test_builder_does_not_apply_state_for_epoch_cancelled_source(consumer_page):
    page = consumer_page
    page.add_script_tag(content=_BUILDER.read_text(encoding="utf-8"))
    page.evaluate(
        "spec => { window.__old = window.__phenotypicNodePreview.mountViewer('stage', spec); }",
        _spec("A", image_visible=False),
    )
    page.wait_for_function("window.__sourcePending.length === 1")
    page.evaluate(
        "spec => { window.__new = window.__phenotypicNodePreview.mountViewer('stage', spec); }",
        _spec("B", image_visible=True),
    )
    page.wait_for_function("window.__sourcePending.length === 2")
    page.evaluate("() => window.__sourcePending[0].resolve(undefined)")
    page.evaluate("() => window.__old")
    assert page.evaluate("window.__displayCalls") == []

    page.evaluate("() => window.__sourcePending[1].resolve({})")
    page.evaluate("() => window.__new")
    assert page.evaluate("window.__displayCalls") == [
        ["visible", "stage", "image", True],
        ["visible", "stage", "labels", True],
    ]
