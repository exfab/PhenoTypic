"""Sanity-check that the picker JS asset is shipped + the namespace is declared."""

from __future__ import annotations

from pathlib import Path


def test_point_picker_js_exists():
    js = Path("src/phenotypic/gui/builder/assets/point_picker.js")
    assert js.exists()
    text = js.read_text()
    # Stub markers must be gone.
    assert "STUB" not in text
    # Required namespace + methods.
    assert "__phenotypicBuilderPointPicker" in text
    for method in ("mountViewer", "redrawOverlay", "disposeViewer"):
        assert method in text, f"JS missing method {method}"
    # OpenSeadragon bootstrap is present.
    assert "OpenSeadragon" in text or "openseadragon" in text
    # Click handler wires up the staged store via set_props.
    assert "set_props" in text
    assert "picker-staged-store" in text
