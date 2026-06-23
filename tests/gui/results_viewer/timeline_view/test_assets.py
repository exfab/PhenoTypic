"""TAB_TIMELINE_ID presence + the vendored timeline assets stay in sync with Browse."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.results_viewer import _ids

# NB: resolve the asset files via the OWNING PACKAGE's __file__ + "/_assets/…".
# Do NOT `import phenotypic.gui.browse._assets` — asset directories carry no
# __init__.py, so importing one raises ModuleNotFoundError (S1).


def _browse_asset(name: str) -> Path:
    import phenotypic.gui.browse as browse

    return Path(browse.__file__).parent / "_assets" / name


def _viewer_asset(name: str) -> Path:
    import phenotypic.gui.results_viewer as rv

    return Path(rv.__file__).parent / "_assets" / name


def test_tab_timeline_id_present_and_unique() -> None:
    assert isinstance(_ids.TAB_TIMELINE_ID, str) and _ids.TAB_TIMELINE_ID
    tab_ids = {
        _ids.TAB_PLATE_ID,
        _ids.TAB_COLONY_ID,
        _ids.TAB_QC_ID,
        _ids.TAB_HEATMAP_ID,
        _ids.TAB_ERROR_ID,
        _ids.TAB_TIMELINE_ID,
    }
    assert len(tab_ids) == 6  # all six tab ids distinct
    assert "TAB_TIMELINE_ID" in _ids.__all__


def test_viewer_timeline_js_is_byte_identical_to_browse() -> None:
    # The viewer vendors its own copy (separate assets_folder); the CI guard
    # keeps it from drifting from the Browse-authored controller (Decision #1).
    assert _viewer_asset("timeline.js").read_bytes() == _browse_asset("timeline.js").read_bytes()


def test_viewer_timeline_css_is_byte_identical_to_browse() -> None:
    # Addition A: the Phase 2 fix pass authored browse/_assets/timeline.css
    # (focus highlight, hover-gated ⤢, badge — all surface-agnostic
    # .timeline-* selectors over var(--color-*) tokens). The viewer vendors
    # its own byte-identical copy alongside the controller.
    assert _viewer_asset("timeline.css").read_bytes() == _browse_asset("timeline.css").read_bytes()
