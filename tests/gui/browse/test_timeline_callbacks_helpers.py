"""Pure helpers behind the Browse Timeline callbacks."""
from __future__ import annotations

from phenotypic.gui.browse._callbacks import (
    pattern_preview_rows,
    render_timeline_grid,
    strip_popout_nonce,
    timeline_thumb_url,
    warnings_alert_state,
)


def test_thumb_url_targets_browse_thumb_segment_with_bucket() -> None:
    url = timeline_thumb_url("/browse/", "TOKEN", 128)
    assert url == "/browse/thumb/TOKEN?size=128"


def test_render_timeline_grid_returns_component_for_records() -> None:
    records = [
        {"row_value": "r1", "time_value": "1", "cell_ref": "imgs/a.png"},
        {"row_value": "r1", "time_value": "2", "cell_ref": "imgs/b.png"},
    ]
    component = render_timeline_grid(records, display_size=120, prefix="/browse/")
    # build_timeline_grid returns (component, grid_order); render_* returns the
    # component only, ready to drop into BROWSE_TL_GRID.
    assert component is not None
    assert hasattr(component, "children")


def test_pattern_preview_rows_returns_component() -> None:
    # Live-preview of the plate-identity pattern over the dataset's stems.
    datasets = {"runX": ["plateA_t01.tif", "plateA_t02.tif", "plateB_t01.tif"]}
    component = pattern_preview_rows(datasets, "{plate}_t{time}", advanced=False)
    assert component is not None
    assert hasattr(component, "children")


def test_warnings_alert_state_hidden_when_empty() -> None:
    # No warnings → alert stays closed with no body.
    assert warnings_alert_state(None) == (None, False)
    assert warnings_alert_state([]) == (None, False)


def test_warnings_alert_state_opens_with_warning_text() -> None:
    # A non-empty warning list opens the alert and surfaces each line.
    warning = (
        "CSV axis: stem(s) appear in multiple folders and cannot be "
        "disambiguated per folder: plateA"
    )
    children, is_open = warnings_alert_state([warning])
    assert is_open is True
    # Each warning becomes a stacked line carrying the text.
    rendered = str(children)
    assert "plateA" in rendered
    assert "multiple folders" in rendered


def test_strip_popout_nonce_recovers_token() -> None:
    # timeline.js appends `#<nonce>`; the server strips it before decoding.
    # base64url tokens never contain `#`, so the split is unambiguous.
    token = "c3JjL3QwL3BsYXRlQS5wbmc"  # base64url, no '#'
    assert strip_popout_nonce(token + "#1") == token
    assert strip_popout_nonce(token + "#42") == token
    # A bare token (no nonce) is returned unchanged.
    assert strip_popout_nonce(token) == token
    # Two clicks on the SAME cell differ as raw values but strip to one token.
    assert strip_popout_nonce(token + "#1") == strip_popout_nonce(token + "#2")
