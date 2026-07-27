"""Pure helpers behind the Browse Timeline callbacks."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.browse._callbacks import (
    SourceRevisionAuthority,
    TimelineRevisionAuthority,
    authorize_revision_candidate,
    pattern_preview_rows,
    render_timeline_grid,
    resolve_popout_event,
    source_reset_values,
    strip_popout_nonce,
    timeline_revision_token,
    timeline_thumb_url,
    warnings_alert_state,
)
from phenotypic.gui.browse._source_render import encode_token
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import source_payload_from_path


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


def test_timeline_revision_changes_with_source_metadata_or_grid_inputs() -> None:
    base = timeline_revision_token(
        {"relative_path": "source-a"},
        {"relative_path": "metadata.csv"},
        "folder",
    )
    assert base == timeline_revision_token(
        {"relative_path": "source-a"},
        {"relative_path": "metadata.csv"},
        "folder",
    )
    assert base != timeline_revision_token(
        {"relative_path": "source-b"},
        {"relative_path": "metadata.csv"},
        "folder",
    )
    assert base != timeline_revision_token(
        {"relative_path": "source-a"},
        {"relative_path": "other.csv"},
        "folder",
    )
    assert base != timeline_revision_token(
        {"relative_path": "source-a"},
        {"relative_path": "metadata.csv"},
        "pattern",
    )


def test_source_reset_transaction_clears_all_timeline_dependent_state() -> None:
    reset = source_reset_values(
        {"relative_path": "source-b", "selected_at": "revision-2"}
    )

    assert reset[0:8] == (
        "folder",
        "exif",
        None,
        None,
        None,
        "",
        [],
        reset[7],
    )
    assert "Enter a pattern to preview matches." in str(reset[7])
    assert reset[8].endswith(" px")
    assert isinstance(reset[9], int)
    assert "Loading current source" in str(reset[10])
    assert reset[11] == []
    assert isinstance(reset[12], str) and reset[12].endswith(":reset")
    assert isinstance(reset[13], str) and reset[13]
    assert reset[14:] == (None, None, False, None, "")


def test_source_reset_revision_changes_when_shared_refresh_changes() -> None:
    payload = {"relative_path": "source-b", "selected_at": "revision-2"}

    first = source_reset_values(payload, 7)
    repeated = source_reset_values(payload, 7)
    refreshed = source_reset_values(payload, 8)

    assert first[13] == repeated[13]
    assert first[13] != refreshed[13]
    assert first[12] != refreshed[12]


def test_source_revision_authority_isolates_browser_sessions() -> None:
    authority = SourceRevisionAuthority()
    authority.ensure_session("tab-a")
    authority.ensure_session("tab-b")

    assert authority.authorize_grid("tab-a", None, "tab-a-grid")
    assert authority.authorize_grid("tab-b", None, "tab-b-grid")
    generation = authority.begin_reset("tab-a")
    assert authority.publish_reset("tab-a", generation, "tab-a-refresh")

    assert not authority.grid_is_current("tab-a", "tab-a-grid")
    assert authority.grid_is_current("tab-b", "tab-b-grid")
    assert authority.is_current("tab-a", "tab-a-refresh")
    assert authority.is_current("tab-b", None)


def test_source_revision_authority_rejects_out_of_order_reset_and_recovers() -> None:
    authority = SourceRevisionAuthority()
    older = authority.begin_reset("tab-a")
    newer = authority.begin_reset("tab-a")

    assert authority.publish_reset("tab-a", newer, "refresh-2")
    assert not authority.publish_reset("tab-a", older, "refresh-1")
    assert authority.is_current("tab-a", "refresh-2")
    assert authority.authorize_grid("tab-a", "refresh-2", "refresh-2-grid")
    assert authority.grid_is_current("tab-a", "refresh-2-grid")


def test_source_revision_authority_blocks_initial_render_through_reset_gap() -> None:
    authority = SourceRevisionAuthority()
    authority.ensure_session("tab-a")
    assert authority.is_current("tab-a", None)
    assert authority.authorize_grid("tab-a", None, "initial-grid")

    generation = authority.begin_reset("tab-a")

    assert not authority.is_current("tab-a", None)
    assert not authority.authorize_grid("tab-a", None, "delayed-initial-grid")
    assert not authority.grid_is_current("tab-a", "initial-grid")
    assert not authority.grid_is_current("tab-a", "delayed-initial-grid")

    assert authority.publish_reset("tab-a", generation, "refresh-1")
    assert authority.is_current("tab-a", "refresh-1")
    assert authority.authorize_grid("tab-a", "refresh-1", "refresh-1-grid")


def test_popout_event_is_revision_bound_and_current_source_contained(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image = source / "plate.png"
    image.write_bytes(b"image")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    sandbox = SandboxRoot.from_path(tmp_path)
    source_payload = source_payload_from_path(
        sandbox,
        source,
        source="manual",
    )
    assert source_payload is not None
    authority = TimelineRevisionAuthority()
    authorized = authorize_revision_candidate(
        authority,
        sandbox,
        {
            "session_id": "browser-1",
            "generation": 2,
            "revision": "grid-2",
        },
        source_payload,
    )
    assert authorized is not None

    valid = resolve_popout_event(
        sandbox,
        authority,
        {
            "session_id": "browser-1",
            "generation": 2,
            "revision": "grid-2",
            "sequence": 1,
            "token": encode_token("source/plate.png"),
        },
    )
    assert valid == {
        "session_id": "browser-1",
        "generation": 2,
        "revision": "grid-2",
        "sequence": 1,
        "token": encode_token("source/plate.png"),
        "label": "source/plate.png",
    }
    assert (
        resolve_popout_event(
            sandbox,
            authority,
            {
                "session_id": "browser-1",
                "generation": 1,
                "revision": "grid-1",
                "sequence": 2,
                "token": encode_token("source/plate.png"),
            },
        )
        is None
    )
    authority.retire("browser-1")
    assert (
        resolve_popout_event(
            sandbox,
            authority,
            {
                "session_id": "browser-1",
                "generation": 2,
                "revision": "grid-2",
                "sequence": 4,
                "token": encode_token("source/plate.png"),
            },
        )
        is None
    )
    assert (
        resolve_popout_event(
            sandbox,
            authority,
            {
                "session_id": "browser-1",
                "generation": 2,
                "revision": "grid-2",
                "sequence": 3,
                "token": encode_token("outside.png"),
            },
        )
        is None
    )


def test_revision_authority_rejects_delayed_older_callback(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    first_payload = source_payload_from_path(sandbox, first, source="manual")
    second_payload = source_payload_from_path(sandbox, second, source="manual")
    assert first_payload is not None and second_payload is not None
    authority = TimelineRevisionAuthority()

    newer = authorize_revision_candidate(
        authority,
        sandbox,
        {
            "session_id": "browser-1",
            "generation": 8,
            "revision": "revision-b",
        },
        second_payload,
    )
    delayed_older = authorize_revision_candidate(
        authority,
        sandbox,
        {
            "session_id": "browser-1",
            "generation": 7,
            "revision": "revision-a",
        },
        first_payload,
    )

    assert newer is not None
    assert delayed_older is None
    assert authority.current("browser-1", 8, "revision-b") is not None
    assert authority.current("browser-1", 7, "revision-a") is None
