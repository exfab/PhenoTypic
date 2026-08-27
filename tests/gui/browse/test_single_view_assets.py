"""Static contracts for the dependency-free Browse browser controller."""

from pathlib import Path


_ASSET = (
    Path(__file__).parents[3]
    / "src/phenotypic/gui/browse/_assets/browse.js"
)


def test_single_viewer_uses_a_reused_named_handle() -> None:
    """Browse keeps ONE named OSD handle, reused rather than re-created.

    The popout half of this test went with the Timeline: ``ns.popoutViewer``
    and ``destroyViewer("popout")`` existed only for the Timeline pop-out
    modal, whose consumer was deleted in phase 2. Leaving those assertions
    would have pinned ~190 lines of dead code in place, and the next person
    to delete it would have "fixed" the test by restoring the code.

    ``ns.viewer`` stays asserted-absent: it is the pre-refactor unnamed
    handle, and its return would be a real regression.
    """
    script = _ASSET.read_text(encoding="utf-8")

    assert "ns.singleViewer" in script
    assert "viewer.open(url)" in script
    assert "ns.viewer" not in script


def test_keyboard_navigation_uses_scoped_dash_event_bridge() -> None:
    script = _ASSET.read_text(encoding="utf-8")

    assert 'key !== "j" && key !== "k"' in script
    assert "ev.shiftKey ? 10 : 1" in script
    assert "KEY_REPEAT_INTERVAL_MS = 80" in script
    assert "editingTarget(ev.target)" in script
    assert "visibleModal()" in script
    assert "singleViewVisible()" in script
    assert 'kind: "offset"' in script
    assert "set_props(NAV_EVENT_ID" in script


def test_progressive_preview_and_equal_dimension_restore_are_guarded() -> None:
    script = _ASSET.read_text(encoding="utf-8")

    assert "showPreview(payload, generation)" in script
    assert "hidePreview(generation)" in script
    assert "generation !== ns[generationProperty]" in script
    assert "equalDimensions(ns.singleState.dimensions, dimensions)" in script
    assert "keepPositionEnabled()" in script


def test_revision_asset_urls_are_scoped_to_tab_and_generation() -> None:
    script = _ASSET.read_text(encoding="utf-8")

    assert 'searchParams.set("client_id", clientId())' in script
    assert 'searchParams.set("generation", String(generation))' in script
    assert "new URL(rawUrl, window.location.href)" in script


def test_filmstrip_is_bounded_and_uses_cache_only_payload_urls() -> None:
    script = _ASSET.read_text(encoding="utf-8")

    assert "FILMSTRIP_RADIUS = 4" in script
    assert "boundedFilmstrip(items, activeValue)" in script
    assert "item.preview_url" in script
    assert 'kind: "select"' in script
    assert "fetch(" not in script
