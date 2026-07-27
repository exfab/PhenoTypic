"""Browse consumes the shared filesystem refresh revision without taking authority."""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any

import dash
import pytest
from dash.exceptions import PreventUpdate

from phenotypic.gui.browse import _ids as browse_ids
from phenotypic.gui.browse import _callbacks as browse_callbacks
from phenotypic.gui.browse._source_render import encode_token
from phenotypic.gui.browse._callbacks import (
    SourceRevisionAuthority,
    TimelineRevisionAuthority,
    authorize_current_revision_candidate,
    register_callbacks,
)
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import source_payload_from_path


def _callback_named(app: dash.Dash, name: str) -> tuple[Any, dict[str, Any]]:
    for metadata in app.callback_map.values():
        callback = metadata.get("callback")
        if callback is not None and callback.__wrapped__.__name__ == name:
            return callback.__wrapped__, metadata
    raise AssertionError(f"callback {name!r} was not registered")


@pytest.mark.parametrize("payload_version", [1, 2])
def test_refresh_rescans_exact_selected_v1_or_v2_source_without_rewriting(
    tmp_path: Path,
    payload_version: int,
) -> None:
    source = tmp_path / "selected"
    source.mkdir()
    (source / "first.png").write_bytes(b"first")
    sandbox = SandboxRoot.from_path(tmp_path)
    v2_payload = source_payload_from_path(sandbox, source, source="manual")
    assert v2_payload is not None
    payload: dict[str, object]
    if payload_version == 1:
        payload = {
            "version": 1,
            "abs_path": str(source.resolve()),
            "rel_path": "selected",
            "label": "selected",
            "validated": True,
        }
    else:
        payload = dict(v2_payload)
    original_payload = deepcopy(payload)

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_callbacks(app, sandbox)
    load_datasets, metadata = _callback_named(app, "_load_datasets")
    assert {item["id"] for item in metadata["inputs"]} == {
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_CLASSIFIER_CACHE_STORE,
    }

    before = load_datasets(payload, 3)
    assert before[0] == {".": ["first.png"]}

    (source / "second.tif").write_bytes(b"second")
    after = load_datasets(payload, 4)

    assert after[0] == {".": ["first.png", "second.tif"]}
    assert payload == original_payload


def test_timeline_reset_consumes_shared_refresh_revision(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_callbacks(app, sandbox)

    reset_timeline, metadata = _callback_named(
        app,
        "_reset_timeline_for_source",
    )

    assert {item["id"] for item in metadata["inputs"]} == {
        SHELL_SOURCE_IMAGE_ROOT_STORE,
        SHELL_CLASSIFIER_CACHE_STORE,
    }
    first = reset_timeline({"version": 1}, 10, "browser-1")
    refreshed = reset_timeline({"version": 1}, 11, "browser-1")
    assert first[13] != refreshed[13]
    assert refreshed[10].children == "Loading current source…"
    assert refreshed[11] == []
    assert refreshed[14] is None
    assert refreshed[15:] == (None, False, None, "")
    assert browse_ids.BROWSE_TL_SOURCE_REVISION in str(metadata["output"])


def test_out_of_order_timeline_reset_cannot_roll_back_and_next_reset_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "image.png").write_bytes(b"image")
    sandbox = SandboxRoot.from_path(tmp_path)
    source_payload = source_payload_from_path(
        sandbox,
        source,
        source="manual",
    )
    assert source_payload is not None
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_callbacks(app, sandbox)
    reset_timeline, _metadata = _callback_named(
        app,
        "_reset_timeline_for_source",
    )
    render_grid, _render_metadata = _callback_named(app, "_render_grid")
    authorize_revision, _authorize_metadata = _callback_named(
        app,
        "_authorize_revision",
    )
    approve_popout, _approve_metadata = _callback_named(
        app,
        "_approve_popout",
    )
    original_reset = browse_callbacks.source_reset_values
    older_started = threading.Event()
    release_older = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def _ordered_reset(payload: object, revision: object) -> tuple[object, ...]:
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            older_started.set()
            assert release_older.wait(timeout=5)
        return original_reset(payload, revision)

    monkeypatch.setattr(browse_callbacks, "source_reset_values", _ordered_reset)
    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(
            reset_timeline,
            {"relative_path": "older"},
            20,
            "browser-1",
        )
        assert older_started.wait(timeout=5)
        with pytest.raises(PreventUpdate):
            render_grid(
                "timeline",
                "folder",
                "exif",
                None,
                None,
                None,
                "",
                [],
                128,
                None,
                None,
                None,
                "browser-1",
            )
        newer = reset_timeline(
            source_payload,
            21,
            "browser-1",
        )
        with app.server.test_request_context("/"):
            _component, _warnings, new_grid_revision = render_grid(
                "timeline",
                "folder",
                "exif",
                None,
                None,
                None,
                "",
                [],
                128,
                newer[13],
                None,
                source_payload,
                "browser-1",
            )
        authorized = authorize_revision(
            {
                "session_id": "browser-1",
                "generation": 3,
                "revision": new_grid_revision,
            },
            source_payload,
        )
        assert authorized["revision"] == new_grid_revision
        release_older.set()
        with pytest.raises(PreventUpdate):
            older.result(timeout=5)

    assert newer[13] == original_reset(source_payload, 21)[13]
    approved = approve_popout(
        {
            "session_id": "browser-1",
            "generation": 3,
            "revision": new_grid_revision,
            "sequence": 1,
            "token": encode_token("source/image.png"),
        }
    )
    assert approved["label"] == "source/image.png"
    recovered = reset_timeline(
        source_payload,
        22,
        "browser-1",
    )
    assert recovered[13] == original_reset(
        source_payload,
        22,
    )[13]


def test_reset_between_source_check_and_timeline_authorize_cannot_resurrect_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    source_payload = source_payload_from_path(
        sandbox,
        source,
        source="manual",
    )
    assert source_payload is not None
    source_authority = SourceRevisionAuthority()
    timeline_authority = TimelineRevisionAuthority()
    source_authority.ensure_session("browser-1")
    assert source_authority.authorize_grid(
        "browser-1",
        None,
        "old-grid",
    )
    candidate = {
        "session_id": "browser-1",
        "generation": 2,
        "revision": "old-grid",
    }
    old_authorized = threading.Event()
    release_old = threading.Event()
    original_authorize = timeline_authority.authorize

    def _blocked_authorize(
        session_id: str,
        generation: int,
        revision: str,
        source_root: Path,
    ) -> bool:
        accepted = original_authorize(
            session_id,
            generation,
            revision,
            source_root,
        )
        if revision == "old-grid":
            old_authorized.set()
            assert release_old.wait(timeout=5)
        return accepted

    monkeypatch.setattr(timeline_authority, "authorize", _blocked_authorize)
    with ThreadPoolExecutor(max_workers=1) as executor:
        stale = executor.submit(
            authorize_current_revision_candidate,
            source_authority,
            timeline_authority,
            sandbox,
            candidate,
            source_payload,
        )
        assert old_authorized.wait(timeout=5)

        reset_generation = source_authority.begin_reset("browser-1")
        timeline_authority.retire("browser-1")
        assert source_authority.publish_reset(
            "browser-1",
            reset_generation,
            "refresh-1",
        )
        assert source_authority.authorize_grid(
            "browser-1",
            "refresh-1",
            "new-grid",
        )
        assert original_authorize(
            "browser-1",
            3,
            "new-grid",
            source,
        )
        release_old.set()
        assert stale.result(timeout=5) is None

    assert timeline_authority.current("browser-1", 2, "old-grid") is None
    assert timeline_authority.current("browser-1", 3, "new-grid") is not None
