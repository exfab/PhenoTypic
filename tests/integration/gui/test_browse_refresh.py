"""Browse consumes the shared filesystem refresh revision without taking authority."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import dash
import pytest

from phenotypic.gui.browse import _ids as browse_ids
from phenotypic.gui.browse._callbacks import register_callbacks
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
    first = reset_timeline({"version": 1}, 10)
    refreshed = reset_timeline({"version": 1}, 11)
    assert first[13] != refreshed[13]
    assert refreshed[10].children == "Loading current source…"
    assert refreshed[11] == []
    assert refreshed[14] is None
    assert browse_ids.BROWSE_TL_SOURCE_REVISION in str(metadata["output"])
